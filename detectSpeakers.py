#!/usr/bin/python
# -*- coding: utf-8 -*-
# Optimized Version: Real-Time Batch Processing with Single TalkNet Initialization
# Processes live camera input, detects and tracks multiple faces in fixed-length segments,
# evaluates lip-sync using TalkNet, annotates frames with results, and displays them.
import queue
import torch
import numpy as np
import time
import argparse
import os
import cv2
import pickle
import json
from collections import defaultdict, deque
from scipy.signal import medfilt
from moviepy.editor import VideoFileClip, ImageSequenceClip, AudioFileClip, CompositeVideoClip
import glob
from pydub import AudioSegment
import subprocess
from sort import Sort
from scipy import signal
from scipy.io import wavfile
import math
import python_speech_features
from shutil import rmtree
from detectors import S3FD
from datetime import datetime
import logging
from talkNet import talkNet
import threading
from tqdm import tqdm
import pyaudio
import wave

# ==================== Logging Configuration ====================

def setup_logger(log_file='processing.log'):
    """
    Sets up the logger to log messages to both console and a file.

    Args:
        log_file (str): The file path for the log file.
    """
    logger = logging.getLogger('POCTrackGeneratorLogger')
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file, mode='w')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

logger = setup_logger()

# ==================== TalkNetInstance Class ====================

class TalkNetInstance:
    def __init__(self, model_path):
        """
        Initializes the TalkNetInstance by loading the TalkNet model.

        Args:
            model_path (str): Path to the pretrained TalkNet model file.
        """
        self.logger = logging.getLogger('TalkNetInstance')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.logger.info("CUDA is available. Using GPU for TalkNet.")
        else:
            self.logger.info("CUDA is not available. Using CPU for TalkNet.")

        self.talkNet = talkNet().to(self.device)
        self.loadParameters(model_path)
        self.talkNet.eval()
        self.logger.info(f"TalkNet model loaded from {model_path}.")

    def resample_video_frames(self, videoFeature, original_fps=30, target_fps=25):
        """
        Resamples video frames from original_fps to target_fps.

        Args:
            videoFeature (np.ndarray): Array of video frames with shape (num_frames, height, width).
            original_fps (int): Original frame rate of the video.
            target_fps (int): Desired frame rate after resampling.

        Returns:
            np.ndarray: Resampled video frames with shape (new_num_frames, height, width).
        """
        num_original_frames = videoFeature.shape[0]
        duration = num_original_frames / original_fps
        num_target_frames = int(np.round(duration * target_fps))

        if num_target_frames == 0:
            self.logger.error("Resampling resulted in zero frames. Check video duration and frame rate.")
            return np.array([])

        target_indices = np.linspace(0, num_original_frames, num=num_target_frames, endpoint=False)
        target_indices = np.floor(target_indices).astype(int)
        target_indices = np.clip(target_indices, 0, num_original_frames - 1)

        resampled_video = videoFeature[target_indices]
        self.logger.debug(f"Resampled video from {original_fps} fps to {target_fps} fps.")
        return resampled_video


    def evaluate(self, frames_dir, audio_file):
        """
        Evaluates the lip-sync using TalkNet on the provided frames and audio.

        Args:
            frames_dir (str): Directory containing the cropped face frames.
            audio_file (str): Path to the corresponding audio segment.

        Returns:
            list: List of confidence scores for each frame.
        """


        # ========== Load audio ==========
        _, audio = wavfile.read(audio_file)

        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)

        # ========== Load video frames ==========
        videoFeature = []
        flist = glob.glob(os.path.join(frames_dir, '*.jpg'))
        logger.debug(f'Frames directory: {frames_dir}')
        flist.sort()

        for fname in flist:
            frames = cv2.imread(fname)
            if frames is None:
                self.logger.warning(f"Failed to read frame: {fname}. Skipping.")
                continue
            face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            height, width = face.shape
            y_center, x_center = height // 2, width // 2
            half_size = 56
            y_min = max(y_center - half_size, 0)
            y_max = min(y_center + half_size, height)
            x_min = max(x_center - half_size, 0)
            x_max = min(x_center + half_size, width)
            face = face[y_min:y_max, x_min:x_max]
            videoFeature.append(face)

        videoFeature = np.array(videoFeature)

        videoFeature = self.resample_video_frames(videoFeature, original_fps=30, target_fps=25)

        expected_audio_len = (videoFeature.shape[0] * 100) // 25
        if audioFeature.shape[0] < expected_audio_len:
            padding_len = expected_audio_len - audioFeature.shape[0]
            print(f'padding values at last because padding len found to be {padding_len}')
            last_values = np.tile(audioFeature[-1:], (padding_len, 1))
            audioFeature = np.vstack((audioFeature, last_values))
        length = min((audioFeature.shape[0] // 100), (videoFeature.shape[0] // 25))
        if length != 1:
            print(f'INFO: length found should have been 1 but found as {length}')

        with torch.no_grad():

            inputA = torch.FloatTensor(audioFeature[0:100, :]).unsqueeze(0).to(self.device)
            inputV = torch.FloatTensor(videoFeature[0:25, :, :]).unsqueeze(0).to(self.device)

            embedA = self.talkNet.model.forward_audio_frontend(inputA)
            embedV = self.talkNet.model.forward_visual_frontend(inputV)
            embedA, embedV = self.talkNet.model.forward_cross_attention(embedA, embedV)
            out = self.talkNet.model.forward_audio_visual_backend(embedA, embedV)
            score = self.talkNet.lossAV.forward(out, labels=None)
            # print(f'score values is {score}')
            if isinstance(score, torch.Tensor):
                score = score.cpu().numpy()

        roundedScores = np.round(score, 2).astype(float)

        # print(f'roundedScores value is {roundedScores} and type is {type(roundedScores)}')

        return roundedScores
    
    def loadParameters(self, path):
        """
        Loads pretrained parameters into the TalkNet model.

        Args:
            path (str): Path to the pretrained model file.
        """
        self.logger.debug(f"Loading TalkNet parameters from: {path}")
        try:
            loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
            self_state = self.talkNet.state_dict()

            for name, param in loaded_state.items():
                if name not in self_state:
                    name = name.replace("module.", "")
                if name in self_state:
                    if self_state[name].shape == param.shape:
                        self_state[name].copy_(param)
                        self.logger.debug(f"Loaded parameter: {name}")
                    else:
                        self.logger.warning(f"Shape mismatch for parameter: {name}. Expected {self_state[name].shape}, got {param.shape}.")
                else:
                    self.logger.warning(f"Parameter {name} not found in the model.")
            self.logger.info("All parameters loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load TalkNet parameters: {e}")

# ==================== POCTrackGenerator Class ====================

class POCTrackGenerator:
    def __init__(self, args):
        self.args = args
        self.tmp_dir = args.tmp_dir

        self.all_frames_dir = os.path.join(self.tmp_dir, 'all_frames')
        self.annotated_frames_dir = os.path.join(self.tmp_dir, 'annotated_frames')

        os.makedirs(self.all_frames_dir, exist_ok=True)
        os.makedirs(self.annotated_frames_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera.")
            exit(1)

        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug(f"Camera FPS: {self.video_fps}, Frame Size: {self.frame_width}x{self.frame_height}")

        self.processing_queue = queue.Queue()
        self.batch_queue = queue.Queue()
        logger.debug("Initialized processing_queue and batch_queue for real-time visualization.")

        self.tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info('Using GPU for processing')
        else:
            logger.info('Using CPU for processing')
        self.s3fd = S3FD(device=self.device.type)

        self.talkNet = TalkNetInstance(args.talknet_model)

        self.current_tracks = defaultdict(lambda: {'sub_track_count': 1, 'frames': [], 'bboxes': []})
        self.processed_tracks = []

        self.batch_size = 30
        self.kernel_size = args.kernel_size
        self.crop_scale = args.crop_scale

        self.processing_done = False

        self.lock = threading.Lock()

        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()
        logger.debug("Display thread started.")

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=args.audio_format,
                                  channels=1,
                                  rate=args.audio_rate,
                                  input=True,
                                  input_device_index=3,
                                  frames_per_buffer=args.chunk_size)
        logger.info("Audio stream opened.")

        self.processing_thread = threading.Thread(target=self.process_data, daemon=True)
        self.processing_thread.start()
        logger.debug("Processing thread started.")

    def detect_faces_s3fd(self, frame):
        """
        Detect faces in a frame using S3FD.

        Args:
            frame (np.array): The input video frame.

        Returns:
            List of bounding boxes [x_min, y_min, x_max, y_max, score].
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        bboxes = self.s3fd.detect_faces(img, conf_th=0.7, scales=[0.25])

        detections = []
        if bboxes is not None:
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox[:-1]
                prob = bbox[-1]
                detections.append([x_min, y_min, x_max, y_max, prob])
        logger.debug(f"Detected {len(detections)} faces in current frame.")
        return detections

    def run_talknet_evaluation(self, track_id, frames, bboxes, audio_segment_path):
        """
        Run TalkNet evaluation on the given track.

        Args:
            track_id (int): Unique identifier for the track.
            frames (list): List of frames (as np.array) in the track.
            bboxes (list): List of bounding boxes for each frame.
            audio_segment_path (str): Path to the audio segment corresponding to the track.

        Returns:
            dict: TalkNet scores and other relevant data.
        """

        temp_dir = os.path.join(self.tmp_dir, f'temp_track_{track_id}_{self.current_tracks[track_id]["sub_track_count"]}')
        os.makedirs(temp_dir, exist_ok=True)

        cropped_frames_dir = os.path.join(temp_dir, 'cropped_frames')
        os.makedirs(cropped_frames_dir, exist_ok=True)

        video_filepath = os.path.join(temp_dir, 'cropped_video.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filepath, fourcc, self.video_fps, (224, 224))

        for idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
            x_min, y_min, x_max, y_max = map(int, bbox)
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            if cropped_frame.size == 0:
                logger.warning(f"Cropped frame {idx} has zero size for Track {track_id}. Skipping.")
                continue

            try:
                cropped_frame = cv2.resize(cropped_frame, (224, 224))
            except Exception as e:
                logger.error(f"Error resizing frame {idx} for Track {track_id}: {e}")
                continue

            out.write(cropped_frame)

            frame_filename = f'{idx + 1:06d}.jpg'
            frame_filepath = os.path.join(cropped_frames_dir, frame_filename)
            cv2.imwrite(frame_filepath, cropped_frame)

        out.release()

        try:
            conf_score = self.talkNet.evaluate(
                frames_dir=cropped_frames_dir,
                audio_file=audio_segment_path,
            )
            # print(f'talkNet conf_score is {conf_score}')
            # interpolation/ extrapolation to 30 fps because our camera records at 30 while talknet evaluates at 25
            conf_score = np.interp(
                np.linspace(0, len(conf_score) - 1, int(len(conf_score) * 30 / 25)),
                np.arange(len(conf_score)),
                conf_score
            )
            talkNet_results = {
                'frame_confidences': conf_score.tolist(),
            }
            return talkNet_results

        except Exception as e:
            logger.error(f"Error during TalkNet evaluation for Track {track_id}: {e}")
            return None

    def annotate_and_save_frames(self, frame_number, frame, detections):
        """
        Annotate the frame with bounding boxes and confidence scores.

        Args:
            frame_number (int): The current frame number.
            frame (np.array): The frame image.
            detections (list): List of detections with bounding boxes and scores.

        Returns:
            np.array: Annotated frame.
        """
        if not detections:
            annotated_frame_path = os.path.join(self.annotated_frames_dir, f'annotated_frame_{frame_number:06d}.jpg')
            frame_number_text = f"Frame {frame_number:06d}"
            text_position = (10, frame.shape[0] - 10)
            cv2.putText(frame, frame_number_text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
            cv2.imwrite(annotated_frame_path, frame)
            logger.debug(f"No detections for frame {frame_number}. Saved unannotated frame.")
            return frame

        max_conf_score = max([det['frame_confidence'] for det in detections])
        max_conf_score = max(max_conf_score, 1)

        for det in detections:
            track_num = det['track_number']
            bbox = det['bounding_box']
            conf_score = det['frame_confidence']

            x_min, y_min, x_max, y_max = map(int, bbox)

            scaled_conf = max(min((conf_score / max_conf_score) * 255, 255), 0)
            red = int(255 - scaled_conf)
            green = int(scaled_conf)
            color = (0, green, red)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 4)

            text = f"Track {track_num}, Conf {conf_score:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            text_position = (x_min, y_min - 10 if y_min - 10 > text_height else y_min + text_height + 10)
            cv2.putText(frame, text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            frame_number_text = f"Frame {frame_number:06d}"
            text_position = (10, frame.shape[0] - 10)
            cv2.putText(frame, frame_number_text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

        annotated_frame_path = os.path.join(self.annotated_frames_dir, f'annotated_frame_{frame_number:06d}.jpg')
        cv2.imwrite(annotated_frame_path, frame)
        logger.debug(f"Annotated and saved frame {frame_number}.")

        return frame

    def smoothenBoxes(self, bboxes, kernel_size=5, crop_scale=0.25):
        """
        Smooths a list of bounding boxes using a median filter.

        Parameters:
        - bboxes: List of bounding boxes, each represented as [x1, y1, x2, y2]
        - kernel_size: Size of the median filter kernel

        Returns:
        - smoothed_bboxes: List of smoothed bounding boxes
        """
        bboxes = np.array(bboxes)

        x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        y = (bboxes[:, 1] + bboxes[:, 3]) / 2

        s = np.maximum(bboxes[:, 2] - bboxes[:, 0],
                       bboxes[:, 3] - bboxes[:, 1]) / 2

        x_filtered = medfilt(x, kernel_size=kernel_size)
        y_filtered = medfilt(y, kernel_size=kernel_size)
        s_filtered = medfilt(s, kernel_size=kernel_size)

        smoothed_bboxes = []
        for xf, yf, sf in zip(x_filtered, y_filtered, s_filtered):
            padded_s = sf * (1 + 2 * crop_scale)
            x1 = xf - padded_s
            y1 = yf - padded_s
            x2 = xf + padded_s
            y2 = yf + padded_s

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, self.frame_width)
            y2 = min(y2, self.frame_height)
            smoothed_bboxes.append([x1, y1, x2, y2])

        logger.debug("Applied median filter to bounding boxes.")
        return smoothed_bboxes

    def display_frames(self):
        """
        Continuously displays batches of frames from the batch_queue in a cv2 window.
        """
        cv2.namedWindow('Annotated Video', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Annotated Video', self.frame_width, self.frame_height)
        logger.debug("Display window 'Annotated Video' created and resized.")

        while True:
            try:
                batch = self.batch_queue.get(timeout=0.01)
                if batch is None:
                    logger.debug("Display thread received sentinel. Exiting.")
                    break

                logger.debug(f"Display thread received a batch of {len(batch)} frames.")
                for frame in batch:
                    cv2.imshow('Annotated Video', frame)
                    if cv2.waitKey(int(1000 / 45)) & 0xFF == ord('q'):                        
                        self.processing_done = True
                        return
                logger.debug("Completed displaying a batch of frames.")
            except queue.Empty:
                if self.processing_done:
                    logger.debug("Processing done and batch_queue is empty. Exiting display thread.")
                    break
                continue

        cv2.destroyAllWindows()
        logger.debug("Display window 'Annotated Video' closed.")

    def capture_audio(self, stream, frames_per_clip, audio_rate, batchNumber):
        """
        Capture audio for a specific number of frames.
        Args:
            stream: pyaudio stream object.
            frames_per_clip: Number of frames to capture.
            audio_rate: Sampling rate.
        Returns:
            audio frames
        """
        
        seconds = frames_per_clip / self.video_fps
        frames = []
        chunk = 128
        total_frames = int(audio_rate / chunk * seconds)
        for _ in range(total_frames):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        audio_data = b''.join(frames)
        return audio_data

    def capture_video(self, cap, frames_per_clip, batchNumber):
        """
        Capture video frames for a specific number of frames.
        Args:
            cap: OpenCV VideoCapture object.
            frames_per_clip: Number of frames to capture.
        Returns:
            Frames captured
        """
        frames = []
        for i in range(frames_per_clip):
            ret, frame = cap.read()
            print(f'Current Frame number which is being captured is {(batchNumber-1)*30 + i}')
            if not ret:
                logging.error("Failed to read frame from camera.")
                break
            frames.append(frame)

        return frames

    def capture_audio_wrapper(self, stream, frames_per_clip, audio_rate, batchNumber, output_queue):
        audio_data = self.capture_audio(stream, frames_per_clip, audio_rate, batchNumber)
        output_queue.put(audio_data)

    def capture_video_wrapper(self, cap, frames_per_clip, batchNumber, output_queue):
        frames = self.capture_video(cap, frames_per_clip, batchNumber)
        output_queue.put(frames)


    def process_data(self):
        """
        Continuously processes captured audio and video data from the processing_queue.
        Utilizes threading to perform TalkNet evaluations for multiple tracks in parallel.
        """
        
        while True:
            try:
                data = self.processing_queue.get(timeout=0.0)
            except queue.Empty:
                if self.processing_done:
                    logger.debug("Processing thread detected processing_done flag. Exiting.")
                    break
                continue

            if data is None:
                logger.debug("Processing thread received sentinel. Exiting.")
                break
            startTimeOneBatch = datetime.now()
            frames, audio_data, start_frame_number = data
            # audioCompiled += audio_data # This is just temporary for troubleshooting
            bboxes = []
            current_batch_tracked_objects = set()
            curr_frame_number = start_frame_number - 1

            for idx, frame in enumerate(frames):
                bboxes.append([])
                curr_frame_number += 1
                frame_filename = os.path.join(self.all_frames_dir, f'frame_{curr_frame_number:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
                detections = self.detect_faces_s3fd(frame)
                bboxes[idx] = detections
                current_frame_number = curr_frame_number

                dets = np.array(detections)
                if dets.size == 0:
                    dets = np.empty((0, 5))
                else:
                    dets = dets[dets[:, 4] >= self.args.detect_threshold]

                tracked_objects = self.tracker.update(dets)

                with self.lock:
                    for trk in tracked_objects:
                        x1, y1, x2, y2, track_id = trk
                        track_id = int(track_id)
                        current_batch_tracked_objects.add(track_id)
                        bbox = [float(x1), float(y1), float(x2), float(y2)]

                        self.current_tracks[track_id]['frames'].append(curr_frame_number)
                        self.current_tracks[track_id]['bboxes'].append(bbox)

            detection_data = defaultdict(list)

            tracks_to_process = []
            with self.lock:
                for track_id in current_batch_tracked_objects:
                    if len(self.current_tracks[track_id]['frames']) >= self.batch_size:
                        track_frames_numbers = self.current_tracks[track_id]['frames'][-self.batch_size:]
                        track_bboxes = self.current_tracks[track_id]['bboxes'][-self.batch_size:]
                        tracks_to_process.append((track_id, track_frames_numbers, track_bboxes))

            def process_track(track_id, track_frames_numbers, track_bboxes):
                """
                Processes a single track: smoothens bounding boxes, extracts frames, saves audio,
                runs TalkNet evaluation, and updates detection_data.

                Args:
                    track_id (int): Unique identifier for the track.
                    track_frames_numbers (list): List of frame numbers for the track.
                    track_bboxes (list): List of bounding boxes for the track.
                """
                smoothed_bboxes = self.smoothenBoxes(track_bboxes, self.kernel_size, self.crop_scale)

                start_frame_number_of_batch = curr_frame_number - self.batch_size + 1
                relative_indices = [i - start_frame_number_of_batch for i in track_frames_numbers]
                relative_indices = [i for i in relative_indices if 0 <= i < len(frames)]
                track_frames = [frames[i].copy() for i in relative_indices]

                if not track_frames:
                    logger.warning(f"No valid frames found for Track {track_id}. Skipping TalkNet evaluation.")
                    return

                audio_dir = os.path.join(
                    self.tmp_dir,
                    f'track_{track_id}_subtrack_{self.current_tracks[track_id]["sub_track_count"]}'
                )
                os.makedirs(audio_dir, exist_ok=True)
                audio_segment_path = os.path.join(audio_dir, 'audio.wav')

                try:
                    with wave.open(audio_segment_path, 'wb') as wf:
                        wf.setnchannels(self.args.channels)
                        wf.setsampwidth(self.p.get_sample_size(self.args.audio_format))
                        wf.setframerate(self.args.audio_rate)
                        wf.writeframes(audio_data)
                    logger.debug(f"Saved audio segment for Track {track_id} to {audio_segment_path}.")
                except Exception as e:
                    logger.error(f"Failed to save audio segment for Track {track_id}: {e}")
                    audio_segment_path = None

                if audio_segment_path and os.path.exists(audio_segment_path):

                    start_time_eval = datetime.now()
                    talkNet_results = self.run_talknet_evaluation(
                        track_id, track_frames, smoothed_bboxes, audio_segment_path
                    )
                    end_time_eval = datetime.now()
                    time_taken = end_time_eval - start_time_eval
                    time_taken_seconds = time_taken.total_seconds()
                    time_taken_milliseconds = time_taken.microseconds / 1000
                    logger.info(f"Time taken to run one batch of TalkNet evaluation: {time_taken_seconds:.3f} seconds ({time_taken_milliseconds:.3f} milliseconds)")

                    if talkNet_results:
                        for i in range(len(track_frames_numbers)):
                            current_frame_number_sync = track_frames_numbers[i]
                            bbox = smoothed_bboxes[i]
                            conf_score = (
                                talkNet_results['frame_confidences'][i]
                                if i < len(talkNet_results['frame_confidences'])
                                else 0.0
                            )

                            with self.lock:
                                detection_data[current_frame_number_sync].append(
                                    {
                                        'track_number': track_id,
                                        'bounding_box': bbox,
                                        'frame_confidence': conf_score
                                    }
                                )
                else:
                    logger.warning(f'Audio segment not found for Track {track_id}. Skipping TalkNet evaluation.')

                with self.lock:
                    self.current_tracks[track_id]['sub_track_count'] += 1
                    self.current_tracks[track_id]['frames'] = []
                    self.current_tracks[track_id]['bboxes'] = []

            threads = []
            for track_id, track_frames_numbers, track_bboxes in tracks_to_process:
                thread = threading.Thread(
                    target=process_track,
                    args=(track_id, track_frames_numbers, track_bboxes)
                )
                threads.append(thread)
                thread.start()

 
            for thread in threads:
                thread.join()

            annotated_batch = []
            for frame_num in range(start_frame_number, curr_frame_number + 1):

                frame_idx = frame_num - start_frame_number
                if 0 <= frame_idx < len(frames):
                    frame_image = frames[frame_idx]
                else:
                    logger.warning(f"Frame number {frame_num} is out of bounds for the current batch.")
                    continue

                detections = detection_data.get(frame_num, [])

                annotated_frame = self.annotate_and_save_frames(frame_num, frame_image, detections)
                annotated_batch.append(annotated_frame)

            if annotated_batch:
                self.batch_queue.put(annotated_batch)
                logger.debug(f"Enqueued a batch of {len(annotated_batch)} frames.")

            start_frame_number = curr_frame_number + 1
            endTimeOneBatch = datetime.now()
            print(f'time taken to run one complete batch with {len(tracks_to_process)} faces is {(endTimeOneBatch - startTimeOneBatch).total_seconds()} seconds')



    def run(self):
        """
        Executes the batch processing pipeline:
        - Reads frames in fixed-length batches from the camera.
        - Detects and tracks faces.
        - Runs TalkNet evaluation.
        - Annotates frames with TalkNet results.
        - Saves annotated frames.
        """
        frame_count = 0
        batchNumber = 0
        start_frame_number = 1
        startTime = datetime.now()
        audioCompiled = b''
        while not self.processing_done:
            batchNumber += 1
            end_frame_number = start_frame_number

            audio_queue = queue.Queue()
            video_queue = queue.Queue()
            audio_thread = threading.Thread(
                target=self.capture_audio_wrapper,
                args=(self.stream, self.batch_size, self.args.audio_rate, batchNumber, audio_queue)
            )
            video_thread = threading.Thread(
                target=self.capture_video_wrapper,
                args=(self.cap, self.batch_size, batchNumber, video_queue)
            )

            audio_thread.start()
            video_thread.start()

            audio_thread.join()
            video_thread.join()
            
            try:
                audio_data = audio_queue.get_nowait()
            except queue.Empty:
                logger.error("Failed to retrieve audio data from audio_thread.")
                audio_data = None

            try:
                frames = video_queue.get_nowait()
            except queue.Empty:
                logger.error("Failed to retrieve frames from video_thread.")
                frames = []

            if not frames:
                logger.warning("No frames captured in this batch. Skipping processing.")
                start_frame_number = end_frame_number
                continue

            if audio_data is None:
                logger.warning("No audio data captured in this batch. Skipping processing.")
                start_frame_number = end_frame_number
                continue
            audioCompiled += audio_data

            try:
                self.processing_queue.put_nowait((frames, audio_data, start_frame_number))
            except queue.Full:
                logger.warning("Processing queue is full. Dropping this batch.")
                continue

            start_frame_number += len(frames)
            if start_frame_number >= 3000:
                self.processing_done = True
                time.sleep(2)
                print(f'*********Creating compiled audio******************')
                combined_audio_dir = self.tmp_dir
                combined_audio_segment_path = os.path.join(combined_audio_dir, 'compiled_audio.wav')

                try:
                    with wave.open(combined_audio_segment_path, 'wb') as wf:
                        wf.setnchannels(self.args.channels)
                        wf.setsampwidth(self.p.get_sample_size(self.args.audio_format))
                        wf.setframerate(self.args.audio_rate)
                        wf.writeframes(audioCompiled)
                    logger.debug(f"Saved combined audio segment to {combined_audio_segment_path}.")

                    print(f'Combining audio and annotated frames to generate video')
                    annotated_frames_dir = os.path.join(self.tmp_dir, 'annotated_frames')
                    video_output_path = os.path.join(self.tmp_dir, "annotated_video.mp4")
                    
                    ffmpeg_command = [
                        "ffmpeg",
                        "-y",
                        "-framerate", "30",
                        "-i", os.path.join(annotated_frames_dir, "annotated_frame_%06d.jpg"),
                        "-i", combined_audio_segment_path,
                        "-c:v", "libx264",
                        "-c:a", "aac",
                        "-shortest",
                        video_output_path,
                    ]                
                    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    

                    logger.info(f"Annotated video with audio saved to {video_output_path}.")
                except Exception as e:
                    logger.error(f"Failed to create annotated video: {e}")
    # ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Optimized Batch Processing with Single TalkNet Initialization")
    parser.add_argument('--talknet_model', type=str, required=True, help='Path to TalkNet model file')  # Made required
    parser.add_argument('--tmp_dir', type=str, default='tmp_poc', help='Temporary directory for processing')
    parser.add_argument('--detect_threshold', type=float, default=0.5, help='Detection confidence threshold (0-1)')
    parser.add_argument('--kernel_size', type=int, default=9, help='Kernel size for image smoothing')
    parser.add_argument('--crop_scale', type=float, default=0.25, help='Crop scale for face bounding box')
    parser.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of threads for ffmpeg processing')
    parser.add_argument('--pyframesPath', type=str, default='tmp_poc/all_frames', help='Path to the frames directory')
    parser.add_argument('--pyaviPath', type=str, default='tmp_poc', help='Path to save the video output')

    # Audio capturing arguments
    parser.add_argument('--audio_format', type=int, default=pyaudio.paInt16, help='Audio format')
    parser.add_argument('--channels', type=int, default=1, help='Number of audio channels')
    parser.add_argument('--audio_rate', type=int, default=16000, help='Audio sampling rate')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Audio chunk size for buffering')

    args = parser.parse_args()

    if not os.path.isfile(args.talknet_model):
        logger.error(f"TalkNet model file {args.talknet_model} does not exist.")
        exit(1)
    if not (0.0 < args.crop_scale < 1.0):
        logger.error("crop_scale must be between 0 and 1.")
        exit(1)
    if args.kernel_size % 2 == 0 or args.kernel_size < 1:
        logger.error("kernel_size must be a positive odd integer.")
        exit(1)

    track_generator = POCTrackGenerator(args)
    try:
        start_time_eval = datetime.now()
        track_generator.run()
        end_time_eval = datetime.now()
        time_taken = end_time_eval - start_time_eval
        time_taken_seconds = time_taken.total_seconds()
        time_taken_milliseconds = time_taken.microseconds / 1000
        logger.info(f"Time taken to run complete code is: {time_taken_seconds:.3f} seconds ({time_taken_milliseconds:.3f} milliseconds)")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        logger.info("Processing finished.")

        if not track_generator.batch_queue.empty():
            track_generator.batch_queue.put(None)

        # Waiting for display thread to finish
        track_generator.display_thread.join()

        saving_track_dict = dict(track_generator.current_tracks)
        track_info_pkl_path = os.path.join(track_generator.tmp_dir, "trackInfo.pkl")
        track_info_txt_path = os.path.join(track_generator.tmp_dir, "trackInfo.txt")

        try:
            with open(track_info_pkl_path, 'wb') as pkl_file:
                pickle.dump(saving_track_dict, pkl_file)
            logger.debug(f"Saved track information to {track_info_pkl_path}.")
        except Exception as e:
            logger.error(f"Failed to save trackInfo.pkl: {e}")

        try:
            with open(track_info_txt_path, 'w') as txt_file:
                for track_id, track_data in saving_track_dict.items():
                    txt_file.write(f"Track ID: {track_id}, Data: {track_data}\n")
            logger.debug(f"Saved track information to {track_info_txt_path}.")
        except Exception as e:
            logger.error(f"Failed to save trackInfo.txt: {e}")

        track_generator.stream.stop_stream()
        track_generator.stream.close()
        track_generator.p.terminate()
        logger.info("Audio stream closed.")

        # Releasing resources
        track_generator.cap.release()
        logger.debug("Released video capture resource.")

        cv2.destroyAllWindows()
        logger.info("Code execution completed. Please check the annotated frames and the display window.")

if __name__ == "__main__":
    main()