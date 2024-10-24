#!/usr/bin/python
# -*- coding: utf-8 -*-
# Optimized Version: Sequential Batch Processing with Single TalkNet Initialization
# Processes a saved video file, detects and tracks multiple faces in fixed-length segments,
# evaluates lip-sync using TalkNet, annotates frames with results, and saves them.

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
from moviepy.editor import VideoFileClip
import glob
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
import queue

# ==================== Logging Configuration ====================

def setup_logger(log_file='processing.log'):
    """
    Sets up the logger to log messages to both console and a file.

    Args:
        log_file (str): The file path for the log file.
    """
    logger = logging.getLogger('POCTrackGeneratorLogger')
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of log messages

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file, mode='w')  # Overwrite log file each run
    c_handler.setLevel(logging.INFO)  # Console handler set to INFO level
    f_handler.setLevel(logging.DEBUG)  # File handler set to DEBUG level

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

# Initialize logger
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
        # self.logger.info(f"Initializing TalkNetInstance with model at: {model_path}")
        
        # Initialize TalkNet
        self.talkNet = talkNet().cuda()
        self.loadParameters(model_path)
        self.talkNet.eval()
        # logger.info(f"TalkNet model loaded from {model_path}.")

    def evaluate(self, frames_dir, audio_file):
        """
        Evaluates the lip-sync using TalkNet on the provided frames and audio.

        Args:
            frames_dir (str): Directory containing the cropped face frames.
            audio_file (str): Path to the corresponding audio segment.

        Returns:
            list: List of confidence scores for each frame.
        """
        # Remove duplicates in durationSet
        durationSet = {1, 2, 3, 4, 5, 6}

        # ========== Load audio ==========
        sample_rate, audio = wavfile.read(audio_file)
        if sample_rate != 16000:
            self.logger.warning(f"Expected audio sample rate of 16000 Hz, but got {sample_rate} Hz.")
    
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
            # Ensure the cropping coordinates are within frame dimensions
            height, width = face.shape
            y_center, x_center = height // 2, width // 2
            half_size = 56  # Since 112/2 = 56
            y_min = max(y_center - half_size, 0)
            y_max = min(y_center + half_size, height)
            x_min = max(x_center - half_size, 0)
            x_max = min(x_center + half_size, width)
            face = face[y_min:y_max, x_min:x_max]
            videoFeature.append(face)

        videoFeature = np.array(videoFeature)

        expected_audio_len = (videoFeature.shape[0] * 100) // 25
        if audioFeature.shape[0] < expected_audio_len:
            padding_len = expected_audio_len - audioFeature.shape[0]
            # print(f'padding values at last because padding len found to be {padding_len}')
            last_values = np.tile(audioFeature[-1:], (padding_len, 1))
            audioFeature = np.vstack((audioFeature, last_values))
        length = min((audioFeature.shape[0] // 100), (videoFeature.shape[0] // 25))
        audioFeature = audioFeature[:int(length * 100), :]
        videoFeature = videoFeature[:int(length * 25), :, :]
        allScore = []  # Evaluation use TalkNet 
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    start_a = i * duration * 100
                    end_a = (i + 1) * duration * 100
                    start_v = i * duration * 25
                    end_v = (i + 1) * duration * 25

                    inputA = torch.FloatTensor(audioFeature[start_a:end_a, :]).unsqueeze(0).cuda()
                    inputV = torch.FloatTensor(videoFeature[start_v:end_v, :, :]).unsqueeze(0).cuda()
                    
                    embedA = self.talkNet.model.forward_audio_frontend(inputA)
                    embedV = self.talkNet.model.forward_visual_frontend(inputV)
                    embedA, embedV = self.talkNet.model.forward_cross_attention(embedA, embedV)
                    out = self.talkNet.model.forward_audio_visual_backend(embedA, embedV)
                    score = self.talkNet.lossAV.forward(out, labels=None)
                    # print(f'score values is {score}')
                    if isinstance(score, torch.Tensor):
                        score = score.cpu().numpy()
                    scores.extend(score)
            # print(f'confidence score for this batch is {scores}')
            allScore.append(scores)
        
        allScore = np.array(allScore)
        meanScores = np.mean(allScore, axis=0)
        roundedScores = np.round(meanScores, 1).astype(float)
        self.logger.debug("Aggregation of confidence scores completed.")

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
                # Handle 'module.' prefix if present
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
            # self.logger.info("All parameters loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load TalkNet parameters: {e}")

# ==================== POCTrackGenerator Class ====================

class POCTrackGenerator:
    def __init__(self, args):
        self.args = args
        self.tmp_dir = args.tmp_dir  # temp directory is actually the output directory

        # Define directories
        self.all_frames_dir = os.path.join(self.tmp_dir, 'all_frames')
        self.annotated_frames_dir = os.path.join(self.tmp_dir, 'annotated_frames')

        # Create directories if they don't exist
        os.makedirs(self.all_frames_dir, exist_ok=True)
        os.makedirs(self.annotated_frames_dir, exist_ok=True)

        # Initialize video capture first to get frame dimensions
        self.cap = cv2.VideoCapture(args.video_path)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video file located at: {args.video_path}")
            exit(1)

        # Get video properties
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25  # Default to 25 if unable to get FPS
        if self.video_fps != 25:
            logger.warning(f'Video frame rate is {self.video_fps}, which is not equal to 25 FPS')
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.debug(f"Video FPS: {self.video_fps}, Total Frames: {self.total_frames}, Frame Size: {self.frame_width}x{self.frame_height}")

        # Initialize shared batch queue for real-time visualization
        self.batch_queue = queue.Queue()
        logger.debug("Initialized batch_queue for real-time visualization.")

        # Initialize audio extraction before starting the display thread
        self.audio_path = os.path.join(self.tmp_dir, 'audio.wav')
        self.extract_audio(args.video_path, self.audio_path)

        # Initialize SORT tracker
        self.tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.5)

        # Initialize S3FD detector
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.info('Using GPU for processing')
        else:
            logger.info('Using CPU for processing')
        self.s3fd = S3FD(device='cuda')

        # Initialize TalkNetInstance once
        self.talkNet = TalkNetInstance(args.talknet_model)
        # logger.info(f"TalkNet model loaded.")

        # Initialize tracks data structures
        self.current_tracks = defaultdict(lambda: {'sub_track_count': 1, 'frames': [], 'bboxes': []})
        self.processed_tracks = []  # List to hold all processed tracks

        # Batch size
        self.batch_size = args.clipped_video_len  # Number of frames per TalkNet evaluation
        self.kernel_size = args.kernel_size
        self.crop_scale = args.crop_scale

        # Initialize processing_done flag
        self.processing_done = False

        # Start the display thread after all necessary attributes are initialized
        self.display_thread = threading.Thread(target=self.display_frames, daemon=True)
        self.display_thread.start()
        logger.debug("Display thread started.")

    def extract_audio(self, video_path, audio_path):
        """
        Extract audio from the video file using ffmpeg.
        """
        if not os.path.exists(audio_path):
            command = f"ffmpeg -y -i {video_path} -ac 1 -vn -acodec pcm_s16le -ar 16000 {audio_path}"
            # logger.info(f"Extracting audio to {audio_path}...")
            try:
                completed_process = subprocess.run(
                    command, shell=True, check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                # logger.info("Audio extracted successfully.")
                self.audio_available = True
                self.audio_clip = VideoFileClip(video_path).audio
            except subprocess.CalledProcessError as e:
                logger.error(f"ffmpeg failed: {e.stderr.decode().strip()}")
                self.audio_available = False
                self.audio_clip = None
        else:
            # logger.info(f"Audio file already exists at {audio_path}.")
            self.audio_available = True
            self.audio_clip = VideoFileClip(video_path).audio

    def detect_faces_s3fd(self, frame):
        """
        Detect faces in a frame using S3FD.

        Args:
            frame (np.array): The input video frame.

        Returns:
            List of bounding boxes [x_min, y_min, x_max, y_max, score].
        """
        # Convert the frame from BGR (OpenCV) to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        bboxes = self.s3fd.detect_faces(img, conf_th=0.9, scales=[0.25])

        detections = []
        if bboxes is not None:
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox[:-1]  # Just the bounding box coordinates without confidence score
                prob = bbox[-1]  # Extract confidence score
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
        # logger.info(f"Running TalkNet for Track {track_id}...")

        # Save frames to a temporary directory
        temp_dir = os.path.join(self.tmp_dir, f'temp_track_{track_id}_{self.current_tracks[track_id]["sub_track_count"]}')
        os.makedirs(temp_dir, exist_ok=True)

        cropped_frames_dir = os.path.join(temp_dir, 'cropped_frames')
        os.makedirs(cropped_frames_dir, exist_ok=True)

        video_filepath = os.path.join(temp_dir, 'cropped_video.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filepath, fourcc, self.video_fps, (224, 224))

        # Crop and save frames
        for idx, (frame, bbox) in enumerate(zip(frames, bboxes)):
            x_min, y_min, x_max, y_max = map(int, bbox)
            # Crop the face region
            cropped_frame = frame[y_min:y_max, x_min:x_max]

            # Handle boundary conditions
            if cropped_frame.size == 0:
                logger.warning(f"Cropped frame {idx} has zero size for Track {track_id}. Skipping.")
                continue

            # Resize to 224x224
            try:
                cropped_frame = cv2.resize(cropped_frame, (224, 224))
            except Exception as e:
                logger.error(f"Error resizing frame {idx} for Track {track_id}: {e}")
                continue

            # Write to video
            out.write(cropped_frame)

            # Save individual cropped frames
            frame_filename = f'{idx + 1:06d}.jpg'
            frame_filepath = os.path.join(cropped_frames_dir, frame_filename)
            cv2.imwrite(frame_filepath, cropped_frame)

        out.release()
        # logger.debug(f"Saved cropped frames and video for Track {track_id}.")

        # Run TalkNet evaluation using the pre-initialized instance
        try:
            conf_score = self.talkNet.evaluate(
                frames_dir=cropped_frames_dir,
                audio_file=audio_segment_path,
            )

            # Collect TalkNet scores
            talkNet_results = {
                'frame_confidences': conf_score.tolist(),
            }

            logger.debug(f'Number of confidence scores returned: {len(talkNet_results["frame_confidences"])}')
            # logger.info(f"TalkNet evaluation completed for Track {track_id}.")

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
            # No detections to annotate
            annotated_frame_path = os.path.join(self.annotated_frames_dir, f'annotated_frame_{frame_number:06d}.jpg')
            cv2.imwrite(annotated_frame_path, frame)
            logger.debug(f"No detections for frame {frame_number}. Saved unannotated frame.")
            return frame

        # Determine the maximum confidence score in current detections for dynamic scaling
        max_conf_score = max([det['frame_confidence'] for det in detections])
        max_conf_score = max(max_conf_score, 1)  # Avoid division by zero

        for det in detections:
            track_num = det['track_number']
            bbox = det['bounding_box']
            conf_score = det['frame_confidence']

            x_min, y_min, x_max, y_max = map(int, bbox)

            # Calculate confidence color (green to red) with dynamic scaling
            scaled_conf = max(min((conf_score / max_conf_score) * 255, 255), 0)
            red = int(255 - scaled_conf)
            green = int(scaled_conf)
            color = (0, green, red)  # BGR format

            # Draw bounding box with increased thickness
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 4)

            # Prepare text with increased font scale and thickness
            text = f"Track {track_num}, Conf {conf_score:.2f}"
            # Calculate text size to adjust positioning
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            text_position = (x_min, y_min - 10 if y_min - 10 > text_height else y_min + text_height + 10)
            cv2.putText(frame, text, text_position,
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Save annotated frame
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
        bboxes = np.array(bboxes)  # Shape: (num_frames, 4)

        # Calculate center coordinates
        x = (bboxes[:, 0] + bboxes[:, 2]) / 2  # Center x
        y = (bboxes[:, 1] + bboxes[:, 3]) / 2  # Center y

        # Calculate size (s) as half of the maximum dimension
        s = np.maximum(bboxes[:, 2] - bboxes[:, 0],
                      bboxes[:, 3] - bboxes[:, 1]) / 2  # Size of the bounding box

        # Apply median filter
        x_filtered = medfilt(x, kernel_size=kernel_size)
        y_filtered = medfilt(y, kernel_size=kernel_size)
        s_filtered = medfilt(s, kernel_size=kernel_size)

        # Reconstruct smoothed bounding boxes
        smoothed_bboxes = []
        for xf, yf, sf in zip(x_filtered, y_filtered, s_filtered):
            padded_s = sf * (1 + 2 * crop_scale)
            x1 = xf - padded_s
            y1 = yf - padded_s
            x2 = xf + padded_s
            y2 = yf + padded_s
            smoothed_bboxes.append([int(x1), int(y1), int(x2), int(y2)])

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
                # Wait for the next batch; timeout to periodically check for termination
                batch = self.batch_queue.get(timeout=1)
                if batch is None:
                    logger.debug("Display thread received sentinel. Exiting.")
                    break

                logger.debug(f"Display thread received a batch of {len(batch)} frames.")
                for frame in batch:
                    cv2.imshow('Annotated Video', frame)
                    if cv2.waitKey(int(1000 / self.video_fps)) & 0xFF == ord('q'):
                        # logger.info("User requested termination via 'q' key.")
                        return
                logger.debug("Completed displaying a batch of frames.")
            except queue.Empty:
                if self.processing_done:
                    logger.debug("Processing done and batch_queue is empty. Exiting display thread.")
                    break
                continue

        cv2.destroyAllWindows()
        logger.debug("Display window 'Annotated Video' closed.")

    def run(self):
        """
        Executes the batch processing pipeline:
        - Reads frames in fixed-length batches.
        - Detects and tracks faces.
        - Runs TalkNet evaluation.
        - Annotates frames with TalkNet results.
        - Saves annotated frames.
        """
        frame_count = 0
        batchNumber = 0
        start_frame_number = 1

        while True:
            batchNumber += 1
            end_frame_number = start_frame_number 
            frames = []
            bboxes = []

            # Read a batch of frames
            for _ in range(self.batch_size):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                end_frame_number += 1
                frames.append(frame)
                bboxes.append([])  # Placeholder for detections

            print(f'For batch Number: {batchNumber}, starting frame number is: {start_frame_number}, and ending frame number is: {end_frame_number-1}')

            if not frames:
                logger.info("No more frames to process.")
                break

            current_batch_tracked_objects = set()

            # Detect faces in each frame
            for idx, frame in enumerate(frames):
                detections = self.detect_faces_s3fd(frame)
                bboxes[idx] = detections

                current_frame_number = frame_count - self.batch_size + idx + 1
                print(f'Current frame number is {current_frame_number}')

                # Update tracker with detections
                dets = np.array(detections)
                if dets.size == 0:
                    dets = np.empty((0, 5))
                else:
                    dets = dets[dets[:, 4] >= self.args.detect_threshold]

                tracked_objects = self.tracker.update(dets)
                # print(f'tracked_objects dictionary value is {tracked_objects}')

                # Update current_tracks
                for trk in tracked_objects:
                    x1, y1, x2, y2, track_id = trk
                    track_id = int(track_id)
                    current_batch_tracked_objects.add(track_id)
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    # Initialize track if new (handled by defaultdict)
                    self.current_tracks[track_id]['frames'].append(current_frame_number)
                    self.current_tracks[track_id]['bboxes'].append(bbox)
                    # print(f'current track that is generated is {self.current_tracks}')

            # Collect detection data for annotation
            detection_data = defaultdict(list)     
            for track_id in current_batch_tracked_objects:
                # Check if track has enough frames for TalkNet evaluation
                if len(self.current_tracks[track_id]['frames']) >= self.batch_size:
                    # Extract frames and bboxes for the track
                    track_frames_numbers = self.current_tracks[track_id]['frames'][-self.batch_size:]
                    track_bboxes = self.current_tracks[track_id]['bboxes'][-self.batch_size:]

                    track_bboxes = self.smoothenBoxes(track_bboxes, self.kernel_size, self.crop_scale)

                    # Extract the actual frame images
                    start_frame_number_of_batch = frame_count - self.batch_size + 1
                    relative_indices = [i - start_frame_number_of_batch for i in track_frames_numbers]
                    # Ensure indices are within the current batch
                    relative_indices = [i for i in relative_indices if 0 <= i < len(frames)]
                    track_frames = [frames[i].copy() for i in relative_indices]

                    if not track_frames:
                        logger.warning(f"No valid frames found for Track {track_id}. Skipping TalkNet evaluation.")
                        continue

                    # Define audio segment path
                    start_time = (track_frames_numbers[0] - 1) / self.video_fps
                    end_time = track_frames_numbers[-1] / self.video_fps
                    logger.debug(f'Track {track_id} - Start time: {start_time}s, End time: {end_time}s')

                    audio_dir = os.path.join(self.tmp_dir, f'track_{track_id}_subtrack_{self.current_tracks[track_id]["sub_track_count"]}')
                    os.makedirs(audio_dir, exist_ok=True)
                    audio_segment_path = os.path.join(audio_dir, 'audio.wav')
                    logger.debug(f'Audio segment path: {audio_segment_path}')

                    audio_clip_path = self.audio_path
                    if self.audio_available:
                        try:
                            # Construct the ffmpeg command for audio clipping
                            command = (
                                "ffmpeg -y -i %s -ss %.3f -to %.3f -ar 16000 -ac 1 -f wav %s" % 
                                (audio_clip_path, start_time, end_time, audio_segment_path)
                            )
                            
                            # Run the command using subprocess.run with error handling
                            completed_process = subprocess.run(
                                command, shell=True, check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE
                            )
                            
                            logger.debug(f"Extracted audio segment for Track {track_id}.")
                            
                        except subprocess.CalledProcessError as e:
                            logger.error(f"ffmpeg failed for Track {track_id}: {e.stderr.decode().strip()}")
                            audio_segment_path = None
                    else:
                        audio_segment_path = None

                    # Run TalkNet evaluation
                    start_time_eval = datetime.now()
                    if audio_segment_path and os.path.exists(audio_segment_path):
                        talkNet_results = self.run_talknet_evaluation(track_id, track_frames, track_bboxes, audio_segment_path)
                    else:
                        talkNet_results = None
                        logger.warning(f'Audio segment not found for Track {track_id}. Skipping TalkNet evaluation.')
                    end_time_eval = datetime.now()
                    time_taken = end_time_eval - start_time_eval
                    time_taken_seconds = time_taken.total_seconds()
                    time_taken_milliseconds = time_taken.microseconds / 1000 
                    logger.info(f"Time taken to run one batch of TalkNet evaluation: {time_taken_seconds:.3f} seconds ({time_taken_milliseconds:.3f} milliseconds)")

                    # Annotate and collect frames with TalkNet results
                    if talkNet_results:
                        for i in range(len(track_frames_numbers)):
                            current_frame_number_sync = track_frames_numbers[i]
                            frame_image = track_frames[i].copy()
                            bbox = track_bboxes[i]
                            conf_score = talkNet_results['frame_confidences'][i] if i < len(talkNet_results['frame_confidences']) else 0.0

                            # Prepare detection data (excluding frame_image)
                            detection_data[current_frame_number_sync].append(
                                {
                                    'track_number': track_id,
                                    'bounding_box': bbox,
                                    'frame_confidence': conf_score
                                }
                            )

                    # Increment sub_track_count and reset frames if needed
                    self.current_tracks[track_id]['sub_track_count'] += 1
                    # Reset frames and bboxes for this track to prevent re-processing
                    self.current_tracks[track_id]['frames'] = []
                    self.current_tracks[track_id]['bboxes'] = []

            # Annotate all frames in the current batch with their respective detections
            annotated_batch = []
            for frame_num in range(start_frame_number, end_frame_number):
                # Calculate the index of the frame in the current batch
                frame_idx = frame_num - start_frame_number
                if 0 <= frame_idx < len(frames):
                    frame_image = frames[frame_idx]
                else:
                    logger.warning(f"Frame number {frame_num} is out of bounds for the current batch.")
                    continue

                # Retrieve detections for the current frame
                detections = detection_data.get(frame_num, [])
                
                # Annotate the frame
                annotated_frame = self.annotate_and_save_frames(frame_num, frame_image, detections)
                annotated_batch.append(annotated_frame)

            # Enqueue the annotated batch for display
            if annotated_batch:
                self.batch_queue.put(annotated_batch)
                logger.debug(f"Enqueued a batch of {len(annotated_batch)} frames.")
            
            # Update start_frame_number for the next batch
            start_frame_number = end_frame_number

        # ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Optimized Batch Processing with Single TalkNet Initialization")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--tmp_dir', type=str, default='tmp_poc', help='Temporary directory for processing')
    parser.add_argument('--detect_threshold', type=float, default=0.5, help='Detection confidence threshold (0-1)')
    parser.add_argument('--talknet_model', type=str, required=True, help='Path to TalkNet model file')  # Made required
    parser.add_argument('--clipped_video_len', type=int, default=50, help='Number of frames per TalkNet evaluation')
    parser.add_argument('--kernel_size', type=int, default=9, help='Kernel size for image smoothing')
    parser.add_argument('--crop_scale', type=float, default=0.25, help='Crop scale for face bounding box')
    parser.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of threads for ffmpeg processing')
    parser.add_argument('--pyframesPath', type=str, default='tmp_poc/all_frames', help='Path to the frames directory')
    parser.add_argument('--pyaviPath', type=str, default='tmp_poc', help='Path to save the video output')
    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.video_path):
        logger.error(f"Video file {args.video_path} does not exist.")
        exit(1)
    if not os.path.isfile(args.talknet_model):
        logger.error(f"TalkNet model file {args.talknet_model} does not exist.")
        exit(1)
    if not (0.0 < args.crop_scale < 1.0):
        logger.error("crop_scale must be between 0 and 1.")
        exit(1)
    if args.kernel_size % 2 == 0 or args.kernel_size < 1:
        logger.error("kernel_size must be a positive odd integer.")
        exit(1)

    # Initialize and run the POCTrackGenerator
    track_generator = POCTrackGenerator(args)
    try:
        start_time_eval = datetime.now()
        track_generator.run()
        end_time_eval = datetime.now()
        time_taken = end_time_eval - start_time_eval
        time_taken_seconds = time_taken.total_seconds()
        time_taken_milliseconds = time_taken.microseconds / 1000 
        # logger.info(f"Time taken to run complete code is: {time_taken_seconds:.3f} seconds ({time_taken_milliseconds:.3f} milliseconds)")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        # logger.info("Processing finished.")

        # Signal the display thread to terminate if it's still running
        if not track_generator.batch_queue.empty():
            track_generator.batch_queue.put(None)

        # Wait for the display thread to finish
        track_generator.display_thread.join()

        # Save current_tracks to pickle and text files
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

        # Release resources
        if track_generator.audio_available and hasattr(track_generator, 'audio_clip'):
            track_generator.audio_clip.close()
            logger.debug("Closed audio clip.")
        track_generator.cap.release()
        logger.debug("Released video capture resource.")

        # logger.info("Code execution completed. Please check the annotated frames and the display window.")

if __name__ == "__main__":
    main()