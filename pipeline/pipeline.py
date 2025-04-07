import os, queue, time, subprocess, pickle, logging, json
import numpy as np
import cv2
import threading
import torch
from datetime import datetime
import pyaudio, wave
from collections import defaultdict
from model.sort import Sort
from utils.logger import setup_logger
from model.lipsync import TalkNetInstance
from face.detection import detect_faces_s3fd
from face.recognition import load_face_database, recognize_face, get_track_label
from streamer.videostream import capture_video_wrapper, annotate_and_save_frames, display_frames
from streamer.audiostream import capture_audio_wrapper
from model.faceDetector import S3FD 
from insightface.app import FaceAnalysis
from utils.speaking_segments import extract_speaking_segments
from utils.tools import smoothenBoxes, interpolate_bboxes

class TrackGenerator:
    def __init__(self, args):
        self.args = args
        self.logger = setup_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.logger.info('Using GPU for processing inside trackGenerator')
        else:
            self.logger.info('Using CPU for processing inside trackGenerator')
        self.tmp_dir = args.tmp_dir
        self.face_masking = args.face_masking
        self.window_size = 30  # Fixed parameter
        self.max_unrecognized_frames = 15 
        self.kernel_size = args.kernel_size
        self.audio_rate = 16000 # Fixed parameter
        self.audio_channels = 1 # Fixed parameter
        self.audio_frames_per_buffer = 1024 # Fixed parameter
        self.face_pad_scale = args.face_pad_scale
        self.mask_blur_kernel = args.mask_blur_kernel
        self.max_frames = args.max_frames
        self.s3fd_conf_threshold = args.s3fd_conf_threshold

        self.talkNet = TalkNetInstance(args.talknet_model)
        self.processing_done = False
        self.lock = threading.Lock()
        self.all_frames_dir = os.path.join(self.tmp_dir, 'all_frames')
        self.annotated_frames_dir = os.path.join(self.tmp_dir, 'annotated_frames')
        os.makedirs(self.all_frames_dir, exist_ok=True)
        os.makedirs(self.annotated_frames_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.logger.error("Cannot open camera.")
            exit(1)
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.debug(f"Camera FPS: {self.video_fps}, Frame Size: {self.frame_width}x{self.frame_height}")
        self.logger.info("cv2 Video stream opened.")

        self.s3fd = S3FD(device=self.device)
        self.tracker = Sort(max_age=int(args.sort_max_age), min_hits=int(args.sort_min_hits), iou_threshold=float(args.sort_iou_threshold))
        self.face_app = FaceAnalysis(name=args.face_model_name)
        self.face_app.prepare(ctx_id=int(args.face_app_ctx_id))
        self.face_database = load_face_database(args.registered_faces, self.face_app, self.logger)
        self.p = pyaudio.PyAudio()
        self.audio_stream = self.p.open(format=args.audio_format,
                                  channels=self.audio_channels,
                                  rate=self.audio_rate,
                                  input=True,
                                  input_device_index=args.input_device_index,
                                  frames_per_buffer=self.audio_frames_per_buffer)
        self.logger.info("PyAudio stream opened.")
        self.processing_queue = queue.Queue()
        self.batch_queue = queue.Queue()
        self.current_tracks = defaultdict(lambda: {'sub_track_count': 1, 'label': "", 'frames': [], 'bboxes': [], 'bboxes2': defaultdict(list), 'scores': defaultdict(float)})
        self.processed_tracks = []

        self.processing_done_flag = [False]
        self.talknet_evaluations_done = False
        self.display_thread = threading.Thread(target=display_frames, args=(self.batch_queue, self.frame_width, self.frame_height, self.processing_done_flag, self.logger), daemon=True)
        self.display_thread.start()
        self.logger.debug("Display thread started.")
        self.processing_thread = threading.Thread(target=self.process_data, daemon=True)
        self.processing_thread.start()
        self.logger.debug("Lip Sync Processing thread started.")

        self.speaking_gap_threshold = args.speaking_gap_threshold
        self.speaking_min_frame_length = args.speaking_min_frame_length

    def process_track(self, track_id, track_frames_numbers, track_bboxes, curr_frame_number, frames, audio_data, detection_data, assigned_label):
        smoothed_bboxes = smoothenBoxes(track_bboxes, self.kernel_size, self.face_pad_scale, self.frame_width, self.frame_height, self.logger)
        start_frame_number_of_batch = curr_frame_number - self.window_size + 1
        track_frames = [frame.copy() for frame in frames[:self.window_size]]
        if not track_frames:
            self.logger.warning(f"No valid frames for Track {track_id}. Skipping TalkNet evaluation.")
            return
        audio_dir = os.path.join(self.tmp_dir, f'track_{track_id}_subtrack_{self.current_tracks[track_id]["sub_track_count"]}')
        os.makedirs(audio_dir, exist_ok=True)
        audio_segment_path = os.path.join(audio_dir, 'audio.wav')
        try:
            with wave.open(audio_segment_path, 'wb') as wf:
                wf.setnchannels(self.audio_channels)
                wf.setsampwidth(self.p.get_sample_size(self.args.audio_format))
                wf.setframerate(self.audio_rate)
                wf.writeframes(audio_data)
            self.logger.debug(f"Saved audio segment for Track {track_id} to {audio_segment_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save audio segment for Track {track_id}: {e}")
            audio_segment_path = None
        if audio_segment_path and os.path.exists(audio_segment_path):
            start_time_eval = datetime.now()
            cropped_frames_dir = os.path.join(audio_dir, 'cropped_frames')
            os.makedirs(cropped_frames_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_filepath = os.path.join(audio_dir, 'cropped_video.avi')
            out = cv2.VideoWriter(video_filepath, fourcc, self.video_fps, (224, 224))
            for idx, (frame_img, bbox) in enumerate(zip(track_frames, smoothed_bboxes)):
                x_min, y_min, x_max, y_max = map(int, bbox)
                cropped = frame_img[y_min:y_max, x_min:x_max]
                if cropped.size == 0:
                    self.logger.warning(f"Cropped frame {idx} has zero size for Track {track_id}.")
                    continue
                try:
                    cropped = cv2.resize(cropped, (224, 224))
                except Exception as e:
                    self.logger.error(f"Error resizing frame {idx} for Track {track_id}: {e}")
                    continue
                out.write(cropped)
                frame_filename = f'{idx + 1:06d}.jpg'
                cv2.imwrite(os.path.join(cropped_frames_dir, frame_filename), cropped)
            out.release()
            talkNet_results = self.talkNet.evaluate(start_frame_number_of_batch, frames_dir=cropped_frames_dir, audio_file=audio_segment_path)
            end_time_eval = datetime.now()
            if talkNet_results is not None:
                for i in range(len(track_frames_numbers)):
                    current_frame_number_sync = track_frames_numbers[i]
                    bbox = smoothed_bboxes[i]
                    conf_score = talkNet_results[i] if i < len(talkNet_results) else 0.0
                    with self.lock:
                        detection_data[current_frame_number_sync].append({
                            'track_number': track_id,
                            'bounding_box': bbox,
                            'frame_confidence': conf_score,
                            'label': assigned_label
                        })
                        self.current_tracks[track_id]['scores'][current_frame_number_sync] = conf_score
        else:
            self.logger.warning(f'Audio segment not found for Track {track_id}. Skipping TalkNet evaluation.')
        with self.lock:
            self.current_tracks[track_id]['sub_track_count'] += 1

    def process_data(self):
        while True:
            try:
                data = self.processing_queue.get(timeout=0.0)
            except queue.Empty:
                if self.processing_done:
                    self.logger.debug("Processing thread detected done flag. Exiting.")
                    print('***Processing thread detected done flag.***')
                    
                    break
                continue
            if data is None:
                self.logger.debug("Processing thread received sentinel. Exiting.")
                print(f'****Processing thread now received no data, hence all 1500 frames are recorded but maybe not processed completely****')
                break
            startTimeOneBatch = datetime.now()
            frames, audio_data, start_frame_number = data
            detection_data = defaultdict(list)
            tracks_to_process = []
            current_batch_tracked_objects = set()
            curr_frame_number = start_frame_number - 1
            for idx, frame in enumerate(frames):
                curr_frame_number += 1
                detections = detect_faces_s3fd(frame, self.s3fd, conf_threshold=self.s3fd_conf_threshold, scales=[0.25])
                dets = np.array(detections) if detections else np.empty((0,5))
                tracked_objects = self.tracker.update(dets)
                with self.lock:
                    for trk in tracked_objects:
                        x1, y1, x2, y2, track_id = trk
                        track_id = int(track_id)
                        current_batch_tracked_objects.add(track_id)
                        bbox = [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)]
                        self.current_tracks[track_id]['frames'].append(curr_frame_number)
                        self.current_tracks[track_id]['bboxes2'][curr_frame_number] = bbox
                        self.current_tracks[track_id]['bboxes'].append(bbox)
            for track_id in current_batch_tracked_objects:
                track_frames_numbers = self.current_tracks[track_id]['frames'][-self.window_size:]
                track_bboxes = self.current_tracks[track_id]['bboxes'][-self.window_size:]
                # filled_bboxes = interpolate_bboxes(track_frames_numbers, track_bboxes, start_frame_number, self.window_size, self.max_unrecognized_frames, self.logger)
                filled_bboxes = interpolate_bboxes(track_frames_numbers, track_bboxes, start_frame_number, self.window_size, self.max_unrecognized_frames, self.frame_width, self.frame_height,self.logger)
                if filled_bboxes is not None:
                    continuous_frames = list(range(start_frame_number, start_frame_number + self.window_size))
                    for i, frame_num in enumerate(continuous_frames):
                        if frame_num not in track_frames_numbers:
                            self.current_tracks[track_id]['frames'].append(frame_num)
                            self.current_tracks[track_id]['bboxes'].append(filled_bboxes[i])
                            self.current_tracks[track_id]['bboxes2'][frame_num] = filled_bboxes[i]
                    assigned_label = get_track_label(continuous_frames, filled_bboxes, frames, start_frame_number,
                                                     self.face_pad_scale, self.frame_width, self.frame_height,
                                                     self.face_app, lambda emb: recognize_face(emb, self.face_database))
                    if self.face_masking:
                        if assigned_label is not None:
                            self.current_tracks[track_id]['label'] = assigned_label
                            tracks_to_process.append((track_id, continuous_frames, filled_bboxes, assigned_label))
                        else:
                            self.current_tracks[track_id]['label'] = 'Unknown'
                            self.logger.info(f"Track {track_id} did not yield a recognized label; skipping TalkNet evaluation.")
                    elif not self.face_masking:
                        if assigned_label is not None:
                            self.current_tracks[track_id]['label'] = assigned_label
                            tracks_to_process.append((track_id, continuous_frames, filled_bboxes, assigned_label))
                        else:
                            self.current_tracks[track_id]['label'] = 'Unknown'
                            tracks_to_process.append((track_id, continuous_frames, filled_bboxes, assigned_label))
                else:
                    self.logger.info(f"Track {track_id}: insufficient detections for interpolation. Skipping TalkNet evaluation.")
            if self.face_masking:
                for track_id in current_batch_tracked_objects:
                    if 'label' not in self.current_tracks[track_id] or self.current_tracks[track_id].get('label', "Unknown") == "Unknown":
                        for frame_num in range(start_frame_number, curr_frame_number + 1):
                            bbox = self.current_tracks[track_id]['bboxes2'].get(frame_num)
                            if bbox:
                                frame_idx = frame_num - start_frame_number
                                if 0 <= frame_idx < len(frames):
                                    frame = frames[frame_idx]
                                    x1, y1, x2, y2 = map(int, bbox)
                                    w = x2 - x1
                                    h = y2 - y1
                                    pad_w = int(self.face_pad_scale * w)
                                    pad_h = int(self.face_pad_scale * h)
                                    new_x1 = int(max(0, x1 - pad_w))
                                    new_y1 = int(max(0, y1 - pad_h))
                                    new_x2 = int(min(self.frame_width, x2 + pad_w))
                                    new_y2 = int(min(self.frame_height, y2 + pad_h))
                                    roi = frame[new_y1:new_y2, new_x1:new_x2]
                                    if roi.size != 0:
                                        kernel_size = (self.mask_blur_kernel, self.mask_blur_kernel)
                                        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
                                        frame[new_y1:new_y2, new_x1:new_x2] = blurred_roi
                                    frames[frame_idx] = frame
            for idx, frame in enumerate(frames):
                frame_number = start_frame_number + idx
                frame_filename = os.path.join(self.all_frames_dir, f'frame_{frame_number:06d}.jpg')
                cv2.imwrite(frame_filename, frame)
            threads = []
            for track_id, track_frames_numbers, track_bboxes, assigned_label in tracks_to_process:
                thread = threading.Thread(target=self.process_track, args=(track_id, track_frames_numbers, track_bboxes, curr_frame_number, frames, audio_data, detection_data, assigned_label))
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
                    self.logger.warning(f"Frame number {frame_num} is out of bounds for current batch.")
                    continue
                detections = detection_data.get(frame_num, [])
                annotated_frame = annotate_and_save_frames(frame_num, frame_image, detections, self.annotated_frames_dir, self.logger)
                annotated_batch.append(annotated_frame)
            if annotated_batch:
                self.batch_queue.put(annotated_batch)
                self.logger.debug(f"Enqueued a batch of {len(annotated_batch)} frames.")
            start_frame_number = curr_frame_number + 1
            endTimeOneBatch = datetime.now()
            self.logger.info(f"Batch processed in {(endTimeOneBatch - startTimeOneBatch).total_seconds()} seconds.")
        self.talknet_evaluations_done = True

    def run(self):
        batchNumber = 0
        start_frame_number = 1
        audioCompiled = b''
        while not self.processing_done:
            batchNumber += 1
            audio_queue = queue.Queue()
            video_queue = queue.Queue()
            audio_thread = threading.Thread(target=capture_audio_wrapper, args=(self.audio_stream, self.window_size, self.audio_rate, self.video_fps, audio_queue, self.logger))
            video_thread = threading.Thread(target=capture_video_wrapper, args=(self.cap, self.window_size, batchNumber, video_queue, self.logger))
            audio_thread.start()
            video_thread.start()
            audio_thread.join()
            video_thread.join()
            try:
                audio_data = audio_queue.get_nowait()
            except queue.Empty:
                self.logger.error("Failed to retrieve audio data.")
                audio_data = None
            try:
                frames = video_queue.get_nowait()
            except queue.Empty:
                self.logger.error("Failed to retrieve video frames.")
                frames = []
            if not frames:
                self.logger.warning("No frames captured. Skipping batch.")
                continue
            if audio_data is None:
                self.logger.warning("No audio data captured. Skipping batch.")
                continue
            audioCompiled += audio_data
            try:
                self.processing_queue.put_nowait((frames, audio_data, start_frame_number))
            except queue.Full:
                self.logger.warning("Processing queue full. Dropping batch.")
                self.logger.info("Processing queue full. Dropping batch.")
                continue
            start_frame_number += len(frames)
            if start_frame_number >= self.max_frames:
                self.processing_done = True
                self.logger.info(f"Processing done as either {self.max_frames} reached or process was manually terminated")
                time.sleep(2)

        # if not self.batch_queue.empty():
        self.batch_queue.put(None)
        self.processing_thread.join()
        self.processing_done_flag[0] = True
        self.display_thread.join()
        print(f'******dispay thread has been joined*****')
        saving_track_dict = dict(self.current_tracks)
        track_info_pkl_path = os.path.join(self.tmp_dir, "trackInfo.pkl")
        track_info_txt_path = os.path.join(self.tmp_dir, "trackInfo.txt")
        self.logger.info(f"Tracks saved at {self.tmp_dir}")
        try:
            with open(track_info_pkl_path, 'wb') as pkl_file:
                pickle.dump(saving_track_dict, pkl_file)
            self.logger.debug(f"Saved track info to {track_info_pkl_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save track info pkl: {e}")
        try:
            with open(track_info_txt_path, 'w') as txt_file:
                for track_id, track_data in saving_track_dict.items():
                    txt_file.write(f"Track ID: {track_id}, Data: {track_data}\n")
            self.logger.debug(f"Saved track info to {track_info_txt_path}.")
        except Exception as e:
            self.logger.error(f"Failed to save track info txt: {e}")
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.p.terminate()
        self.logger.info("Audio stream closed.")
        self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Video capture released. Processing finished.")
        combined_audio_segment_path = os.path.join(self.tmp_dir, 'compiled_audio.wav')
        try:
            with wave.open(combined_audio_segment_path, 'wb') as wf:
                wf.setnchannels(self.audio_channels)
                wf.setsampwidth(self.p.get_sample_size(self.args.audio_format))
                wf.setframerate(self.audio_rate)
                wf.writeframes(audioCompiled)
            self.logger.debug(f"Saved combined audio to {combined_audio_segment_path}.")
            video_output_path = os.path.join(self.tmp_dir, "annotated_video.mp4")
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-framerate", "30",
                "-i", os.path.join(self.annotated_frames_dir, "annotated_frame_%06d.jpg"),
                "-i", combined_audio_segment_path,
                "-c:v", "libx264",
                "-c:a", "aac",
                "-shortest",
                video_output_path,
            ]
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.logger.info(f"Annotated video saved to {video_output_path}.")
        except Exception as e:
            self.logger.error(f"Failed to create annotated video: {e}")
        # --- Compile speaking segments ---
        speaking_results = {}
        for track_id, data in self.current_tracks.items():
            sorted_frames = sorted(data['scores'].keys())
            person_id = data['label']
            if not sorted_frames:
                continue
            binary_scores = [1 if data['scores'][frame] >= 0.30 else 0 for frame in sorted_frames]
            segments = extract_speaking_segments(sorted_frames, binary_scores, gap_threshold=self.speaking_gap_threshold, min_segment_length=self.speaking_min_frame_length)
            if person_id in speaking_results:
                speaking_results[person_id] = speaking_results[person_id] + segments
            else:
                speaking_results[person_id] = segments
            print(f"Track: {track_id}, Person Identity: {person_id} speaking segments: {segments}")
        try:
            with open(os.path.join(self.tmp_dir, "speaking_segments.json"), "w") as f:
                json.dump(speaking_results, f, indent=4)
            print("Speaking segments saved to speaking_segments.json")
        except Exception as e:
            self.logger.error(f"Failed to save speaking segments: {e}")



def main():
    from utils.args import parse_args
    args = parse_args()
    if not os.path.isfile(args.talknet_model):
        logging.getLogger('TrackGeneratorLogger').error(f"TalkNet model file {args.talknet_model} does not exist.")
        exit(1)
    if not (0.0 <= args.face_pad_scale <= 1.0):
        logging.getLogger('TrackGeneratorLogger').error("Face padding scale must be between 0 and 1.")
        exit(1)
    if args.kernel_size % 2 == 0 or args.kernel_size < 1:
        logging.getLogger('TrackGeneratorLogger').error("kernel_size must be a positive odd integer.")
        exit(1)
    track_generator = TrackGenerator(args)
    try:
        start_time_eval = datetime.now()
        track_generator.run()
        end_time_eval = datetime.now()
        elapsed = end_time_eval - start_time_eval
        logging.getLogger('TrackGeneratorLogger').info(f"Total processing time: {elapsed.total_seconds()} seconds.")
    except KeyboardInterrupt:
        logging.getLogger('TrackGeneratorLogger').warning("Processing interrupted by user.")
    except Exception as e:
        logging.getLogger('TrackGeneratorLogger').exception(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
