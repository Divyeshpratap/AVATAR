import cv2
import numpy as np

def capture_video(cap, frames_per_clip, batchNumber, logger):
    frames = []
    for i in range(frames_per_clip):
        ret, frame = cap.read()
        logger.info(f'Capturing frame {(batchNumber-1)*frames_per_clip + i}')
        if not ret:
            logger.error("Failed to read frame from camera.")
            break
        frames.append(frame)
    return frames

def capture_video_wrapper(cap, frames_per_clip, batchNumber, output_queue, logger):
    frames = capture_video(cap, frames_per_clip, batchNumber, logger)
    output_queue.put(frames)

def annotate_and_save_frames(frame_number, frame, detections, annotated_frames_dir, logger):
    if not detections:
        annotated_frame_path = f'{annotated_frames_dir}/annotated_frame_{frame_number:06d}.jpg'
        frame_number_text = f"Frame {frame_number:06d}"
        text_position = (10, frame.shape[0] - 10)
        cv2.putText(frame, frame_number_text, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
        cv2.imwrite(annotated_frame_path, frame)
        logger.debug(f"No detections for frame {frame_number}. Saved unannotated frame.")
        return frame
    for det in detections:
        track_num = det['track_number']
        bbox = det['bounding_box']
        conf_score = det['frame_confidence']
        person_identity = det['label']
        person_identity = person_identity if person_identity else 'Unknown'
        x_min, y_min, x_max, y_max = map(int, bbox)
        if conf_score < 0.25:
            green = int((conf_score / 0.4) * 255)
            red = 255
        else:
            green = 255
            red = int(255 * (1 - (conf_score - 0.4) / 0.6))
        color = (0, green, red)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 4)
        # text = f"T: {track_num}, P: {person_identity}, C: {conf_score:.2f}"
        text = f"{person_identity} {'is speaking' if conf_score >= 0.3 else ''}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        text_position = (x_min, y_min - 10 if y_min - 10 > text_height else y_min + text_height + 10)
        cv2.putText(frame, text, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        frame_number_text = f"Frame {frame_number:06d}"
        text_position = (10, frame.shape[0] - 10)
        cv2.putText(frame, frame_number_text, text_position,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)
    annotated_frame_path = f'{annotated_frames_dir}/annotated_frame_{frame_number:06d}.jpg'
    cv2.imwrite(annotated_frame_path, frame)
    logger.debug(f"Annotated and saved frame {frame_number}.")
    return frame

def display_frames(batch_queue, frame_width, frame_height, processing_done_flag, logger):
    cv2.namedWindow('Annotated Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Annotated Video', frame_width, frame_height)
    logger.debug("Display window 'Annotated Video' created and resized.")
    while True:
        try:
            batch = batch_queue.get(timeout=0.01)
            if batch is None:
                logger.debug("Display thread received sentinel. Exiting.")
                break
            logger.debug(f"Display thread received a batch of {len(batch)} frames.")
            for frame in batch:
                cv2.imshow('Annotated Video', frame)
                if cv2.waitKey(int(1000 / 45)) & 0xFF == ord('q'):
                    processing_done_flag[0] = True
                    return
            logger.debug("Completed displaying a batch of frames.")
        except Exception:
            if processing_done_flag[0]:
                logger.debug("Processing done and batch_queue is empty. Exiting display thread.")
                break
            continue
    cv2.destroyAllWindows()
    logger.debug("Display window 'Annotated Video' closed.")