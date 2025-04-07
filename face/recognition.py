import os
import cv2
import numpy as np

def load_face_database(face_db_path, face_app, logger):
    face_database = {}
    if not os.path.isdir(face_db_path):
        logger.error(f"Face database directory '{face_db_path}' does not exist.")
        return face_database
    for person in os.listdir(face_db_path):
        person_dir = os.path.join(face_db_path, person)
        if os.path.isdir(person_dir):
            embeddings = []
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"Failed to load {image_path}")
                    continue
                faces = face_app.get(img)
                if len(faces) == 0:
                    logger.warning(f"No face detected in {image_path}")
                    continue
                for face in faces:
                    embeddings.append(face.normed_embedding)
            if embeddings:
                face_database[person] = embeddings
    logger.info(f"Face database loaded. Persons found: {list(face_database.keys())}")
    return face_database

def recognize_face(embedding, face_database, threshold=0.35):
    identity = "Unknown"
    best_score = -1.0
    for person, embeddings in face_database.items():
        for db_emb in embeddings:
            score = np.dot(embedding, db_emb)
            if score > best_score:
                best_score = score
                if score > threshold:
                    identity = person
    return identity, best_score

def get_track_label(track_frames_numbers, track_bboxes, frames, start_frame_number, face_pad_scale, frame_width, frame_height, face_app, recognize_face_func):
    sample_indices = [0, 8, 16, 24]
    votes = []
    for i in sample_indices:
        if i < len(track_bboxes):
            bbox = track_bboxes[i]
            frame_index = track_frames_numbers[i] - start_frame_number
            if frame_index < 0 or frame_index >= len(frames):
                continue
            frame = frames[frame_index]
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            pad_w = int(face_pad_scale * w)
            pad_h = int(face_pad_scale * h)
            new_x1 = int(max(0, x1 - pad_w))
            new_y1 = int(max(0, y1 - pad_h))
            new_x2 = int(min(frame_width, x2 + pad_w))
            new_y2 = int(min(frame_height, y2 + pad_h))
            crop = frame[new_y1:new_y2, new_x1:new_x2]
            if crop.size == 0:
                continue
            faces = face_app.get(crop)
            if faces is not None and len(faces) > 0:
                for face in faces:
                    embedding = face.normed_embedding
                    identity, score = recognize_face_func(embedding)
                    if identity != "Unknown":
                        votes.append(identity)
                        break
    if votes:
        return max(set(votes), key=votes.count)
    else:
        return None
