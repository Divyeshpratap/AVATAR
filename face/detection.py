import cv2
import numpy as np

def detect_faces_s3fd(frame, s3fd, conf_threshold=0.55, scales=[0.25]):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes = s3fd.detect_faces(img, conf_th=conf_threshold, scales=scales)
    detections = []
    if bboxes is not None:
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox[:-1]
            prob = bbox[-1]
            detections.append([x_min, y_min, x_max, y_max, prob])
    return detections
