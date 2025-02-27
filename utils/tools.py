import numpy as np
from scipy.signal import medfilt

def smoothenBoxes(bboxes, kernel_size, crop_scale, frame_width, frame_height, logger):
    bboxes_arr = np.array(bboxes)
    x = (bboxes_arr[:, 0] + bboxes_arr[:, 2]) / 2
    y = (bboxes_arr[:, 1] + bboxes_arr[:, 3]) / 2
    s = np.maximum(bboxes_arr[:, 2] - bboxes_arr[:, 0],
                   bboxes_arr[:, 3] - bboxes_arr[:, 1]) / 2
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
        x2 = min(x2, frame_width)
        y2 = min(y2, frame_height)
        smoothed_bboxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)])
    logger.debug("Applied median filter to bounding boxes.")
    return smoothed_bboxes

def interpolate_bboxes(track_frames_numbers, track_bboxes, batch_start, batch_size, window_size, max_unrecognized_frames, logger):
    filled = [None] * batch_size
    for tn, bbox in zip(track_frames_numbers, track_bboxes):
        idx = tn - batch_start
        if 0 <= idx < batch_size:
            filled[idx] = bbox
    first_idx = next((i for i, v in enumerate(filled) if v is not None), None)
    last_idx = next((i for i, v in enumerate(reversed(filled)) if v is not None), None)
    if first_idx is None or last_idx is None:
        return None
    last_idx = batch_size - 1 - last_idx
    if first_idx > 0:
        contiguous_from_end = 0
        for i in range(batch_size - 1, -1, -1):
            if filled[i] is not None:
                contiguous_from_end += 1
            else:
                break
        if contiguous_from_end >= window_size - max_unrecognized_frames:
            for i in range(0, first_idx):
                filled[i] = filled[first_idx]
        else:
            return None
    if last_idx < batch_size - 1:
        contiguous_from_start = 0
        for i in range(0, batch_size):
            if filled[i] is not None:
                contiguous_from_start += 1
            else:
                break
        if contiguous_from_start >= window_size - max_unrecognized_frames:
            for i in range(last_idx + 1, batch_size):
                filled[i] = filled[last_idx]
        else:
            return None
    i = 0
    while i < batch_size:
        if filled[i] is None:
            gap_start = i - 1
            j = i
            while j < batch_size and filled[j] is None:
                j += 1
            gap_end = j
            gap_length = gap_end - i
            if gap_length > max_unrecognized_frames:
                return None
            if gap_start < 0 or gap_end >= batch_size:
                return None
            bbox_before = np.array(filled[gap_start])
            bbox_after = np.array(filled[gap_end])
            avg_bbox = ((bbox_before + bbox_after) / 2).tolist()
            for k in range(i, gap_end):
                filled[k] = avg_bbox
            i = gap_end
        else:
            i += 1
    if any(v is None for v in filled):
        return None
    return filled
