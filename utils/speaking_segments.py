from scipy.signal import medfilt

def run_length_encode(seq):
    if len(seq) == 0:
        return []
    segments = []
    start = 0
    prev_val = seq[0]
    for i, val in enumerate(seq[1:], start=1):
        if val != prev_val:
            segments.append((start, i - 1, prev_val))
            start = i
            prev_val = val
    segments.append((start, len(seq) - 1, prev_val))
    return segments

def extract_speaking_segments(frame_numbers, scores, gap_threshold=59, min_segment_length=31):
    smoothed = medfilt(scores, kernel_size=gap_threshold)
    segments = run_length_encode(smoothed)
    merged_segments = []
    i = 0
    while i < len(segments):
        start_idx, end_idx, val = segments[i]
        if val == 1:
            j = i + 1
            while j + 1 < len(segments):
                gap_start, gap_end, gap_val = segments[j]
                next_start, next_end, next_val = segments[j+1]
                if gap_val == 0 and (gap_end - gap_start + 1) <= gap_threshold and next_val == 1:
                    end_idx = segments[j+1][1]
                    j += 2
                else:
                    break
            merged_segments.append((start_idx, end_idx, 1))
            i = j
        else:
            i += 1
    speaking_segments = []
    for seg in merged_segments:
        seg_start_idx, seg_end_idx, val = seg
        if (seg_end_idx - seg_start_idx + 1) >= min_segment_length:
            speaking_segments.append((frame_numbers[seg_start_idx]- 15, frame_numbers[seg_end_idx] + 15))
    return speaking_segments
