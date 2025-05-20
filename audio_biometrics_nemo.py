#!/usr/bin/env python3
"""
Complete pipeline
=================
input  : <folder>/video_name
voices : <folder>/person_name/*.wav|*.mp3   (enrolment utterances)
output :
    <folder>/predicted_segments.txt   (raw diarizer output)
    <folder>/timeline_labeled.txt     (after biometrics)
    <folder>/annotated_video.mp4      (video with names on screen)

Requires
--------
ffmpeg (CLI) • torch • nemo_toolkit[asr] • pandas • opencv-python
GPU mimimum 16 GB
"""
import logging
import os
import warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"      # HF transformers
warnings.filterwarnings("ignore")                  # hide Python warnings

logging.basicConfig(level=logging.ERROR)           # root logger
logging.getLogger("nemo_logger").setLevel(logging.ERROR)  
import itertools
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from nemo.collections.asr.models import (
    EncDecSpeakerLabelModel,
    SortformerEncLabelModel,
)

# ------------------------------------------------------------------ #
# 1.  CLI args
# ------------------------------------------------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="folder where video is located and where all processing will be done")
ap.add_argument("--video_name", required=True, help="name of the video along with extension")
ap.add_argument("--sim_threshold", type = float, default=0.5, help="Threshold for speaker recognition between 0 and 1")
ap.add_argument("--min_clean_segment", type = float, default=1, help="Minimum clean segment length in audio to perform ASR")
args = ap.parse_args()

# --------------------------------------------------------------------------- #
# Parameters for audio diarization and speaker recogntion
# --------------------------------------------------------------------------- #
MIN_CLEAN_SEC = float(args.min_clean_segment)      # minimal length for evidence segment
SIM_THRESHOLD = float(args.sim_threshold)       # ≥ → accept, else "unknown"
EMBED_MODEL   = "titanet_small"  # any NeMo speaker-verification model


# --------------------------------------------------------------------------- #
# 0.  Generic helpers
# --------------------------------------------------------------------------- #
def ffmpeg_extract(src: Path, dst: Path, start: float, dur: float):
    """lossless PCM 16-kHz mono extraction"""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()


# --------------------------------------------------------------------------- #
# 1.  Segment / timeline utilities
# --------------------------------------------------------------------------- #
def parse_segment_lines(lines: List[str]) -> pd.DataFrame:
    starts, ends, spks = [], [], []
    for ln in lines:
        ln = ln.replace(",", " ")
        p = ln.strip().split()
        if len(p) < 3:
            continue
        starts.append(float(p[0]))
        ends.append(float(p[1]))
        spks.append(p[2])
    return pd.DataFrame({"Start": starts, "End": ends, "Speaker": spks})


def dataframe_to_lines(df: pd.DataFrame) -> List[str]:
    return [f"{r.Start:.3f} {r.End:.3f} {r.Speaker}" for r in df.itertuples()]


def build_intervals(df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    m: Dict[str, List[Tuple[float, float]]] = {}
    for _, r in df.iterrows():
        m.setdefault(r["Speaker"], []).append((r["Start"], r["End"]))
    return m


def non_overlapping_segments(
    target_ivl: List[Tuple[float, float]],
    other_ivls: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Return sub-intervals of target that do not overlap *any* in other_ivls."""
    clean = []
    for s, e in target_ivl:
        # subtract any overlap with every 'other'
        segments = [(s, e)]
        for os, oe in other_ivls:
            new = []
            for xs, xe in segments:
                if oe <= xs or os >= xe:       # no overlap
                    new.append((xs, xe))
                else:  # split
                    if xs < os:
                        new.append((xs, os))
                    if oe < xe:
                        new.append((oe, xe))
            segments = new
        clean.extend(segments)
    return clean


# --------------------------------------------------------------------------- #
# 2.  Video annotation  (unchanged except label list)
# --------------------------------------------------------------------------- #
def annotate_video(
    video_in: Path,
    segment_lines: List[str],
    video_out: Path,
    font_scale: float = 1.0,
    thick: int = 2,
):
    df = parse_segment_lines(segment_lines)
    intervals = build_intervals(df)
    speakers = sorted(intervals)

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        sys.exit(f"❌  Cannot open {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ff_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-i",
        str(video_in),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-movflags",
        "+faststart",
        str(video_out),
    ]
    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)

    pad = 10
    line_h = int(30 * font_scale) + 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    header = "Audio Diarization and Biometrics"
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps

        # ─── header ──────────────────────────────────────────────
        header_y = H - pad - line_h              # one row above first speaker
        cv2.putText(
            frame,
            header,
            (10, header_y),
            font,
            font_scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )

        for i, spk in enumerate(reversed(speakers)):
            y = H - pad - i * line_h
            color = (0, 255, 0) if any(s <= t < e for s, e in intervals[spk]) else (0, 0, 255)
            cv2.putText(frame, spk, (10, y), font, font_scale, color, thick, cv2.LINE_AA)

        proc.stdin.write(frame.tobytes())
        frame_idx += 1

    cap.release()
    proc.stdin.close()
    proc.wait()
    print(f"✅  Annotated video saved → {video_out}")


# --------------------------------------------------------------------------- #
# 3.  Main pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(folder: str, video_name):
    folder = Path(folder).expanduser().resolve()
    video_path = folder / video_name
    audio_path = folder / "audio.wav"
    diar_segments_txt = folder / "predicted_segments.txt"
    labeled_txt = folder / "timeline_labeled.txt"
    out_video = folder / "annotated_video.mp4"
    voice_dir = Path("voice_samples")

    if not video_path.exists():
        sys.exit(f"❌  Video File: {video_path} not found")
    if not voice_dir.exists():
        sys.exit(f"❌  {voice_dir} not found (enrolment samples)")

    # 3.1 extract full audio
    print("🔊  Extracting full audio …")
    ffmpeg_extract(video_path, audio_path, 0.0, 1e9)

    # 3.2 diarization
    print("🤖  Diarizing …")
    diarizer = SortformerEncLabelModel.restore_from(
        restore_path=str(
            Path.home()
            / ".cache/huggingface/hub/models--nvidia--diar_sortformer_4spk-v1"
            / "snapshots"
            / "4cb5954e59a1a6527e6ec061a0568b61efa8babd"
            / "diar_sortformer_4spk-v1.nemo"
        ),
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        strict=False,
    )
    diarizer.eval()
    raw_segments: List[str] = diarizer.diarize(audio=str(audio_path), batch_size=1)
    raw_segments = raw_segments[0]
    diar_segments_txt.write_text("\n".join(raw_segments))
    print(f"📄  Raw segments → {diar_segments_txt}")

    # 3.3 choose clean evidence for each diarized speaker
    print("🎙️  Selecting clean evidence per diarized stream …")
    timeline_df = parse_segment_lines(raw_segments)
    intervals_by_spk = build_intervals(timeline_df)

    clean_samples: Dict[str, Tuple[float, float]] = {}
    for spk, ivl in intervals_by_spk.items():
        others = list(
            itertools.chain.from_iterable(
                v for k, v in intervals_by_spk.items() if k != spk
            )
        )
        candidates = [
            (s, e) for s, e in non_overlapping_segments(ivl, others) if (e - s) >= MIN_CLEAN_SEC
        ]
        if not candidates:
            print(f"⚠️  No clean ≥{MIN_CLEAN_SEC}s segment for {spk}")
        else:
            # pick the longest
            clean_samples[spk] = max(candidates, key=lambda x: x[1] - x[0])

    # 3.4 load speaker-embedding model
    print("🔬  Loading speaker-embedding model …")
    spk_model: EncDecSpeakerLabelModel = EncDecSpeakerLabelModel.from_pretrained(
        model_name=EMBED_MODEL, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    spk_model.eval()

    # 3.5 embed enrolment voices
    print("📑  Embedding enrolment samples …")
    enrol_embeddings = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for audio_file in voice_dir.iterdir():
            if audio_file.suffix.lower() not in {".wav", ".mp3"}:
                continue
            wav = tmp / f"{audio_file.stem}.wav"
            ffmpeg_extract(audio_file, wav, 0.0, 1e9)
            emb = spk_model.get_embedding(str(wav)).cpu()
            enrol_embeddings[audio_file.stem] = emb

    # 3.6 embed clean samples & match
    print("🔍  Matching diarized speakers with enrolment …")
    mapping: Dict[str, str] = {}    # diarized_stream -> final label
    unknown_counter = 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for spk, (s, e) in clean_samples.items():
            seg_wav = tmp / f"{spk}.wav"
            ffmpeg_extract(video_path, seg_wav, s, e - s)
            seg_emb = spk_model.get_embedding(str(seg_wav)).cpu()

            best_name = None
            best_score = -1.0
            for name, ref_emb in enrol_embeddings.items():
                score = cosine(seg_emb, ref_emb)
                print(f'******score value with name {name} is {score}********')
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= SIM_THRESHOLD:
                mapping[spk] = best_name
                print(f"✅  {spk} → {best_name}  (score={best_score:.3f})")
            else:
                unknown_counter += 1
                unk_label = f"unknown_{unknown_counter}"
                mapping[spk] = unk_label
                print(f"❔  {spk} remains {unk_label}  (best score={best_score:.3f})")

    # speakers that lacked clean evidence stay as unknown_N
    for spk in intervals_by_spk:
        if spk not in mapping:
            unknown_counter += 1
            mapping[spk] = f"unknown_{unknown_counter}"

    # 3.7 relabel full timeline
    timeline_df["Speaker"] = timeline_df["Speaker"].map(mapping)
    labeled_lines = dataframe_to_lines(timeline_df)
    labeled_txt.write_text("\n".join(labeled_lines))
    print(f"📄  Labeled timeline → {labeled_txt}")

    # 3.8 burn labels on the video
    annotate_video(video_path, labeled_lines, out_video)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    folder_loc = Path(args.dir).expanduser().resolve()
    video_name = args.video_name
    if not folder_loc.exists():
        sys.exit(f"❌  Folder location: {folder_loc} not found")
    
    run_pipeline(folder_loc, video_name)
