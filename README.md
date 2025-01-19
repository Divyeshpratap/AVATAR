# AVATAR

**Audio-Visual Active Tracking and Annotation Rendering**

AVATAR is a near real-time active speaker detection system that continuously captures frames from your camera, detects and tracks multiple faces, evaluates their lip-sync accuracy using TalkNet, and annotates the video stream in real time. Its design combines an efficient producer-consumer pipeline, multi-threading, and single-initialization batch processing to ensure smooth performance and scalability.

## Table of Contents

1. [Introduction](#introduction)  
2. [What's New](#whats-new)  
3. [Key Features](#key-features)  
4. [How It Works (High-Level Overview)](#how-it-works-high-level-overview)  
5. [System Architecture](#system-architecture)  
   - [Real-Time Producer-Consumer Model](#real-time-producer-consumer-model)  
   - [Lip-Sync Evaluation with TalkNet](#lip-sync-evaluation-with-talknet)  
6. [Installation](#installation)  
7. [Usage](#usage)  
8. [Output](#output)  
9. [Dependencies](#dependencies)  
10. [License](#license)  
11. [Acknowledgements](#acknowledgements)

---

## Introduction

**AVATAR** (Audio-Visual Active Tracking and Annotation Rendering) is designed for live camera input or saved videos, allowing you to automatically detect, track, and evaluate multiple people’s lip-sync accuracy. The results are shown via bounding boxes and confidence scores, overlaid in real time on the video stream.

This pipeline can be a starting point for advanced audio-visual understanding tasks such as speaker diarization, content creation tools, or live-stream moderation. If you are working in an area where identifying who is speaking is key, this project provides a solid, extensible foundation.

---

## What's New

- **S3FD Face Detection**: Switched from MTCNN to [S3FD](https://github.com/sfzhang15/SFD) for robust, real-time face detection.
- **SORT Tracking**: Integrated [SORT](https://github.com/abewley/sort) for lightweight, accurate multi-face tracking.
- **Single TalkNet Initialization**: The TalkNet model is loaded only once, reducing overhead in lip-sync scoring.
- **Real-Time Camera Capture**: The system now supports direct camera input (`cv2.VideoCapture(0)`).
- **Multi-Threaded Visualization**: A dedicated thread fetches annotated frames from a queue and displays them in real time.
- **Batch-Based Processing**: Frames and audio are processed in fixed-sized batches (e.g., 30 frames), balancing responsiveness with efficiency.

---

## Key Features

- **Live Camera Integration**: Captures video (and audio via PyAudio) straight from your webcam.
- **Active Speaker Identification**: Uses TalkNet to generate lip-sync scores, helping determine who is speaking in each frame.
- **Multi-Face Detection & Tracking**: Employs S3FD for face detection and SORT for ID-based tracking across frames.
- **Real-Time Annotations**: Bounding boxes with color-coded confidence scores are rendered on each frame, displayed in a dedicated OpenCV window.
- **Producer-Consumer with Threads**: One thread handles processing (face detection, lip-sync evaluation), another handles visualization to avoid pipeline bottlenecks.
- **Flexible Configuration**: User-defined arguments for detection thresholds, bounding-box smoothing, batch size, and audio parameters.

---

## How It Works (High-Level Overview)

1. **Capture Audio and Video**  
   - Frames are read in batches (e.g., 30 frames each) from your webcam.  
   - Corresponding audio chunks are captured via PyAudio.

2. **Face Detection & Tracking**  
   - S3FD locates faces in each frame.  
   - SORT assigns track IDs to each detected face, maintaining consistent identities across frames.

3. **Lip-Sync Scoring**  
   - TalkNet evaluates each tracked face’s lip movements against captured audio, returning a score representing lip-sync accuracy.

4. **Annotation & Display**  
   - Bounding boxes are drawn around faces with color-coded confidence.  
   - An OpenCV window renders these annotated frames in real time.

5. **Batch Cleanup & Logging**  
   - Processed data (frames, audio segments) are saved or cleared, and logs capture any warnings or performance metrics.

---

## System Architecture

### Real-Time Producer-Consumer Model

- **Producer** (Main/Processing Thread):  
  - Reads frames from the camera in batches.  
  - Runs S3FD face detection and SORT tracking.  
  - Performs TalkNet lip-sync evaluation.  
  - Annotates each frame and enqueues them for display.

- **Consumer** (Display Thread):  
  - Dequeues and displays annotated frames in an OpenCV window.  
  - Prevents the main pipeline from stalling due to visualization delays.

### Lip-Sync Evaluation with TalkNet

- **TalkNetInstance** Class:  
  - Loads the pre-trained TalkNet model once (GPU or CPU).  
  - Extracts audio features (MFCC) and video features (face crops).  
  - Generates lip-sync confidence scores for frames in each batch.

- **Single Initialization**:  
  - TalkNet is loaded at startup, significantly cutting down on repetitive overhead.

---

## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/AVATAR.git
   cd AVATAR

### Demo

Run the script using the below command

```bash
python detectSpeakers.py --tmp_dir output/<projectName> --talknet_model pretrain_TalkSet.model
```

## License
This repository is private and intended solely for Center for Unified Biometrics and Sensors Lab (University at Buffalo) use. Unauthorized reproduction, distribution, or commercial use of this code is prohibited.


## Acknowledgment
This project leverages trained ASD model from [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) model.

## Contact
For more information, contact the University at Buffalo.

