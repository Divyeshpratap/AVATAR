# AVATAR

**Audio-Visual Active Tracking and Annotation Rendering**

AVATAR is a near real-time active speaker detection system that continuously captures frames from your camera, detects and tracks multiple faces, evaluates their lip-sync accuracy using TalkNet, and annotates the video stream in real time with the active speaker. The design combines an efficient producer-consumer pipeline, and multi-threading to ensure video capture, active speaker localization, and frame rendering can be performed simultaneously.

## DEMO
<div align="center">
  <video src="https://github.com/user-attachments/assets/af9571b2-113a-4d4c-a12e-00f08d8ffcfa" controls width="50%">
    Your browser does not support the video tag.
  </video>
  <p><i>Active Speaker Localization Demo</i></p>
</div>

## High Level System Architecture:

<div align="center">
  <img src="https://github.com/user-attachments/assets/4386602c-c44f-4dd6-918d-df33c2089b8a" alt="Inferencing Architecture" width="100%">
  <p><i>Model Inferencing Architecture</i></p>
</div>

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
10. [License](#license)  
11. [Acknowledgements](#acknowledgements)

---

## Introduction

**AVATAR** (Audio-Visual Active Tracking and Annotation Rendering) is designed for live camera input or saved videos, allowing you to automatically detect, track, and evaluate multiple people’s lip-sync accuracy. The results are shown via bounding boxes and confidence scores, overlaid in real time on the video stream.

This pipeline is a starting point for advanced audio-visual understanding tasks such as speaker diarization, content creation tools, or live-stream moderation. Some of the applications of the project are:
1. Annotating which person spoke which sentence in a multi speaker meeting scenario. This can be useful to generate speaker specific Minutes of Meeting.
2. Speaker Source separation.
3. Identifying the contribution of a speaker in a meeting.
4. 

---


## Key Features

- **Live Camera Integration**: Captures video (and audio via PyAudio) from your webcam.
- **Multi-Face Detection & Tracking**: Employs S3FD for face detection and SORT for ID-based tracking across frames.
- **Active Speaker Identification**: Uses TalkNet to generate lip-sync scores, to determine who is speaking in each frame.
- **Real-Time Annotations**: Bounding boxes with color-coded confidence scores are rendered on each frame, displayed in a dedicated OpenCV window.
- **Producer-Consumer with Threads**: Producer thread handles processing (face detection and tracking, lip-sync evaluation), consumer handles visualization to avoid pipeline bottlenecks. Lip-sync evaluation dynamically allocates threads based on number of speakers identified to speed up ML inferencing.

---

## Major Steps

1. **Capture Audio and Video**  
   - Frames are read in batches (currently 30 frames each at 30fps) from your webcam.  
   - Corresponding audio chunks (at 16 kHz) are captured via PyAudio.
   - Both thread data is joined for synchronization.

2. **Face Detection & Tracking**  
   - S3FD locates faces in each frame.  
   - SORT assigns track IDs to each detected face, maintaining consistent identities across frames.

3. **Lip-Sync Scoring**  
   - TalkNet evaluates each tracked face’s lip movements against captured audio, returning a score representing lip-sync accuracy.

4. **Annotation & Display**  
   - Bounding boxes are drawn around faces with color-coded confidence scores from talkNet.  
   - An OpenCV window renders these annotated frames in real time.

5. **Repeat steps 1 to 4**
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


## Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/YourUsername/AVATAR.git
   cd AVATAR

### Usage

Run the inferencing script using the below command

```bash
python detectSpeakers.py --tmp_dir output/<projectName> --talknet_model pretrain_TalkSet.model
```

## License
This repository is private and intended solely for Center for Unified Biometrics and Sensors Lab (University at Buffalo) use. Unauthorized reproduction, distribution, or commercial use of this code is prohibited.


## Acknowledgment
This project leverages trained ASD model from [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) model.

## Contact
For more information, contact the University at Buffalo.

