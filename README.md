# AVATAR

**Audio-Visual Active Tracking and Annotation Rendering**

AVATAR is a near real-time active speaker detection system that streams camera feeds, tracks all faces and applies facial recongition, evaluates their lip-sync accuracy using TalkNet, and annotates the video stream in real time with the active speaker. The system uses an efficient producer-consumer pipeline to ensure simultaneous image + audio capture, lip-sync score evaluation, and live frame annotation and rendering.The system also generates speaking segments tagged to speaker identity which can be later used for video captioning. It also integrates an optional face masking feature if the person identity is not present in the database.
## DEMO
<div align="center">
  <video src="https://github.com/user-attachments/assets/af9571b2-113a-4d4c-a12e-00f08d8ffcfa" controls width="50%">
    Your browser does not support the video tag.
  </video>
  <p><i>Active Speaker Localization Demo</i></p>
</div>

## High Level System Architecture:

<div align="center">
  <img src="https://github.com/user-attachments/assets/ee7442c8-a0df-4e89-9030-ab129a96b187" alt="Inferencing Architecture" width="100%">
  <p><i>Model Inferencing Architecture</i></p>
</div>

## Table of Contents

1. [Introduction](#introduction)  
2. [Key Features](#key-features)  
3. [Major Steps)](#major-steps)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [License](#license)  
7. [Acknowledgements](#acknowledgements)

---

## Introduction

**AVATAR** (Audio-Visual Active Tracking and Annotation Rendering) serves as a foundation model for advanced audio-visual understanding tasks such as speaker diarization, content creation tools, or live-stream moderation. Some of the applications of the project are:
1. **Annotating Captions in Meetings**: Automatically tag each utterance in a multi-speaker meeting, making it easier to generate speaker-specific minutes of meeting or transcripts.
2. **Speaker Source separation**: In complex audio environments where multiple people talk simultaneously, the system can help isolate or label each speaker’s audio track.
3. **Interactive Learning or Tele-Education**: In virtual classrooms, highlight the student currently speaking, or track group discussions to get speaker contribution metrics.
4. **Subtitle or Caption Automation**: Integrate with real-time transcription services to automatically generate speaker-attributed subtitles, useful for online lectures, panels, or conferences.

---

## Key Features

- **Live Camera Integration**: Captures video (and audio via PyAudio) from your webcam.
- **Multi-Face Detection & Tracking**: Integrates [S3FD](https://github.com/sfzhang15/SFD) for face detection, recognition using [Insightface](https://github.com/deepinsight/insightface) and [SORT](https://github.com/abewley/sort) for ID-based tracking across frames.
- **Active Speaker Identification**: Uses [TalkNet](https://github.com/TaoRuijie/TalkNet-ASD) to generate lip-sync scores,idicating active speakers in a frame.
- **Real-Time Annotations**: Bounding boxes with color-coded confidence scores are rendered real-time for each frame, and displayed in a dedicated OpenCV window.
- **Producer-Consumer with Threads**: Producer threads handles processing (face detection, recognition and tracking, lip-sync evaluation), consumer handles visualization for concurrent execution. Lip-sync evaluation dynamically allocates separate threads based on number of speakers identified to speed up ML inferencing.
- **Speaking Segment Compilation:** After processing the entire video stream, speaking segments are computed and saved to JSON, summarizing active speaker intervals.
---

## Major Steps

1. **Capture Audio and Video**  
   - Frames are read in batches (currently 30 frames at 30fps) from webcam.  
   - Corresponding audio chunks (at 16 kHz) are captured via PyAudio.
   - Both thread data is joined for synchronization.

2. **Face Detection & Tracking**  
   - S3FD locates faces in each frame.
   - SORT assigns track IDs to each detected face, maintaining consistent identities across frames.
   - InsightFace allocated Track IDs with person Identity.

3. **Lip-Sync Scoring**  
   - TalkNet evaluates each tracked face’s lip movements against captured audio, returning a score representing lip-sync synchronization.

4. **Annotation & Display**  
   - Bounding boxes are drawn around faces with color-coded confidence scores from talkNet.  
   - An OpenCV window renders these annotated frames in real time.

5. **Repeat steps 1 to 4**

6. **Speaking Segments Compilation**  
   - For each Person identity, frame scores are converted to a binary sequence based on a threshold.  
   - Speaking segments are extracted (merging small gaps and enforcing a minimum segment length) and saved as a JSON file.
---

## Installation

1. Clone the Repository
   ```bash
   git clone https://github.com/Divyeshpratap/AVATAR.git
   pip install -r requirements.txt
   cd AVATAR
   
2. Setup the environment.
  ```bash
  conda create -n avlocalizer python=3.9 -y
  conda activate avlocalizer
  ```

Install [ffmpeg](https://ffmpeg.org/) on your system.
```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

```
InsightFace requires [onnx](https://pypi.org/project/onnxruntime-gpu/) installation for gpu-acceleration which requires the following dependencies. Please install these before onnx installation.
```bash
sudo apt-get update
sudo apt-get install libgtk2.0-dev pkg-config

3. Install Python Packages
  ```bash
  pip install -r requirements.txt
  ```


### Directory Structure
```
${ROOT} 
├── detectSpeakers.py          # Entry point
├── face                       # Face detection and recognition scripts
│   ├── detection.py
│   └── recognition.py
├── model                      # LipSync Scripts
│   ├── talkNet.py
│   ├── sort.py
│   ├── lipsync.py
│   └── … (other model files)
├── pipeline                   # Main Pipeline
│   └── pipeline.py
├── registeredFaces            # Directory to store registered faces (identities)
│   ├── <identity1>
│   │   └── <1>.jpeg
│   └── <identity2>
│       ├── <1>.jpeg
│       ├── <2>.jpeg
│       └── … (other images)
├── streamer                   # Camera Feed Scripts
│   ├── audiostream.py
│   └── videostream.py
├── utils                      # Utility functions
│   ├── args.py
│   ├── logger.py
│   ├── speaking_segments.py
│   └── tools.py
└── weights                    # LipSync Model Weights
    ├── pretrain_AVA.model
    └── pretrain_TalkSet.model
```


### Usage

Run the inferencing script using the below command

```bash
python detectSpeakers.py --tmp_dir output/<projectName>
pip install -r requirements.txt
```



## License
This repository is private and intended solely for Center for Unified Biometrics and Sensors Lab (University at Buffalo) use. Unauthorized reproduction, distribution, or commercial use of this code is prohibited.


## Acknowledgment
This project adopts trained ASD model from [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD), face detection from [S3FD](https://github.com/sfzhang15/SFD), face recognition from [Insightface](https://github.com/deepinsight/insightface), and tracking code from [SORT](https://github.com/abewley/sort) 

## Contact
For more information, contact the University at Buffalo.

