# AVATAR

**Audio-Visual Active Tracking and Annotation Rendering**
Note: The project is still in development. Please revisit for exciting future developments.

## DEMO

<div align="center">
  <video src="https://github.com/user-attachments/assets/af9571b2-113a-4d4c-a12e-00f08d8ffcfa" controls width="75%">
    Your browser does not support the video tag.
  </video>
  <p><i>Acrive Speaker Detection Demo</i></p>
</div>


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Innovative Aspects](#innovative-aspects)
- [Pipeline](#pipeline)
- [Architecture](#architecture)
  - [Real-Time Visualization Strategy](#real-time-visualization-strategy)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

**AVATAR** (Audio-Visual Active Tracking and Annotation Rendering) is an advanced system designed for near real-time audio-visual active speaker localization. It processes video files to detect and track multiple faces, evaluates lip-sync accuracy using the TalkNet model, annotates frames with relevant information, and renders the results both visually and in saved outputs. AVATAR is optimized for efficiency, leveraging sequential batch processing with a single initialization of the TalkNet model to ensure scalability and performance.

## Features

- **Multi-Face Detection and Tracking:** Utilizes state-of-the-art detectors (S3FD) and the SORT algorithm to identify and follow multiple faces across video frames.
- **Lip-Sync Evaluation:** Implements the TalkNet model to assess the synchronization between audio and lip movements.
- **Real-Time Visualization:** Displays annotated video frames in real-time using a Producer-Consumer model with multi-threading for seamless performance.
- **Comprehensive Logging:** Detailed logging of processing steps, errors, and performance metrics.
- **Batch Processing:** Efficiently handles video processing in fixed-length segments to optimize resource usage.
- **Sliding Window Mechanism:** Processes frames in a sliding window to achieve real-time near active speaker localization.
- **Extensible Architecture:** Designed to incorporate additional features like real-time diarization in the future.

## Innovative Aspects

AVATAR introduces several innovations in the field of audio-visual processing:

1. **Sliding Window Real-Time Localization:**
   - **First of Its Kind:** AVATAR is the first system to implement a sliding window mechanism for real-time near active speaker localization, enabling continuous and efficient processing of streaming data.
   - **Enhanced Performance:** By processing frames in overlapping batches, AVATAR ensures that active speakers are localized with minimal latency and high accuracy.

2. **Producer-Consumer Model for Real-Time Visualization:**
   - **Multi-Threading:** Employs a Producer-Consumer model using multi-threading to handle real-time visualization without interrupting the main processing pipeline.
   - **Thread-Safe Operations:** Utilizes thread-safe queues and locking mechanisms to manage shared resources, ensuring smooth and synchronized frame rendering.

3. **Efficient Batch Processing with Single TalkNet Initialization:**
   - **Resource Optimization:** Initializes the TalkNet model only once, significantly reducing computational overhead and enhancing processing speed.
   - **Scalability:** Designed to handle large-scale video processing tasks without performance degradation.

4. **Comprehensive Annotation and Rendering:**
   - **Detailed Visualization:** Annotates frames with bounding boxes and confidence scores based on lip-sync evaluations, providing clear visual insights into active speaker localization.
   - **Real-Time Display:** Renders annotated frames in a dedicated OpenCV window, allowing users to monitor processing in real-time.

## Pipeline

AVATAR's processing pipeline is meticulously designed to ensure accurate and efficient audio-visual analysis. Below is a step-by-step overview:

1. **Video and Audio Extraction:**
   - **Video Capture:** Reads input video files and extracts frame information such as FPS, resolution, and total frame count.
   - **Audio Extraction:** Utilizes `ffmpeg` to extract audio from the video, converting it to a suitable format for processing.

2. **Face Detection and Tracking:**
   - **Detection:** Employs the S3FD detector to identify faces within each frame.
   - **Tracking:** Implements the SORT algorithm to track detected faces across frames, assigning unique track IDs.

3. **Audio Processing:**
   - **Feature Extraction:** Computes MFCC features from the extracted audio to represent audio characteristics.

4. **Lip-Sync Evaluation:**
   - **TalkNet Integration:** Loads the pre-trained TalkNet model to evaluate lip-sync accuracy between detected faces and audio segments.
   - **Batch Evaluation:** Processes frames and corresponding audio in batches to optimize performance.

5. **Annotation and Rendering:**
   - **Frame Annotation:** Draws bounding boxes around detected faces with color-coded confidence scores based on TalkNet evaluations.
   - **Real-Time Display:** Renders annotated frames in a dedicated OpenCV window, supporting real-time monitoring.
   - **Frame Saving:** Saves annotated frames to the specified output directory for further analysis or record-keeping.

6. **Logging and Reporting:**
   - **Detailed Logs:** Captures comprehensive logs of all processing steps, including warnings, errors, and performance metrics.
   - **Track Information:** Saves tracking information in both pickle and text formats for easy access and analysis.

## Architecture

AVATAR's architecture is modular, ensuring each component handles a specific aspect of the processing pipeline. Below is an overview of the main components:

### 1. **Input Module**

- **Video Capture:** Uses OpenCV to read the video file, extracting frame rate, total frames, and frame dimensions.
- **Audio Extraction:** Employs `ffmpeg` to extract audio from the video file, converting it to a single-channel WAV file with a 16kHz sample rate.

### 2. **Detection and Tracking Module**

- **S3FD Detector:** Detects faces in each video frame with high accuracy. It processes frames to identify bounding boxes around faces.
- **SORT Tracker:** Tracks detected faces across frames, assigning unique track IDs to maintain consistent tracking.

### 3. **Audio Processing Module**

- **MFCC Feature Extraction:** Converts raw audio into Mel-Frequency Cepstral Coefficients (MFCC) features, which represent the audio's spectral properties.

### 4. **Lip-Sync Evaluation Module**

- **TalkNetInstance Class:** Manages the TalkNet model, performing lip-sync evaluation by analyzing the synchronization between audio and visual data (i.e., lip movements).
- **Batch Processing:** Processes audio and video frames in batches to optimize computational efficiency.

### 5. **Annotation and Rendering Module**

- **Frame Annotation:** Draws bounding boxes around detected faces, color-coded based on confidence scores from TalkNet evaluations.
- **Real-Time Display:** Utilizes a separate thread to render annotated frames in real-time using OpenCV, ensuring smooth visualization without interrupting the main processing pipeline.
- **Frame Saving:** Saves annotated frames to the output directory for offline analysis or record-keeping.

### 6. **Logging Module**

- **Logger Configuration:** Sets up logging to capture detailed information about the processing pipeline, including debug messages, warnings, and errors.
- **Log Management:** Logs are written to both the console and a file (`processing.log`) for comprehensive tracking and debugging.

### 7. **Output Module**

- **Annotated Frames:** Stores all annotated frames in a designated directory (`annotated_frames`), with each frame labeled sequentially.
- **Track Information:** Saves tracking data in both pickle (`trackInfo.pkl`) and text (`trackInfo.txt`) formats for easy access and analysis.

### 8. **Display Module**

- **Producer-Consumer Model:**
  - **Producer (Processing Thread):** Handles reading frames, detecting faces, running inference, annotating frames, and saving them. Annotated frames are enqueued into a shared buffer (`batch_queue`).
  - **Consumer (Display Thread):** Continuously dequeues frames from the shared buffer and displays them in a real-time OpenCV window at the desired frame rate (e.g., 25 FPS).

### 9. **Error Handling and Resource Management**

- **Graceful Termination:** Handles user interruptions and errors gracefully, ensuring all resources like video capture and audio clips are properly released.
- **Resource Cleanup:** Cleans up temporary directories and files post-processing to maintain a tidy working environment.

## Real-Time Visualization Strategy

AVATAR employs a **Producer-Consumer** model using multi-threading to achieve real-time visualization alongside the existing processing pipeline. Here's a detailed strategy:

### a. Utilize Multi-Threading

- **Processing Thread (Producer):**
  - Continues with the existing pipeline: reading frames, detecting faces, running inference, annotating, and saving frames.
  - After annotating each frame, it appends the frame to a shared buffer (`batch_queue`).

- **Display Thread (Consumer):**
  - Continuously reads frames from the shared buffer.
  - Displays frames in an OpenCV window at the desired frame rate (e.g., 25 FPS).
  - Maintains a sliding window using a deque to manage the frames being displayed, ensuring older frames are removed as new ones are added.

### b. Implement a Shared Buffer with Thread-Safe Operations

- **Shared Buffer:** Utilizes a `Queue` from Python's `queue` module to act as the shared buffer between the producer and consumer.
- **Thread Safety:** The `Queue` inherently handles thread-safe operations, ensuring that frames are safely enqueued and dequeued without race conditions.

### **Benefits of This Strategy:**

- **Seamless Real-Time Visualization:** By decoupling the processing and display tasks into separate threads, AVATAR ensures that visualization does not hinder the main processing pipeline.
- **Efficient Resource Utilization:** The producer can process frames at its own pace, while the consumer displays them as quickly as possible, leading to optimized performance.
- **Scalability:** This model allows for easy scaling, enabling the integration of additional features like real-time diarization without significant architectural changes.

## Installation

### Prerequisites

- **Python Version:** Python 3.7 or higher.
- **CUDA:** (Optional) NVIDIA GPU with CUDA support for accelerated processing.


### Clone the Repository

```bash
https://github.com/Divyeshpratap/AVATAR.git
cd avatar
```

### Demo

Run the script using the below command

```bash
python talkNetLive.py --video_path input/<videoname.mp4/avi> --tmp_dir output/<projectName> --clipped_video_len 50 --talknet_model pretrain_TalkSet.model
```

## License
This repository is private and intended solely for Center for Unified Biometrics and Sensors Lab (University at Buffalo) use. Unauthorized reproduction, distribution, or commercial use of this code is prohibited.


## Acknowledgment
This project leverages trained ASD model from [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) model.

## Contact
For more information, contact the University at Buffalo.

