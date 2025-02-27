import torch
import numpy as np
import cv2
import glob
import os
import logging
from datetime import datetime
import python_speech_features
from scipy.io import wavfile
from model.talkNet import talkNet

class TalkNetInstance:
    def __init__(self, model_path):
        self.logger = logging.getLogger('TalkNetInstance')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            self.logger.info("CUDA is available. Using GPU for TalkNet.")
        else:
            self.logger.info("CUDA is not available. Using CPU for TalkNet.")
        self.talkNet = talkNet().to(self.device)
        self.loadParameters(model_path)
        self.talkNet.eval()
        self.logger.info(f"TalkNet model loaded from {model_path}.")

    def evaluate(self, start_frame_number_of_batch, frames_dir, audio_file):
        _, audio = wavfile.read(audio_file)
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.0083)
        videoFeature = []
        flist = glob.glob(os.path.join(frames_dir, '*.jpg'))
        flist.sort()
        for fname in flist:
            frame = cv2.imread(fname)
            if frame is None:
                self.logger.warning(f"Failed to read frame: {fname}. Skipping.")
                continue
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = face.shape
            y_center, x_center = height // 2, width // 2
            half_size = 56
            y_min = max(y_center - half_size, 0)
            y_max = min(y_center + half_size, height)
            x_min = max(x_center - half_size, 0)
            x_max = min(x_center + half_size, width)
            face = face[y_min:y_max, x_min:x_max]
            videoFeature.append(face)
        videoFeature = np.array(videoFeature)
        expected_audio_len = (videoFeature.shape[0] * 120) // 30
        if audioFeature.shape[0] < expected_audio_len:
            padding_len = expected_audio_len - audioFeature.shape[0]
            last_values = np.tile(audioFeature[-1:], (padding_len, 1))
            audioFeature = np.vstack((audioFeature, last_values))
        length = min((audioFeature.shape[0] // 120), (videoFeature.shape[0] // 30))
        if length != 1:
            print(f'Warning: length found should have been 1 but found as {length}')
        with torch.no_grad():
            inputA = torch.FloatTensor(audioFeature[0:120, :]).unsqueeze(0).to(self.device)
            inputV = torch.FloatTensor(videoFeature[0:30, :, :]).unsqueeze(0).to(self.device)
            embedA = self.talkNet.model.forward_audio_frontend(inputA)
            embedV = self.talkNet.model.forward_visual_frontend(inputV)
            embedA, embedV = self.talkNet.model.forward_cross_attention(embedA, embedV)
            out = self.talkNet.model.forward_audio_visual_backend(embedA, embedV)
            label, score = self.talkNet.lossAV.forward(out, labels=None)
        return score
    
    def loadParameters(self, path):
        self.logger.debug(f"Loading TalkNet parameters from: {path}")
        try:
            loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
            self_state = self.talkNet.state_dict()
            for name, param in loaded_state.items():
                if name not in self_state:
                    name = name.replace("module.", "")
                if name in self_state:
                    if self_state[name].shape == param.shape:
                        self_state[name].copy_(param)
                        self.logger.debug(f"Loaded parameter: {name}")
                    else:
                        self.logger.warning(f"Shape mismatch for parameter: {name}. Expected {self_state[name].shape}, got {param.shape}.")
                else:
                    self.logger.warning(f"Parameter {name} not found in the model.")
            self.logger.info("All parameters loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load TalkNet parameters: {e}")
