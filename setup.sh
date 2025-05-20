#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Fully reproducible environment bootstrap for this project.
#   • Creates & activates conda env “nemoEnv” (Python 3.10.12)
#   • Installs system-level libs with apt
#   • Pulls CUDA-11.8 PyTorch wheels and every Python dependency
#   • Order is **identical** to the list you supplied
#
# Usage (from repo root):   bash setup.sh
# ---------------------------------------------------------------------------

set -euo pipefail

log() { echo -e "\033[1;32m[setup]\033[0m $*"; }

# 1. Conda environment --------------------------------------------------------
log "Creating conda environment 'nemoEnv' (Python 3.10.12)…"
conda create -n nemoEnv python=3.10.12 -y

# Activate conda inside a non-interactive script
eval "$(conda shell.bash hook)"
conda activate nemoEnv
log "Activated conda env 'nemoEnv'."

# 2. System-level prerequisites ----------------------------------------------
log "Running 'sudo apt-get update'…"
sudo apt-get update

log "Installing libgtk2.0-dev and pkg-config (build/GUI headers)…"
sudo apt-get install -y libgtk2.0-dev pkg-config

# 3. GPU PyTorch stack (CUDA 11.8 wheels) ------------------------------------
log "Installing torch / torchvision / torchaudio (CUDA 11.8)…"
pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
           torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0

# 4. FFmpeg toolchain + compilers via conda-forge -----------------------------
log "Installing FFmpeg 6.1.1, Cython and GNU compilers from conda-forge…"
conda install -y -c conda-forge \
      ffmpeg=6.1.1 cython pkg-config gcc_linux-64 gxx_linux-64

# 5. Extra audio codec runtime ------------------------------------------------
log "Installing libsndfile1…"
sudo apt-get install -y libsndfile1

# 6. Python libs --------------------------------------------------------------
log "Building PyAV 12.3.0 from source (no binary wheels)…"
pip install "av==12.3.0" --no-binary av --no-cache-dir

log "Installing remaining Python packages…"
pip install \
    opencv-python \
    scipy \
    python_speech_features \
    insightface \
    onnxruntime-gpu==1.19.2

# PyAudio is easiest via conda-forge because of native bits
log "Installing PyAudio + PortAudio via conda-forge…"
conda install -y -c conda-forge pyaudio portaudio

# NeMo (ASR subset) from GitHub main
log "Installing NVIDIA NeMo (toolkit[asr]) from GitHub…"
pip install "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]"

# SceneDetect
log "Installing PySceneDetect 0.6.4…"
pip install scenedetect==0.6.4

log "🎉  All done!  Activate later with:  conda activate nemoEnv"
