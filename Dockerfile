# AlgoAI360 — LTX-Video serverless worker
# Base: CUDA 12.1 runtime on Ubuntu 22.04 (matches torch 2.4 cu121 wheels).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# HF_HOME points at the RunPod volume/host-cache path so a downloaded model persists
# and is reused by later workers (RunPod caches HF models on the host). Small image.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    PIP_NO_CACHE_DIR=1

# Python + ffmpeg (imageio needs the ffmpeg binary for mp4 encoding).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg git && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY requirements.txt .
# Install torch from the cu121 index so the CUDA build matches the base image.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# NOTE: We do NOT bake the model into the image — that makes the image so large that
# RunPod's "export layers" step exceeds the 30-min build cap (learned the hard way).
# Instead, keep the image SMALL (code + deps only) and cache the model on the HOST via
# RunPod's "Cached model" field (set to huggingface.co/Lightricks/LTX-Video on the
# endpoint) OR a network volume. The handler downloads on first cold start into
# HF_HOME, and RunPod's host cache makes that fast on subsequent workers.

COPY handler.py .

# RunPod invokes the handler; no CMD web server needed for serverless.
CMD ["python", "-u", "handler.py"]
