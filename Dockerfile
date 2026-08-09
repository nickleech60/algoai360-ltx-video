# AlgoAI360 — LTX-Video serverless worker
# Base: CUDA 12.1 runtime on Ubuntu 22.04 (matches torch 2.4 cu121 wheels).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

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

COPY handler.py .

# RunPod invokes the handler; no CMD web server needed for serverless.
CMD ["python", "-u", "handler.py"]
