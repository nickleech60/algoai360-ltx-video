# AlgoAI360 — LTX-Video serverless worker
# Base: CUDA 12.1 runtime on Ubuntu 22.04 (matches torch 2.4 cu121 wheels).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# HF_HOME points INSIDE the image so the model we bake in at build time is found at
# runtime (no re-download on cold start — the whole point). NOT /runpod-volume.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
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

# ── BAKE THE MODEL INTO THE IMAGE (build time, once) ─────────────────────────
# This is the fix for the "worker hangs downloading the model at runtime" problem.
# The ~several-GB LTX-Video weights are pulled NOW, during build, into /opt/hf.
# At runtime the worker finds them locally → starts in seconds, never downloads.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Lightricks/LTX-Video')"

COPY handler.py .

# RunPod invokes the handler; no CMD web server needed for serverless.
CMD ["python", "-u", "handler.py"]
