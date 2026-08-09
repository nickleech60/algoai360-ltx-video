"""
AlgoAI360 — LTX-Video RunPod Serverless worker.

Contract (MUST match the Cloudflare Worker exactly — worker.js handleVideoGenerate):
  INPUT   { "input": { "prompt": str|null, "image": <b64 or data-uri or null>,
                       "seconds": int (2-10), "fps": int } }
  OUTPUT  { "url": "<public https url to the .mp4>" }
          (the Worker's status poll reads output.url / output.video_url)

Runs LTX-Video (Lightricks) via the diffusers pipeline. Text-to-video when only a
prompt is given; image-to-video when an image is provided. The finished mp4 is
uploaded to an S3-compatible bucket (Cloudflare R2) and the public URL is returned —
the Worker/app needs a URL, not base64 (video is too big for base64 round-trips).

Env vars the endpoint needs (set on the RunPod endpoint, see the deploy guide):
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_PUBLIC_BASE
  (If R2 vars are absent, falls back to returning a base64 data URI so it still works
   for a quick test — but set R2 for production so URLs are small and streamable.)
"""
import base64
import io
import os
import time
import uuid

import runpod
import torch

# ── Model load (once per cold start; stays warm via FlashBoot) ───────────────
_PIPE = None
_PIPE_I2V = None


def _load_pipes():
    """Lazy-load the LTX-Video pipelines. Text-to-video + image-to-video share weights."""
    global _PIPE, _PIPE_I2V
    if _PIPE is not None:
        return
    from diffusers import LTXPipeline, LTXImageToVideoPipeline
    model_id = os.environ.get("LTX_MODEL", "Lightricks/LTX-Video")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    _PIPE = LTXPipeline.from_pretrained(model_id, torch_dtype=dtype)
    _PIPE.to("cuda" if torch.cuda.is_available() else "cpu")
    # Image-to-video reuses the same underlying components (no second download).
    _PIPE_I2V = LTXImageToVideoPipeline(**_PIPE.components)
    _PIPE_I2V.to("cuda" if torch.cuda.is_available() else "cpu")


def _decode_image(img):
    """Accept a data-uri or raw base64 → PIL.Image (RGB)."""
    from PIL import Image
    if not img:
        return None
    if isinstance(img, str) and img.startswith("data:"):
        img = img.split(",", 1)[1]
    raw = base64.b64decode(img)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _upload_r2(mp4_bytes, key):
    """Upload the mp4 to Cloudflare R2 (S3 API). Returns a public URL, or None if R2
    isn't configured (caller then falls back to a data URI)."""
    acct = os.environ.get("R2_ACCOUNT_ID")
    ak = os.environ.get("R2_ACCESS_KEY_ID")
    sk = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET")
    public_base = os.environ.get("R2_PUBLIC_BASE")  # e.g. https://media.algoai360.com
    if not all([acct, ak, sk, bucket, public_base]):
        return None
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
        region_name="auto",
    )
    s3.put_object(Bucket=bucket, Key=key, Body=mp4_bytes, ContentType="video/mp4")
    return f"{public_base.rstrip('/')}/{key}"


def _frames_to_mp4(frames, fps):
    """Encode a list of PIL frames to mp4 bytes via imageio/ffmpeg."""
    import imageio.v3 as iio
    import numpy as np
    arr = np.stack([np.asarray(f) for f in frames])
    buf = io.BytesIO()
    iio.imwrite(buf, arr, extension=".mp4", fps=fps, codec="libx264")
    return buf.getvalue()


def handler(event):
    """RunPod entrypoint. event['input'] = the Worker's JSON input."""
    inp = event.get("input", {}) or {}
    prompt = (inp.get("prompt") or "").strip()
    image_in = inp.get("image")
    seconds = int(inp.get("seconds") or 4)
    seconds = max(2, min(10, seconds))
    fps = int(inp.get("fps") or 8)
    fps = max(6, min(30, fps))

    if not prompt and not image_in:
        return {"error": "prompt or image required"}

    # LTX generates in frames; num_frames must be 8k+1 for its temporal compression.
    num_frames = seconds * fps
    num_frames = ((num_frames - 1) // 8) * 8 + 1  # snap to 8n+1

    try:
        _load_pipes()
    except Exception as e:
        return {"error": f"model load failed: {e}"}

    neg = "worst quality, blurry, jittery, distorted, watermark, text"
    # LTX-Video is trained at 768x512-ish; keep it modest for speed/VRAM.
    width, height = 768, 512

    try:
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        common = dict(
            prompt=prompt or "a high quality cinematic video",
            negative_prompt=neg,
            width=width, height=height,
            num_frames=num_frames,
            num_inference_steps=int(os.environ.get("LTX_STEPS", "40")),
            generator=gen,
        )
        img = _decode_image(image_in)
        if img is not None:
            result = _PIPE_I2V(image=img, **common)
        else:
            result = _PIPE(**common)
        frames = result.frames[0]  # list of PIL images
    except Exception as e:
        return {"error": f"generation failed: {e}"}

    try:
        mp4 = _frames_to_mp4(frames, fps)
    except Exception as e:
        return {"error": f"encode failed: {e}"}

    key = f"video/{time.strftime('%Y%m%d')}/{uuid.uuid4().hex}.mp4"
    url = _upload_r2(mp4, key)
    if url:
        return {"url": url}
    # Fallback for testing without R2 configured: return a data URI (large, but works).
    b64 = base64.b64encode(mp4).decode()
    return {"url": f"data:video/mp4;base64,{b64}"}


runpod.serverless.start({"handler": handler})
