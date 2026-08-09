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

print("[ltx] handler process started — worker is up", flush=True)

# ── Model load (once per cold start; stays warm via FlashBoot) ───────────────
_PIPE = None
_PIPE_I2V = None


def _load_pipes():
    """Lazy-load the LTX-Video pipelines. Text-to-video + image-to-video share weights."""
    global _PIPE, _PIPE_I2V
    if _PIPE is not None:
        return
    # Use the FP8-quantized 13B — the SAME checkpoint Bob runs locally on a 24GB 3090
    # (proven-good quality Nick has seen). fp8 is ~half the memory of bf16, so the 13B
    # FITS 24GB (the bf16 13B OOM'd; the fp8 13B is what makes 24GB work — that's why
    # local works). Load the fp8 transformer via from_single_file, plug into the base
    # pipeline (VAE + T5 from the diffusers repo). License: free commercial <$10M ARR.
    base_repo = os.environ.get("LTX_BASE", "Lightricks/LTX-Video")
    ckpt = os.environ.get(
        "LTX_CKPT",
        "https://huggingface.co/Lightricks/LTX-Video/blob/main/ltxv-13b-0.9.7-dev-fp8.safetensors",
    )
    have_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if have_cuda else torch.float32
    print(f"[ltx] loading FP8 13B transformer from {ckpt} ...", flush=True)
    from diffusers import LTXPipeline, LTXImageToVideoPipeline
    from diffusers import LTXVideoTransformer3DModel
    transformer = LTXVideoTransformer3DModel.from_single_file(ckpt, torch_dtype=dtype)
    print(f"[ltx] transformer loaded; building pipeline from {base_repo} ...", flush=True)
    _PIPE = LTXPipeline.from_pretrained(base_repo, transformer=transformer, torch_dtype=dtype)
    print("[ltx] weights loaded, placing on device ...", flush=True)

    if have_cuda:
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        # fp8 13B (~13GB) fits 24GB → full-GPU = FAST (like Nick's local). VAE tiling
        # still on to keep the decode from spiking. Offload only if a small card.
        if free_gb >= 16:
            _PIPE.to("cuda")
            print(f"[ltx] full-GPU mode ({free_gb:.0f}GB free) — fp8 13B fits", flush=True)
        else:
            _PIPE.enable_model_cpu_offload()
            print(f"[ltx] CPU-offload ({free_gb:.0f}GB free)", flush=True)
        try:
            _PIPE.vae.enable_tiling()
            _PIPE.vae.enable_slicing()
            print("[ltx] VAE tiling+slicing on", flush=True)
        except Exception as _e:
            print(f"[ltx] VAE tiling unavailable: {_e}", flush=True)
    else:
        _PIPE.to("cpu")

    # Image-to-video reuses the same underlying components (no second download).
    _PIPE_I2V = LTXImageToVideoPipeline(**_PIPE.components)
    print("[ltx] pipelines ready", flush=True)


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
    # Default fps 24 (cinematic/smooth). 8fps looked like a stutter-y slideshow; LTX
    # is trained on real motion so it wants 24-30. Overridable per-request + by env.
    fps = int(inp.get("fps") or os.environ.get("LTX_FPS", "24"))
    fps = max(6, min(30, fps))

    if not prompt and not image_in:
        return {"error": "prompt or image required"}

    # Light prompt enhancement only for very short prompts. The 13B model follows
    # prompts well on its own — over-stuffing (the old long suffix) can HURT coherence.
    if prompt and inp.get("enhance", True) and len(prompt) < 80:
        prompt = f"{prompt}, high quality, detailed, smooth natural motion, cinematic"

    # LTX generates in frames; num_frames must be 8k+1 for its temporal compression.
    num_frames = seconds * fps
    num_frames = ((num_frames - 1) // 8) * 8 + 1  # snap to 8n+1

    try:
        _load_pipes()
    except Exception as e:
        return {"error": f"model load failed: {e}"}

    neg = ("worst quality, low quality, blurry, jittery, distorted, deformed, "
           "watermark, text, low resolution, pixelated, choppy motion, artifacts")
    # Quality-tier resolution. LTX-2 handles 1216x704 (720p-class) well on a 24GB card.
    # Must be multiples of 32. Overridable via env for a cheaper "draft" tier later.
    # 768x512 = old cheap/ugly default; 1216x704 = the real "standard" quality tier.
    width = int(os.environ.get("LTX_WIDTH", "1216"))
    height = int(os.environ.get("LTX_HEIGHT", "704"))

    try:
        # With CPU offload the generator must live on CPU (the model modules move
        # on/off GPU as needed; a cuda generator would mismatch the offloaded state).
        gen = torch.Generator(device="cpu")
        common = dict(
            prompt=prompt or "a high quality cinematic video",
            negative_prompt=neg,
            width=width, height=height,
            num_frames=num_frames,
            # 30 steps: the DISTILLED 13B is designed for ~30 (not 50). Fewer steps =
            # much faster (fixes the 6-min problem) with no quality loss on distilled.
            # 13B-dev (non-distilled) uses standard ~30 steps. Tunable via env.
            num_inference_steps=int(os.environ.get("LTX_STEPS", "30")),
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
