# AlgoAI360 — ComfyUI serverless worker for LTX-Video (the GOOD 13B, run the SAME way
# Builder Bob runs it locally). Base = RunPod's official worker-comfyui (serverless
# ComfyUI API). We ONLY add the LTX custom nodes here; the MODELS live on the network
# volume (/runpod-volume/models/...), auto-detected — no model baking.
#
# Why ComfyUI (not diffusers): the fp8 13B checkpoint is a ComfyUI-format single file.
# diffusers can't load it (shape mismatch); ComfyUI loads it natively — exactly like
# Bob's proven local pipeline. Same model, same loader, same good output.
FROM runpod/worker-comfyui:5.8.6-base

# Add the LTX-Video custom nodes (CheckpointLoaderSimple works for the ckpt, but the
# LTX-specific nodes — EmptyLTXVLatentVideo, LTXVConditioning, LTXVAddGuide, the "ltxv"
# CLIP type, VAEDecodeTiled temporal args — come from ComfyUI-LTXVideo).
# LTX nodes (EmptyLTXVLatentVideo, LTXVConditioning, "ltxv" CLIP, etc.).
RUN comfy-node-install comfyui-ltxvideo
# VideoHelperSuite (VHS_VideoCombine) so ComfyUI outputs a finished MP4 directly.
# Separate RUN so one failing doesn't skip the other, and so we see which one errors.
RUN comfy-node-install comfyui-videohelpersuite

# Models are NOT baked — they sit on the network volume at:
#   /runpod-volume/models/checkpoints/ltxv-13b-0.9.7-dev-fp8.safetensors
#   /runpod-volume/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors
#   /runpod-volume/models/vae/  (LTX ckpt bundles its VAE, so this may be optional)
# ComfyUI auto-detects them. Keeps the image small = fast build.
