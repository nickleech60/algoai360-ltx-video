# ComfyUI-LTX serverless worker — deploy guide

Runs the GOOD fp8 13B LTX-Video the SAME way Bob runs it locally (ComfyUI), so cloud
output = proven local output. Diffusers couldn't load the fp8 13B; ComfyUI loads it natively.

## Files
- `Dockerfile` — worker-comfyui:5.8.6-base + `comfy-node-install comfyui-ltxvideo` (LTX nodes).
- `workflow_t2v.json` — Bob's proven LTX-13B text-to-video graph (fp8 13B + fp8 T5, 30 steps,
  cfg 3.2, tiled decode). `__PROMPT__` = injected by the CF Worker per request.

## Deploy steps
1. **GitHub repo** `algoai360-comfyui-ltx` (public) → push these files. RunPod builds from it.
2. **Serverless → New endpoint** from that repo. Attach `algoai360-models` volume (60GB, EU-RO-1).
   24GB GPU (RTX 3090/A5000/PRO4500 — fp8 13B fits via ComfyUI + tiled decode). Active 0 / Max 1.
3. **Arrange models on the volume in ComfyUI's layout** (temp pod + notebook):
   - `/workspace/models/checkpoints/ltxv-13b-0.9.7-dev-fp8.safetensors`  ← already downloaded (in hf cache; move/copy it here)
   - `/workspace/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors`  ← FETCH THIS (comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors)
   - (VAE: LTX ckpt bundles its own VAE, so a separate vae file is likely NOT needed — verify)
   Note: serverless mounts the volume at /runpod-volume, so ComfyUI sees /runpod-volume/models/...
4. **Wire endpoint id** into CF: `echo <ID> | npx wrangler secret put RUNPOD_VIDEO_ENDPOINT_ID`.
5. **CF Worker dispatch rewrite** (worker.js handleVideoGenerate): send
   `{ "input": { "workflow": <workflow_t2v.json with __PROMPT__ replaced> } }`. Update status
   poll to read the output image(s)/video from worker-comfyui's response shape.

## ⚠️ OPEN ITEM — video output format
Bob's graph ends in `SaveImage` = saves FRAMES (Bob assembles the mp4 itself afterward via
ffmpeg/NVENC). worker-comfyui returns whatever SaveImage produces (frames as base64/S3). Two paths:
  A) Add a video-combine node to the workflow (e.g. ComfyUI-VideoHelperSuite `VHS_VideoCombine`)
     so ComfyUI outputs an mp4 directly → add `comfy-node-install comfyui-videohelpersuite` to the
     Dockerfile. CLEANEST — one mp4 back.
  B) Keep SaveImage (frames) and have the CF Worker / a tiny step assemble frames→mp4.
RECOMMEND A: add VideoHelperSuite, swap `save` node to VHS_VideoCombine (fps=24, format=video/h264-mp4).

## Test
Requests tab first: `{ "input": { "workflow": {<the graph, __PROMPT__ replaced with a real prompt>} } }`.
Watch RunPod Logs for ComfyUI loading the ckpt + T5, sampling 30 steps, decode, save. Then the
end-to-end `/api/video/generate`. Watch a video PLAY before calling it done.

## Proven-good reference (Bob local)
E:\Builder Bob\studios\video-studio\generate_video.py (the graph) + first_run_setup.py (model files).
Model files: ltxv-13b-0.9.7-dev-fp8.safetensors + t5xxl_fp8_e4m3fn.safetensors + type "ltxv" CLIP.
