# Deploy the LTX-Video worker to RunPod Serverless

This worker generates AI video (text→video and image→video) and returns a URL the
AlgoAI360 Worker already knows how to read. Deploying it makes "cloud video" real.

## What you need
- RunPod account with a funded balance (you have this)
- A GitHub account (RunPod builds the worker from a git repo — easiest path)
- ~15 min

## The plan
1. Push this folder to a GitHub repo.
2. RunPod builds a serverless endpoint from that repo.
3. Copy the endpoint ID → paste into Cloudflare as `RUNPOD_VIDEO_ENDPOINT_ID`.
4. Test: generate one real video. Only "done" when we SEE the video.

---

## Step 1 — Put this folder in a GitHub repo
The three files (`handler.py`, `requirements.txt`, `Dockerfile`) must be at the repo root
(or note the subfolder path for step 2). Simplest: a new repo `algoai360-ltx-video` with
just these three files.

## Step 2 — Create the RunPod endpoint
RunPod → **Serverless** → **+ New Endpoint** → **Import Git Repository** (or "Custom / Docker").
- **Repo:** your `algoai360-ltx-video` repo. RunPod builds the Dockerfile automatically.
- **GPU:** **24 GB (RTX 4090 / A5000 — NON-PRO, High Supply).** LTX-Video fits 24GB.
  ⚠️ Do NOT pick a PRO/Blackwell/MIG card (the "no kernel image" CUDA error).
- **Active workers: 0**  ·  **Max workers: 1**  ·  **Idle timeout: 5s**  ·  **FlashBoot: ON**
- **Container disk:** 40 GB (room for the LTX weights + CUDA).
- **Env vars** (Settings → add these so videos upload to R2 and return small URLs):
  ```
  R2_ACCOUNT_ID        = <your Cloudflare account id>
  R2_ACCESS_KEY_ID     = <R2 API token access key>
  R2_SECRET_ACCESS_KEY = <R2 API token secret>
  R2_BUCKET            = <bucket for generated media, e.g. algoai360-media>
  R2_PUBLIC_BASE       = https://media.algoai360.com   (the bucket's public URL)
  ```
  (If you skip R2 for a first test, the worker returns a base64 data-URI instead — big,
  but it proves generation works. Set R2 before real customers use it.)
- **Deploy.** First build takes several minutes (installs torch + diffusers).

## Step 3 — Wire the endpoint ID into Cloudflare
Copy the new endpoint's **ID**, then:
Cloudflare → Workers & Pages → `algoai360-api` → Settings → Variables & Secrets →
add secret `RUNPOD_VIDEO_ENDPOINT_ID = <the id>` → **Save and Deploy**.

## Step 4 — Prove it (the real test — we do this together)
From the app or a direct curl with a valid beta license key:
```
POST https://algoai360-api.nickmeet282.workers.dev/api/video/generate
{ "license_key": "<beta key>", "prompt": "a golden retriever running on a beach", "duration": 4, "fps": 8 }
```
→ returns `{ ok, job_id }`. Then poll:
```
GET /api/video/status/<job_id>
```
→ eventually `{ status: "completed", url: "<video url>" }`. Open the url — **watch the video.**
First run is a cold start (model downloads onto the worker) — can be a few minutes; after
that it's fast. Watch the RunPod meter: it bills only while running, $0 idle.

## If it fails
The Worker auto-refunds the credits on any failed job (idempotent). Check the RunPod
endpoint **Logs** tab — the handler prints the exact error (model load / generation /
encode / upload). Paste that to Claude and we fix the handler.
