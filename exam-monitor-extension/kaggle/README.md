# Running cv_service.py on Kaggle (free GPU)

`cv_service.py` (in the parent folder) is a headless HTTP wrapper around this
project's `computer_vision/` proctoring pipeline. This doc runs it inside a
Kaggle notebook — for its free GPU — and exposes it to the local extension
over a tunnel, the same way `VLM_URL` already points `server.js` at a
Cloudflare-tunneled Colab notebook for the screenshot VLM.

This is a copy-pasteable set of notebook cells, not a committed `.ipynb`
(notebook JSON doesn't diff/review well, and Kaggle notebooks are edited in
Kaggle's own UI anyway).

## 1. One-time: package the model assets as a Kaggle Dataset

The three required model files + the L2CS-Net source live on the project's
Google Drive (see `computer_vision/models/README.md`). Downloading them from
Drive inside the notebook every session is slow and Drive's large-file
consent screen gets in the way of `gdown`. Instead, upload them **once** as a
private Kaggle Dataset so they persist across notebook restarts:

1. Download from Drive to your own machine:
   - `original_5e_best.pt` (YOLO)
   - `L2CSNet_gaze360.pkl` (L2CS gaze)
   - `face_landmarker.task` (MediaPipe)
2. On kaggle.com -> **Create -> New Dataset**, upload the three files, keep
   this layout so the paths below match:
   ```text
   teep-cv-models/
   ├── yolo/original_5e_best.pt
   ├── l2cs/L2CSNet_gaze360.pkl
   └── mediapipe/face_landmarker.task
   ```
3. Name it (e.g. `teep-cv-models`), keep it **private**, publish.
4. In your inference notebook, **Add Input -> your dataset** — it lands at
   `/kaggle/input/teep-cv-models/...`.

Verify checksums against `computer_vision/models/README.md`'s Integrity
Record after upload and again after attaching, to confirm nothing got
corrupted in transit.

## 2. Notebook cells

Enable **Internet** and a **GPU accelerator** (Settings, right sidebar) before
running anything below.

**Clone the repo (code only — the model dataset is attached separately):**
```bash
!git clone -b exam-monitor-integration https://github.com/<org>/online-assessment-supervision /kaggle/working/repo
```

**Clone L2CS-Net source** (the adapter imports from this checked-out
directory via `sys.path.insert`, not from the `l2cs` pip package — see
`computer_vision/models/README.md`):
```bash
!git clone https://github.com/Ahmednull/L2CS-Net /kaggle/working/L2CS-Net
```

**Install Python dependencies.** Kaggle's GPU image already ships a working
CUDA-linked `torch`/`torchvision` — do **not** force-reinstall them, that
risks breaking the preinstalled CUDA wiring. Install only what's missing,
matching `computer_vision/environments/teep_integration.yml`'s versions
where practical:
```bash
!pip install --no-deps l2cs==0.0.1
!pip install flask ultralytics==8.4.86 mediapipe==0.10.35 opencv-python-headless==5.0.0.93
```
If `cv_service.py` fails to boot with a torch/torchvision incompatibility
error, that's the point to investigate pinning them — don't pre-emptively
pin without seeing an actual failure first.

**Point the service at the Kaggle layout and start it in the background:**
```bash
import os, subprocess

env = os.environ.copy()
env.update({
    "CV_DEVICE": "cuda",
    "CV_YOLO_MODEL_DIR": "/kaggle/input/teep-cv-models/yolo",
    "CV_L2CS_ROOT": "/kaggle/working/L2CS-Net",
    "CV_L2CS_SNAPSHOT": "/kaggle/input/teep-cv-models/l2cs/L2CSNet_gaze360.pkl",
    "CV_MEDIAPIPE_MODEL": "/kaggle/input/teep-cv-models/mediapipe/face_landmarker.task",
    # CV_CANONICAL14_CSV and CV_REVIEW_CONFIG default to the repo-relative
    # paths under /kaggle/working/repo/computer_vision/, no override needed.
})
subprocess.Popen(
    ["python", "/kaggle/working/repo/exam-monitor-extension/cv_service.py"],
    env=env,
    stdout=open("/kaggle/working/cv_service.log", "w"),
    stderr=subprocess.STDOUT,
)
```
Wait ~10-20s (model loading), then check the log for `serving on
http://0.0.0.0:8789` before continuing:
```bash
!tail -n 40 /kaggle/working/cv_service.log
!curl -s http://localhost:8789/health
```

**Expose it via a `cloudflared` tunnel** (same pattern as the existing
`VLM_URL`/`trycloudflare.com` comment in `server.js`):
```bash
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
get_ipython().system_raw('./cloudflared tunnel --url http://localhost:8789 > cloudflared.log 2>&1 &')
```
Wait a few seconds, then grab the printed URL:
```bash
!grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' cloudflared.log | head -n 1
```

## 3. Point the extension's server at it

On the machine running `server.js`:
```bash
CV_URL=https://<the-printed-subdomain>.trycloudflare.com node exam-monitor-extension/server.js
```

## 4. Verify before touching the extension

```bash
curl https://<subdomain>.trycloudflare.com/health
# then, with a real base64 JPEG:
curl -X POST https://<subdomain>.trycloudflare.com/frame \
  -H "Content-Type: application/json" \
  -d '{"studentId":"smoke-test","image":"<base64 jpeg>"}'
```
Confirm `calibration_phase` eventually reaches `"READY"` after several
frames, and that `review_score` responds to deliberately holding up a
phone/book to the camera.

## Known operational limitations

- **Kaggle free-GPU quota** is roughly 30 GPU-hours/week per account — plan
  test sessions accordingly.
- **Notebook sessions time out** (idle timeout, and a hard runtime cap) —
  the service and tunnel die with the notebook.
- **The tunnel URL changes every restart.** There's no way around this with
  the free `cloudflared` quick-tunnel approach — `CV_URL` must be updated
  manually each time the notebook (and therefore the tunnel) restarts. This
  is the same manual workflow the repo already has for `VLM_URL` with its
  Colab-hosted VLM service; it isn't a new limitation this feature
  introduces.
- This is a prototype control-flow, not a production deployment — there's no
  auth on `cv_service.py`'s endpoints, so anyone with the tunnel URL can post
  frames to it. Fine for a research demo on a throwaway URL; not fine beyond
  that without adding at least a shared-secret header.
