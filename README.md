# TEEP Online Assessment Monitoring Project

Shared research and prototype repository for the online assessment monitoring project. Module-specific code, documentation, evaluation results, and setup instructions will be added progressively.

## Modules

- [`exam-monitor-extension/`](exam-monitor-extension/) — Browser extension (MV3) that captures screen-side proctoring signals (tab switches, navigation, focus, idle, screenshots), a Node.js proctor server + live dashboard, and VLM screenshot-analysis services (single-model and a two-model cascade) that turn each screenshot into a `category` / `exam_relevance` / `is_cheating` verdict.
- [`computer_vision/`](computer_vision/src/integration/README.md) — Webcam-side computer-vision pipeline: reliability-aware head pose + coarse gaze estimation, YOLO object-cue detection, temporal event management, and a configurable multi-cue review score. Exposed to the extension over HTTP by `exam-monitor-extension/cv_service.py`.

## Getting Started

The two modules run independently and can be started in either order. Each
section below is the short path — follow the linked docs for details,
troubleshooting, and the full model/environment registry.

### Computer Vision (webcam) side

The CV pipeline needs its own Python environment, the frozen model assets,
and an external source checkout before it can run. If you already have the
`teep-integration` Conda environment set up per
[`computer_vision/environments/README.md`](computer_vision/environments/README.md),
skip step 1 and jump to step 4.

**1. Create the local Python environment (plain `venv`, no Conda)**

This is the quick alternative to the Conda setup — useful when Conda isn't
available. It is not yet tracked in the repo (see the note at the end of this
section), so for now build it from the pinned package list below.

```bash
cd computer_vision
python -m venv .venv
```

Activate it:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```
```bash
# macOS/Linux
source .venv/bin/activate
```

Then install the pinned dependencies:

```bash
pip install -r requirements_local.txt
```

This mirrors the versions in the canonical `teep-integration` Conda
environment (Python 3.9, NumPy 2.0.2, MediaPipe 0.10.35, OpenCV 5.0.0.93,
PyTorch 2.8.0, Ultralytics 8.4.86 — see the
[Key Versions table](computer_vision/environments/README.md#key-versions)).
The pipeline's canonical baseline runs on **CPU**, so no CUDA setup is
required for this path.

> **Note:** `computer_vision/requirements_local.txt` is currently a local,
> untracked file — it won't be there yet on a fresh clone. Ask whoever set up
> this environment to share it, or generate an equivalent with
> `pip freeze > requirements_local.txt` from a working environment, until
> it's committed to the repo.

**2. Download the model assets and clone the L2CS-Net source**

Follow [`computer_vision/models/README.md`](computer_vision/models/README.md)
to download `original_5e_best.pt`, `L2CSNet_gaze360.pkl`,
`face_landmarker.task` from the project's shared Drive folder, and clone
`Ahmednull/L2CS-Net` into `computer_vision/external/L2CS-Net/`. The runner
checks for these paths at startup and fails fast if any are missing.

**3. Verify the environment** (optional but recommended — see
[`computer_vision/environments/README.md`](computer_vision/environments/README.md#verify-the-installation)
for the exact commands.)

**4. Run the integrated demo**

```bash
cd computer_vision/src/integration
python -B run_integrated_demo.py
```

A webcam window opens showing head-pose/gaze/object cues and the live review
score. Press `Q` to quit; see
[`computer_vision/src/integration/README.md`](computer_vision/src/integration/README.md#window-modes-and-controls)
for the rest of the keyboard controls and structured output locations.

### VLM (screen) side

**1. Start the VLM (+ CV, optional) service on Kaggle**

Upload [`exam-monitor-extension/kaggle/teep_proctoring_services.ipynb`](exam-monitor-extension/kaggle/teep_proctoring_services.ipynb)
to Kaggle (attach the CV dataset bundle per
[`exam-monitor-extension/kaggle/README.md`](exam-monitor-extension/kaggle/README.md)),
enable **GPU** + **Internet**, and run all cells top to bottom.

**2. Note the tunnel URL(s) printed by the last cell**

The notebook prints one `cloudflared` URL per running service (VLM, and CV if
attached) — these change every time the notebook restarts.

**3. Point the proctor server at them and start it**

```bash
cd exam-monitor-extension
VLM_URL=https://xxxx.trycloudflare.com CV_URL=https://yyyy.trycloudflare.com node server.js
```

(Requires Node.js. On Windows PowerShell, set each var with
`$env:VLM_URL="https://xxxx.trycloudflare.com"` before running `node server.js`,
or edit the defaults directly in `server.js`.)

**4. Open the dashboard**

The server prints a local URL (`http://localhost:8787`) — open it in a
browser to reach the live proctor dashboard.

**5. Install the extension on each test-taker's machine**

Load the unpacked extension from `exam-monitor-extension/` (via
`manifest.json`) into their browser — see
[`exam-monitor-extension/README.md`](exam-monitor-extension/README.md#firefox--zen-browser)
for the exact Firefox/Zen and Chrome/Edge steps, since the manifest needs a
small edit for Chrome/Edge before loading.
