# Running the proctoring services on Kaggle (free GPU)

[`teep_proctoring_services.ipynb`](teep_proctoring_services.ipynb) runs
**both** model services this project needs, in one Kaggle notebook, using
Kaggle's free GPU instead of your own machine:

- `vlm_service.py` — Qwen2.5-VL-3B screenshot analysis (port `8788`)
- `cv_service.py` — head/gaze + YOLO + review-score pipeline (port `8789`)

Each is exposed through its own `cloudflared` quick tunnel, the same pattern
already used for `VLM_URL` with a Colab-hosted VLM. The notebook embeds both
services' source directly (`%%writefile` cells, kept in sync with
`exam-monitor-extension/vlm_service.py` / `cv_service.py`) so it's
self-contained — it does not `git clone` this (private) repo, avoiding any
GitHub-credential entanglement inside the notebook.

## 1. One-time: package the CV assets as a Kaggle Dataset

The notebook needs the CV pipeline's model binaries and its four integration
source files. Prepare this **once**, upload as a private Kaggle Dataset, and
reuse it across notebook restarts:

```text
teep-kaggle-bundle/
├── integration/                            # from computer_vision/src/integration/
│   ├── 01_head_gaze_adapter.py
│   ├── 02_yolo_output_adapter.py
│   ├── 03_event_manager.py
│   └── 05_multi_cue_review_score.py
├── configs/
│   └── multi_cue_review_score_v1_1.json    # from computer_vision/configs/
├── resources/
│   └── mediapipe/
│       └── mediapipe_expanded_subset_14.csv    # from computer_vision/resources/mediapipe/
└── models/                                 # flat — no yolo/l2cs/mediapipe subfolders
    ├── original_5e_best.pt
    ├── L2CSNet_gaze360.pkl
    └── face_landmarker.task
```

Note: Kaggle's newer "Add Input" flow may mount the dataset at
`/kaggle/input/datasets/<your-username>/<slug>/` instead of the older
`/kaggle/input/<slug>/` — if the notebook's dataset-check cell reports
everything under `integration/`/`configs/` as found but the three files
under `models/`/`resources/` as missing, that usually means the top-level
`DATASET_DIR` prefix is already correct and only those subpaths are
wrong (check the actual layout with `os.walk` and adjust the notebook's
`CV_*` env vars in the "start services" cell to match, rather than
re-uploading the dataset).

- `integration/`, `configs/`, `resources/` are copied straight from this
  repo's `computer_vision/` folder as-is (read-only source, nothing to
  rewrite — `computer_vision/` itself is never modified by this feature).
- `models/` are the three files from the project's Google Drive folder (see
  `computer_vision/models/README.md` for the download link). Verify their
  SHA-256 against that file's Integrity Record after upload and again after
  attaching, to confirm nothing got corrupted in transit.

On kaggle.com: **Create → New Dataset**, upload with the layout above, name
it (e.g. `teep-kaggle-bundle`), keep it **private**, publish. In the
notebook: **Add Input → your dataset** → lands at
`/kaggle/input/teep-kaggle-bundle/...`. If your slug differs, change
`DATASET_SLUG` in the notebook's dataset-check cell.

### Known L2CS-Net dependency gotcha

`l2cs/pipeline.py` (inside the cloned L2CS-Net source, step 3 below) does
`from face_detection import RetinaFace` — a separate package that `pip
install l2cs` does **not** pull in on its own, and it is *not* the same
`face_detection` package you'd get from a plain `pip install face_detection`
on PyPI (that's an unrelated package with the same name). Without it,
`HeadGazeAdapter.start()` fails on every `/frame` request with
`ModuleNotFoundError: No module named 'face_detection'`. Install the correct
one explicitly:
```bash
pip install git+https://github.com/elliottzheng/face-detection.git@master
```
The notebook's dependency-install cell already includes this.

## 2. Run the notebook

1. Upload `teep_proctoring_services.ipynb` to Kaggle (or open it there
   directly), attach the dataset from step 1.
2. **Settings → Accelerator → GPU**, **Internet → On**.
3. Run all cells top to bottom. `vlm_service.py` additionally downloads
   `Qwen/Qwen2.5-VL-3B-Instruct` from Hugging Face on first run (a few GB) —
   unrelated to the attached dataset.
4. The last cell prints both tunnel URLs. On the machine running `server.js`:
   ```bash
   VLM_URL=https://xxxx.trycloudflare.com CV_URL=https://yyyy.trycloudflare.com node server.js
   ```

## 3. Verify before touching the extension

```bash
curl https://<cv-subdomain>.trycloudflare.com/health
curl -X POST https://<cv-subdomain>.trycloudflare.com/frame \
  -H "Content-Type: application/json" \
  -d '{"studentId":"smoke-test","image":"<base64 jpeg>"}'
```
Confirm `calibration_phase` eventually reaches `"READY"` after several
frames, and that `review_score` responds to deliberately holding up a
phone/book to the camera. Same idea for the VLM: `curl
https://<vlm-subdomain>.trycloudflare.com/health`.

## Known operational limitations

- **Kaggle free-GPU quota** is roughly 30 GPU-hours/week per account — plan
  test sessions accordingly.
- **Notebook sessions time out** (idle timeout, and a hard runtime cap) —
  both services and both tunnels die with the notebook.
- **Tunnel URLs change every restart.** There's no way around this with the
  free `cloudflared` quick-tunnel approach — re-run the tunnel cells and
  update `VLM_URL`/`CV_URL` on the `server.js` machine each time. This is
  the same manual workflow the repo already had for `VLM_URL` alone.
- This is a prototype control-flow, not a production deployment — there's no
  auth on either service's endpoints, so anyone with a tunnel URL can post
  frames/images to it. Fine for a research demo on a throwaway URL; not fine
  beyond that without adding at least a shared-secret header.
- Running both models on one GPU is untested for peak combined VRAM. If you
  hit an OOM, the notebook's limitations note points at lowering
  `MAX_PIXELS` in the `vlm_service.py` cell as the first thing to try.
