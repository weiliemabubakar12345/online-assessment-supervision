# Running the proctoring services on Kaggle (free GPU)

Three notebooks, all using Kaggle's free GPU instead of your own machine.
Each embeds its service source directly (`%%writefile` cells, kept in sync
with the corresponding file under `exam-monitor-extension/`) so it's
self-contained — none of them `git clone` this (private) repo, avoiding any
GitHub-credential entanglement inside the notebook.

| Notebook | Runs | Ports |
|---|---|---|
| [`teep_proctoring_services.ipynb`](teep_proctoring_services.ipynb) | `vlm_service.py` (single-model, Qwen2.5-VL-3B) + `cv_service.py` | VLM `8788`, CV `8789` |
| [`teep_cascade_vlm_service.ipynb`](teep_cascade_vlm_service.ipynb) | `vlm_service_cascade.py` only — no CV, no evaluation | VLM `8791` |
| [`teep_proctoring_services_cascade.ipynb`](teep_proctoring_services_cascade.ipynb) | `vlm_service_cascade.py` + `cv_service.py` | VLM `8791`, CV `8789` |

`vlm_service_cascade.py` runs a small model (SmolVLM-500M) that screens every
screenshot, escalating only the ones it cannot confidently clear to the large
model (Qwen2.5-VL) — see
[`../README.md`'s VLM cheating decision section](../README.md#vlm-cheating-decision-is_cheating)
for the routing logic. The same file also serves two single-model baselines
via `MODE=large_only` / `MODE=small_only` (set in each cascade notebook's
configuration section), so an accuracy/latency comparison isn't confounded by
a different code path. Pick the plain `teep_proctoring_services.ipynb` for the
already-deployed single-model service, or one of the cascade notebooks when
evaluating/running the cascade architecture instead.

Each service is exposed through its own `cloudflared` quick tunnel, the same
pattern already used for `VLM_URL` with a Colab-hosted VLM.

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
one explicitly, with `--no-deps`:
```bash
pip install --no-deps git+https://github.com/elliottzheng/face-detection.git@master
```
The notebook's dependency-install cell already includes this. The `--no-deps`
matters here specifically: this package's own `setup.py` pulls in an
unpinned `torch`, which can silently replace Kaggle's preinstalled
CUDA-linked `torch` build with a CPU-only one. If that happens, `cv_service.py`
now fails fast at boot with a clear message (rather than deep inside the
first `/frame` request) when `CV_DEVICE=cuda` is requested but
`torch.cuda.is_available()` is `False`. The notebook also has a standalone
GPU sanity-check cell right after the installs — if it fails, restart the
kernel and re-run from the top before continuing.

## 2. Run the notebook

1. Upload whichever notebook you need to Kaggle (or open it there directly).
   Attach the dataset from step 1 **unless** you're running
   `teep_cascade_vlm_service.ipynb` — that one has no CV service, so it needs
   no dataset.
2. **Settings → Accelerator → GPU**, **Internet → On**.
3. Run all cells top to bottom. The VLM service(s) additionally download
   their model(s) from Hugging Face on first run (a few GB each —
   `Qwen/Qwen2.5-VL-3B-Instruct` for the single-model service,
   `Qwen/Qwen2.5-VL-7B-Instruct` + `HuggingFaceTB/SmolVLM-500M-Instruct` for
   the cascade) — unrelated to the attached dataset.
4. The last cell prints one tunnel URL per running service. On the machine
   running `server.js`:
   ```bash
   # single-model notebook:
   VLM_URL=https://xxxx.trycloudflare.com CV_URL=https://yyyy.trycloudflare.com node server.js
   # cascade notebook (VLM_URL now points at the cascade service instead):
   VLM_URL=https://zzzz.trycloudflare.com CV_URL=https://yyyy.trycloudflare.com node server.js
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
