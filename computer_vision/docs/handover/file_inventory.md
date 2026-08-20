# Computer Vision Handover File Inventory

## 1. Purpose and Audit Basis

This inventory records the canonical Computer Vision files present in the
combined `computer-vision` branch archive audited on 20 August 2026. It replaces
the earlier pre-merge inventory, which referred to local development paths and
files that had not yet been selected.

The audit archive contained tracked files only. An item marked **not present**
may exist elsewhere in the project; the label means only that it was not in the
audited branch archive.

| Status | Meaning |
| --- | --- |
| **Tracked** | Present in the audited branch and suitable for ordinary Git tracking |
| **External** | Required locally but intentionally distributed or cloned outside ordinary Git tracking |
| **Not present** | Expected handover evidence or documentation not found in the audited archive |
| **Excluded** | Private, raw, temporary, redundant, or otherwise unsuitable for the shared repository |

## 2. Canonical Tracked Files

### 2.1 Entry, Policy, and Configuration

| Path | Status | Role |
| --- | --- | --- |
| `computer_vision/README.md` | Tracked | Component overview, Quick Start, links, and handover status |
| `computer_vision/.gitignore` | Tracked | Module-specific data, model, media, output, and environment exclusions |
| `computer_vision/configs/multi_cue_review_score_v1_1.json` | Tracked | Authoritative review-score weights, factors, bonuses, and levels |

### 2.2 Documentation

| Path | Status | Role |
| --- | --- | --- |
| `computer_vision/docs/project_scope.md` | Tracked | Project boundary, objectives, evaluation scope, and responsible use |
| `computer_vision/docs/module_overview.md` | Tracked | Architecture, interfaces, event lifecycle, scoring, and output flow |
| `computer_vision/docs/datasets/class_mapping.md` | Tracked | Seven-class model and integration-facing label mapping |
| `computer_vision/docs/datasets/dataset_sources.md` | Tracked | Dataset provenance and Dataset V3 composition |
| `computer_vision/docs/handover/file_inventory.md` | Tracked | This combined-branch inventory |

### 2.3 Environment, Models, and Resources

| Path | Status | Role |
| --- | --- | --- |
| `computer_vision/environments/README.md` | Tracked | Conda creation, verification, and clean-setup status |
| `computer_vision/environments/teep_integration.yml` | Tracked | Frozen environment snapshot |
| `computer_vision/models/README.md` | Tracked | Required assets, canonical paths, source, integrity, and distribution policy |
| `computer_vision/resources/mediapipe/mediapipe_expanded_subset_14.csv` | Tracked | Canonical 14 geometry used by the head-pose pipeline |

### 2.4 Frozen Integration Runtime

| Path | Status | Role |
| --- | --- | --- |
| `computer_vision/src/integration/README.md` | Tracked | Required setup, model download link, runtime commands, outputs, and verification |
| `computer_vision/src/integration/01_head_gaze_adapter.py` | Tracked | Head pose, L2CS gaze, calibration, eye reliability, and normalized output |
| `computer_vision/src/integration/02_yolo_output_adapter.py` | Tracked | YOLO inference, class mapping, checkpoint selection, and freshness metadata |
| `computer_vision/src/integration/03_event_manager.py` | Tracked | Cross-module validation, temporal rules, and event lifecycle |
| `computer_vision/src/integration/04_event_logger.py` | Tracked | Event, candidate, score, metadata, CSV, and summary logging |
| `computer_vision/src/integration/05_multi_cue_review_score.py` | Tracked | Experimental Visual-Cue Review Score implementation |
| `computer_vision/src/integration/run_integrated_demo.py` | Tracked | Canonical webcam entry point and runtime orchestration |

The `multi_cue` filenames are retained for code and configuration compatibility.
Project-facing documentation uses **Experimental Visual-Cue Review Score**.

### 2.5 Evaluation and Tests

| Path | Status | Role |
| --- | --- | --- |
| `computer_vision/evaluation/README.md` | Tracked | Evaluation folder guide and evidence-status boundary |
| `computer_vision/evaluation/protocols/final_end_to_end_protocol.md` | Tracked | Frozen 30-scenario, 90-trial protocol |
| `computer_vision/evaluation/scripts/05_structured_integration_evaluation_protocol.py` | Tracked | Interactive timing, prompting, resume, and trial-log helper |
| `computer_vision/evaluation/templates/end_to_end_trial_log.csv` | Tracked | Blank formal trial-log template; not completed results |
| `computer_vision/tests/integration/test_multi_cue_review_score.py` | Tracked | Synthetic score/configuration checks |
| `computer_vision/tests/integration/test_review_score_event_logging.py` | Tracked | Logger compatibility and score-audit persistence checks |

Both synthetic test scripts passed during the 20 August 2026 archive audit.
The evaluation helper passed syntax checking and matched all 30 scenarios and
90 template rows. All integration and test Python files also passed
`py_compile` syntax checking. This does not replace a clean-machine webcam
smoke test with the required model assets.

## 3. Required External Runtime Items

These items are required for the full runtime but must remain outside ordinary
project Git tracking:

| Canonical local path | Status | Source / verification |
| --- | --- | --- |
| `computer_vision/models/yolo/original_5e_best.pt` | External | Project shared Drive; verify size and SHA-256 in `models/README.md` |
| `computer_vision/models/l2cs/L2CSNet_gaze360.pkl` | External | Project shared Drive; verify size and SHA-256 |
| `computer_vision/models/mediapipe/face_landmarker.task` | External | Project shared Drive; verify size and SHA-256 |
| `computer_vision/external/L2CS-Net/` | External | Clone official upstream revision `a4d8f7fa5436a2b2b9f088471623b552a85811bd` |

Optional 15e, 30e, and 50e YOLO comparison checkpoints are documented in the
model registry and should be distributed only when comparison reproduction is
required.

## 4. Handover Artifacts Not Present in the Audited Archive

The following should be considered for addition before final repository
handover. Do not invent or reconstruct numerical results from the blank
template.

| Recommended destination | Status | Required action |
| --- | --- | --- |
| `computer_vision/evaluation/results/final_end_to_end_trial_log.csv` | Not present | Add the reviewed, filled formal master log after removing private paths or identifiers |
| `computer_vision/evaluation/results/final_end_to_end_summary.csv` | Not present | Add scenario-level outcomes with denominator and metric definitions |
| `computer_vision/evaluation/results/README.md` | Not present | Explain evaluation date, hardware, frozen configuration, result files, and interpretation |
| `computer_vision/docs/limitations.md` | Not present | Consolidate fragmentation, multi-cue sensitivity, identity ambiguity, pose/gaze limits, and object-class limits |
| `computer_vision/docs/architecture/` | Not present | Add the reviewed final architecture source and a shareable export if available |
| `computer_vision/docs/methodology/` | Not present | Add the final research-method description if it is part of the handover scope |
| `computer_vision/results/` | Not present | Add only selected, reviewed, anonymized tables or figures; never raw session logs or private frames |

If the final presentation is required for project handover, place a reviewed
copy under the repository's agreed project-level presentation area rather than
duplicating it inside the runtime source directory.

## 5. Verification Still Pending

- Confirm intended-recipient access to the shared model folder.
- Test a clean clone at the recorded L2CS-Net revision
  `a4d8f7fa5436a2b2b9f088471623b552a85811bd`.
- Recreate the Conda environment from `teep_integration.yml` on a clean machine.
- Run both synthetic tests and a full webcam startup/calibration/quit smoke test
  using the documented repository-relative asset layout.
- Confirm the session closes with matched lifecycle transitions and no open
  events.
- Review every proposed result or media artifact for identifiers and private
  filesystem paths before staging.

Pending verification must stay labelled as pending; it should not be converted
into a success statement without evidence.

## 6. Explicit Exclusions

Do not upload:

- raw or redistributed datasets;
- identifiable webcam recordings, screenshots, or assessment content;
- raw integration session directories and debug logs;
- model binaries or the external L2CS-Net checkout;
- credentials, `.env` files, or developer-specific absolute paths;
- local Conda/virtual-environment directories and caches;
- superseded `_updated`, `_final`, or duplicate experimental scripts when a
  canonical tracked implementation already exists;
- unreviewed notebook outputs; or
- Daily Worklogs unless the project owner explicitly approves them for the
  shared repository.

## 7. Final Pre-Push Check

From the repository root:

```bat
python -B computer_vision\tests\integration\test_multi_cue_review_score.py
python -B computer_vision\tests\integration\test_review_score_event_logging.py
git diff --check
git status --short
```

Inspect the staged file list before committing. A clean documentation commit
must not contain model files, archives, private media, external repositories,
session logs, or cache directories.
