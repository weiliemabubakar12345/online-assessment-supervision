# Computer Vision Integration Runtime (Updated 17 August 2026)

This directory contains the frozen computer-vision integration baseline for the
online assessment monitoring prototype. It combines reliability-aware head pose
and coarse gaze estimation, YOLO object-cue detection, temporal event
management, configurable multi-cue review scoring, and structured audit logs.

The runtime reports visual cues for later human review. Its review score is an
experimental prioritisation indicator, not a calibrated probability of
cheating, and must not be used as an automatic verdict.

## Start Here

Complete the following steps from the repository root before launching the
integration runtime:

1. Create and activate the tested Conda environment by following
   [`../../environments/README.md`](../../environments/README.md).
2. Download the required model assets, clone the external L2CS-Net source, and
   verify their checksums by following
   [`../../models/README.md`](../../models/README.md).
3. Confirm that the tracked Canonical 14 geometry file exists at
   `computer_vision/resources/mediapipe/mediapipe_expanded_subset_14.csv`.
4. Run the integration tests listed in [Verification](#verification).
5. Start the webcam demo using the command in
   [Run the Frozen Baseline](#run-the-frozen-baseline).

The model and L2CS setup in step 2 is required. Installing the Python
environment alone does not provide the local model binaries or external L2CS
source code.

## Runtime Files

| File | Responsibility |
| --- | --- |
| `01_head_gaze_adapter.py` | Head pose, L2CS gaze, calibration, eye reliability, and normalized head/gaze output |
| `02_yolo_output_adapter.py` | YOLO inference, checkpoint selection, class normalization, and object-cue output |
| `03_event_manager.py` | Frozen temporal rules, event eligibility, lifecycle management, and cross-module person validation |
| `04_event_logger.py` | Session metadata, event/candidate records, review-score audit records, completed-event CSV, and session summary |
| `05_multi_cue_review_score.py` | Configurable contribution calculation, concurrent-cue bonus, review levels, and session peak tracking |
| `run_integrated_demo.py` | Canonical webcam entry point, asynchronous YOLO worker, UI, controls, and runtime orchestration |

The numbered modules are loaded dynamically by `run_integrated_demo.py`. Do not
rename them without updating the entry point.

The review-score parameters are stored outside the Python implementation in:

```text
computer_vision/configs/multi_cue_review_score_v1_1.json
```

This JSON file is the authoritative location for cue weights, confidence and
duration factors, concurrency bonuses, and review-level thresholds.

## Required Local Assets

Model binaries and the external L2CS checkout are intentionally excluded from
ordinary Git tracking. Follow [`../../models/README.md`](../../models/README.md)
for download sources, placement instructions, integrity checks, and licensing
notes.

The final local layout must include:

```text
computer_vision/
├── external/
│   └── L2CS-Net/
├── models/
│   ├── yolo/
│   │   └── original_5e_best.pt
│   ├── l2cs/
│   │   └── L2CSNet_gaze360.pkl
│   └── mediapipe/
│       └── face_landmarker.task
└── resources/
    └── mediapipe/
        └── mediapipe_expanded_subset_14.csv
```

The Canonical 14 CSV is a small tracked geometry resource, not a trained model
binary. Keep it under `resources/`; do not move it into an ignored model or
results directory.

At startup, the runner checks the required paths and reports any missing asset
before module initialization.

## Environment

Use the tested environment documented in
[`../../environments/README.md`](../../environments/README.md). From the
repository root:

```bat
conda activate teep-integration
```

Do not commit a local Conda environment folder. Only the cleaned environment
definition and its documentation belong in Git.

## Run the Frozen Baseline

When the required assets use the default repository-relative layout, run from
the repository root:

```bat
python -B computer_vision\src\integration\run_integrated_demo.py
```

No developer-specific absolute path is required. The frozen runtime settings
are:

| Setting | Frozen value |
| --- | --- |
| YOLO checkpoint | `5e` (`original_5e_best.pt`) |
| YOLO device | CPU |
| YOLO runtime mode | Asynchronous latest-frame inference |
| Base confidence threshold | `0.25` |
| Input size | `640` |
| IoU threshold | `0.45` |
| Audio-device formal-event threshold | `0.43` plus its temporal rule |
| Review-score configuration | `multi_cue_review_score_v1_1.json` |

Keep the checkpoint, runtime settings, event rules, and score configuration
fixed within a formal evaluation session.

### Optional Path Overrides

Use command-line overrides only when an asset is intentionally stored elsewhere
or when a controlled experiment requires an alternative. For example:

```bat
python -B computer_vision\src\integration\run_integrated_demo.py ^
  --l2cs-root "D:\path\to\L2CS-Net" ^
  --snapshot "D:\path\to\L2CSNet_gaze360.pkl" ^
  --mediapipe-model "D:\path\to\face_landmarker.task" ^
  --canonical14-csv "D:\path\to\mediapipe_expanded_subset_14.csv"
```

Run the following command for all available options:

```bat
python -B computer_vision\src\integration\run_integrated_demo.py --help
```

Do not save developer-specific absolute paths in committed code or
documentation.

## Review Score

For each currently active and temporally confirmed cue:

```text
cue contribution = cue weight x confidence factor x duration factor
```

The current score is:

```text
review score = min(1.0, sum of cue contributions + concurrency bonus)
```

Important scoring behaviour:

- duplicate detections of the same object class contribute only once;
- a cross-domain bonus may be added when different cue domains are active at
  the same time;
- release-grace, stale, filtered, or completed cues do not contribute;
- an `END` event removes that cue's current contribution; and
- the session peak is retained separately after the current score falls.

The default levels are loaded from the JSON configuration:

| Score | Level |
| --- | --- |
| `< 0.25` | `LOW` |
| `0.25` to `< 0.50` | `MODERATE` |
| `0.50` to `< 0.75` | `HIGH` |
| `>= 0.75` | `VERY_HIGH` |

These values are research configuration parameters, not statistical evidence of
misconduct. Review any flagged interval together with its underlying cues,
timestamps, reliability fields, and source recording.

## Window Modes and Controls

The default startup mode is fullscreen. Add `--windowed` for a resizable window;
optional dimensions can be supplied with `--window-width` and
`--window-height`.

| Key | Action |
| --- | --- |
| `C` | Confirm the neutral head baseline when calibration requests it |
| `R` | Restart calibration |
| `S` | Save a screenshot |
| `F` | Toggle fullscreen/windowed mode |
| `Q` | Quit cleanly and close the event-log session |

## Terminal Output

The default terminal mode prints startup information, calibration transitions,
event `START`/`END` records, and the final session directory. The live state is
shown in the UI and detailed records are written to disk.

Useful debugging options include:

```bat
--print-interval 5
--print-interval 5 --verbose-json
--print-active-events
```

`--print-active-events` can produce frequent output and is intended for
lifecycle debugging. MediaPipe or TensorFlow Lite may also print XNNPACK,
feedback-manager, or offline telemetry messages; these do not by themselves
mean that local inference or session saving failed.

## Structured Outputs

Each run creates a unique session under:

```text
computer_vision/results/integration/event_logs/<session_id>/
```

The session can contain:

- `session_metadata.json`
- `events.jsonl`
- `completed_events.csv`
- `filtered_candidates.jsonl`
- `review_scores.jsonl`
- `session_summary.json`

`review_scores.jsonl` stores per-frame score audit data, including active cues,
cue weights, confidence and duration factors, individual contributions,
concurrency bonus, current score, review level, and session peak.

Screenshots are stored under:

```text
computer_vision/results/integration/screenshots/
```

The `results/` directory contains runtime evidence and must not be committed.
Review logs and screenshots before sharing because they may contain sensitive or
identifiable assessment data.

## Verification

Run these checks from the repository root after completing the environment and
model setup:

```bat
python -B computer_vision\tests\integration\test_multi_cue_review_score.py
python -B computer_vision\tests\integration\test_review_score_event_logging.py
```

The tests verify external JSON loading, neutral scoring, same-class
deduplication, cross-domain concurrency, stale/release handling, `END` removal,
session peak retention, review-level thresholds, per-frame audit persistence,
and compatibility with the completed-events CSV.

Before committing, also run:

```bat
git diff --check
git status --short
```

Confirm that no model binaries, external source checkout, session logs,
screenshots, private recordings, or `__pycache__` directories are staged.

## Frozen Runtime Notes

- The trained YOLO `laptop` label is normalized to `computer_device` by the
  output adapter. It must not be described as a validated general monitor
  detector.
- The cross-module person check can suppress a YOLO `NO_PERSON` candidate when
  the head/gaze adapter still provides reliable face evidence.
- The `audio_device` rule is most reliable for larger headphones and headsets.
  Small earbuds remain unreliable under full-frame webcam inference.
- Raw detections, filtered candidates, formal temporal events, and the review
  score are different evidence stages and must not be interpreted as equivalent
  outputs.
