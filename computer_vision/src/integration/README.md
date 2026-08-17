# Computer Vision Integration Runtime

This directory contains the frozen computer-vision integration baseline for
the online assessment monitoring prototype. It combines reliability-aware head
pose and coarse gaze estimation, YOLO object-cue detection, temporal event
management, configurable multi-cue review scoring, and structured audit logs.

The runtime reports suspicious visual cues for later human review. The review
score is an experimental prioritisation indicator: it is not a calibrated
probability of cheating and must not be used as an automatic verdict.

## Runtime Files

| File | Responsibility |
| --- | --- |
| `01_head_gaze_adapter.py` | Head pose, L2CS gaze, calibration, eye reliability, and normalized head/gaze output |
| `02_yolo_output_adapter.py` | YOLO inference, checkpoint selection, class normalization, and object-cue output |
| `03_event_manager.py` | Frozen temporal rules, event eligibility, lifecycle management, and cross-module person validation |
| `04_event_logger.py` | Session metadata, event/candidate records, review-score audit records, completed-event CSV, and session summary |
| `05_multi_cue_review_score.py` | Configurable contribution calculation, concurrent-cue bonus, review levels, and session peak tracking |
| `run_integrated_demo.py` | Canonical webcam entry point, asynchronous YOLO worker, UI, controls, and runtime orchestration |

The numbered filenames are loaded dynamically by `run_integrated_demo.py`.
Do not rename them without updating the entry point.

The review-score parameters are stored outside the Python implementation in:

```text
computer_vision/configs/multi_cue_review_score_v1_1.json
```

This JSON file is the main place to tune cue weights, confidence factors,
duration factors, concurrency bonuses, and review-level thresholds.

## Required Local Assets

Model binaries and the external L2CS source checkout are intentionally not
tracked in ordinary Git. Place the approved local assets at these default
repository-relative locations:

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

The Canonical 14 CSV is tracked in Git because it is a small runtime geometry
resource rather than a trained model binary. Keep it under `resources/`; do not
place it under an ignored model or results directory.

## Environment

Use the tested conda environment documented in
[`../../environments/README.md`](../../environments/README.md). From the
repository root:

```bat
conda activate teep-integration
```

## Run the Frozen Baseline

When the assets use the default layout above, run this command from the
repository root:

```bat
python -B computer_vision\src\integration\run_integrated_demo.py
```

The script resolves its defaults relative to the repository, so the README
does not need a developer-specific absolute path. The canonical baseline uses
the `5e` YOLO checkpoint, CPU execution, and asynchronous latest-frame YOLO
inference.

Use command-line arguments only when an asset is stored elsewhere or when a
controlled experiment requires an override. For example:

```bat
python -B computer_vision\src\integration\run_integrated_demo.py ^
  --l2cs-root "D:\path\to\L2CS-Net" ^
  --snapshot "D:\path\to\L2CSNet_gaze360.pkl" ^
  --mediapipe-model "D:\path\to\face_landmarker.task" ^
  --canonical14-csv "D:\path\to\mediapipe_expanded_subset_14.csv"
```

Run `python -B computer_vision\src\integration\run_integrated_demo.py --help`
for all optional inference, device, logging, screenshot, and window settings.
Keep the checkpoint and scoring configuration fixed within an evaluation
session so that results remain comparable.

## Review Score

The review score combines the suspicious cues that are currently active. In
simple terms, each cue contributes:

```text
cue contribution = class weight × confidence factor × duration factor
```

The current score is then:

```text
review score = min(1.0, sum of active cue contributions + concurrency bonus)
```

Important scoring behaviour:

- duplicate detections of the same object class contribute only once;
- a cross-domain bonus may be added when different cue types are active at the
  same time, such as a phone plus looking away;
- released, stale, or completed cues do not contribute to the current score;
- an `END` event removes that cue's current contribution; and
- the session peak is retained separately even after the current score falls.

The default review levels are configured in JSON:

| Score | Level |
| --- | --- |
| `< 0.25` | `LOW` |
| `0.25` to `< 0.50` | `MODERATE` |
| `0.50` to `< 0.75` | `HIGH` |
| `>= 0.75` | `VERY_HIGH` |

These thresholds and weights are configuration values, not statistical proof
of misconduct. Any flagged interval requires human review together with its
underlying cues and timestamps.

## Window Modes and Controls

The default startup mode is fullscreen. Add `--windowed` for a resizable
window. Optional dimensions may be supplied with `--window-width` and
`--window-height`.

| Key | Action |
| --- | --- |
| `C` | Confirm the neutral head baseline when calibration requests it |
| `R` | Restart calibration |
| `S` | Save a screenshot |
| `F` | Toggle fullscreen/windowed mode |
| `Q` | Quit cleanly and close the event-log session |

## Terminal Output

The default terminal mode is intentionally concise. It prints startup details,
calibration transitions, event `START`/`END` records, and the final session
directory. The live state remains visible in the UI and structured records are
written to disk.

Enable a one-line status every five seconds when debugging:

```bat
--print-interval 5
```

Enable full periodic adapter and event-manager JSON:

```bat
--print-interval 5 --verbose-json
```

Use `--print-active-events` only for detailed lifecycle debugging because it
can produce frequent terminal output.

MediaPipe or TensorFlow Lite may print XNNPACK and feedback-manager messages at
startup. A Clearcut telemetry upload failure may also appear after shutdown in
an offline or restricted network. These third-party messages do not indicate
that the local models, event logger, or saved session failed.

## Structured Outputs

By default, each run creates a unique session under:

```text
computer_vision/results/integration/event_logs/<session_id>/
```

The session can contain:

- `session_metadata.json`;
- `events.jsonl`;
- `completed_events.csv`;
- `filtered_candidates.jsonl`;
- `review_scores.jsonl`; and
- `session_summary.json`.

Screenshots are stored under:

```text
computer_vision/results/integration/screenshots/
```

The `results/` directory contains runtime evidence and must not be committed.
Review logs and screenshots before sharing because they may contain sensitive
or identifiable assessment data.

## Verification

Run the integration checks from the repository root before committing:

```bat
python -B computer_vision\tests\integration\test_multi_cue_review_score.py
python -B computer_vision\tests\integration\test_review_score_event_logging.py
```

The review-score test verifies external JSON loading, neutral scoring,
same-class deduplication, cross-domain concurrency, stale/release handling,
`END` removal, session peak retention, and JSON review-level thresholds.

## Frozen Runtime Notes

- The trained YOLO `laptop` label is normalized to `computer_device` by the
  output adapter.
- The cross-module person check can suppress a YOLO `NO_PERSON` candidate when
  the head/gaze adapter still provides reliable face evidence.
- The `audio_device` event rule is most reliable for larger headphones and
  headsets. Small earbuds remain unreliable under full-frame webcam inference.
- Full-frame detections, raw candidates, filtered candidates, formal temporal
  events, and the review score represent different evidence stages and should
  not be interpreted as equivalent outputs.
