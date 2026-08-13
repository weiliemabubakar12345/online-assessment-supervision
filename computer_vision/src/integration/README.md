# Computer Vision Integration Runtime

This directory contains the frozen computer-vision integration baseline for
the online assessment monitoring prototype. It combines reliability-aware head
pose and coarse gaze estimation, YOLO object-cue detection, temporal event
management, and structured event logging.

The runtime reports suspicious visual cues for later review. It does not make
an automatic conclusion that a student is cheating.

## Runtime Files

| File | Responsibility |
| --- | --- |
| `01_head_gaze_adapter.py` | Head pose, L2CS gaze, calibration, eye reliability, and normalized head/gaze output |
| `02_yolo_output_adapter.py` | YOLO inference, checkpoint selection, class normalization, and object-cue output |
| `03_event_manager.py` | Frozen temporal rules, event eligibility, lifecycle management, and cross-module person validation |
| `04_event_logger.py` | Session metadata, JSONL event records, filtered candidates, completed-event CSV, and session summary |
| `run_integrated_demo.py` | Canonical webcam entry point, asynchronous YOLO worker, UI, controls, and runtime orchestration |

The numbered filenames are loaded dynamically by `run_integrated_demo.py`.
Do not rename them without updating the entry point.

## Required Local Assets

Model binaries are not tracked in ordinary Git. Download the approved assets
described in [`../../models/README.md`](../../models/README.md) and place them
in the following repository-relative locations:

```text
computer_vision/
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
resource rather than a trained model binary.

The current implementation also imports L2CS source code from a local
`L2CS-Net` checkout. Supply its root directory through `--l2cs-root`.

## Environment

Use the tested conda environment documented in
[`../../environments/README.md`](../../environments/README.md). From the
repository root:

```bat
conda activate teep-integration
```

## Run the Frozen Baseline

Run the following command from the repository root in Anaconda Prompt. Replace
the `--l2cs-root` value if the external L2CS checkout is stored elsewhere.

```bat
python -B computer_vision\src\integration\run_integrated_demo.py ^
  --l2cs-root "D:\Intern\NTUST\computer_vision\external_models\L2CS-Net" ^
  --snapshot "computer_vision\models\l2cs\L2CSNet_gaze360.pkl" ^
  --mediapipe-model "computer_vision\models\mediapipe\face_landmarker.task" ^
  --canonical14-csv "computer_vision\resources\mediapipe\mediapipe_expanded_subset_14.csv" ^
  --checkpoint 5e ^
  --head-gaze-device cpu ^
  --yolo-device cpu ^
  --async-yolo
```

The canonical baseline uses checkpoint `5e`, CPU execution, and asynchronous
latest-frame YOLO inference. Keep the checkpoint fixed within an evaluation
session.

## Window Modes and Controls

The default startup mode is fullscreen. Add `--windowed` for a 1600 x 900
resizable window:

```bat
--windowed
```

Optional dimensions may be supplied with `--window-width` and
`--window-height`. The UI reserves the lower panel area for controls and
compacts long event text in a 900-pixel-high window.

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
directory. Periodic status output is disabled by default because the live state
is already visible in the UI and structured records are written to disk.

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
- `filtered_candidates.jsonl`; and
- `session_summary.json`.

Screenshots are stored under:

```text
computer_vision/results/integration/screenshots/
```

The `results/` directory is runtime output and must not be committed.

## Frozen Runtime Notes

- The trained YOLO `laptop` label is normalized to `computer_device` by the
  output adapter.
- The cross-module person check can suppress a YOLO `NO_PERSON` candidate when
  the head/gaze adapter still provides reliable face evidence.
- The `audio_device` event rule is most reliable for larger headphones and
  headsets. Small earbuds remain unreliable under full-frame webcam inference.
- Full-frame detections, raw candidates, filtered candidates, and formal
  temporal events are separate outputs and should not be interpreted as the
  same level of evidence.
