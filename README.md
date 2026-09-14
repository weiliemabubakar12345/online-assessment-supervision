# TEEP Online Assessment Monitoring Project

Shared research and prototype repository for the online assessment monitoring project. Module-specific code, documentation, evaluation results, and setup instructions will be added progressively.

## Modules

- [`exam-monitor-extension/`](exam-monitor-extension/) — Browser extension (MV3) that captures screen-side proctoring signals (tab switches, navigation, focus, idle, screenshots), a Node.js proctor server + live dashboard, and VLM screenshot-analysis services (single-model and a two-model cascade) that turn each screenshot into a `category` / `exam_relevance` / `is_cheating` verdict.
- [`computer_vision/`](computer_vision/src/integration/README.md) — Webcam-side computer-vision pipeline: reliability-aware head pose + coarse gaze estimation, YOLO object-cue detection, temporal event management, and a configurable multi-cue review score. Exposed to the extension over HTTP by `exam-monitor-extension/cv_service.py`.
