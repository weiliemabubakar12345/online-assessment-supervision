"""Headless CV proctoring inference service for exam-monitor-extension.

Wraps the ../computer_vision project's integration modules (01_head_gaze_adapter,
02_yolo_output_adapter, 03_event_manager, 05_multi_cue_review_score) behind a
small stateful HTTP API, so inference can run on a remote GPU (e.g. a Kaggle
notebook, see kaggle/README.md) instead of the student's own machine.

This file only IMPORTS those modules by path (the same dynamic-loading trick
run_integrated_demo.py uses, since filenames starting with digits aren't valid
Python identifiers). It never modifies anything under computer_vision/ — that
directory belongs to a teammate's module.

Endpoints:
  GET  /health                                  -> service + session status
  POST /session/start  { studentId }            -> warm up a session
  POST /frame  { studentId, image, timestamp? } -> one review-score reading
  POST /session/end    { studentId }            -> release a session's resources

Run:
  python cv_service.py            # serves http://localhost:8789

Configuration is entirely via CV_* environment variables (see the block below)
because the Kaggle filesystem layout differs from this repo's local layout.

The review_score/review_level values below come straight from
05_multi_cue_review_score.py, whose own module docstring is explicit that this
is "an experimental prioritisation indicator, not a calibrated probability of
cheating, and must not be used as an automatic verdict" — keep that framing in
any downstream UI/logs; the "disclaimer" field in every /frame response is
that exact string, passed through verbatim.
"""
import base64
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

import cv2
import numpy as np
from flask import Flask, request, jsonify

EXT_DIR = Path(__file__).resolve().parent
CV_ROOT = EXT_DIR.parent / "computer_vision"
# Overridable because a Kaggle notebook lays the integration modules out flat
# (from an attached Dataset) rather than in this repo's computer_vision/src/
# integration/ layout.
INTEGRATION_DIR = Path(os.environ.get("CV_INTEGRATION_DIR") or (CV_ROOT / "src" / "integration"))

PORT = int(os.environ.get("CV_PORT", "8789"))
IDLE_SESSION_TIMEOUT_S = float(os.environ.get("CV_IDLE_SESSION_TIMEOUT_S", "600"))

L2CS_ROOT = Path(os.environ.get("CV_L2CS_ROOT") or (CV_ROOT / "external" / "L2CS-Net"))
L2CS_SNAPSHOT = Path(os.environ.get("CV_L2CS_SNAPSHOT") or (CV_ROOT / "models" / "l2cs" / "L2CSNet_gaze360.pkl"))
MEDIAPIPE_MODEL = Path(os.environ.get("CV_MEDIAPIPE_MODEL") or (CV_ROOT / "models" / "mediapipe" / "face_landmarker.task"))
CANONICAL14_CSV = Path(os.environ.get("CV_CANONICAL14_CSV") or (CV_ROOT / "resources" / "mediapipe" / "mediapipe_expanded_subset_14.csv"))
YOLO_MODEL_DIR = Path(os.environ.get("CV_YOLO_MODEL_DIR") or (CV_ROOT / "models" / "yolo"))
YOLO_CHECKPOINT = os.environ.get("CV_YOLO_CHECKPOINT", "5e")
REVIEW_CONFIG_PATH = Path(os.environ.get("CV_REVIEW_CONFIG") or (CV_ROOT / "configs" / "multi_cue_review_score_v1_1.json"))


def _load_local_module(path: Path, module_name: str) -> ModuleType:
    """Import a numbered integration module by file path (identifiers can't start with 01/02/...)."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Required integration module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_device() -> str:
    env_device = os.environ.get("CV_DEVICE")
    if env_device:
        # An explicit CV_DEVICE=cuda is trusted at face value here, but NOT
        # left unvalidated -- see the torch.cuda.is_available() check right
        # after this function is called below. Without that check, a stale
        # CUDA_VISIBLE_DEVICES or a pip install that clobbered the CUDA-linked
        # torch build (e.g. an unpinned transitive dependency) only surfaces
        # as a confusing failure deep inside the first /frame request
        # ("Invalid CUDA 'device=0' requested") instead of a clear boot error.
        return env_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _adapter_device_string(device: str) -> str:
    """Normalize a device string for the adapters below.

    01_head_gaze_adapter.py calls l2cs's own select_device(), an
    older YOLOv5-vintage helper that only understands '', 'cpu', an index
    ('0'), or a comma list ('0,1') -- NOT a bare 'cuda' with no index. Given
    'cuda', that helper ends up doing `os.environ['CUDA_VISIBLE_DEVICES'] =
    'cuda'` (a non-numeric value), which the CUDA driver then rejects with
    "Invalid device id". A bare 'cuda' is normalized to '0' (first visible
    GPU) here; an explicit index, 'cuda:0', or 'cpu' passes through
    unchanged. Applied to both adapters' device config for consistency, even
    though Ultralytics YOLO's own device parsing already tolerates 'cuda'.
    """
    return "0" if device.strip().lower() == "cuda" else device


def _require_paths(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print("cv_service: missing required CV assets:")
        for m in missing:
            print("  -", m)
        print(
            "See computer_vision/models/README.md for local setup, or "
            "exam-monitor-extension/kaggle/README.md for the Kaggle layout, "
            "and set the CV_* environment variables if assets live elsewhere."
        )
        sys.exit(1)


print("cv_service: loading integration modules from", INTEGRATION_DIR)
head_module = _load_local_module(INTEGRATION_DIR / "01_head_gaze_adapter.py", "teep_head_gaze_adapter")
yolo_module = _load_local_module(INTEGRATION_DIR / "02_yolo_output_adapter.py", "teep_yolo_output_adapter")
event_module = _load_local_module(INTEGRATION_DIR / "03_event_manager.py", "teep_event_manager")
review_module = _load_local_module(INTEGRATION_DIR / "05_multi_cue_review_score.py", "teep_multi_cue_review_score")

DEVICE = _resolve_device()

if DEVICE == "cuda":
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except Exception as e:
        cuda_ok = False
        print(f"cv_service: torch import/CUDA check failed: {e}")
    if not cuda_ok:
        print(
            "cv_service: CV_DEVICE=cuda was requested but "
            "torch.cuda.is_available() is False -- refusing to start with a "
            "device that will fail on the first real request instead of here.\n"
            "Common causes on Kaggle: GPU accelerator not enabled for this "
            "session, or a pip install (e.g. an unpinned transitive "
            "dependency) silently replaced the preinstalled CUDA-linked "
            "torch build with a CPU-only one -- try restarting the kernel "
            "and re-running the install cell with --no-deps where possible.\n"
            "Set CV_DEVICE=cpu to run on CPU instead (slow) once you've "
            "confirmed that's actually what you want."
        )
        sys.exit(1)

_require_paths(L2CS_ROOT, L2CS_SNAPSHOT, MEDIAPIPE_MODEL, CANONICAL14_CSV, REVIEW_CONFIG_PATH)

print(f"cv_service: device={DEVICE} | yolo_checkpoint={YOLO_CHECKPOINT}")

REVIEW_SCORE_CONFIG = review_module.load_review_score_config(REVIEW_CONFIG_PATH)

# Stateless per frame (no calibration, no temporal state) -> one shared instance
# is safe across every student session and avoids reloading the checkpoint per student.
print("cv_service: loading YOLO checkpoint...")
_yolo_adapter = yolo_module.YoloOutputAdapter(
    config=yolo_module.YoloAdapterConfig(checkpoint_id=YOLO_CHECKPOINT, device=_adapter_device_string(DEVICE)),
    model_dir=YOLO_MODEL_DIR,
)
_yolo_adapter.start()

# Frozen defaults (README-documented event thresholds) — not overridden here.
_EVENT_RULES = event_module.build_initial_integration_rules(enable_audio_device_event=True)
_OBJECT_EVENT_ELIGIBILITY = event_module.build_initial_object_event_eligibility()


class _Session:
    """Per-studentId state. HeadGazeAdapter/EventManager/MultiCueReviewScorer are
    all stateful (calibration, temporal cue tracking, session-peak score), so
    each student needs their own instance — unlike the shared YOLO adapter above.
    """

    __slots__ = ("student_id", "head_adapter", "event_manager", "review_scorer", "last_seen", "frame_count")

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.head_adapter = head_module.HeadGazeAdapter(
            l2cs_root=L2CS_ROOT,
            l2cs_snapshot=L2CS_SNAPSHOT,
            mediapipe_model=MEDIAPIPE_MODEL,
            canonical14_csv=CANONICAL14_CSV,
            config=head_module.HeadGazeConfig(device=_adapter_device_string(DEVICE), mirror_input=False),
        )
        self.head_adapter.start()
        self.event_manager = event_module.EventManager(
            _EVENT_RULES,
            object_event_eligibility=_OBJECT_EVENT_ELIGIBILITY,
            suppress_no_person_if_face_present=True,
        )
        self.review_scorer = review_module.MultiCueReviewScorer(config=REVIEW_SCORE_CONFIG)
        self.last_seen = time.time()
        self.frame_count = 0

    def close(self) -> None:
        try:
            self.head_adapter.close()
        except Exception as e:
            print(f"cv_service: error closing session {self.student_id!r}: {e}")


_sessions: Dict[str, _Session] = {}
_sessions_lock = threading.Lock()
# One Kaggle GPU behind this service -> serialize every model call across
# sessions, same reasoning vlm_service.py already applies with its own lock.
_inference_lock = threading.Lock()


def _get_or_create_session(student_id: str) -> _Session:
    with _sessions_lock:
        session = _sessions.get(student_id)
        if session is not None:
            return session

    # Cold-starting a session calls HeadGazeAdapter.start(), which loads L2CS +
    # MediaPipe onto the GPU. Serialize that against in-flight /frame inference
    # through the same _inference_lock _process_frame uses below -- one Kaggle
    # GPU, avoid a new session's model load racing an active inference call
    # for CUDA memory (this raced uninstrumented before and could crash the
    # request with an unhandled exception -- see the try/except in the routes
    # that call this function).
    with _inference_lock:
        with _sessions_lock:
            session = _sessions.get(student_id)
            if session is not None:
                return session  # created by another thread while we waited
        print(f"cv_service: starting session for studentId={student_id!r}")
        session = _Session(student_id)
        with _sessions_lock:
            _sessions[student_id] = session
        return session


def _drop_session(student_id: str) -> bool:
    with _sessions_lock:
        session = _sessions.pop(student_id, None)
    if session is None:
        return False
    session.close()
    return True


def _evict_idle_sessions_loop() -> None:
    # No "press Q to quit" teardown here (unlike the desktop demo), so idle
    # sessions must be reaped or each one leaks a worker thread + landmarker.
    while True:
        time.sleep(60.0)
        now = time.time()
        with _sessions_lock:
            stale_ids = [sid for sid, s in _sessions.items() if now - s.last_seen > IDLE_SESSION_TIMEOUT_S]
        for sid in stale_ids:
            print(f"cv_service: evicting idle session studentId={sid!r}")
            _drop_session(sid)


threading.Thread(target=_evict_idle_sessions_loop, name="cv-session-reaper", daemon=True).start()


def _decode_frame(image_field: str) -> np.ndarray:
    s = image_field
    if isinstance(s, str) and s.strip().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image as JPEG/PNG.")
    return frame


def _process_frame(session: _Session, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
    with _inference_lock:
        head_output = session.head_adapter.process_frame(frame)
        calibration = head_output.get("calibration") or {}
        if calibration.get("head_confirmation_required"):
            # Scripted equivalent of the desktop demo's "press C" step: as soon
            # as a stable neutral baseline is ready, confirm it automatically —
            # there's no interactive user here to press a key.
            session.head_adapter.confirm_head_baseline()
            calibration = session.head_adapter.calibration_status()

        yolo_output = _yolo_adapter.process_frame(frame, timestamp=timestamp)
        event_output = session.event_manager.process(head_output, yolo_output, timestamp=timestamp)
        review_output = session.review_scorer.process(event_output)

    session.frame_count += 1
    session.last_seen = time.time()

    return {
        # Explicit float(): 05_multi_cue_review_score.py may hand back a
        # numpy scalar depending on how it computed the value internally, and
        # jsonify() raises (uncaught, since this happens during response
        # serialization *after* the route's own try/except already
        # succeeded) if it hits one of those instead of a native float.
        "review_score": float(review_output["score"]),
        "review_level": review_output["review_level"],
        "session_peak_score": float(review_output["session_peak_score"]),
        "active_cues": list(review_output["active_cues"]),
        "calibration_phase": calibration.get("phase"),
        "frame_count": int(session.frame_count),
        "disclaimer": review_output["disclaimer"],
        # Raw per-frame YOLO detections (already confidence-thresholded by
        # 02_yolo_output_adapter.py itself) so the dashboard can draw
        # bounding boxes over the webcam thumbnail. bbox_xyxy is in the pixel
        # space of the frame the extension captured (currently 320x240) --
        # the dashboard reads the displayed <img>'s natural size rather than
        # assuming that resolution, so this stays correct if it ever changes.
        "detections": [
            {
                "label": str(d.get("label", "unknown")),
                "confidence": float(d.get("confidence", 0.0)),
                "bbox_xyxy": [float(v) for v in d.get("bbox_xyxy", [0, 0, 0, 0])],
            }
            for d in (yolo_output.get("detections") or [])
        ],
    }


app = Flask(__name__)


@app.errorhandler(Exception)
def _handle_unexpected_error(e):
    # Global safety net: ANY unhandled exception anywhere in a route (not
    # just the ones explicitly try/except'd below) must still come back as
    # JSON, never Flask/Werkzeug's default HTML error page -- server.js's
    # analyzeWithCV() does response.json() and has no HTML fallback, so an
    # HTML response there previously surfaced as an opaque "Unexpected
    # token '<'" instead of the actual error.
    import traceback
    print("cv_service: unhandled exception:")
    traceback.print_exc()
    return jsonify({"error": "unhandled server error: " + str(e)}), 500


@app.route("/health")
def health():
    with _sessions_lock:
        active = len(_sessions)
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "yolo_checkpoint": YOLO_CHECKPOINT,
        "sessions_active": active,
    })


@app.route("/session/start", methods=["POST"])
def session_start():
    data = request.get_json(force=True, silent=True) or {}
    student_id = str(data.get("studentId") or "").strip()
    if not student_id:
        return jsonify({"error": "missing 'studentId'"}), 400
    try:
        session = _get_or_create_session(student_id)
    except Exception as e:
        print(f"cv_service: /session/start error for studentId={student_id!r}: {e}")
        return jsonify({"error": "session start failed: " + str(e)}), 500
    return jsonify({"ok": True, "calibration_phase": session.head_adapter.calibration_status().get("phase")})


@app.route("/frame", methods=["POST"])
def frame_route():
    data = request.get_json(force=True, silent=True) or {}
    student_id = str(data.get("studentId") or "").strip()
    if not student_id:
        return jsonify({"error": "missing 'studentId'"}), 400
    if "image" not in data:
        return jsonify({"error": "missing 'image'"}), 400

    try:
        decoded_frame = _decode_frame(data["image"])
    except Exception as e:
        return jsonify({"error": "bad image: " + str(e)}), 400

    timestamp = data.get("timestamp")
    # Client sends epoch milliseconds (JS Date.now()); event durations only
    # need monotonically increasing seconds, so epoch seconds works directly.
    ts = float(timestamp) / 1000.0 if isinstance(timestamp, (int, float)) else time.time()

    try:
        session = _get_or_create_session(student_id)
    except Exception as e:
        print(f"cv_service: /frame session-start error for studentId={student_id!r}: {e}")
        return jsonify({"error": "session start failed: " + str(e)}), 500

    try:
        result = _process_frame(session, decoded_frame, ts)
    except Exception as e:
        print(f"cv_service: /frame error for studentId={student_id!r}: {e}")
        return jsonify({"error": "inference failed: " + str(e)}), 500

    return jsonify(result)


@app.route("/session/end", methods=["POST"])
def session_end():
    data = request.get_json(force=True, silent=True) or {}
    student_id = str(data.get("studentId") or "").strip()
    if not student_id:
        return jsonify({"error": "missing 'studentId'"}), 400
    dropped = _drop_session(student_id)
    return jsonify({"ok": True, "dropped": dropped})


if __name__ == "__main__":
    print(f"cv_service: serving on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
