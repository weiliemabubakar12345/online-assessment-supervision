# UPDATED — Week 8 Day 3 selected audio-device integration configuration
# Default global checkpoint remains 5e via normal CLI/default selection.
# audio_device temporal event enabled by default at event_conf=0.43.

"""
run_integrated_demo.py

Live integration test for:
    01_head_gaze_adapter.py
    02_yolo_output_adapter.py
    03_event_manager.py
    04_event_logger.py

Purpose
-------
Run one webcam frame through the frozen head/gaze adapter and frozen YOLO
adapter, then pass both normalized outputs to the event manager.

This is an integration/evaluation tool, not a cheating-decision system.

Important design choices
------------------------
- One shared webcam frame is used by both adapters.
- --mirror flips the shared frame BEFORE both adapters, so YOLO boxes and
  head/gaze geometry remain in the same coordinate space.
- Event-manager timestamps use one integration clock.
- The initial event thresholds are engineering defaults for live testing,
  not final research thresholds.
- audio_device event is enabled by default with a 0.43 event-eligibility gate.
- 04 persists START/END events and filtered-candidate transitions.

Controls
--------
C : confirm prepared neutral head baseline
R : reset head/gaze calibration
S : save integrated screenshot
Q : quit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Mapping, Optional

import cv2


INTEGRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEAD_GAZE_ADAPTER_PATH = INTEGRATION_DIR / "01_head_gaze_adapter.py"
YOLO_ADAPTER_PATH = INTEGRATION_DIR / "02_yolo_output_adapter.py"
EVENT_MANAGER_PATH = INTEGRATION_DIR / "03_event_manager.py"
EVENT_LOGGER_PATH = INTEGRATION_DIR / "04_event_logger.py"

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "integration"
)

DEFAULT_SCREENSHOT_DIR = (
    DEFAULT_OUTPUT_DIR
    / "screenshots"
)

DEFAULT_EVENT_LOG_DIR = (
    DEFAULT_OUTPUT_DIR
    / "event_logs"
)

WINDOW_NAME = "TEEP Integrated Visual Cue Demo | UI v7"
UI_VERSION = "v7-large-readable-panel"


def _get_screen_size() -> tuple[int, int]:
    """Best-effort desktop size detection with safe fallback."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
    except Exception:
        pass

    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = int(root.winfo_screenwidth())
        height = int(root.winfo_screenheight())
        root.destroy()
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass

    return 1920, 1080


def _load_local_module(path: Path, module_name: str) -> ModuleType:
    """
    Import a numbered integration module safely.

    Python identifiers cannot normally start with 01/02/03, so the integration
    runner loads these files explicitly instead of renaming the already-frozen
    adapter files.
    """
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



class _AsyncYoloWorker:
    """Latest-frame YOLO worker with frame dropping and generation-safe reset."""

    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter
        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._pending: Optional[tuple[int, int, Any, float]] = None
        self._latest: Optional[Dict[str, Any]] = None
        self._error: Optional[BaseException] = None
        self._input_sequence = 0
        self._result_sequence = 0
        self._generation = 0
        self._submitted = 0
        self._dropped_pending = 0
        self._completed = 0
        self._inference_ms_ema = 0.0
        self._worker_fps_ema = 0.0
        self._last_completion_time: Optional[float] = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                return
            self._stop_requested = False
            self._thread = threading.Thread(
                target=self._run,
                name="teep-yolo-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(self, frame: Any, timestamp: float) -> int:
        frame_copy = frame.copy()
        with self._condition:
            self._input_sequence += 1
            sequence_id = self._input_sequence
            self._submitted += 1
            if self._pending is not None:
                self._dropped_pending += 1
            self._pending = (
                self._generation,
                sequence_id,
                frame_copy,
                float(timestamp),
            )
            self._condition.notify_all()
            return sequence_id

    def clear(self) -> None:
        """Discard pending/latest results; in-flight old-generation output is ignored."""
        with self._condition:
            self._generation += 1
            self._pending = None
            self._latest = None
            self._error = None
            self._condition.notify_all()

    def get_latest(self, now_timestamp: float) -> Optional[Dict[str, Any]]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError("Async YOLO worker failed") from self._error
            if self._latest is None:
                return None
            item = dict(self._latest)

        source_ts = float(item["source_timestamp"])
        item["result_age_s"] = max(0.0, float(now_timestamp) - source_ts)
        return item

    def get_stats(self) -> Dict[str, Any]:
        with self._condition:
            return {
                "submitted_frames": int(self._submitted),
                "dropped_pending_frames": int(self._dropped_pending),
                "completed_inferences": int(self._completed),
                "worker_inference_ms": float(self._inference_ms_ema),
                "worker_fps": float(self._worker_fps_ema),
            }

    def stop(self) -> None:
        thread: Optional[threading.Thread]
        with self._condition:
            self._stop_requested = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        with self._condition:
            self._thread = None

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending is None and not self._stop_requested:
                        self._condition.wait(timeout=0.2)
                    if self._stop_requested:
                        return
                    job = self._pending
                    self._pending = None

                if job is None:
                    continue

                generation, source_sequence_id, frame, source_timestamp = job
                inference_start = time.perf_counter()
                output = self.adapter.process_frame(
                    frame,
                    timestamp=source_timestamp,
                )
                completion_time = time.perf_counter()
                inference_ms = (completion_time - inference_start) * 1000.0

                with self._condition:
                    # A reset may happen while inference is running.
                    if generation != self._generation:
                        continue

                    self._result_sequence += 1
                    self._completed += 1
                    alpha = 0.15
                    if self._inference_ms_ema <= 0.0:
                        self._inference_ms_ema = inference_ms
                    else:
                        self._inference_ms_ema = (
                            (1.0 - alpha) * self._inference_ms_ema
                            + alpha * inference_ms
                        )

                    if self._last_completion_time is not None:
                        dt = max(1e-6, completion_time - self._last_completion_time)
                        fps = 1.0 / dt
                        if self._worker_fps_ema <= 0.0:
                            self._worker_fps_ema = fps
                        else:
                            self._worker_fps_ema = (
                                (1.0 - alpha) * self._worker_fps_ema
                                + alpha * fps
                            )
                    self._last_completion_time = completion_time

                    self._latest = {
                        "result_sequence_id": int(self._result_sequence),
                        "source_frame_sequence_id": int(source_sequence_id),
                        "source_timestamp": float(source_timestamp),
                        "completion_monotonic": float(completion_time),
                        "inference_ms": float(inference_ms),
                        "output": output,
                    }
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()


def _clip_text(text: str, max_length: int = 88) -> str:
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _object_summary(yolo_output: Mapping[str, Any]) -> str:
    detections = yolo_output.get("detections", [])
    if not isinstance(detections, list) or not detections:
        return "none"

    grouped: Dict[str, int] = {}
    for detection in detections:
        if not isinstance(detection, Mapping):
            continue
        label = str(detection.get("label", "unknown"))
        if label == "person":
            continue
        grouped[label] = grouped.get(label, 0) + 1

    if not grouped:
        return "none"

    parts = []
    for label in sorted(grouped):
        count = grouped[label]
        parts.append(label if count == 1 else f"{label}x{count}")
    return ", ".join(parts)


def _label_color(label: str) -> tuple[int, int, int]:
    # Bright BGR colours chosen so BLACK text is readable on the label badge.
    palette = {
        "person": (170, 235, 180),
        "phone": (120, 205, 255),
        "computer_device": (120, 225, 255),
        "book_notes": (245, 210, 145),
        "calculator": (225, 180, 245),
        "watch": (155, 235, 235),
        "audio_device": (235, 175, 225),
    }
    return palette.get(str(label), (220, 220, 220))


def _draw_text_with_bg(
    frame: Any,
    text: str,
    origin: tuple[int, int],
    *,
    font_scale: float = 0.58,
    text_color: tuple[int, int, int] = (20, 20, 20),
    bg_color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 2,
    padding: int = 5,
) -> None:
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x1 = max(0, x - padding)
    y1 = max(0, y - text_h - baseline - padding)
    x2 = min(frame.shape[1] - 1, x + text_w + padding)
    y2 = min(frame.shape[0] - 1, y + padding)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(
        frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        text_color, thickness, cv2.LINE_AA
    )


def _draw_yolo_boxes(frame: Any, yolo_output: Mapping[str, Any]) -> None:
    height, width = frame.shape[:2]
    detections = yolo_output.get("detections", [])

    if not isinstance(detections, list):
        return

    for detection in detections:
        if not isinstance(detection, Mapping):
            continue

        bbox = detection.get("bbox_xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1_i = max(0, min(width - 1, int(round(x1))))
        y1_i = max(0, min(height - 1, int(round(y1))))
        x2_i = max(0, min(width - 1, int(round(x2))))
        y2_i = max(0, min(height - 1, int(round(y2))))

        label = str(detection.get("label", "unknown"))
        color = _label_color(label)

        cv2.rectangle(
            frame,
            (x1_i, y1_i),
            (x2_i, y2_i),
            color,
            2,
        )

        model_label = str(detection.get("model_label", label))
        confidence = float(detection.get("confidence", 0.0))

        if label == model_label:
            box_text = f"{label} {confidence:.2f}"
        else:
            box_text = f"{label} ({model_label}) {confidence:.2f}"

        text_y = max(24, y1_i - 8)
        _draw_text_with_bg(
            frame,
            box_text,
            (x1_i + 2, text_y),
            font_scale=0.58,
            text_color=(15, 15, 15),
            bg_color=color,
            thickness=2,
            padding=5,
        )



def _draw_status_panel(
    frame: Any,
    *,
    checkpoint_id: str,
    head_gaze_output: Mapping[str, Any],
    yolo_output: Mapping[str, Any],
    event_output: Mapping[str, Any],
    processing_fps: float,
    canvas_width: int,
    canvas_height: int,
    calibration_targets: Mapping[str, int],
) -> Any:
    """
    UI v5

    Layout:
        [ large webcam / YOLO view ][ larger light status panel ]

    During calibration, a large centre-screen instruction and target are shown
    directly over the webcam so the user does NOT need to look sideways at the
    status panel. After calibration reaches READY, the centre instruction
    disappears and the webcam stays unobstructed.
    """
    import numpy as _np

    canvas_w = max(1200, int(canvas_width))
    canvas_h = max(700, int(canvas_height))

    # Wider than v4 so the right-side text can be genuinely readable, while
    # still leaving ~70% of a 1920-wide display for the webcam.
    panel_w = max(610, min(700, int(round(canvas_w * 0.34))))
    video_area_w = canvas_w - panel_w

    canvas = _np.full((canvas_h, canvas_w, 3), 18, dtype=_np.uint8)

    video = frame.copy()
    _draw_yolo_boxes(video, yolo_output)

    src_h, src_w = video.shape[:2]
    scale = min(video_area_w / float(src_w), canvas_h / float(src_h))
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(video, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

    x0 = max(0, (video_area_w - dst_w) // 2)
    y0 = max(0, (canvas_h - dst_h) // 2)
    canvas[y0:y0 + dst_h, x0:x0 + dst_w] = resized

    # -----------------------------
    # Current standardized outputs
    # -----------------------------
    calibration = head_gaze_output.get("calibration", {})
    phase = str(calibration.get("phase", "UNKNOWN"))
    head = head_gaze_output.get("head_pose", {})
    gaze = head_gaze_output.get("gaze", {})

    observed = event_output.get("observed_active_event_keys", [])
    release = event_output.get("release_grace_event_keys", [])
    candidates = event_output.get(
        "observed_cue_candidates",
        event_output.get("cue_candidates", {}),
    )
    candidate_keys = (
        sorted(str(k) for k in candidates.keys())
        if isinstance(candidates, Mapping)
        else []
    )

    events = event_output.get("events", [])
    lifecycle_events = []
    if isinstance(events, list):
        lifecycle_events = [
            f"{e.get('lifecycle')} {e.get('cue_key')}"
            for e in events
            if isinstance(e, Mapping)
            and e.get("lifecycle") in {"START", "END"}
        ]

    # ---------------------------------------------------------------
    # Guided calibration overlay: user keeps looking at SCREEN CENTRE
    # ---------------------------------------------------------------
    if phase != "READY":
        eye_target = max(1, int(calibration_targets.get("eye", 45)))
        gaze_target = max(1, int(calibration_targets.get("gaze", 8)))
        head_target = max(1, int(calibration_targets.get("head", 30)))

        eye_samples = int(calibration.get("eye_samples", 0) or 0)
        gaze_samples = int(calibration.get("gaze_samples", 0) or 0)
        head_samples = int(calibration.get("head_samples", 0) or 0)

        if phase == "EYE_CALIBRATION":
            step_title = "STEP 1 / 4   EYE CALIBRATION"
            instruction = "LOOK AT THE +   |   KEEP EYES NATURALLY OPEN"
            detail = "Stay facing forward. Normal blinking is OK."
            progress = min(1.0, eye_samples / float(eye_target))
            progress_text = f"Accepted eye samples: {eye_samples} / {eye_target}"

        elif phase == "GAZE_CALIBRATION":
            step_title = "STEP 2 / 4   GAZE CENTRE"
            instruction = "LOOK ONLY AT THE +   |   KEEP YOUR HEAD STILL"
            detail = "Do not look at the side panel. This step is automatic."
            progress = min(1.0, gaze_samples / float(gaze_target))
            progress_text = f"Stable gaze samples: {gaze_samples} / {gaze_target}"

        elif phase == "HEAD_POSE_CALIBRATION":
            step_title = "STEP 3 / 4   HEAD BASELINE"
            instruction = "FACE THE + STRAIGHT   |   HOLD YOUR HEAD STILL"
            detail = "If you move too much, the head samples restart."
            progress = min(1.0, head_samples / float(head_target))
            progress_text = f"Stable head samples: {head_samples} / {head_target}"

        elif phase == "HEAD_POSE_CONFIRMATION":
            step_title = "STEP 4 / 4   CONFIRM NEUTRAL HEAD"
            instruction = "KEEP LOOKING AT THE +   |   PRESS  C  NOW"
            detail = "C confirms this position as your neutral head baseline."
            progress = 1.0
            progress_text = "Neutral baseline is ready to confirm."

        else:
            step_title = "CALIBRATION"
            instruction = "LOOK AT THE +   |   STAY FACING FORWARD"
            detail = f"Current state: {phase}"
            progress = 0.0
            progress_text = ""

        vcx = x0 + dst_w // 2
        vcy = y0 + dst_h // 2

        # Large centre target. Bright cyan/blue remains visible on light and dark
        # webcam backgrounds.
        target_colour = (255, 170, 0)
        cv2.circle(canvas, (vcx, vcy), 22, target_colour, 3)
        cv2.line(canvas, (vcx - 34, vcy), (vcx + 34, vcy), target_colour, 3)
        cv2.line(canvas, (vcx, vcy - 34), (vcx, vcy + 34), target_colour, 3)

        # Large light instruction card positioned around the screen centre.
        card_w = min(video_area_w - 80, 960)
        card_h = 215
        card_x1 = max(24, vcx - card_w // 2)
        card_x2 = min(video_area_w - 24, card_x1 + card_w)
        card_y1 = min(canvas_h - card_h - 30, vcy + 70)
        card_y1 = max(30, card_y1)
        card_y2 = card_y1 + card_h

        overlay = canvas.copy()
        cv2.rectangle(
            overlay,
            (card_x1, card_y1),
            (card_x2, card_y2),
            (238, 246, 252),
            -1,
        )
        cv2.addWeighted(overlay, 0.94, canvas, 0.06, 0.0, canvas)
        cv2.rectangle(
            canvas,
            (card_x1, card_y1),
            (card_x2, card_y2),
            (80, 100, 120),
            3,
        )

        card_center = (card_x1 + card_x2) // 2

        def centre_text(
            msg: str,
            y: int,
            font_scale: float,
            colour: tuple[int, int, int],
            thickness: int,
        ) -> None:
            (tw, _), _baseline = cv2.getTextSize(
                msg,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            tx = max(card_x1 + 15, card_center - tw // 2)
            cv2.putText(
                canvas,
                msg,
                (tx, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                colour,
                thickness,
                cv2.LINE_AA,
            )

        centre_text(step_title, card_y1 + 43, 0.82, (25, 25, 25), 2)
        centre_text(instruction, card_y1 + 89, 0.72, (15, 45, 85), 2)
        centre_text(detail, card_y1 + 128, 0.56, (65, 65, 65), 1)
        centre_text(progress_text, card_y1 + 160, 0.55, (35, 35, 35), 1)

        # Progress bar
        bar_margin = 42
        bar_x1 = card_x1 + bar_margin
        bar_x2 = card_x2 - bar_margin
        bar_y1 = card_y1 + 178
        bar_y2 = card_y1 + 197
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (195, 195, 195), -1)
        fill_x = bar_x1 + int(round((bar_x2 - bar_x1) * max(0.0, min(1.0, progress))))
        if fill_x > bar_x1:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (fill_x, bar_y2), (60, 145, 80), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (120, 120, 120), 1)

    else:
        # Small READY badge only; no large overlay after calibration.
        badge = "CALIBRATION READY - TEST CAN START"
        (bw, bh), _ = cv2.getTextSize(
            badge, cv2.FONT_HERSHEY_SIMPLEX, 0.63, 2
        )
        bx1 = max(18, x0 + 18)
        by1 = max(18, y0 + 18)
        cv2.rectangle(
            canvas,
            (bx1, by1),
            (bx1 + bw + 28, by1 + bh + 24),
            (220, 248, 225),
            -1,
        )
        cv2.putText(
            canvas,
            badge,
            (bx1 + 14, by1 + bh + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.63,
            (25, 95, 40),
            2,
            cv2.LINE_AA,
        )

    # -----------------------------
    # Larger right-side status panel
    # -----------------------------
    px0 = video_area_w
    canvas[:, px0:] = (245, 245, 245)
    cv2.line(canvas, (px0, 0), (px0, canvas_h), (145, 145, 145), 2)

    panel_x = px0 + 34
    panel_right = canvas_w - 28
    dark = (20, 20, 20)
    muted = (75, 75, 75)
    accent = (115, 75, 25)
    line = (185, 185, 185)

    def put(
        text: str,
        y: int,
        scale_: float = 0.72,
        colour=dark,
        thick: int = 2,
        max_lines: int = 2,
        line_step: int = 38,
    ) -> int:
        text = str(text)
        max_chars = 42 if panel_w < 650 else 48
        chunks = []
        rest = text

        while rest and len(chunks) < max_lines:
            if len(rest) <= max_chars:
                chunks.append(rest)
                rest = ""
                break
            cut = rest.rfind(" ", 0, max_chars + 1)
            if cut < 8:
                cut = max_chars
            chunks.append(rest[:cut])
            rest = rest[cut:].lstrip()

        if rest and chunks:
            chunks[-1] = chunks[-1][:-3] + "..."

        for chunk in chunks or ["none"]:
            cv2.putText(
                canvas,
                chunk,
                (panel_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale_,
                colour,
                thick,
                cv2.LINE_AA,
            )
            y += line_step
        return y

    def heading(text: str, y: int) -> int:
        # Light section bar makes each block easy to scan from a distance.
        bar_top = y - 25
        bar_bottom = y + 10
        cv2.rectangle(
            canvas,
            (panel_x - 8, bar_top),
            (panel_right, bar_bottom),
            (232, 238, 244),
            -1,
        )
        cv2.putText(
            canvas,
            text,
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            accent,
            2,
            cv2.LINE_AA,
        )
        return y + 38

    cv2.putText(
        canvas,
        "TEEP VISUAL MONITOR",
        (panel_x, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.00,
        dark,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"YOLO {checkpoint_id}  |  {processing_fps:.1f} FPS  |  UI v7",
        (panel_x, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        muted,
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (panel_x, 108), (panel_right, 108), line, 1)

    y = 144
    y = heading("CALIBRATION", y)
    y = put(phase, y, 0.78, dark, 2)

    if phase != "READY":
        y = put(
            "Keep looking at the centre +",
            y,
            0.68,
            (30, 75, 115),
            2,
        )

    y += 10
    y = heading("HEAD / GAZE", y)
    y = put(
        f"Head: {head.get('label', 'UNKNOWN')} [{head.get('confidence', 'UNKNOWN')}]",
        y,
        0.72,
        dark,
        2,
    )
    y = put(
        f"Gaze: {gaze.get('label', 'UNKNOWN')} [{gaze.get('confidence', 'UNKNOWN')}]",
        y,
        0.72,
        dark,
        2,
    )
    y = put(
        f"Eyes: {gaze.get('eye_status', 'UNKNOWN')}",
        y,
        0.64,
        muted,
        1,
    )

    y += 10
    y = heading("PERSON STATUS", y)

    raw_person_state = str(
        yolo_output.get("person_state", "UNKNOWN")
    )
    raw_person_count = int(
        yolo_output.get("person_count", 0) or 0
    )

    cross_validation = event_output.get(
        "cross_module_validation", {}
    )
    if not isinstance(cross_validation, Mapping):
        cross_validation = {}

    face_evidence = cross_validation.get(
        "head_gaze_face_evidence", {}
    )
    if not isinstance(face_evidence, Mapping):
        face_evidence = {}

    face_present = bool(face_evidence.get("present", False))
    no_person_suppressed = bool(
        cross_validation.get("no_person_suppressed", False)
    )

    if raw_person_state == "MULTIPLE_PERSONS":
        integrated_person_text = "Integrated: MULTIPLE PERSONS"
        integrated_person_colour = (45, 45, 150)
    elif raw_person_state == "ONE_PERSON":
        integrated_person_text = "Integrated: PERSON PRESENT"
        integrated_person_colour = (35, 105, 45)
    elif raw_person_state == "NO_PERSON" and no_person_suppressed:
        integrated_person_text = (
            "Integrated: PERSON PRESENT [FACE EVIDENCE]"
        )
        integrated_person_colour = (35, 105, 45)
    elif raw_person_state == "NO_PERSON":
        integrated_person_text = "Integrated: NO PERSON"
        integrated_person_colour = (45, 45, 150)
    else:
        integrated_person_text = "Integrated: UNKNOWN"
        integrated_person_colour = dark

    y = put(
        integrated_person_text,
        y,
        0.64,
        integrated_person_colour,
        2,
    )
    y = put(
        f"YOLO raw: {raw_person_state} ({raw_person_count})",
        y,
        0.62,
        muted,
        1,
    )
    y = put(
        "Face evidence: "
        + ("PRESENT" if face_present else "ABSENT"),
        y,
        0.62,
        muted,
        1,
    )

    y += 10
    y = heading("OBJECTS", y)
    y = put(
        f"Detected: {_object_summary(yolo_output)}",
        y,
        0.72,
        dark,
        2,
    )

    y += 10
    y = heading("EVENTS", y)

    rejected = event_output.get("rejected_event_candidates", {})
    rejected_parts = []
    if isinstance(rejected, Mapping) and rejected:
        for key, item in rejected.items():
            if not isinstance(item, Mapping):
                continue

            short_label = key.split(":", 1)[-1]
            reason = str(item.get("reason", ""))

            if reason == "head_gaze_face_evidence_present":
                face = item.get("face_evidence", {})
                if not isinstance(face, Mapping):
                    face = {}
                eye_status = str(face.get("eye_status", ""))
                rejected_parts.append(
                    f"{short_label}: suppressed by face evidence"
                    + (f" ({eye_status})" if eye_status else "")
                )
            else:
                rejected_parts.append(
                    f"{short_label} "
                    f"{float(item.get('max_confidence', 0.0)):.2f}"
                    f"<{float(item.get('required_min_confidence', 0.0)):.2f}"
                )

    candidate_text = (
        "Candidates: "
        + (", ".join(candidate_keys) if candidate_keys else "none")
    )
    active_text = "Active: " + (", ".join(observed) if observed else "none")

    if canvas_h <= 950:
        # A 1600x900 window has limited vertical room. Keep the two operationally
        # important rows visible, then use only genuinely available space for
        # diagnostics. Single-line truncation prevents event text from entering
        # the fixed controls area.
        event_bottom = canvas_h - 150
        event_line_step = 32
        available_rows = max(
            1,
            int((event_bottom - y) // event_line_step) + 1,
        )

        event_rows = [
            (candidate_text, 0.61, dark, 1),
            (active_text, 0.64, dark, 2),
        ]
        if rejected_parts:
            event_rows.append(
                (
                    "Filtered: " + ", ".join(rejected_parts),
                    0.57,
                    (70, 70, 150),
                    1,
                )
            )
        if release:
            event_rows.append(
                ("Grace: " + ", ".join(release), 0.57, muted, 1)
            )
        if lifecycle_events:
            event_rows.append(
                ("Now: " + ", ".join(lifecycle_events), 0.59, accent, 2)
            )

        if available_rows == 1:
            event_rows = [event_rows[1]]

        for text, scale_, colour, thick in event_rows[:available_rows]:
            y = put(
                text,
                y,
                scale_,
                colour,
                thick,
                max_lines=1,
                line_step=event_line_step,
            )
    else:
        y = put(candidate_text, y, 0.64, dark, 1)
        if rejected_parts:
            y = put(
                "Filtered: " + ", ".join(rejected_parts),
                y,
                0.61,
                (70, 70, 150),
                1,
                max_lines=3,
            )
        y = put(active_text, y, 0.68, dark, 2)
        if release:
            y = put("Grace: " + ", ".join(release), y, 0.60, muted, 1)
        if lifecycle_events:
            y += 3
            y = put(
                "Now: " + ", ".join(lifecycle_events),
                y,
                0.63,
                accent,
                2,
            )

    # If a very verbose event state reaches the controls area, draw a small
    # visual separator rather than letting text crowd the bottom controls.
    if y > canvas_h - 145:
        cv2.line(
            canvas,
            (panel_x, canvas_h - 145),
            (panel_right, canvas_h - 145),
            (205, 205, 205),
            1,
        )

    # Controls stay large enough to read.
    bottom_y = canvas_h - 118
    cv2.line(canvas, (panel_x, bottom_y), (panel_right, bottom_y), line, 1)
    put(
        "C confirm   R recalibrate",
        bottom_y + 38,
        0.66,
        dark,
        2,
        line_step=36,
    )
    put(
        "S screenshot   F window   Q quit",
        bottom_y + 78,
        0.61,
        muted,
        1,
        line_step=34,
    )

    return canvas


def _save_screenshot(
    frame: Any,
    screenshot_dir: Path,
    checkpoint_id: str,
) -> Path:
    screenshot_dir = Path(screenshot_dir).expanduser().resolve()
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = screenshot_dir / f"integrated_demo_{checkpoint_id}_{stamp}.jpg"

    if not cv2.imwrite(str(path), frame):
        raise IOError(f"Could not save screenshot: {path}")

    return path


def _compact_status(
    *,
    timestamp: float,
    head_gaze_output: Mapping[str, Any],
    yolo_output: Mapping[str, Any],
    event_output: Mapping[str, Any],
) -> Dict[str, Any]:
    head = head_gaze_output.get("head_pose", {})
    gaze = head_gaze_output.get("gaze", {})
    calibration = head_gaze_output.get("calibration", {})

    return {
        "timestamp": round(float(timestamp), 3),
        "calibration": calibration.get("phase"),
        "head": head.get("label"),
        "head_confidence": head.get("confidence"),
        "gaze": gaze.get("label"),
        "gaze_confidence": gaze.get("confidence"),
        "eye_status": gaze.get("eye_status"),
        "person_state": yolo_output.get("person_state"),
        "person_count": yolo_output.get("person_count"),
        "objects": _object_summary(yolo_output),
        "candidate_keys": sorted(
            event_output.get(
                "observed_cue_candidates",
                event_output.get("cue_candidates", {}),
            ).keys()
        ),
        "rejected_event_candidates": {
            key: (
                {
                    "reason": value.get("reason"),
                    "max_confidence": round(
                        float(value.get("max_confidence", 0.0)), 3
                    ),
                    "required_min_confidence": round(
                        float(value.get("required_min_confidence", 0.0)), 3
                    ),
                }
                if value.get("reason")
                == "below_event_confidence_threshold"
                else {
                    "reason": value.get("reason"),
                    "face_evidence": value.get("face_evidence", {}),
                }
            )
            for key, value in event_output.get(
                "rejected_event_candidates", {}
            ).items()
            if isinstance(value, Mapping)
        },
        "active_event_keys": event_output.get("active_event_keys", []),
        "observed_active_event_keys": event_output.get(
            "observed_active_event_keys", []
        ),
        "release_grace_event_keys": event_output.get(
            "release_grace_event_keys", []
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live 01 + 02 + 03 + 04 integration test. "
            "Uses the frozen integration event baseline."
        )
    )

    # Head/gaze resources: same explicit inputs used by 01.
    parser.add_argument("--l2cs-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mediapipe-model", type=Path, required=True)
    parser.add_argument("--canonical14-csv", type=Path, required=True)
    parser.add_argument("--head-gaze-device", default="cpu")

    # YOLO.
    parser.add_argument(
        "--checkpoint",
        choices=("5e", "15e", "30e", "50e"),
        default="5e",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help=(
            "YOLO checkpoint directory. If omitted, 02's project-relative "
            "default is used."
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--yolo-device", default="cpu")
    yolo_runtime_group = parser.add_mutually_exclusive_group()
    yolo_runtime_group.add_argument(
        "--async-yolo",
        dest="async_yolo",
        action="store_true",
        help=(
            "Use the validated latest-frame background YOLO worker "
            "(default runtime mode)."
        ),
    )
    yolo_runtime_group.add_argument(
        "--sync-yolo",
        dest="async_yolo",
        action="store_false",
        help=(
            "Use synchronous YOLO inference as a compatibility/debug fallback."
        ),
    )
    parser.set_defaults(async_yolo=True)
    parser.add_argument(
        "--yolo-stale-after",
        type=float,
        default=1.0,
        help=(
            "Seconds after which the latest async YOLO result is treated as "
            "stale for simultaneous-cue context. Default: 1.0"
        ),
    )

    # Provisional per-class EVENT eligibility gates.
    # These do not change 02's raw YOLO confidence threshold.
    parser.add_argument(
        "--computer-device-event-conf",
        type=float,
        default=0.45,
        help=(
            "Minimum computer_device confidence to enter 03 temporal "
            "event tracking. Raw 02 output is unchanged. Default: 0.45"
        ),
    )
    parser.add_argument(
        "--book-notes-event-conf",
        type=float,
        default=0.40,
        help=(
            "Minimum book_notes confidence to enter 03 temporal "
            "event tracking. Raw 02 output is unchanged. Default: 0.40"
        ),
    )

    parser.add_argument(
        "--audio-device-event-conf",
        type=float,
        default=0.43,
        help=(
            "Minimum audio_device confidence to enter 03 temporal event "
            "tracking. Raw 02 output remains at its normal threshold. "
            "Selected Week 8 default: 0.43"
        ),
    )

    # Shared runtime.
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror the shared frame before BOTH adapters.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.0,
        help=(
            "Seconds between terminal status updates. "
            "Use 0 to disable periodic status output. Default: 0."
        ),
    )
    parser.add_argument(
        "--verbose-json",
        action="store_true",
        help="Print full 01/02/03 JSON periodically instead of compact status.",
    )
    parser.add_argument(
        "--print-active-events",
        action="store_true",
        help="Also print per-frame ACTIVE lifecycle records. Default: START/END only.",
    )
    audio_event_group = parser.add_mutually_exclusive_group()
    audio_event_group.add_argument(
        "--enable-audio-event",
        dest="enable_audio_event",
        action="store_true",
        help="Enable the selected audio_device temporal event rule (default).",
    )
    audio_event_group.add_argument(
        "--disable-audio-event",
        dest="enable_audio_event",
        action="store_false",
        help="Disable audio_device temporal events for debugging/ablation.",
    )
    parser.set_defaults(enable_audio_event=True)
    parser.add_argument(
        "--event-log-dir",
        type=Path,
        default=DEFAULT_EVENT_LOG_DIR,
        help=(
            "Base directory for structured 04 event logs. "
            "A unique session subfolder is created automatically."
        ),
    )
    parser.add_argument(
        "--disable-event-logging",
        action="store_true",
        help="Disable 04 structured event logging for a temporary debug run.",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=DEFAULT_SCREENSHOT_DIR,
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Start in a normal resizable window instead of fullscreen.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1600,
        help="Windowed-mode width. Default: 1600",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Windowed-mode height. Default: 900",
    )
    parser.add_argument(
        "--profile-performance",
        action="store_true",
        help=(
            "Print rolling per-stage runtime timings without changing "
            "detection/event logic."
        ),
    )
    parser.add_argument(
        "--profile-interval",
        type=float,
        default=5.0,
        help=(
            "Seconds between performance reports when "
            "--profile-performance is enabled. Default: 5.0"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load numbered local modules without renaming the frozen files.
    head_module = _load_local_module(
        HEAD_GAZE_ADAPTER_PATH,
        "teep_head_gaze_adapter",
    )
    yolo_module = _load_local_module(
        YOLO_ADAPTER_PATH,
        "teep_yolo_output_adapter",
    )
    event_module = _load_local_module(
        EVENT_MANAGER_PATH,
        "teep_event_manager",
    )
    logger_module = _load_local_module(
        EVENT_LOGGER_PATH,
        "teep_event_logger",
    )

    rules = event_module.build_initial_integration_rules(
        enable_audio_device_event=args.enable_audio_event,
    )
    object_event_eligibility = (
        event_module.build_initial_object_event_eligibility(
            computer_device_min_confidence=args.computer_device_event_conf,
            book_notes_min_confidence=args.book_notes_event_conf,
            audio_device_min_confidence=args.audio_device_event_conf,
        )
    )
    event_manager = event_module.EventManager(
        rules,
        object_event_eligibility=object_event_eligibility,
        suppress_no_person_if_face_present=True,
    )

    # Important: mirror_input=False because --mirror is applied once to the
    # shared frame before BOTH modules.
    head_config = head_module.HeadGazeConfig(
        device=args.head_gaze_device,
        mirror_input=False,
    )
    head_adapter = head_module.HeadGazeAdapter(
        l2cs_root=args.l2cs_root,
        l2cs_snapshot=args.snapshot,
        mediapipe_model=args.mediapipe_model,
        canonical14_csv=args.canonical14_csv,
        config=head_config,
    )

    yolo_config = yolo_module.YoloAdapterConfig(
        checkpoint_id=args.checkpoint,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        iou_threshold=args.iou,
        device=args.yolo_device,
    )
    yolo_adapter = yolo_module.YoloOutputAdapter(
        config=yolo_config,
        model_dir=args.model_dir,
    )

    cap: Optional[Any] = None
    event_logger: Optional[Any] = None
    yolo_worker: Optional[_AsyncYoloWorker] = None
    last_consumed_yolo_result_sequence: Optional[int] = None
    last_frame_timestamp: Optional[float] = None

    print("Loading head/gaze models...")
    head_adapter.start()

    try:
        print("Loading YOLO checkpoint...")
        yolo_adapter.start()

        yolo_metadata = yolo_adapter.get_session_metadata()
        if args.async_yolo:
            yolo_worker = _AsyncYoloWorker(yolo_adapter)
            yolo_worker.start()

        rule_metadata = event_module.describe_rules(rules)
        object_eligibility_metadata = (
            event_module.describe_object_event_eligibility(
                object_event_eligibility
            )
        )

        session_metadata = {
            "integration_mode": "01_head_gaze + 02_yolo + 03_event_manager + 04_event_logger",
            "yolo": yolo_metadata,
            "event_rules": rule_metadata,
            "object_event_eligibility": object_eligibility_metadata,
            "person_state_cross_module_validation": {
                "suppress_no_person_if_head_gaze_face_present": True,
                "face_evidence_source": "01_head_gaze_adapter",
            },
            "audio_device_event_enabled": bool(args.enable_audio_event),
            "audio_device_event_confidence": float(args.audio_device_event_conf),
            "audio_device_temporal_rule": {
                "min_duration_s": 0.75,
                "release_duration_s": 0.60,
                "cooldown_s": 0.50,
            },
            "audio_device_limitation": (
                "Most reliable for larger headphones/headsets; small earbuds "
                "remain unreliable under full-frame webcam inference."
            ),
            "mirror_shared_frame": bool(args.mirror),
            "head_gaze_device": str(args.head_gaze_device),
            "yolo_runtime_mode": ("async_latest_frame" if args.async_yolo else "synchronous"),
            "yolo_stale_after_s": float(args.yolo_stale_after),
        }
        if args.verbose_json:
            print("\n[INTEGRATION_SESSION_METADATA]")
            print(json.dumps(session_metadata, indent=2, ensure_ascii=False))
        else:
            print(
                "\n[SESSION] "
                f"YOLO={args.checkpoint} | "
                f"mode={session_metadata['yolo_runtime_mode']} | "
                f"head/gaze={args.head_gaze_device}"
            )

        if not args.disable_event_logging:
            event_logger = logger_module.EventLogger(
                base_dir=args.event_log_dir,
                session_metadata=session_metadata,
            )
            logger_paths = event_logger.get_paths()

            if args.verbose_json:
                print("\n[EVENT_LOGGER]")
                print(
                    json.dumps(
                        logger_paths,
                        indent=2,
                        ensure_ascii=False,
                    )
                )
            else:
                print(f"[EVENT LOGGER] {logger_paths['session_dir']}")
        else:
            print("[EVENT LOGGER] disabled")

        cap = cv2.VideoCapture(args.cam)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {args.cam}.")

        screen_width, screen_height = _get_screen_size()
        is_fullscreen = not bool(args.windowed)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if is_fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            canvas_width, canvas_height = screen_width, screen_height
        else:
            cv2.resizeWindow(
                WINDOW_NAME,
                max(1200, int(args.window_width)),
                max(700, int(args.window_height)),
            )
            canvas_width = max(1200, int(args.window_width))
            canvas_height = max(700, int(args.window_height))

        print(f"\n[UI] {UI_VERSION} | canvas={canvas_width}x{canvas_height} | fullscreen={is_fullscreen}")
        print("[UI CHECK] Window title must contain: UI v7")
        print("Integrated webcam runtime started.")
        print("Controls: C = confirm baseline | R = recalibrate | S = screenshot | F = fullscreen/windowed | Q = quit")
        if args.enable_audio_event:
            print(
                "audio_device event: ENABLED | "
                f"event_conf={float(args.audio_device_event_conf):.2f} | "
                "temporal=0.75s min / 0.60s release / 0.50s cooldown"
            )
        else:
            print("audio_device event: DISABLED (debug/ablation mode).")
        if args.async_yolo:
            print(
                "YOLO runtime: ASYNC latest-frame worker | "
                f"stale-after={float(args.yolo_stale_after):.2f}s"
            )
        else:
            print("YOLO runtime: synchronous baseline")
        if args.profile_performance:
            print(
                "Performance profiling enabled: "
                f"report every {max(0.5, float(args.profile_interval)):.1f}s."
            )

        integration_start = time.perf_counter()
        last_print = -1e9
        last_loop_end = time.perf_counter()
        fps_ema = 0.0
        last_calibration_phase = None

        # Optional main-thread profiling. These measurements intentionally
        # observe the existing pipeline rather than changing its scheduling.
        perf_last_print = -1e9
        perf_alpha = 0.10
        perf_ema_ms = {
            "capture": 0.0,
            "head_gaze": 0.0,
            "yolo_main": 0.0,
            "event_manager": 0.0,
            "event_logger": 0.0,
            "ui_draw": 0.0,
            "display_wait": 0.0,
            "total_loop": 0.0,
        }
        perf_total_fps_ema = 0.0

        def _update_perf_ema(name: str, value_ms: float) -> None:
            previous = float(perf_ema_ms.get(name, 0.0))
            perf_ema_ms[name] = (
                float(value_ms)
                if previous <= 0.0
                else (1.0 - perf_alpha) * previous
                + perf_alpha * float(value_ms)
            )

        while True:
            loop_start = time.perf_counter()

            capture_start = time.perf_counter()
            ok, frame = cap.read()
            capture_end = time.perf_counter()
            _update_perf_ema(
                "capture",
                (capture_end - capture_start) * 1000.0,
            )
            if not ok or frame is None:
                print("Warning: webcam frame read failed.")
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_timestamp = time.perf_counter() - integration_start

            # In async mode, submit the latest frame before head/gaze work so
            # YOLO can overlap with the main-thread MediaPipe/UI pipeline.
            stage_start = time.perf_counter()
            if args.async_yolo:
                assert yolo_worker is not None
                yolo_worker.submit(frame, frame_timestamp)
            _update_perf_ema(
                "yolo_main",
                (time.perf_counter() - stage_start) * 1000.0,
            )

            stage_start = time.perf_counter()
            head_gaze_output = head_adapter.process_frame(frame)
            _update_perf_ema(
                "head_gaze",
                (time.perf_counter() - stage_start) * 1000.0,
            )

            yolo_observation_is_new = True
            yolo_result_sequence_id: Optional[int] = None
            yolo_result_age_s: Optional[float] = 0.0

            if args.async_yolo:
                assert yolo_worker is not None
                stage_start = time.perf_counter()
                latest_yolo = yolo_worker.get_latest(frame_timestamp)
                _update_perf_ema(
                    "yolo_main",
                    (time.perf_counter() - stage_start) * 1000.0,
                )

                if latest_yolo is None:
                    yolo_output = {
                        "timestamp": float(frame_timestamp),
                        "detections": [],
                        "person_count": 0,
                        "person_state": "UNKNOWN",
                    }
                    yolo_for_event = None
                    yolo_observation_is_new = False
                    yolo_result_age_s = None
                else:
                    yolo_output = latest_yolo["output"]
                    yolo_result_sequence_id = int(
                        latest_yolo["result_sequence_id"]
                    )
                    yolo_result_age_s = float(
                        latest_yolo["result_age_s"]
                    )
                    yolo_observation_is_new = (
                        last_consumed_yolo_result_sequence
                        != yolo_result_sequence_id
                    )
                    yolo_for_event = (
                        yolo_output if yolo_observation_is_new else None
                    )
                    if yolo_observation_is_new:
                        last_consumed_yolo_result_sequence = (
                            yolo_result_sequence_id
                        )
            else:
                stage_start = time.perf_counter()
                yolo_output = yolo_adapter.process_frame(
                    frame,
                    timestamp=frame_timestamp,
                )
                _update_perf_ema(
                    "yolo_main",
                    (time.perf_counter() - stage_start) * 1000.0,
                )
                yolo_for_event = yolo_output

            # One integration clock drives lifecycle emission; 03 separately
            # tracks the observation timestamp of each fresh YOLO result.
            stage_start = time.perf_counter()
            event_output = event_manager.process(
                head_gaze_output=head_gaze_output,
                yolo_output=yolo_for_event,
                timestamp=frame_timestamp,
                yolo_observation_is_new=yolo_observation_is_new,
                yolo_result_sequence_id=yolo_result_sequence_id,
                yolo_result_age_s=yolo_result_age_s,
                yolo_stale_after_s=float(args.yolo_stale_after),
            )
            _update_perf_ema(
                "event_manager",
                (time.perf_counter() - stage_start) * 1000.0,
            )

            last_frame_timestamp = float(frame_timestamp)
            stage_start = time.perf_counter()
            if event_logger is not None:
                event_logger.process(
                    event_output=event_output,
                    frame_timestamp=frame_timestamp,
                )
            _update_perf_ema(
                "event_logger",
                (time.perf_counter() - stage_start) * 1000.0,
            )

            current_calibration_phase = str(
                head_gaze_output.get("calibration", {}).get("phase", "UNKNOWN")
            )
            if current_calibration_phase != last_calibration_phase:
                calibration_messages = {
                    "EYE_CALIBRATION": (
                        "STEP 1/4 EYE: look at the centre + and keep eyes naturally open."
                    ),
                    "GAZE_CALIBRATION": (
                        "STEP 2/4 GAZE: keep head still and look only at the centre +."
                    ),
                    "HEAD_POSE_CALIBRATION": (
                        "STEP 3/4 HEAD: face the centre + straight and hold your head still."
                    ),
                    "HEAD_POSE_CONFIRMATION": (
                        "STEP 4/4 CONFIRM: keep looking at the centre + and press C now."
                    ),
                    "READY": (
                        "READY: calibration complete. Live monitoring can start."
                    ),
                }
                print(
                    "\n[CALIBRATION GUIDE] "
                    + calibration_messages.get(
                        current_calibration_phase,
                        current_calibration_phase,
                    )
                )
                last_calibration_phase = current_calibration_phase

            # Avoid unnecessary per-frame JSON serialization here.
            # Full JSON remains available through --verbose-json.

            loop_end = time.perf_counter()
            loop_dt = max(1e-6, loop_end - last_loop_end)
            instant_fps = 1.0 / loop_dt
            fps_ema = (
                instant_fps
                if fps_ema <= 0.0
                else 0.90 * fps_ema + 0.10 * instant_fps
            )
            last_loop_end = loop_end

            # Keep normal logs readable: print START/END immediately.
            # Per-frame ACTIVE records are available only when explicitly requested.
            lifecycle_events = event_output.get("events", [])
            if lifecycle_events:
                events_to_print = [
                    event
                    for event in lifecycle_events
                    if args.print_active_events
                    or event.get("lifecycle") in {"START", "END"}
                ]
                if events_to_print:
                    if args.verbose_json:
                        print("\n[EVENT_LIFECYCLE]")
                        print(
                            json.dumps(
                                events_to_print,
                                indent=2,
                                ensure_ascii=False,
                            )
                        )
                    else:
                        for event in events_to_print:
                            lifecycle = event.get("lifecycle", "UNKNOWN")
                            cue_key = event.get("cue_key", "unknown")
                            duration = float(event.get("duration_s") or 0.0)
                            context = event.get("context") or {}
                            confidence = context.get("max_confidence")

                            message = (
                                f"[EVENT {lifecycle}] {cue_key} "
                                f"| duration={duration:.2f}s"
                            )
                            if confidence is not None:
                                message += (
                                    f" | confidence={float(confidence):.2f}"
                                )

                            print(message)

            now = time.perf_counter()
            if (
                args.print_interval > 0.0
                and now - last_print >= args.print_interval
            ):
                if args.verbose_json:
                    print("\n[HEAD_GAZE_OUTPUT]")
                    print(json.dumps(head_gaze_output, indent=2, ensure_ascii=False))
                    print("\n[YOLO_OUTPUT]")
                    print(json.dumps(yolo_output, indent=2, ensure_ascii=False))
                    print("\n[EVENT_MANAGER_OUTPUT]")
                    print(json.dumps(event_output, indent=2, ensure_ascii=False))
                else:
                    status = _compact_status(
                        timestamp=frame_timestamp,
                        head_gaze_output=head_gaze_output,
                        yolo_output=yolo_output,
                        event_output=event_output,
                    )
                    active = status.get("active_event_keys") or []
                    active_text = ",".join(active) if active else "none"

                    print(
                        "[STATUS] "
                        f"t={float(status['timestamp']):.1f}s | "
                        f"{status['calibration']} | "
                        f"head={status['head']} | "
                        f"gaze={status['gaze']} | "
                        f"person={status['person_count']} | "
                        f"objects={status['objects']} | "
                        f"active={active_text}"
                    )
                last_print = now

            stage_start = time.perf_counter()
            display = _draw_status_panel(
                frame,
                checkpoint_id=args.checkpoint,
                head_gaze_output=head_gaze_output,
                yolo_output=yolo_output,
                event_output=event_output,
                processing_fps=fps_ema,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                calibration_targets={
                    "eye": int(head_config.calibration_samples),
                    "gaze": int(head_config.gaze_calibration_samples),
                    "head": int(getattr(head_module, "HEAD_CALIBRATION_FRAMES", 30)),
                },
            )
            _update_perf_ema(
                "ui_draw",
                (time.perf_counter() - stage_start) * 1000.0,
            )

            stage_start = time.perf_counter()
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            _update_perf_ema(
                "display_wait",
                (time.perf_counter() - stage_start) * 1000.0,
            )

            total_loop_dt = max(
                1e-6,
                time.perf_counter() - loop_start,
            )
            _update_perf_ema(
                "total_loop",
                total_loop_dt * 1000.0,
            )
            total_fps_now = 1.0 / total_loop_dt
            perf_total_fps_ema = (
                total_fps_now
                if perf_total_fps_ema <= 0.0
                else (1.0 - perf_alpha) * perf_total_fps_ema
                + perf_alpha * total_fps_now
            )

            if args.profile_performance:
                perf_now = time.perf_counter()
                if perf_now - perf_last_print >= max(
                    0.5,
                    float(args.profile_interval),
                ):
                    measured_stage_ms = (
                        perf_ema_ms["capture"]
                        + perf_ema_ms["head_gaze"]
                        + perf_ema_ms["yolo_main"]
                        + perf_ema_ms["event_manager"]
                        + perf_ema_ms["event_logger"]
                        + perf_ema_ms["ui_draw"]
                        + perf_ema_ms["display_wait"]
                    )
                    other_ms = max(
                        0.0,
                        perf_ema_ms["total_loop"] - measured_stage_ms,
                    )
                    print("\n[PERFORMANCE_PROFILE]")
                    print(
                        json.dumps(
                            {
                                "total_fps": round(
                                    perf_total_fps_ema,
                                    2,
                                ),
                                "total_loop_ms": round(
                                    perf_ema_ms["total_loop"],
                                    2,
                                ),
                                "capture_ms": round(
                                    perf_ema_ms["capture"],
                                    2,
                                ),
                                "head_gaze_main_ms": round(
                                    perf_ema_ms["head_gaze"],
                                    2,
                                ),
                                "yolo_runtime_mode": (
                                    "async_latest_frame"
                                    if args.async_yolo
                                    else "synchronous"
                                ),
                                "yolo_main_ms": round(
                                    perf_ema_ms["yolo_main"],
                                    2,
                                ),
                                "yolo_worker_inference_ms": (
                                    round(
                                        yolo_worker.get_stats()[
                                            "worker_inference_ms"
                                        ],
                                        2,
                                    )
                                    if yolo_worker is not None
                                    else None
                                ),
                                "yolo_worker_fps": (
                                    round(
                                        yolo_worker.get_stats()["worker_fps"],
                                        2,
                                    )
                                    if yolo_worker is not None
                                    else None
                                ),
                                "yolo_result_age_ms": (
                                    round(float(yolo_result_age_s) * 1000.0, 2)
                                    if yolo_result_age_s is not None
                                    else None
                                ),
                                "yolo_submitted_frames": (
                                    yolo_worker.get_stats()["submitted_frames"]
                                    if yolo_worker is not None
                                    else None
                                ),
                                "yolo_dropped_pending_frames": (
                                    yolo_worker.get_stats()[
                                        "dropped_pending_frames"
                                    ]
                                    if yolo_worker is not None
                                    else None
                                ),
                                "yolo_completed_inferences": (
                                    yolo_worker.get_stats()[
                                        "completed_inferences"
                                    ]
                                    if yolo_worker is not None
                                    else None
                                ),
                                "event_manager_ms": round(
                                    perf_ema_ms["event_manager"],
                                    3,
                                ),
                                "event_logger_ms": round(
                                    perf_ema_ms["event_logger"],
                                    3,
                                ),
                                "ui_draw_ms": round(
                                    perf_ema_ms["ui_draw"],
                                    2,
                                ),
                                "display_wait_ms": round(
                                    perf_ema_ms["display_wait"],
                                    2,
                                ),
                                "other_python_ms": round(
                                    other_ms,
                                    2,
                                ),
                                "note": (
                                    "Main-thread timings plus async YOLO worker "
                                    "statistics when enabled. Background L2CS worker "
                                    "time is not measured separately."
                                ),
                            },
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                    perf_last_print = perf_now

            if key in (ord("c"), ord("C")):
                if head_adapter.confirm_head_baseline():
                    print("Head baseline confirmed.")
                else:
                    print("Head baseline is not ready to confirm.")

            elif key in (ord("r"), ord("R")):
                head_adapter.reset_calibration()
                event_manager.reset()
                if yolo_worker is not None:
                    yolo_worker.clear()
                    last_consumed_yolo_result_sequence = None
                print(
                    "Head/gaze calibration reset; event-manager and async "
                    "YOLO cached state reset. Logger session continues."
                )

            elif key in (ord("s"), ord("S")):
                try:
                    screenshot_path = _save_screenshot(
                        display,
                        args.screenshot_dir,
                        args.checkpoint,
                    )
                    print(f"Screenshot saved: {screenshot_path}")
                except IOError as exc:
                    print(f"Screenshot error: {exc}")

            elif key in (ord("f"), ord("F")):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
                    canvas_width, canvas_height = screen_width, screen_height
                else:
                    cv2.setWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL
                    )
                    cv2.resizeWindow(
                        WINDOW_NAME,
                        max(1200, int(args.window_width)),
                        max(700, int(args.window_height)),
                    )
                    canvas_width = max(1200, int(args.window_width))
                    canvas_height = max(700, int(args.window_height))
                print(f"UI fullscreen: {is_fullscreen}")

            elif key in (ord("q"), ord("Q")):
                break

    finally:
        if event_logger is not None:
            try:
                event_logger.close(
                    final_frame_timestamp=last_frame_timestamp,
                )
                closed_paths = event_logger.get_paths()
                if args.verbose_json:
                    print("\n[EVENT_LOGGER CLOSED]")
                    print(
                        json.dumps(
                            closed_paths,
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                else:
                    print(
                        "\n[SESSION CLOSED] "
                        f"{closed_paths['session_dir']}"
                    )
            except Exception as exc:
                print(f"Event logger close warning: {exc}")

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if yolo_worker is not None:
            yolo_worker.stop()
        yolo_adapter.close()
        head_adapter.close()


if __name__ == "__main__":
    main()
