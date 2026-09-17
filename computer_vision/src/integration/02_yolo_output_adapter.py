"""
02_yolo_output_adapter.py

YOLO object-cue adaptor for the TEEP assessment prototype.

Responsibilities:
- load one selected Ultralytics checkpoint per session;
- validate the current seven-class model schema;
- run frame-level object detection;
- convert Ultralytics tensors/arrays into JSON-safe Python values;
- preserve raw model labels and normalised integration labels;
- expose person_count/person_state and session metadata;
- provide a lightweight webcam smoke test with screenshot support.

Non-responsibilities:
- no duration/event rules;
- no head-gaze fusion;
- no cheating decision;
- no automatic checkpoint switching;
- no special audio-device threshold beyond the configured raw YOLO threshold.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import cv2


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "yolo"


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "5e": {
        "name": "original_5e_best",
        "role": "default_balanced",
        "filename": "original_5e_best.pt",
    },
    "15e": {
        "name": "extended_epoch10",
        "role": "recall_oriented",
        "filename": "extended_epoch10.pt",
    },
    "30e": {
        "name": "extended_epoch25",
        "role": "background_conservative",
        "filename": "extended_epoch25.pt",
    },
    "50e": {
        "name": "extended_best",
        "role": "computer_audio_oriented",
        "filename": "extended_best.pt",
    },
}

DEFAULT_CHECKPOINT_ID = "5e"


# ---------------------------------------------------------------------------
# Frozen current model schema
# ---------------------------------------------------------------------------

EXPECTED_MODEL_NAMES: Dict[int, str] = {
    0: "person",
    1: "phone",
    2: "laptop",
    3: "book_notes",
    4: "calculator",
    5: "watch",
    6: "audio_device",
}

FINAL_INTEGRATION_LABELS = {
    "person",
    "phone",
    "computer_device",
    "book_notes",
    "calculator",
    "watch",
    "audio_device",
}

# Current checkpoint labels plus explicit optional aliases.
LABEL_MAPPING: Dict[str, str] = {
    "person": "person",
    "student": "person",
    "extra_person": "person",

    "phone": "phone",
    "mobile_phone": "phone",

    "laptop": "computer_device",
    "computer_device": "computer_device",
    "keyboard": "computer_device",
    "mouse": "computer_device",

    "book": "book_notes",
    "paper": "book_notes",
    "notes": "book_notes",
    "book_notes": "book_notes",

    "calculator": "calculator",
    "watch": "watch",

    "headphone": "audio_device",
    "headphones": "audio_device",
    "earphone": "audio_device",
    "earphones": "audio_device",
    "earbuds": "audio_device",
    "headset": "audio_device",
    "audio_device": "audio_device",
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class YoloAdapterConfig:
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID
    confidence_threshold: float = 0.25
    image_size: int = 640
    iou_threshold: float = 0.45
    device: str = "cpu"

    def validate(self) -> None:
        if self.checkpoint_id not in CHECKPOINTS:
            raise ValueError(
                f"Unknown checkpoint_id={self.checkpoint_id!r}. "
                f"Choose from: {sorted(CHECKPOINTS)}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")


DetectionDict = Dict[str, Any]
FrameOutput = Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _python_int(value: Any) -> int:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _python_xyxy(value: Any) -> List[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()

    if (
        isinstance(value, Sequence)
        and len(value) == 1
        and isinstance(value[0], Sequence)
    ):
        value = value[0]

    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {value!r}")

    return [float(v) for v in value]


def normalise_label(model_label: str) -> str:
    key = str(model_label).strip()

    if key not in LABEL_MAPPING:
        raise ValueError(
            f"Unknown YOLO model label {key!r}. "
            "Update LABEL_MAPPING explicitly before using this checkpoint."
        )

    label = LABEL_MAPPING[key]

    if label not in FINAL_INTEGRATION_LABELS:
        raise ValueError(
            f"Label mapping produced invalid integration label {label!r}."
        )

    return label


def person_state_from_count(person_count: int) -> str:
    if person_count <= 0:
        return "NO_PERSON"
    if person_count == 1:
        return "ONE_PERSON"
    return "MULTIPLE_PERSONS"


def _normalise_model_names(
    names: Union[Mapping[Any, Any], Sequence[Any]],
) -> Dict[int, str]:
    if isinstance(names, Mapping):
        return {int(class_id): str(label) for class_id, label in names.items()}

    return {int(class_id): str(label) for class_id, label in enumerate(names)}


def validate_model_names(model_names: Mapping[int, str]) -> None:
    actual = {int(class_id): str(label) for class_id, label in model_names.items()}

    if actual != EXPECTED_MODEL_NAMES:
        raise ValueError(
            "Checkpoint class schema does not match the frozen integration schema.\n"
            f"Expected: {EXPECTED_MODEL_NAMES}\n"
            f"Actual:   {actual}\n"
            "Do not continue until the model schema and adapter mapping are reviewed."
        )

    for model_label in actual.values():
        normalise_label(model_label)


# ---------------------------------------------------------------------------
# Main adaptor
# ---------------------------------------------------------------------------

class YoloOutputAdapter:
    """Stable YOLO frame-output interface for later integration."""

    def __init__(
        self,
        config: Optional[YoloAdapterConfig] = None,
        model_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or YoloAdapterConfig()
        self.config.validate()

        self.model_dir = (
            Path(model_dir).expanduser().resolve()
            if model_dir is not None
            else DEFAULT_MODEL_DIR.resolve()
        )

        checkpoint_info = CHECKPOINTS[self.config.checkpoint_id]
        self.checkpoint_id = self.config.checkpoint_id
        self.checkpoint_name = checkpoint_info["name"]
        self.checkpoint_role = checkpoint_info["role"]
        self.checkpoint_path = (
            self.model_dir / checkpoint_info["filename"]
        ).resolve()

        self.model: Any = None
        self.model_names: Dict[int, str] = {}
        self._started = False

    def start(self) -> None:
        """Load one selected checkpoint and validate its class schema."""
        if self._started:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "YOLO checkpoint not found:\n"
                f"  {self.checkpoint_path}\n"
                "Check --model-dir or the project model folder."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed in the active environment."
            ) from exc

        self.model = YOLO(str(self.checkpoint_path))
        self.model_names = _normalise_model_names(self.model.names)
        validate_model_names(self.model_names)
        self._started = True

    def close(self) -> None:
        self.model = None
        self._started = False

    def get_session_metadata(self) -> Dict[str, Any]:
        """Return session-level checkpoint and inference configuration."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_role": self.checkpoint_role,
            "checkpoint_path": str(self.checkpoint_path),
            "confidence_threshold": float(self.config.confidence_threshold),
            "image_size": int(self.config.image_size),
            "iou_threshold": float(self.config.iou_threshold),
            "device": str(self.config.device),
            "model_names": {
                str(class_id): model_label
                for class_id, model_label in self.model_names.items()
            },
        }

    def adapt_result(self, result: Any) -> List[DetectionDict]:
        """
        Convert one Ultralytics Result into JSON-safe detection dictionaries.

        Empty result -> []
        """
        detections: List[DetectionDict] = []

        if result is None:
            return detections

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            class_id = _python_int(box.cls[0])
            confidence = _python_float(box.conf[0])
            bbox_xyxy = _python_xyxy(box.xyxy)

            if class_id not in self.model_names:
                raise ValueError(
                    f"Detection class_id={class_id} is not present in model.names."
                )

            model_label = self.model_names[class_id]
            label = normalise_label(model_label)

            detections.append(
                {
                    "class_id": class_id,
                    "model_label": model_label,
                    "label": label,
                    "confidence": confidence,
                    "bbox_xyxy": bbox_xyxy,
                }
            )

        return detections

    def process_frame(
        self,
        frame: Any,
        timestamp: Optional[float] = None,
    ) -> FrameOutput:
        """Run YOLO on one frame and return neutral object-cue state only."""
        if not self._started or self.model is None:
            raise RuntimeError(
                "YoloOutputAdapter.start() must be called before process_frame()."
            )

        if frame is None:
            raise ValueError("frame must not be None.")

        results = self.model.predict(
            source=frame,
            conf=float(self.config.confidence_threshold),
            imgsz=int(self.config.image_size),
            iou=float(self.config.iou_threshold),
            device=str(self.config.device),
            verbose=False,
        )

        detections = self.adapt_result(results[0]) if results else []

        person_count = sum(
            1 for detection in detections if detection["label"] == "person"
        )

        frame_timestamp = (
            float(timestamp)
            if timestamp is not None
            else float(time.monotonic())
        )

        return {
            "timestamp": frame_timestamp,
            "detections": detections,
            "person_count": int(person_count),
            "person_state": person_state_from_count(person_count),
        }


# ---------------------------------------------------------------------------
# Smoke-test visualisation
# ---------------------------------------------------------------------------

def draw_frame_output(
    frame: Any,
    frame_output: FrameOutput,
    metadata: Mapping[str, Any],
) -> Any:
    display = frame.copy()

    cv2.putText(
        display,
        f"YOLO Adapter | {metadata['checkpoint_id']}: {metadata['checkpoint_name']}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        (
            f"conf={metadata['confidence_threshold']:.2f} "
            f"imgsz={metadata['image_size']} "
            f"iou={metadata['iou_threshold']:.2f} "
            f"device={metadata['device']}"
        ),
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        f"Person: {frame_output['person_state']} ({frame_output['person_count']})",
        (18, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        display,
        "S screenshot | Q quit",
        (18, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    height, width = display.shape[:2]

    for detection in frame_output["detections"]:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        x1_i = max(0, min(width - 1, int(round(x1))))
        y1_i = max(0, min(height - 1, int(round(y1))))
        x2_i = max(0, min(width - 1, int(round(x2))))
        y2_i = max(0, min(height - 1, int(round(y2))))

        cv2.rectangle(display, (x1_i, y1_i), (x2_i, y2_i), (255, 255, 255), 2)

        model_label = detection["model_label"]
        label = detection["label"]
        confidence = detection["confidence"]

        if model_label == label:
            text = f"{label} {confidence:.2f}"
        else:
            text = f"{label} ({model_label}) {confidence:.2f}"

        cv2.putText(
            display,
            text,
            (x1_i, max(20, y1_i - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return display


def save_smoke_test_screenshot(
    frame: Any,
    screenshot_dir: Path,
    checkpoint_id: str,
) -> Path:
    screenshot_dir = Path(screenshot_dir).expanduser().resolve()
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = screenshot_dir / f"yolo_adapter_{checkpoint_id}_{stamp}.jpg"

    if not cv2.imwrite(str(path), frame):
        raise IOError(f"Could not save screenshot: {path}")

    return path


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO output adaptor smoke test. "
            "Normalises detections only; no event logic."
        )
    )

    parser.add_argument(
        "--checkpoint",
        choices=sorted(CHECKPOINTS),
        default=DEFAULT_CHECKPOINT_ID,
        help="Checkpoint ID to load. Default: 5e.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=(
            "Directory containing checkpoint .pt files. "
            "Default: computer_vision/models/yolo"
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--print-interval", type=float, default=1.0)
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "results"
            / "integration"
            / "yolo_adapter_test"
            / "screenshots"
        ),
    )

    return parser.parse_args()


def run_smoke_test() -> None:
    args = parse_args()

    config = YoloAdapterConfig(
        checkpoint_id=args.checkpoint,
        confidence_threshold=args.conf,
        image_size=args.imgsz,
        iou_threshold=args.iou,
        device=args.device,
    )

    adapter = YoloOutputAdapter(config=config, model_dir=args.model_dir)

    print("Loading YOLO checkpoint...")
    adapter.start()

    metadata = adapter.get_session_metadata()
    json.dumps(metadata)  # explicit serialisability check

    print("\n[SESSION_METADATA]")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        adapter.close()
        raise RuntimeError(f"Could not open webcam index {args.cam}.")

    print("\nSmoke test started.")
    print("Controls: S = screenshot | Q = quit")

    last_print_time = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: webcam frame read failed.")
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            output = adapter.process_frame(frame)
            json.dumps(output)  # explicit serialisability check

            now = time.monotonic()
            if now - last_print_time >= args.print_interval:
                print("\n[YOLO_OUTPUT]")
                print(json.dumps(output, indent=2, ensure_ascii=False))
                last_print_time = now

            display = draw_frame_output(frame, output, metadata)
            cv2.imshow("YOLO Output Adapter Smoke Test", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("s"), ord("S")):
                try:
                    path = save_smoke_test_screenshot(
                        display,
                        args.screenshot_dir,
                        adapter.checkpoint_id,
                    )
                    print(f"Screenshot saved: {path}")
                except IOError as error:
                    print(f"Screenshot error: {error}")

            elif key in (ord("q"), ord("Q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        adapter.close()


if __name__ == "__main__":
    run_smoke_test()
