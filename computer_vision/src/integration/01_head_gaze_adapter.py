"""
Reusable head-pose + gaze adapter for TEEP multi-module integration.

This GitHub-ready module keeps the verified head-pose, eye-reliability and
L2CS gaze logic, but intentionally excludes the structured evaluation
protocol, ground-truth CSV recorder and audio cues.

It returns a standardized dictionary for the later event manager. It does
not make a cheating decision.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Verified project constants
# ---------------------------------------------------------------------------

LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)
LEFT_BLINK_NAME = "eyeBlinkLeft"
RIGHT_BLINK_NAME = "eyeBlinkRight"

HEAD_CALIBRATION_FRAMES = 30
HEAD_CALIBRATION_MAX_AXIS_RANGE_DEG = 5.0

MIN_FACE_AREA_RATIO = 0.05
MAX_FACE_AREA_RATIO = 0.45
MAX_FACE_CENTER_X_OFFSET = 0.18
MAX_FACE_CENTER_Y_OFFSET = 0.20
MIN_FRAME_MARGIN_RATIO = 0.02
MAX_NORMALIZED_REPROJECTION_ERROR = 0.10

DEFAULT_HEAD_FORWARD_MAX_ABS_PITCH_DEG = 20.0
DEFAULT_HEAD_FORWARD_MAX_ABS_YAW_DEG = 15.0
DEFAULT_HEAD_FORWARD_MAX_ABS_ROLL_DEG = 12.0
DEFAULT_HEAD_PITCH_THRESHOLD_DEG = 12.0
DEFAULT_HEAD_YAW_THRESHOLD_DEG = 15.0
DEFAULT_HEAD_ROLL_THRESHOLD_DEG = 15.0
DEFAULT_HEAD_ROTATION_JUMP_THRESHOLD_DEG = 75.0
DEFAULT_HEAD_RELATIVE_ROLL_FLIP_THRESHOLD_DEG = 90.0

RX_180 = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# L2CS worker data structures
# ---------------------------------------------------------------------------

@dataclass
class L2CSResult:
    """Latest completed L2CS inference result."""

    frame_id: int = -1
    captured_time: float = 0.0
    completed_time: float = 0.0
    bboxes: np.ndarray | None = None
    pitch: np.ndarray | None = None
    yaw: np.ndarray | None = None
    inference_fps: float = 0.0
    error: str | None = None


class LatestFrameExchange:
    """Thread-safe latest-frame input and latest-result output."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.new_frame_event = threading.Event()

        self.frame_id = -1
        self.frame_time = 0.0
        self.frame: np.ndarray | None = None

        self.result = L2CSResult()

    def submit(
        self,
        frame_id: int,
        frame_time: float,
        frame: np.ndarray,
    ) -> None:
        """Replace the pending worker input with the newest frame."""

        with self.lock:
            self.frame_id = frame_id
            self.frame_time = frame_time
            self.frame = frame.copy()

        self.new_frame_event.set()

    def take_latest_frame(
        self,
    ) -> tuple[int, float, np.ndarray] | None:
        """Return the newest submitted frame."""

        with self.lock:
            if self.frame is None:
                return None

            return (
                self.frame_id,
                self.frame_time,
                self.frame.copy(),
            )

    def store_result(self, result: L2CSResult) -> None:
        """Store the newest completed inference result."""

        with self.lock:
            self.result = result

    def read_result(self) -> L2CSResult:
        """Return a copy of the latest result metadata and arrays."""

        with self.lock:
            result = self.result

            return L2CSResult(
                frame_id=result.frame_id,
                captured_time=result.captured_time,
                completed_time=result.completed_time,
                bboxes=(
                    None
                    if result.bboxes is None
                    else result.bboxes.copy()
                ),
                pitch=(
                    None
                    if result.pitch is None
                    else result.pitch.copy()
                ),
                yaw=(
                    None
                    if result.yaw is None
                    else result.yaw.copy()
                ),
                inference_fps=result.inference_fps,
                error=result.error,
            )


def l2cs_worker(
    *,
    pipeline: Pipeline,
    exchange: LatestFrameExchange,
    stop_event: threading.Event,
) -> None:
    """Run L2CS-Net in a background thread using only the newest frame."""

    last_processed_frame_id = -1

    with torch.no_grad():
        while not stop_event.is_set():
            exchange.new_frame_event.wait(timeout=0.1)

            if stop_event.is_set():
                break

            exchange.new_frame_event.clear()
            item = exchange.take_latest_frame()

            if item is None:
                continue

            frame_id, captured_time, frame = item

            if frame_id == last_processed_frame_id:
                continue

            last_processed_frame_id = frame_id
            inference_start = time.perf_counter()

            try:
                result = pipeline.step(frame)
                elapsed = time.perf_counter() - inference_start

                exchange.store_result(
                    L2CSResult(
                        frame_id=frame_id,
                        captured_time=captured_time,
                        completed_time=time.perf_counter(),
                        bboxes=np.asarray(result.bboxes).copy(),
                        pitch=np.asarray(result.pitch).copy(),
                        yaw=np.asarray(result.yaw).copy(),
                        inference_fps=(
                            1.0 / elapsed if elapsed > 0 else 0.0
                        ),
                    )
                )

            except Exception as error:  # Worker must not silently die.
                exchange.store_result(
                    L2CSResult(
                        frame_id=frame_id,
                        captured_time=captured_time,
                        completed_time=time.perf_counter(),
                        error=f"{type(error).__name__}: {error}",
                    )
                )



# ---------------------------------------------------------------------------
# Passive calibration
# ---------------------------------------------------------------------------

class PassiveEyeCalibrator:
    """Collect high-confidence open-eye EAR samples without user actions."""

    def __init__(
        self,
        *,
        minimum_samples: int,
        minimum_duration: float,
        maximum_samples: int = 180,
    ) -> None:
        self.minimum_samples = minimum_samples
        self.minimum_duration = minimum_duration
        self.maximum_samples = maximum_samples
        self.reset()

    def reset(self) -> None:
        """Restart calibration."""

        self.state = "COLLECTING"
        self.started_time = time.perf_counter()
        self.left_samples: deque[float] = deque(
            maxlen=self.maximum_samples
        )
        self.right_samples: deque[float] = deque(
            maxlen=self.maximum_samples
        )
        self.left_baseline = float("nan")
        self.right_baseline = float("nan")

    @property
    def ready(self) -> bool:
        return self.state == "READY"

    @property
    def sample_count(self) -> int:
        return min(
            len(self.left_samples),
            len(self.right_samples),
        )

    def consider_sample(
        self,
        *,
        left_ear: float,
        right_ear: float,
        left_blink: float,
        right_blink: float,
        maximum_blink_score: float,
        minimum_raw_ear: float,
        maximum_eye_ratio: float,
    ) -> None:
        """Add one high-confidence open-eye sample when appropriate."""

        if self.ready:
            return

        if (
            not np.isfinite(left_ear)
            or not np.isfinite(right_ear)
            or not np.isfinite(left_blink)
            or not np.isfinite(right_blink)
        ):
            return

        if (
            left_blink > maximum_blink_score
            or right_blink > maximum_blink_score
        ):
            return

        if (
            left_ear < minimum_raw_ear
            or right_ear < minimum_raw_ear
        ):
            return

        smaller = min(left_ear, right_ear)
        larger = max(left_ear, right_ear)

        if smaller <= 1e-6 or larger / smaller > maximum_eye_ratio:
            return

        self.left_samples.append(float(left_ear))
        self.right_samples.append(float(right_ear))

        elapsed = time.perf_counter() - self.started_time

        if (
            self.sample_count >= self.minimum_samples
            and elapsed >= self.minimum_duration
        ):
            self.left_baseline = float(
                np.median(np.asarray(self.left_samples))
            )
            self.right_baseline = float(
                np.median(np.asarray(self.right_samples))
            )
            self.state = "READY"


class PassiveGazeCenterCalibrator:
    """Collect stable L2CS samples while the user watches the centre target."""

    def __init__(
        self,
        *,
        minimum_samples: int,
        minimum_duration: float,
        maximum_samples: int = 60,
    ) -> None:
        self.minimum_samples = minimum_samples
        self.minimum_duration = minimum_duration
        self.maximum_samples = maximum_samples
        self.reset(waiting=True)

    def reset(self, *, waiting: bool = True) -> None:
        """Reset calibration, optionally waiting for eye calibration first."""

        self.state = "WAITING" if waiting else "COLLECTING"
        self.started_time = (
            0.0 if waiting else time.perf_counter()
        )
        self.pitch_samples: deque[float] = deque(
            maxlen=self.maximum_samples
        )
        self.yaw_samples: deque[float] = deque(
            maxlen=self.maximum_samples
        )
        self.pitch_baseline = float("nan")
        self.yaw_baseline = float("nan")

    def start(self) -> None:
        """Begin centre-gaze sample collection."""

        self.reset(waiting=False)

    @property
    def ready(self) -> bool:
        return self.state == "READY"

    @property
    def collecting(self) -> bool:
        return self.state == "COLLECTING"

    @property
    def sample_count(self) -> int:
        return min(
            len(self.pitch_samples),
            len(self.yaw_samples),
        )

    def consider_sample(
        self,
        *,
        pitch_degrees: float,
        yaw_degrees: float,
    ) -> None:
        """Add one stable, fresh centre-gaze sample."""

        if not self.collecting:
            return

        if (
            not np.isfinite(pitch_degrees)
            or not np.isfinite(yaw_degrees)
        ):
            return

        self.pitch_samples.append(float(pitch_degrees))
        self.yaw_samples.append(float(yaw_degrees))

        elapsed = time.perf_counter() - self.started_time

        if (
            self.sample_count >= self.minimum_samples
            and elapsed >= self.minimum_duration
        ):
            self.pitch_baseline = float(
                np.median(np.asarray(self.pitch_samples))
            )
            self.yaw_baseline = float(
                np.median(np.asarray(self.yaw_samples))
            )
            self.state = "READY"


class GazeCalibrationStabilityGate:
    """Check whether recent raw L2CS angles are stable enough to calibrate."""

    def __init__(
        self,
        *,
        window_size: int,
        maximum_std_degrees: float,
    ) -> None:
        self.window_size = window_size
        self.maximum_std_degrees = maximum_std_degrees
        self.reset()

    def reset(self) -> None:
        self.pitch_history: deque[float] = deque(
            maxlen=self.window_size
        )
        self.yaw_history: deque[float] = deque(
            maxlen=self.window_size
        )
        self.pitch_std = float("nan")
        self.yaw_std = float("nan")
        self.stable = False

    def update(
        self,
        *,
        pitch_degrees: float,
        yaw_degrees: float,
    ) -> bool:
        """Add a fresh result and return whether the window is stable."""

        self.pitch_history.append(float(pitch_degrees))
        self.yaw_history.append(float(yaw_degrees))

        if (
            len(self.pitch_history) < self.window_size
            or len(self.yaw_history) < self.window_size
        ):
            self.pitch_std = float("nan")
            self.yaw_std = float("nan")
            self.stable = False
            return False

        self.pitch_std = float(
            np.std(np.asarray(self.pitch_history))
        )
        self.yaw_std = float(
            np.std(np.asarray(self.yaw_history))
        )

        self.stable = (
            self.pitch_std <= self.maximum_std_degrees
            and self.yaw_std <= self.maximum_std_degrees
        )

        return self.stable


class ExponentialAngleSmoother:
    """Apply exponential smoothing only when a new L2CS result arrives."""

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.pitch = float("nan")
        self.yaw = float("nan")
        self.initialized = False

    def update(
        self,
        pitch_degrees: float,
        yaw_degrees: float,
    ) -> tuple[float, float]:
        """Update and return smoothed pitch/yaw."""

        if not self.initialized:
            self.pitch = float(pitch_degrees)
            self.yaw = float(yaw_degrees)
            self.initialized = True
        else:
            self.pitch = (
                self.alpha * float(pitch_degrees)
                + (1.0 - self.alpha) * self.pitch
            )
            self.yaw = (
                self.alpha * float(yaw_degrees)
                + (1.0 - self.alpha) * self.yaw
            )

        return self.pitch, self.yaw


class DirectionStabilizer:
    """Require repeated L2CS predictions before changing direction label."""

    def __init__(self, confirmations: int) -> None:
        self.confirmations = confirmations
        self.reset()

    def reset(self) -> None:
        self.candidate = "FORWARD"
        self.candidate_count = 0
        self.stable = "FORWARD"

    def update(self, direction: str) -> tuple[str, int]:
        """Update candidate and return stable direction plus count."""

        if direction == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = direction
            self.candidate_count = 1

        if self.candidate_count >= self.confirmations:
            self.stable = self.candidate

        return self.stable, self.candidate_count


def classify_coarse_direction(
    *,
    horizontal_degrees: float,
    vertical_degrees: float,
    horizontal_threshold: float,
    up_threshold: float,
    down_threshold: float,
) -> str:
    """
    Convert corrected L2CS components to one coarse gaze direction.

    L2CS drawing convention used by this repository:
    - positive pitch moves the arrow toward displayed left,
    - negative pitch moves the arrow toward displayed right,
    - positive yaw moves the arrow upward,
    - negative yaw moves the arrow downward.

    The left/right labels therefore currently follow displayed-arrow
    coordinates and must be checked during the structured webcam test.
    """

    horizontal_score = (
        abs(horizontal_degrees) / horizontal_threshold
    )

    vertical_threshold = (
        up_threshold
        if vertical_degrees >= 0
        else down_threshold
    )

    vertical_score = (
        abs(vertical_degrees) / vertical_threshold
    )

    if horizontal_score < 1.0 and vertical_score < 1.0:
        return "FORWARD"

    if horizontal_score >= vertical_score:
        # Live webcam validation showed that the previous labels were
        # reversed for this project display/camera convention.
        return (
            "LOOKING_RIGHT"
            if horizontal_degrees > 0
            else "LOOKING_LEFT"
        )

    return (
        "LOOKING_UP"
        if vertical_degrees > 0
        else "LOOKING_DOWN"
    )


# ---------------------------------------------------------------------------
# Eye-state tracking
# ---------------------------------------------------------------------------

@dataclass
class EyeDecision:
    """Current eye-state output."""

    status: str
    reliability: str
    duration: float
    left_closed_signal: bool
    right_closed_signal: bool
    blink_count: int


class EyeStateTracker:
    """
    Time-based binocular eye-state classifier.

    The state machine separates three different questions:

    1. Is there a short bilateral closure that should count as a blink?
    2. Is there strong sustained evidence that both eyes are closed?
    3. Is one eye persistently unavailable because of closure, occlusion,
       or poor landmark geometry?

    A short relaxed bilateral closure can count as a blink. Sustained
    EYES_CLOSED still requires stricter evidence, which helps avoid treating
    extreme downward gaze as definite eye closure.
    """

    def __init__(self) -> None:
        self.both_start: float | None = None
        self.both_strong_seen = False

        self.left_start: float | None = None
        self.right_start: float | None = None

        self.blink_count = 0
        self.blink_display_until = 0.0

    def reset_temporal_state(self) -> None:
        """Reset incomplete temporal candidates without resetting the count."""

        self.both_start = None
        self.both_strong_seen = False
        self.left_start = None
        self.right_start = None

    def update(
        self,
        *,
        now: float,
        left_normalized_ear: float,
        right_normalized_ear: float,
        left_blink: float,
        right_blink: float,
        strong_closed_ratio: float,
        ambiguous_ratio: float,
        one_eye_ratio: float,
        blink_threshold: float,
        open_blink_threshold: float,
        closed_duration: float,
        one_eye_duration: float,
        minimum_blink_duration: float,
        maximum_normalized_ear: float,
        one_eye_min_open_ratio: float,
        one_eye_min_difference: float,
        one_eye_max_relative_ratio: float,
    ) -> EyeDecision:
        """Update and return the current eye-state decision."""

        finite = all(
            np.isfinite(value)
            for value in (
                left_normalized_ear,
                right_normalized_ear,
                left_blink,
                right_blink,
            )
        )

        if not finite:
            self.reset_temporal_state()

            return EyeDecision(
                status="NO_EYE_MEASUREMENT",
                reliability="UNRELIABLE",
                duration=0.0,
                left_closed_signal=False,
                right_closed_signal=False,
                blink_count=self.blink_count,
            )

        # Large side-profile turns can make projected eye width very small,
        # producing normalized EAR values far above the calibrated open-eye
        # baseline. Such measurements must not be accepted as reliable open
        # eyes.
        if (
            left_normalized_ear > maximum_normalized_ear
            or right_normalized_ear > maximum_normalized_ear
        ):
            self.reset_temporal_state()

            return EyeDecision(
                status="PROFILE_OR_LANDMARK_UNCERTAIN",
                reliability="UNRELIABLE",
                duration=0.0,
                left_closed_signal=False,
                right_closed_signal=False,
                blink_count=self.blink_count,
            )

        average_blink_score = (
            left_blink + right_blink
        ) / 2.0

        # Relaxed bilateral signal: suitable for detecting a short natural
        # blink. Requiring both normalized EAR values to be low prevents a
        # one-eye closure from entering this branch.
        both_relaxed_closed = (
            left_normalized_ear <= ambiguous_ratio
            and right_normalized_ear <= ambiguous_ratio
            and average_blink_score >= blink_threshold
        )

        # Strong bilateral evidence is required before a sustained candidate
        # becomes definite EYES_CLOSED.
        both_strong_closed = (
            left_normalized_ear <= strong_closed_ratio
            and right_normalized_ear <= strong_closed_ratio
            and left_blink >= blink_threshold
            and right_blink >= blink_threshold
        )

        if both_relaxed_closed:
            if self.both_start is None:
                self.both_start = now
                self.both_strong_seen = False

            self.both_strong_seen = (
                self.both_strong_seen
                or both_strong_closed
            )

            self.left_start = None
            self.right_start = None

            duration = now - self.both_start

            if (
                duration >= closed_duration
                and self.both_strong_seen
            ):
                return EyeDecision(
                    status="EYES_CLOSED",
                    reliability="UNRELIABLE",
                    duration=duration,
                    left_closed_signal=True,
                    right_closed_signal=True,
                    blink_count=self.blink_count,
                )

            if duration >= closed_duration:
                # The eyes look narrow for a sustained period, but the strict
                # closed-eye evidence was never reached. This is deliberately
                # kept uncertain because extreme downward gaze can look like
                # this.
                return EyeDecision(
                    status="EYE_STATUS_UNCERTAIN",
                    reliability="UNRELIABLE",
                    duration=duration,
                    left_closed_signal=True,
                    right_closed_signal=True,
                    blink_count=self.blink_count,
                )

            return EyeDecision(
                status="BLINK_CANDIDATE",
                reliability="UNRELIABLE",
                duration=duration,
                left_closed_signal=True,
                right_closed_signal=True,
                blink_count=self.blink_count,
            )

        # The relaxed bilateral closure has ended. A short completed closure
        # is counted once when the eyes reopen.
        if self.both_start is not None:
            completed_duration = now - self.both_start
            completed_was_strong = self.both_strong_seen

            self.both_start = None
            self.both_strong_seen = False

            if (
                minimum_blink_duration
                <= completed_duration
                < closed_duration
            ):
                self.blink_count += 1
                self.blink_display_until = now + 0.25

            elif (
                completed_duration >= closed_duration
                and completed_was_strong
            ):
                # A long deliberate eye closure is not added to the normal
                # blink counter.
                self.blink_display_until = 0.0

        # ------------------------------------------------------------------
        # One-eye unavailability
        # ------------------------------------------------------------------

        smaller_ear = min(
            left_normalized_ear,
            right_normalized_ear,
        )
        larger_ear = max(
            left_normalized_ear,
            right_normalized_ear,
        )

        eye_difference = larger_ear - smaller_ear
        relative_ratio = (
            smaller_ear / larger_ear
            if larger_ear > 1e-6
            else 1.0
        )

        left_is_smaller = (
            left_normalized_ear < right_normalized_ear
        )
        right_is_smaller = (
            right_normalized_ear < left_normalized_ear
        )

        # Very clear geometry: one eye is almost closed while the other eye
        # remains open. Blendshape scores are not required.
        clear_left_geometry = (
            left_is_smaller
            and left_normalized_ear <= strong_closed_ratio
            and right_normalized_ear >= one_eye_min_open_ratio
        )

        clear_right_geometry = (
            right_is_smaller
            and right_normalized_ear <= strong_closed_ratio
            and left_normalized_ear >= one_eye_min_open_ratio
        )

        # Broader persistent asymmetry: useful for hand occlusion or imperfect
        # MediaPipe blink scores. Duration is required before confirmation.
        asymmetric_geometry = (
            smaller_ear <= one_eye_ratio
            and larger_ear >= one_eye_min_open_ratio
            and eye_difference >= one_eye_min_difference
            and relative_ratio <= one_eye_max_relative_ratio
        )

        left_blendshape_support = (
            left_blink >= blink_threshold
            and (
                right_blink <= open_blink_threshold
                or left_blink > right_blink
            )
        )

        right_blendshape_support = (
            right_blink >= blink_threshold
            and (
                left_blink <= open_blink_threshold
                or right_blink > left_blink
            )
        )

        left_one_eye_signal = (
            clear_left_geometry
            or (
                asymmetric_geometry
                and left_is_smaller
            )
            or (
                left_blendshape_support
                and left_normalized_ear <= one_eye_ratio
                and right_normalized_ear
                >= one_eye_min_open_ratio
            )
        )

        right_one_eye_signal = (
            clear_right_geometry
            or (
                asymmetric_geometry
                and right_is_smaller
            )
            or (
                right_blendshape_support
                and right_normalized_ear <= one_eye_ratio
                and left_normalized_ear
                >= one_eye_min_open_ratio
            )
        )

        if left_one_eye_signal and not right_one_eye_signal:
            if self.left_start is None:
                self.left_start = now

            self.right_start = None
            duration = now - self.left_start

            return EyeDecision(
                status=(
                    "ONE_EYE_UNAVAILABLE"
                    if duration >= one_eye_duration
                    else "ONE_EYE_CANDIDATE"
                ),
                reliability="UNRELIABLE",
                duration=duration,
                left_closed_signal=True,
                right_closed_signal=False,
                blink_count=self.blink_count,
            )

        if right_one_eye_signal and not left_one_eye_signal:
            if self.right_start is None:
                self.right_start = now

            self.left_start = None
            duration = now - self.right_start

            return EyeDecision(
                status=(
                    "ONE_EYE_UNAVAILABLE"
                    if duration >= one_eye_duration
                    else "ONE_EYE_CANDIDATE"
                ),
                reliability="UNRELIABLE",
                duration=duration,
                left_closed_signal=False,
                right_closed_signal=True,
                blink_count=self.blink_count,
            )

        self.left_start = None
        self.right_start = None

        if now < self.blink_display_until:
            return EyeDecision(
                status="BLINK",
                reliability="UNRELIABLE",
                duration=0.0,
                left_closed_signal=False,
                right_closed_signal=False,
                blink_count=self.blink_count,
            )

        average_normalized = (
            left_normalized_ear + right_normalized_ear
        ) / 2.0

        ambiguous_eye_opening = (
            average_normalized <= ambiguous_ratio
            or left_normalized_ear <= ambiguous_ratio
            or right_normalized_ear <= ambiguous_ratio
        )

        ambiguous_blink_signal = (
            left_blink >= blink_threshold
            or right_blink >= blink_threshold
        )

        if ambiguous_eye_opening or ambiguous_blink_signal:
            return EyeDecision(
                status="EYE_STATUS_UNCERTAIN",
                reliability="UNRELIABLE",
                duration=0.0,
                left_closed_signal=False,
                right_closed_signal=False,
                blink_count=self.blink_count,
            )

        return EyeDecision(
            status="EYES_OPEN",
            reliability="RELIABLE",
            duration=0.0,
            left_closed_signal=False,
            right_closed_signal=False,
            blink_count=self.blink_count,
        )



# ---------------------------------------------------------------------------
# Canonical 14 head-pose diagnostic
# ---------------------------------------------------------------------------

def angle_diff_degrees(value: float, reference: float) -> float:
    """Return the signed shortest angular difference in degrees."""

    return float(
        (float(value) - float(reference) + 180.0) % 360.0
        - 180.0
    )


def circular_median(values: Sequence[float]) -> float:
    """Return a robust circular median."""

    array = np.asarray(values, dtype=np.float64)

    if array.size == 0:
        raise ValueError("Cannot calculate a median from no angles.")

    total_distances: list[float] = []

    for candidate in array:
        distances = np.abs(
            (array - candidate + 180.0) % 360.0 - 180.0
        )
        total_distances.append(float(np.sum(distances)))

    return float(array[int(np.argmin(total_distances))])


def circular_range(values: Sequence[float]) -> float:
    """Return the angular range after unwrapping around the first sample."""

    if not values:
        return float("inf")

    reference = float(values[0])
    unwrapped = [
        reference + angle_diff_degrees(float(value), reference)
        for value in values
    ]

    return float(max(unwrapped) - min(unwrapped))


def extract_euler_rx_ry_rz(
    matrix: np.ndarray,
) -> tuple[float, float, float]:
    """
    Extract pitch, yaw and roll for:
        R = Rx(pitch) @ Ry(yaw) @ Rz(roll)
    """

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    ).reshape(3, 3)

    yaw = math.asin(
        float(np.clip(matrix[0, 2], -1.0, 1.0))
    )
    cosine_yaw = math.cos(yaw)

    if abs(cosine_yaw) > 1e-8:
        pitch = math.atan2(
            -float(matrix[1, 2]),
            float(matrix[2, 2]),
        )
        roll = math.atan2(
            -float(matrix[0, 1]),
            float(matrix[0, 0]),
        )
    else:
        pitch = math.atan2(
            float(matrix[2, 1]),
            float(matrix[1, 1]),
        )
        roll = 0.0

    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll),
    )


def read_canonical14_subset(
    path: Path,
) -> tuple[list[int], np.ndarray]:
    """Read the fixed Canonical 14 landmark subset."""

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    required = {
        "mediapipe_landmark_id",
        "x_canonical_cm",
        "y_canonical_cm",
        "z_canonical_cm",
    }

    if not rows:
        raise ValueError(f"Canonical 14 subset is empty: {path}")

    if not required.issubset(rows[0]):
        raise ValueError(
            "Canonical 14 subset is missing required columns."
        )

    landmark_ids = [
        int(row["mediapipe_landmark_id"])
        for row in rows
    ]
    model_points = np.asarray(
        [
            [
                float(row["x_canonical_cm"]),
                float(row["y_canonical_cm"]),
                float(row["z_canonical_cm"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    if len(landmark_ids) != 14:
        raise ValueError(
            f"Expected 14 landmarks, found {len(landmark_ids)}."
        )

    return landmark_ids, model_points


def all_landmarks_to_pixels(
    landmarks,
    frame_width: int,
    frame_height: int,
) -> dict[int, tuple[float, float]]:
    """
    Convert normalized MediaPipe landmarks to subpixel image coordinates.

    solvePnP receives float64 coordinates. Integer conversion is deferred
    until drawing so geometric precision is not discarded prematurely.
    """

    return {
        index: (
            float(
                np.clip(
                    float(landmark.x) * frame_width,
                    0.0,
                    float(frame_width - 1),
                )
            ),
            float(
                np.clip(
                    float(landmark.y) * frame_height,
                    0.0,
                    float(frame_height - 1),
                )
            ),
        )
        for index, landmark in enumerate(landmarks)
    }


def largest_face_index(
    faces,
    frame_width: int,
    frame_height: int,
) -> int:
    """Return the index of the largest MediaPipe face."""

    best_index = 0
    best_area = -1.0

    for index, landmarks in enumerate(faces):
        x_values = [
            float(landmark.x) * frame_width
            for landmark in landmarks
        ]
        y_values = [
            float(landmark.y) * frame_height
            for landmark in landmarks
        ]

        area = (
            max(x_values) - min(x_values)
        ) * (
            max(y_values) - min(y_values)
        )

        if area > best_area:
            best_area = area
            best_index = index

    return best_index


def approximate_camera_matrix(
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Build the approximate webcam camera matrix used by the baseline."""

    focal_length = float(frame_width)

    return np.asarray(
        [
            [
                focal_length,
                0.0,
                frame_width / 2.0,
            ],
            [
                0.0,
                focal_length,
                frame_height / 2.0,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def estimate_canonical14_pose(
    landmarks,
    frame_width: int,
    frame_height: int,
    landmark_ids: Sequence[int],
    model_points: np.ndarray,
    *,
    initial_rotation_vector: np.ndarray | None = None,
    initial_translation_vector: np.ndarray | None = None,
) -> dict[str, object] | None:
    """
    Estimate Canonical 14 pitch, yaw and roll.

    When a previous accepted solvePnP pose is available, it is supplied as
    an iterative initial guess. If that solve fails, the function falls back
    to an unseeded solve instead of returning no pose.
    """

    pixel_points = all_landmarks_to_pixels(
        landmarks,
        frame_width,
        frame_height,
    )

    try:
        image_points = np.asarray(
            [
                pixel_points[landmark_id]
                for landmark_id in landmark_ids
            ],
            dtype=np.float64,
        )
    except KeyError:
        return None

    camera_matrix = approximate_camera_matrix(
        frame_width,
        frame_height,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)

    use_initial_guess = (
        initial_rotation_vector is not None
        and initial_translation_vector is not None
    )

    if use_initial_guess:
        rotation_guess = np.asarray(
            initial_rotation_vector,
            dtype=np.float64,
        ).reshape(3, 1).copy()
        translation_guess = np.asarray(
            initial_translation_vector,
            dtype=np.float64,
        ).reshape(3, 1).copy()

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            rotation_guess,
            translation_guess,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success and use_initial_guess:
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success:
        return None

    raw_rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )
    aligned_rotation_matrix = (
        raw_rotation_matrix @ RX_180
    )

    pitch, yaw, roll = extract_euler_rx_ry_rz(
        aligned_rotation_matrix
    )

    projected_points, _ = cv2.projectPoints(
        model_points,
        rotation_vector,
        translation_vector,
        camera_matrix,
        distortion,
    )
    projected_points = projected_points.reshape(-1, 2)

    reprojection_error_px = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (projected_points - image_points) ** 2,
                    axis=1,
                )
            )
        )
    )

    x_values = [
        float(point[0])
        for point in pixel_points.values()
    ]
    y_values = [
        float(point[1])
        for point in pixel_points.values()
    ]

    bbox = (
        max(0.0, min(x_values)),
        max(0.0, min(y_values)),
        min(float(frame_width - 1), max(x_values)),
        min(float(frame_height - 1), max(y_values)),
    )

    bbox_diagonal = max(
        1.0,
        math.hypot(
            float(bbox[2] - bbox[0]),
            float(bbox[3] - bbox[1]),
        ),
    )

    return {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
        "pixel_points": pixel_points,
        "bbox": bbox,
        "normalized_reprojection_error": (
            reprojection_error_px / bbox_diagonal
        ),
        "rotation_matrix": aligned_rotation_matrix,
        "rotation_vector": np.asarray(
            rotation_vector,
            dtype=np.float64,
        ).reshape(3, 1),
        "translation_vector": np.asarray(
            translation_vector,
            dtype=np.float64,
        ).reshape(3, 1),
        "used_initial_guess": bool(use_initial_guess),
    }


def rotation_geodesic_distance_degrees(
    first_rotation: np.ndarray,
    second_rotation: np.ndarray,
) -> float:
    """Return the shortest 3D rotation distance between two matrices."""

    first = np.asarray(
        first_rotation,
        dtype=np.float64,
    ).reshape(3, 3)
    second = np.asarray(
        second_rotation,
        dtype=np.float64,
    ).reshape(3, 3)

    relative_rotation = first.T @ second
    cosine_angle = (
        float(np.trace(relative_rotation)) - 1.0
    ) / 2.0
    cosine_angle = float(
        np.clip(cosine_angle, -1.0, 1.0)
    )

    return math.degrees(math.acos(cosine_angle))


class HeadPoseTemporalGuard:
    """
    Accept only temporally plausible Canonical 14 pose candidates.

    Rejected candidates remain available for CSV logging, but are not allowed
    to update the previous accepted pose or enter the fusion layer.
    """

    def __init__(
        self,
        *,
        maximum_rotation_jump_degrees: float,
        maximum_relative_roll_degrees: float,
    ) -> None:
        self.maximum_rotation_jump_degrees = float(
            maximum_rotation_jump_degrees
        )
        self.maximum_relative_roll_degrees = float(
            maximum_relative_roll_degrees
        )
        self.reset()

    def reset(self) -> None:
        self.previous_rotation_matrix: np.ndarray | None = None
        self.previous_rotation_vector: np.ndarray | None = None
        self.previous_translation_vector: np.ndarray | None = None
        self.accepted_count = 0
        self.rejected_count = 0

    def initial_guess(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return copies of the previous accepted raw solvePnP vectors."""

        rotation = (
            None
            if self.previous_rotation_vector is None
            else self.previous_rotation_vector.copy()
        )
        translation = (
            None
            if self.previous_translation_vector is None
            else self.previous_translation_vector.copy()
        )

        return rotation, translation

    def evaluate(
        self,
        candidate: dict[str, object] | None,
        *,
        neutral_pose: tuple[float, float, float] | None,
    ) -> tuple[
        dict[str, object] | None,
        str,
        str,
        float,
    ]:
        """
        Return:
            accepted_pose,
            temporal_state,
            rejection_reason,
            rotation_jump_degrees
        """

        if candidate is None:
            return (
                None,
                "NO_POSE_CANDIDATE",
                "POSE_NOT_SOLVED",
                float("nan"),
            )

        current_rotation = np.asarray(
            candidate["rotation_matrix"],
            dtype=np.float64,
        ).reshape(3, 3)

        if self.previous_rotation_matrix is None:
            rotation_jump = 0.0
        else:
            rotation_jump = rotation_geodesic_distance_degrees(
                self.previous_rotation_matrix,
                current_rotation,
            )

        rejection_reasons: list[str] = []

        if (
            self.previous_rotation_matrix is not None
            and rotation_jump
            > self.maximum_rotation_jump_degrees
        ):
            rejection_reasons.append(
                "ROTATION_JUMP"
            )

        if neutral_pose is not None:
            relative_roll = angle_diff_degrees(
                float(candidate["roll"]),
                float(neutral_pose[2]),
            )

            if (
                abs(relative_roll)
                > self.maximum_relative_roll_degrees
            ):
                rejection_reasons.append(
                    "RELATIVE_ROLL_FLIP"
                )

        if rejection_reasons:
            self.rejected_count += 1

            return (
                None,
                "POSE_FLIP_REJECTED",
                "|".join(rejection_reasons),
                float(rotation_jump),
            )

        self.previous_rotation_matrix = current_rotation.copy()
        self.previous_rotation_vector = np.asarray(
            candidate["rotation_vector"],
            dtype=np.float64,
        ).reshape(3, 1).copy()
        self.previous_translation_vector = np.asarray(
            candidate["translation_vector"],
            dtype=np.float64,
        ).reshape(3, 1).copy()
        self.accepted_count += 1

        return (
            candidate,
            "POSE_ACCEPTED",
            "",
            float(rotation_jump),
        )


def evaluate_head_framing(
    face_count: int,
    pose: dict[str, object] | None,
    frame_width: int,
    frame_height: int,
) -> dict[str, object]:
    """Apply the safe-framing rules from the Canonical 14 live script."""

    if face_count == 0:
        return {
            "state": "NO_FACE",
            "good": False,
        }

    if face_count > 1:
        return {
            "state": "MULTIPLE_FACES",
            "good": False,
        }

    if pose is None:
        return {
            "state": "POSE_FAILED",
            "good": False,
        }

    x1, y1, x2, y2 = pose["bbox"]

    bbox_width = max(1, int(x2) - int(x1))
    bbox_height = max(1, int(y2) - int(y1))

    face_area_ratio = (
        bbox_width * bbox_height
    ) / float(frame_width * frame_height)

    face_center_x_ratio = (
        (float(x1) + float(x2)) / 2.0
    ) / float(frame_width)
    face_center_y_ratio = (
        (float(y1) + float(y2)) / 2.0
    ) / float(frame_height)

    normalized_error = float(
        pose["normalized_reprojection_error"]
    )

    margin_x = frame_width * MIN_FRAME_MARGIN_RATIO
    margin_y = frame_height * MIN_FRAME_MARGIN_RATIO

    if (
        x1 <= margin_x
        or y1 <= margin_y
        or x2 >= frame_width - 1 - margin_x
        or y2 >= frame_height - 1 - margin_y
    ):
        state = "FACE_NEAR_EDGE"
    elif face_area_ratio < MIN_FACE_AREA_RATIO:
        state = "FACE_TOO_SMALL"
    elif face_area_ratio > MAX_FACE_AREA_RATIO:
        state = "FACE_TOO_LARGE"
    elif (
        abs(face_center_x_ratio - 0.5)
        > MAX_FACE_CENTER_X_OFFSET
        or abs(face_center_y_ratio - 0.5)
        > MAX_FACE_CENTER_Y_OFFSET
    ):
        state = "FACE_NOT_CENTERED"
    elif normalized_error > MAX_NORMALIZED_REPROJECTION_ERROR:
        state = "LOW_POSE_RELIABILITY"
    else:
        state = "GOOD_FRAMING"

    return {
        "state": state,
        "good": state == "GOOD_FRAMING",
    }


def head_forward_state(
    pose: dict[str, object] | None,
    *,
    pitch_limit: float,
    yaw_limit: float,
    roll_limit: float,
) -> str:
    """Check whether a raw head pose is suitable for neutral calibration."""

    if pose is None:
        return "NO_HEAD_POSE"

    pitch = float(pose["pitch"])
    yaw = float(pose["yaw"])
    roll = float(pose["roll"])

    if abs(yaw) > yaw_limit:
        return "LOOK_FORWARD"

    if abs(roll) > roll_limit:
        return "KEEP_HEAD_LEVEL"

    if pitch > pitch_limit:
        return "LOOK_UP_SLIGHTLY"

    if pitch < -pitch_limit:
        return "LOOK_DOWN_SLIGHTLY"

    return "FORWARD_READY"


class HeadPoseNeutralCalibrator:
    """Collect and manually confirm a stable Canonical 14 neutral pose."""

    def __init__(
        self,
        *,
        required_frames: int,
        maximum_axis_range_degrees: float,
    ) -> None:
        self.required_frames = required_frames
        self.maximum_axis_range_degrees = (
            maximum_axis_range_degrees
        )
        self.reset(waiting=True)

    def reset(self, *, waiting: bool = True) -> None:
        self.state = "WAITING" if waiting else "COLLECTING"
        self.samples: deque[
            tuple[float, float, float]
        ] = deque(maxlen=self.required_frames)
        self.pending_neutral: (
            tuple[float, float, float] | None
        ) = None
        self.neutral: (
            tuple[float, float, float] | None
        ) = None

    def start(self) -> None:
        self.reset(waiting=False)

    @property
    def collecting(self) -> bool:
        return self.state == "COLLECTING"

    @property
    def ready_to_confirm(self) -> bool:
        return (
            self.state == "READY"
            and self.pending_neutral is not None
        )

    @property
    def confirmed(self) -> bool:
        return (
            self.state == "CONFIRMED"
            and self.neutral is not None
        )

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def pause_and_clear(self) -> None:
        if self.collecting:
            self.samples.clear()
            self.pending_neutral = None

    def consider(
        self,
        pose: dict[str, object],
    ) -> None:
        if not self.collecting:
            return

        self.samples.append(
            (
                float(pose["pitch"]),
                float(pose["yaw"]),
                float(pose["roll"]),
            )
        )

        if len(self.samples) < self.required_frames:
            return

        pitch_values = [
            sample[0]
            for sample in self.samples
        ]
        yaw_values = [
            sample[1]
            for sample in self.samples
        ]
        roll_values = [
            sample[2]
            for sample in self.samples
        ]

        stable = max(
            circular_range(pitch_values),
            circular_range(yaw_values),
            circular_range(roll_values),
        ) <= self.maximum_axis_range_degrees

        if not stable:
            self.samples.clear()
            self.pending_neutral = None
            return

        self.pending_neutral = (
            circular_median(pitch_values),
            circular_median(yaw_values),
            circular_median(roll_values),
        )
        self.state = "READY"

    def confirm(self) -> None:
        if not self.ready_to_confirm:
            raise RuntimeError(
                "No prepared head-pose baseline to confirm."
            )

        self.neutral = self.pending_neutral
        self.pending_neutral = None
        self.state = "CONFIRMED"

    def relative_pose(
        self,
        pose: dict[str, object] | None,
    ) -> tuple[float, float, float] | None:
        if pose is None or not self.confirmed:
            return None

        assert self.neutral is not None

        return (
            angle_diff_degrees(
                float(pose["pitch"]),
                self.neutral[0],
            ),
            angle_diff_degrees(
                float(pose["yaw"]),
                self.neutral[1],
            ),
            angle_diff_degrees(
                float(pose["roll"]),
                self.neutral[2],
            ),
        )


def classify_head_pose_state(
    relative_pose: tuple[float, float, float] | None,
    *,
    pitch_threshold: float,
    yaw_threshold: float,
    roll_threshold: float,
) -> str:
    """
    Convert relative head pose to a descriptive state.

    Yaw uses POSITIVE/NEGATIVE labels intentionally. The sign-to-left/right
    convention should be verified during this diagnostic before it is frozen.
    """

    if relative_pose is None:
        return "HEAD_POSE_NOT_READY"

    pitch, yaw, roll = relative_pose

    scores = {
        "HEAD_PITCH_POSITIVE": abs(pitch) / pitch_threshold
        if pitch >= 0
        else -1.0,
        "HEAD_PITCH_NEGATIVE": abs(pitch) / pitch_threshold
        if pitch < 0
        else -1.0,
        "HEAD_YAW_POSITIVE": abs(yaw) / yaw_threshold
        if yaw >= 0
        else -1.0,
        "HEAD_YAW_NEGATIVE": abs(yaw) / yaw_threshold
        if yaw < 0
        else -1.0,
        "HEAD_ROLL_POSITIVE": abs(roll) / roll_threshold
        if roll >= 0
        else -1.0,
        "HEAD_ROLL_NEGATIVE": abs(roll) / roll_threshold
        if roll < 0
        else -1.0,
    }

    best_state = max(scores, key=scores.get)

    if scores[best_state] < 1.0:
        return "HEAD_FORWARD"

    return best_state


def describe_head_gaze_relationship(
    *,
    head_state: str,
    gaze_status: str,
) -> str:
    """
    Create a cautious combined head-gaze interpretation.

    Current validated Canonical 14 live yaw convention:
    - HEAD_YAW_NEGATIVE: participant turns left
    - HEAD_YAW_POSITIVE: participant turns right

    L2CS gaze labels are head-relative rather than screen-coordinate labels.
    Therefore, opposite head/gaze directions can indicate possible
    compensation back toward the screen centre. These labels are diagnostic
    only and do not prove the exact visual target.
    """

    if head_state == "HEAD_POSE_NOT_READY":
        return "HEAD_GAZE_NOT_READY"

    if gaze_status in {
        "GAZE_UNRELIABLE",
        "GAZE_RESULT_STALE",
        "NO_L2CS_FACE",
        "L2CS_FACE_INPUT_AMBIGUOUS",
    }:
        return f"{head_state}__GAZE_UNAVAILABLE"

    if head_state == "HEAD_FORWARD":
        return f"HEAD_FORWARD__{gaze_status}"

    if head_state == "HEAD_YAW_NEGATIVE":
        if gaze_status == "LOOKING_LEFT":
            return "GAZE_FOLLOWS_HEAD_LEFT"
        if gaze_status == "LOOKING_RIGHT":
            return "POSSIBLE_SCREEN_CENTRE_COMPENSATION_LEFT_HEAD"
        if gaze_status == "FORWARD":
            return "HEAD_LEFT__GAZE_NEAR_HEAD_CENTRE"

    if head_state == "HEAD_YAW_POSITIVE":
        if gaze_status == "LOOKING_RIGHT":
            return "GAZE_FOLLOWS_HEAD_RIGHT"
        if gaze_status == "LOOKING_LEFT":
            return "POSSIBLE_SCREEN_CENTRE_COMPENSATION_RIGHT_HEAD"
        if gaze_status == "FORWARD":
            return "HEAD_RIGHT__GAZE_NEAR_HEAD_CENTRE"

    return f"{head_state}__{gaze_status}"


def combined_diagnostic_reliability(
    *,
    head_state: str,
    head_framing_good: bool,
    gaze_confidence: str,
) -> str:
    """Reduce confidence during head turns without discarding gaze output."""

    if not head_framing_good:
        return "UNRELIABLE"

    if gaze_confidence == "UNRELIABLE":
        return "UNRELIABLE"

    if head_state == "HEAD_FORWARD":
        return gaze_confidence

    # A stable L2CS direction under head rotation remains visible, but is
    # treated as low-confidence relationship evidence.
    return "LOW"



# ---------------------------------------------------------------------------
# Integration-ready output helpers
# ---------------------------------------------------------------------------

INVALID_GAZE_STATUSES = {
    "CALIBRATING",
    "GAZE_UNRELIABLE",
    "GAZE_RESULT_STALE",
    "NO_L2CS_FACE",
    "L2CS_FACE_INPUT_AMBIGUOUS",
}


def map_head_state_for_integration(head_state: str) -> str:
    """
    Convert internal Canonical 14 labels to user-facing integration labels.

    Verified convention for this project:
    - HEAD_YAW_NEGATIVE -> HEAD_LEFT
    - HEAD_YAW_POSITIVE -> HEAD_RIGHT
    - HEAD_PITCH_NEGATIVE -> HEAD_UP
    - HEAD_PITCH_POSITIVE -> HEAD_DOWN

    Roll labels are retained because roll-event interpretation has not yet
    been frozen.
    """

    mapping = {
        "HEAD_YAW_NEGATIVE": "HEAD_LEFT",
        "HEAD_YAW_POSITIVE": "HEAD_RIGHT",
        "HEAD_PITCH_NEGATIVE": "HEAD_UP",
        "HEAD_PITCH_POSITIVE": "HEAD_DOWN",
        "HEAD_FORWARD": "HEAD_FORWARD",
        "HEAD_POSE_NOT_READY": "HEAD_POSE_NOT_READY",
    }
    return mapping.get(head_state, head_state)


def map_relationship_for_integration(
    relationship: str,
) -> str:
    """Map technical head-state prefixes inside relationship labels."""

    if "__" not in relationship:
        return relationship

    head_part, gaze_part = relationship.split("__", 1)
    mapped_head = map_head_state_for_integration(head_part)
    return f"{mapped_head}__{gaze_part}"


def optional_round(
    value: float | None,
    digits: int = 2,
) -> float | None:
    """Round a finite numeric value, otherwise return None."""

    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric):
        return None

    return round(numeric, digits)


def build_head_gaze_frame_output(
    *,
    timestamp: float,
    internal_head_state: str,
    head_relative_pose: tuple[float, float, float] | None,
    head_pose_candidate: dict[str, object] | None,
    head_pose_accepted: bool,
    head_framing_state: str,
    head_framing_good: bool,
    head_temporal_state: str,
    head_rejection_reason: str,
    head_rotation_jump_degrees: float,
    final_gaze_status: str,
    final_gaze_confidence: str,
    eye_status: str,
    eye_reliability: str,
    stable_direction: str,
    instant_direction: str,
    direction_confirmation_count: int,
    smoothed_pitch_degrees: float,
    smoothed_yaw_degrees: float,
    l2cs_face_count: int,
    l2cs_result_age_seconds: float,
    head_gaze_relationship: str,
    combined_reliability: str,
) -> dict[str, Any]:
    """
    Build one standard integration-ready head-pose and gaze output.

    This function reorganizes existing results only. It does not alter the
    prediction, reliability, calibration, or temporal-guard logic.
    """

    user_head_label = map_head_state_for_integration(
        internal_head_state
    )

    pose_valid = bool(
        head_pose_accepted
        and head_framing_good
        and head_temporal_state == "POSE_ACCEPTED"
        and internal_head_state != "HEAD_POSE_NOT_READY"
    )

    gaze_valid = bool(
        final_gaze_status not in INVALID_GAZE_STATUSES
        and final_gaze_confidence in {"HIGH", "LOW"}
        and l2cs_face_count == 1
    )

    if head_relative_pose is None:
        relative_pitch = None
        relative_yaw = None
        relative_roll = None
    else:
        relative_pitch, relative_yaw, relative_roll = head_relative_pose

    if head_pose_candidate is None:
        raw_pitch = None
        raw_yaw = None
        raw_roll = None
    else:
        raw_pitch = head_pose_candidate.get("pitch")
        raw_yaw = head_pose_candidate.get("yaw")
        raw_roll = head_pose_candidate.get("roll")

    return {
        "timestamp": round(float(timestamp), 3),
        "head_pose": {
            "label": user_head_label,
            "internal_label": internal_head_state,
            "confidence": "HIGH" if pose_valid else "UNRELIABLE",
            "is_valid": pose_valid,
            "relative_pitch_deg": optional_round(relative_pitch),
            "relative_yaw_deg": optional_round(relative_yaw),
            "relative_roll_deg": optional_round(relative_roll),
            "raw_pitch_deg": optional_round(raw_pitch),
            "raw_yaw_deg": optional_round(raw_yaw),
            "raw_roll_deg": optional_round(raw_roll),
            "framing_state": head_framing_state,
            "temporal_state": head_temporal_state,
            "rejection_reason": head_rejection_reason or "",
            "rotation_jump_deg": optional_round(
                head_rotation_jump_degrees
            ),
        },
        "gaze": {
            "label": final_gaze_status,
            "confidence": final_gaze_confidence,
            "is_valid": gaze_valid,
            "eye_status": eye_status,
            "eye_reliability": eye_reliability,
            "instant_direction": instant_direction,
            "stable_direction": stable_direction,
            "direction_confirmation_count": int(
                direction_confirmation_count
            ),
            "smoothed_pitch_deg": optional_round(
                smoothed_pitch_degrees
            ),
            "smoothed_yaw_deg": optional_round(
                smoothed_yaw_degrees
            ),
            "l2cs_face_count": int(l2cs_face_count),
            "result_age_seconds": optional_round(
                l2cs_result_age_seconds,
                digits=3,
            ),
            "relationship": map_relationship_for_integration(
                head_gaze_relationship
            ),
        },
        "combined": {
            "reliability": combined_reliability,
            "frame_is_loggable": True,
            "head_can_trigger_event": pose_valid,
            "gaze_output_available": gaze_valid,
            "gaze_deviation_event_eligible": bool(
                pose_valid
                and user_head_label == "HEAD_FORWARD"
                and gaze_valid
                and eye_status == "EYES_OPEN"
                and final_gaze_status
                in {
                    "LOOKING_LEFT",
                    "LOOKING_RIGHT",
                    "LOOKING_UP",
                    "LOOKING_DOWN",
                }
            ),
        },
    }



def create_face_landmarker(model_path: Path):
    """Create MediaPipe Face Landmarker in VIDEO mode."""

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )


def landmark_to_pixel(
    landmark,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Convert a normalized landmark to pixel coordinates."""

    return np.array(
        [
            float(landmark.x) * frame_width,
            float(landmark.y) * frame_height,
        ],
        dtype=np.float64,
    )


def extract_eye_points(
    landmarks,
    indices: tuple[int, int, int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> list[np.ndarray]:
    """Extract the six eye points used by EAR."""

    return [
        landmark_to_pixel(
            landmarks[index],
            frame_width,
            frame_height,
        )
        for index in indices
    ]


def calculate_ear(points: list[np.ndarray]) -> float:
    """Calculate Eye Aspect Ratio."""

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal <= 1e-6:
        return 0.0

    return float(
        (vertical_1 + vertical_2)
        / (2.0 * horizontal)
    )


def blendshape_scores_to_dict(categories) -> dict[str, float]:
    """Convert blendshape categories to a dictionary."""

    return {
        category.category_name: float(category.score)
        for category in categories
        if category.category_name
    }


def draw_eye_points(
    frame: np.ndarray,
    points: list[np.ndarray],
) -> None:
    """Draw selected eye landmarks."""

    for point in points:
        x, y = point.astype(int)

        cv2.circle(
            frame,
            (x, y),
            2,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def draw_l2cs_result(
    *,
    frame: np.ndarray,
    result: L2CSResult,
    arrow_color: tuple[int, int, int] | None,
    box_color: tuple[int, int, int],
) -> None:
    """Draw the latest L2CS bounding boxes and optional gaze arrows."""

    if (
        result.bboxes is None
        or result.pitch is None
        or result.yaw is None
    ):
        return

    face_count = len(result.bboxes)

    for index in range(face_count):
        bbox = np.asarray(result.bboxes[index]).astype(int)
        x_min, y_min, x_max, y_max = bbox[:4]

        cv2.rectangle(
            frame,
            (int(x_min), int(y_min)),
            (int(x_max), int(y_max)),
            box_color,
            2,
            cv2.LINE_AA,
        )

        if arrow_color is not None:
            width = int(x_max - x_min)
            height = int(y_max - y_min)

            draw_gaze(
                int(x_min),
                int(y_min),
                width,
                height,
                frame,
                (
                    float(result.pitch[index]),
                    float(result.yaw[index]),
                ),
                color=arrow_color,
            )





# ---------------------------------------------------------------------------
# Adapter configuration
# ---------------------------------------------------------------------------

@dataclass
class HeadGazeConfig:
    """Runtime configuration for the reusable integration adapter."""

    device: str = "cpu"
    architecture: str = "ResNet50"
    mirror_input: bool = False

    calibration_samples: int = 45
    calibration_duration: float = 2.0
    strong_closed_ratio: float = 0.15
    ambiguous_ratio: float = 0.35
    one_eye_ratio: float = 0.65
    blink_threshold: float = 0.55
    open_blink_threshold: float = 0.35
    closed_duration: float = 0.35
    one_eye_duration: float = 0.18
    minimum_blink_duration: float = 0.03
    maximum_normalized_ear: float = 1.80
    one_eye_min_open_ratio: float = 0.55
    one_eye_min_difference: float = 0.25
    one_eye_max_relative_ratio: float = 0.65

    gaze_calibration_samples: int = 8
    gaze_calibration_duration: float = 1.8
    gaze_calibration_stability_window: int = 5
    gaze_calibration_max_std: float = 4.0
    gaze_calibration_max_abs_angle: float = 35.0

    smoothing_alpha: float = 0.45
    horizontal_threshold: float = 12.0
    up_threshold: float = 12.0
    down_threshold: float = 15.0
    direction_confirmations: int = 2
    one_eye_direction_confirmations: int = 3

    head_pitch_threshold: float = DEFAULT_HEAD_PITCH_THRESHOLD_DEG
    head_yaw_threshold: float = DEFAULT_HEAD_YAW_THRESHOLD_DEG
    head_roll_threshold: float = DEFAULT_HEAD_ROLL_THRESHOLD_DEG
    head_rotation_jump_threshold: float = DEFAULT_HEAD_ROTATION_JUMP_THRESHOLD_DEG
    head_relative_roll_flip_threshold: float = DEFAULT_HEAD_RELATIVE_ROLL_FLIP_THRESHOLD_DEG
    maximum_gaze_age: float = 0.75


# ---------------------------------------------------------------------------
# Reusable adapter
# ---------------------------------------------------------------------------

class HeadGazeAdapter:
    """Process frames and expose standardized head/gaze integration output."""

    def __init__(
        self,
        *,
        l2cs_root: Path,
        l2cs_snapshot: Path,
        mediapipe_model: Path,
        canonical14_csv: Path,
        config: HeadGazeConfig | None = None,
    ) -> None:
        self.l2cs_root = Path(l2cs_root).expanduser().resolve()
        self.l2cs_snapshot = Path(l2cs_snapshot).expanduser().resolve()
        self.mediapipe_model = Path(mediapipe_model).expanduser().resolve()
        self.canonical14_csv = Path(canonical14_csv).expanduser().resolve()
        self.config = config or HeadGazeConfig()

        for path, label, expect_dir in (
            (self.l2cs_root, "L2CS-Net repository", True),
            (self.l2cs_snapshot, "L2CS checkpoint", False),
            (self.mediapipe_model, "MediaPipe model", False),
            (self.canonical14_csv, "Canonical 14 CSV", False),
        ):
            exists = path.is_dir() if expect_dir else path.is_file()
            if not exists:
                raise FileNotFoundError(f"{label} not found: {path}")

        self.head_landmark_ids, self.head_model_points = (
            read_canonical14_subset(self.canonical14_csv)
        )

        self._started = False
        self._pipeline = None
        self._landmarker = None
        self._exchange = None
        self._stop_event = None
        self._worker = None
        self._session_start = 0.0
        self._frame_id = 0
        self._last_mp_timestamp_ms = -1
        self._last_consumed_l2cs_frame_id = -1
        self._create_state_objects()

    def _create_state_objects(self) -> None:
        cfg = self.config
        self.eye_calibrator = PassiveEyeCalibrator(
            minimum_samples=cfg.calibration_samples,
            minimum_duration=cfg.calibration_duration,
        )
        self.gaze_calibrator = PassiveGazeCenterCalibrator(
            minimum_samples=cfg.gaze_calibration_samples,
            minimum_duration=cfg.gaze_calibration_duration,
        )
        self.gaze_calibration_stability = GazeCalibrationStabilityGate(
            window_size=cfg.gaze_calibration_stability_window,
            maximum_std_degrees=cfg.gaze_calibration_max_std,
        )
        self.head_calibrator = HeadPoseNeutralCalibrator(
            required_frames=HEAD_CALIBRATION_FRAMES,
            maximum_axis_range_degrees=HEAD_CALIBRATION_MAX_AXIS_RANGE_DEG,
        )
        self.head_pose_guard = HeadPoseTemporalGuard(
            maximum_rotation_jump_degrees=cfg.head_rotation_jump_threshold,
            maximum_relative_roll_degrees=cfg.head_relative_roll_flip_threshold,
        )
        self.gaze_smoother = ExponentialAngleSmoother(alpha=cfg.smoothing_alpha)
        self.direction_stabilizer = DirectionStabilizer(
            confirmations=cfg.direction_confirmations
        )
        self.eye_tracker = EyeStateTracker()
        self.eye_calibration_was_ready = False
        self.corrected_pitch_deg = float("nan")
        self.corrected_yaw_deg = float("nan")
        self.smoothed_pitch_deg = float("nan")
        self.smoothed_yaw_deg = float("nan")
        self.instant_direction = "CALIBRATING"
        self.stable_direction = "CALIBRATING"
        self.direction_confirmation_count = 0

    def _load_l2cs_symbols(self):
        root = str(self.l2cs_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module("l2cs")
        return module.Pipeline, module.select_device

    def start(self) -> None:
        """Load models and start the asynchronous L2CS worker."""
        if self._started:
            return

        pipeline_class, select_device = self._load_l2cs_symbols()
        self._pipeline = pipeline_class(
            weights=self.l2cs_snapshot,
            arch=self.config.architecture,
            device=select_device(self.config.device, batch_size=1),
        )
        self._landmarker = create_face_landmarker(self.mediapipe_model)
        self._exchange = LatestFrameExchange()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=l2cs_worker,
            kwargs={
                "pipeline": self._pipeline,
                "exchange": self._exchange,
                "stop_event": self._stop_event,
            },
            name="L2CSWorker",
            daemon=True,
        )
        self._worker.start()
        self._session_start = time.perf_counter()
        self._started = True

    def close(self) -> None:
        """Release model resources and stop the worker."""
        if not self._started:
            return
        assert self._stop_event is not None
        assert self._exchange is not None
        self._stop_event.set()
        self._exchange.new_frame_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)
        if self._landmarker is not None:
            self._landmarker.close()
        self._started = False

    def reset_calibration(self) -> None:
        """Restart eye, gaze and neutral-head calibration."""
        self._create_state_objects()

    def confirm_head_baseline(self) -> bool:
        """Confirm a prepared neutral head pose."""
        if not self.head_calibrator.ready_to_confirm:
            return False
        self.head_calibrator.confirm()
        return True

    def calibration_status(self) -> dict[str, Any]:
        if not self.eye_calibrator.ready:
            phase = "EYE_CALIBRATION"
        elif not self.gaze_calibrator.ready:
            phase = "GAZE_CALIBRATION"
        elif not self.head_calibrator.confirmed:
            phase = (
                "HEAD_POSE_CONFIRMATION"
                if self.head_calibrator.ready_to_confirm
                else "HEAD_POSE_CALIBRATION"
            )
        else:
            phase = "READY"

        return {
            "phase": phase,
            "ready": phase == "READY",
            "eye_samples": self.eye_calibrator.sample_count,
            "gaze_samples": self.gaze_calibrator.sample_count,
            "head_samples": self.head_calibrator.sample_count,
            "head_confirmation_required": self.head_calibrator.ready_to_confirm,
        }

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """Process one BGR frame and return the standardized output."""
        if not self._started:
            raise RuntimeError("Call start() before process_frame().")
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty frame.")

        assert self._landmarker is not None
        assert self._exchange is not None
        cfg = self.config

        inference_frame = cv2.flip(frame, 1) if cfg.mirror_input else frame
        self._frame_id += 1
        captured_time = time.perf_counter()
        self._exchange.submit(self._frame_id, captured_time, inference_frame)

        frame_height, frame_width = inference_frame.shape[:2]
        rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )

        mp_timestamp_ms = int((captured_time - self._session_start) * 1000)
        if mp_timestamp_ms <= self._last_mp_timestamp_ms:
            mp_timestamp_ms = self._last_mp_timestamp_ms + 1
        self._last_mp_timestamp_ms = mp_timestamp_ms
        mp_result = self._landmarker.detect_for_video(mp_image, mp_timestamp_ms)

        mediapipe_face_count = len(mp_result.face_landmarks)
        head_pose_candidate = None
        head_pose = None
        head_temporal_state = "NO_POSE_CANDIDATE"
        head_rejection_reason = ""
        head_rotation_jump_degrees = float("nan")
        head_framing = evaluate_head_framing(
            mediapipe_face_count, None, frame_width, frame_height
        )
        head_forward_readiness = "NO_HEAD_POSE"
        left_ear = right_ear = float("nan")
        left_blink = right_blink = float("nan")

        eye_decision = EyeDecision(
            status="CALIBRATING" if not self.eye_calibrator.ready else "NO_FACE",
            reliability="UNRELIABLE",
            duration=0.0,
            left_closed_signal=False,
            right_closed_signal=False,
            blink_count=self.eye_tracker.blink_count,
        )

        if mp_result.face_landmarks and mp_result.face_blendshapes:
            selected = largest_face_index(
                mp_result.face_landmarks, frame_width, frame_height
            )
            landmarks = mp_result.face_landmarks[selected]
            rotation_guess, translation_guess = self.head_pose_guard.initial_guess()
            head_pose_candidate = estimate_canonical14_pose(
                landmarks,
                frame_width,
                frame_height,
                self.head_landmark_ids,
                self.head_model_points,
                initial_rotation_vector=rotation_guess,
                initial_translation_vector=translation_guess,
            )
            candidate_framing = evaluate_head_framing(
                mediapipe_face_count,
                head_pose_candidate,
                frame_width,
                frame_height,
            )
            if bool(candidate_framing["good"]):
                (
                    head_pose,
                    head_temporal_state,
                    head_rejection_reason,
                    head_rotation_jump_degrees,
                ) = self.head_pose_guard.evaluate(
                    head_pose_candidate, neutral_pose=self.head_calibrator.neutral
                )
                head_framing = (
                    candidate_framing
                    if head_pose is not None
                    else {"state": "POSE_FLIP_REJECTED", "good": False}
                )
            else:
                head_framing = candidate_framing
                head_temporal_state = "POSE_NOT_TEMPORALLY_EVALUATED"

            head_forward_readiness = head_forward_state(
                head_pose,
                pitch_limit=DEFAULT_HEAD_FORWARD_MAX_ABS_PITCH_DEG,
                yaw_limit=DEFAULT_HEAD_FORWARD_MAX_ABS_YAW_DEG,
                roll_limit=DEFAULT_HEAD_FORWARD_MAX_ABS_ROLL_DEG,
            )

            left_points = extract_eye_points(
                landmarks, LEFT_EYE_INDICES, frame_width, frame_height
            )
            right_points = extract_eye_points(
                landmarks, RIGHT_EYE_INDICES, frame_width, frame_height
            )
            left_ear = calculate_ear(left_points)
            right_ear = calculate_ear(right_points)
            scores = blendshape_scores_to_dict(mp_result.face_blendshapes[selected])
            left_blink = scores.get(LEFT_BLINK_NAME, float("nan"))
            right_blink = scores.get(RIGHT_BLINK_NAME, float("nan"))

            self.eye_calibrator.consider_sample(
                left_ear=left_ear,
                right_ear=right_ear,
                left_blink=left_blink,
                right_blink=right_blink,
                maximum_blink_score=0.25,
                minimum_raw_ear=0.15,
                maximum_eye_ratio=1.55,
            )

            if self.eye_calibrator.ready:
                eye_decision = self.eye_tracker.update(
                    now=time.perf_counter(),
                    left_normalized_ear=left_ear / self.eye_calibrator.left_baseline,
                    right_normalized_ear=right_ear / self.eye_calibrator.right_baseline,
                    left_blink=left_blink,
                    right_blink=right_blink,
                    strong_closed_ratio=cfg.strong_closed_ratio,
                    ambiguous_ratio=cfg.ambiguous_ratio,
                    one_eye_ratio=cfg.one_eye_ratio,
                    blink_threshold=cfg.blink_threshold,
                    open_blink_threshold=cfg.open_blink_threshold,
                    closed_duration=cfg.closed_duration,
                    one_eye_duration=cfg.one_eye_duration,
                    minimum_blink_duration=cfg.minimum_blink_duration,
                    maximum_normalized_ear=cfg.maximum_normalized_ear,
                    one_eye_min_open_ratio=cfg.one_eye_min_open_ratio,
                    one_eye_min_difference=cfg.one_eye_min_difference,
                    one_eye_max_relative_ratio=cfg.one_eye_max_relative_ratio,
                )
        else:
            self.eye_tracker.reset_temporal_state()

        if self.eye_calibrator.ready and not self.eye_calibration_was_ready:
            self.gaze_calibrator.start()
            self.gaze_calibration_stability.reset()
        self.eye_calibration_was_ready = self.eye_calibrator.ready

        l2cs_result = self._exchange.read_result()
        l2cs_face_count = 0 if l2cs_result.bboxes is None else len(l2cs_result.bboxes)
        raw_pitch_deg = raw_yaw_deg = float("nan")
        if (
            l2cs_result.pitch is not None
            and l2cs_result.yaw is not None
            and len(l2cs_result.pitch) >= 1
            and len(l2cs_result.yaw) >= 1
        ):
            raw_pitch_deg = float(np.degrees(l2cs_result.pitch[0]))
            raw_yaw_deg = float(np.degrees(l2cs_result.yaw[0]))

        l2cs_result_age = (
            float("inf")
            if l2cs_result.completed_time <= 0
            else time.perf_counter() - l2cs_result.completed_time
        )
        l2cs_fresh = (
            l2cs_result.error is None
            and l2cs_result_age <= cfg.maximum_gaze_age
        )
        new_l2cs_result = (
            l2cs_result.frame_id >= 0
            and l2cs_result.frame_id != self._last_consumed_l2cs_frame_id
        )

        if new_l2cs_result:
            self._last_consumed_l2cs_frame_id = l2cs_result.frame_id

            if self.gaze_calibrator.collecting:
                quality_ok = (
                    l2cs_fresh
                    and l2cs_face_count == 1
                    and eye_decision.status == "EYES_OPEN"
                    and np.isfinite(raw_pitch_deg)
                    and np.isfinite(raw_yaw_deg)
                    and abs(raw_pitch_deg) <= cfg.gaze_calibration_max_abs_angle
                    and abs(raw_yaw_deg) <= cfg.gaze_calibration_max_abs_angle
                )
                if quality_ok:
                    stable = self.gaze_calibration_stability.update(
                        pitch_degrees=raw_pitch_deg,
                        yaw_degrees=raw_yaw_deg,
                    )
                    if stable:
                        self.gaze_calibrator.consider_sample(
                            pitch_degrees=raw_pitch_deg,
                            yaw_degrees=raw_yaw_deg,
                        )
                else:
                    self.gaze_calibration_stability.reset()

            if (
                self.eye_calibrator.ready
                and self.gaze_calibrator.ready
                and l2cs_fresh
                and l2cs_face_count == 1
                and np.isfinite(raw_pitch_deg)
                and np.isfinite(raw_yaw_deg)
            ):
                self.corrected_pitch_deg = (
                    raw_pitch_deg - self.gaze_calibrator.pitch_baseline
                )
                self.corrected_yaw_deg = (
                    raw_yaw_deg - self.gaze_calibrator.yaw_baseline
                )
                (
                    self.smoothed_pitch_deg,
                    self.smoothed_yaw_deg,
                ) = self.gaze_smoother.update(
                    self.corrected_pitch_deg, self.corrected_yaw_deg
                )
                if eye_decision.status in {
                    "EYES_OPEN",
                    "ONE_EYE_UNAVAILABLE",
                    "PROFILE_OR_LANDMARK_UNCERTAIN",
                    "EYE_STATUS_UNCERTAIN",
                }:
                    self.instant_direction = classify_coarse_direction(
                        horizontal_degrees=self.smoothed_pitch_deg,
                        vertical_degrees=self.smoothed_yaw_deg,
                        horizontal_threshold=cfg.horizontal_threshold,
                        up_threshold=cfg.up_threshold,
                        down_threshold=cfg.down_threshold,
                    )
                    (
                        self.stable_direction,
                        self.direction_confirmation_count,
                    ) = self.direction_stabilizer.update(self.instant_direction)

        if self.gaze_calibrator.ready and self.head_calibrator.state == "WAITING":
            self.head_pose_guard.reset()
            self.head_calibrator.start()

        if self.head_calibrator.collecting:
            if (
                bool(head_framing["good"])
                and head_forward_readiness == "FORWARD_READY"
                and head_pose is not None
            ):
                self.head_calibrator.consider(head_pose)
            else:
                self.head_calibrator.pause_and_clear()

        head_relative_pose = self.head_calibrator.relative_pose(head_pose)
        head_pose_state = classify_head_pose_state(
            head_relative_pose,
            pitch_threshold=cfg.head_pitch_threshold,
            yaw_threshold=cfg.head_yaw_threshold,
            roll_threshold=cfg.head_roll_threshold,
        )
        calibration = self.calibration_status()
        ready = bool(calibration["ready"])

        if not ready:
            final_gaze_status, final_confidence = "CALIBRATING", "UNRELIABLE"
        elif not l2cs_fresh:
            final_gaze_status, final_confidence = "GAZE_RESULT_STALE", "UNRELIABLE"
        elif l2cs_face_count == 0:
            final_gaze_status, final_confidence = "NO_L2CS_FACE", "UNRELIABLE"
        elif l2cs_face_count > 1:
            final_gaze_status, final_confidence = (
                "L2CS_FACE_INPUT_AMBIGUOUS", "UNRELIABLE"
            )
        elif eye_decision.status == "EYES_OPEN":
            final_gaze_status, final_confidence = self.stable_direction, "HIGH"
        elif (
            eye_decision.status == "ONE_EYE_UNAVAILABLE"
            and head_pose_state == "HEAD_FORWARD"
            and self.direction_confirmation_count >= cfg.one_eye_direction_confirmations
            and self.stable_direction in {
                "FORWARD", "LOOKING_LEFT", "LOOKING_RIGHT",
                "LOOKING_UP", "LOOKING_DOWN",
            }
        ):
            final_gaze_status, final_confidence = self.stable_direction, "LOW"
        elif (
            eye_decision.status == "PROFILE_OR_LANDMARK_UNCERTAIN"
            and self.direction_confirmation_count >= cfg.direction_confirmations
            and self.stable_direction in {
                "FORWARD", "LOOKING_LEFT", "LOOKING_RIGHT",
                "LOOKING_UP", "LOOKING_DOWN",
            }
        ):
            final_gaze_status, final_confidence = self.stable_direction, "LOW"
        elif (
            eye_decision.status == "EYE_STATUS_UNCERTAIN"
            and self.stable_direction == "LOOKING_DOWN"
            and self.direction_confirmation_count >= max(
                cfg.direction_confirmations, cfg.one_eye_direction_confirmations
            )
            and np.isfinite(self.smoothed_yaw_deg)
            and self.smoothed_yaw_deg <= -cfg.down_threshold
        ):
            final_gaze_status, final_confidence = "LOOKING_DOWN", "LOW"
        elif (
            eye_decision.status == "EYE_STATUS_UNCERTAIN"
            and np.isfinite(self.smoothed_yaw_deg)
            and self.smoothed_yaw_deg <= -cfg.down_threshold
        ):
            final_gaze_status, final_confidence = "POSSIBLE_LOOKING_DOWN", "LOW"
        else:
            final_gaze_status, final_confidence = "GAZE_UNRELIABLE", "UNRELIABLE"

        relationship = describe_head_gaze_relationship(
            head_state=head_pose_state, gaze_status=final_gaze_status
        )
        reliability = combined_diagnostic_reliability(
            head_state=head_pose_state,
            head_framing_good=bool(head_framing["good"]),
            gaze_confidence=final_confidence,
        )

        output = build_head_gaze_frame_output(
            timestamp=captured_time - self._session_start,
            internal_head_state=head_pose_state,
            head_relative_pose=head_relative_pose,
            head_pose_candidate=head_pose_candidate,
            head_pose_accepted=head_pose is not None,
            head_framing_state=str(head_framing["state"]),
            head_framing_good=bool(head_framing["good"]),
            head_temporal_state=head_temporal_state,
            head_rejection_reason=head_rejection_reason,
            head_rotation_jump_degrees=head_rotation_jump_degrees,
            final_gaze_status=final_gaze_status,
            final_gaze_confidence=final_confidence,
            eye_status=eye_decision.status,
            eye_reliability=eye_decision.reliability,
            stable_direction=self.stable_direction,
            instant_direction=self.instant_direction,
            direction_confirmation_count=self.direction_confirmation_count,
            smoothed_pitch_degrees=self.smoothed_pitch_deg,
            smoothed_yaw_degrees=self.smoothed_yaw_deg,
            l2cs_face_count=l2cs_face_count,
            l2cs_result_age_seconds=l2cs_result_age,
            head_gaze_relationship=relationship,
            combined_reliability=reliability,
        )
        output["calibration"] = calibration
        return output


# ---------------------------------------------------------------------------
# Optional smoke test
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the reusable head/gaze adapter."
    )
    parser.add_argument("--l2cs-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--mediapipe-model", type=Path, required=True)
    parser.add_argument("--canonical14-csv", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path(
            "outputs/integration/head_gaze_adapter_test/screenshots"
        ),
        help=(
            "Directory used by the S-key screenshot feature. "
            "Default: outputs/integration/head_gaze_adapter_test/screenshots"
        ),
    )
    return parser.parse_args()


def save_smoke_test_screenshot(
    frame: np.ndarray,
    screenshot_dir: Path,
) -> Path:
    """Save the current smoke-test display frame and return its path."""

    screenshot_dir = Path(screenshot_dir).expanduser().resolve()
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = screenshot_dir / f"head_gaze_adapter_{timestamp}.jpg"

    if not cv2.imwrite(str(path), frame):
        raise IOError(f"Could not save screenshot: {path}")

    return path


def run_smoke_test() -> None:
    """Minimal webcam test: C confirm, R recalibrate, S screenshot, Q quit."""
    args = _parse_args()
    adapter = HeadGazeAdapter(
        l2cs_root=args.l2cs_root,
        l2cs_snapshot=args.snapshot,
        mediapipe_model=args.mediapipe_model,
        canonical14_csv=args.canonical14_csv,
        config=HeadGazeConfig(device=args.device, mirror_input=args.mirror),
    )
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise IOError(f"Cannot open webcam with camera ID {args.cam}.")

    adapter.start()
    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            output = adapter.process_frame(frame)
            display = cv2.flip(frame, 1) if args.mirror else frame.copy()

            lines = [
                f"Calibration: {output['calibration']['phase']}",
                f"Head: {output['head_pose']['label']} ({output['head_pose']['confidence']})",
                f"Gaze: {output['gaze']['label']} ({output['gaze']['confidence']})",
                f"Eye: {output['gaze']['eye_status']}",
                f"Relationship: {output['gaze']['relationship']}",
                "C confirm head baseline | R recalibrate | S screenshot | Q quit",
            ]
            for i, text in enumerate(lines):
                cv2.putText(
                    display, text, (20, 35 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255),
                    2, cv2.LINE_AA,
                )

            now = time.perf_counter()
            if now - last_print >= 1.0:
                print("[HEAD_GAZE_OUTPUT]", output)
                last_print = now

            cv2.imshow("Head Gaze Adapter Smoke Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                print(
                    "Head baseline confirmed."
                    if adapter.confirm_head_baseline()
                    else "Head baseline is not ready to confirm."
                )
            elif key == ord("r"):
                adapter.reset_calibration()
                print("Calibration reset.")
            elif key == ord("s"):
                try:
                    path = save_smoke_test_screenshot(
                        display,
                        args.screenshot_dir,
                    )
                    print(f"Screenshot saved: {path}")
                except IOError as error:
                    print(f"Screenshot error: {error}")
            elif key == ord("q"):
                break
    finally:
        adapter.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_smoke_test()
