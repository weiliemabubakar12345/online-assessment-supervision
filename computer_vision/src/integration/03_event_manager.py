# UPDATED — Week 8 Day 3 selected audio-device integration configuration
# audio_device event enabled by default; event eligibility threshold = 0.43
# temporal rule = 0.75 s min / 0.60 s release / 0.50 s cooldown

"""03_event_manager.py

Temporal event manager for the TEEP visual-cue integration pipeline.

Inputs:
- normalized head/gaze output from 01_head_gaze_adapter.py
- normalized YOLO output from 02_yolo_output_adapter.py

Responsibilities:
- extract neutral cue candidates;
- preserve the no-double-counting head/gaze policy;
- apply configurable temporal confirmation;
- support fresh/stale multi-rate YOLO observations without double-counting;
- emit START / ACTIVE / END lifecycle records;
- expose simultaneous active cues as multi-cue context.

This module does NOT:
- decide that a student is cheating;
- assign a cheating score;
- write CSV files;
- modify either adapter;
- hardcode an audio-device threshold/duration rule.

Final research thresholds are not frozen yet, so no event rules are enabled
unless the caller explicitly supplies them. The built-in synthetic smoke test
uses DEMO-ONLY thresholds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional


NON_FORWARD_HEAD_LABELS = {
    "HEAD_LEFT", "HEAD_RIGHT", "HEAD_UP", "HEAD_DOWN"
}

GAZE_DEVIATION_LABELS = {
    "LOOKING_LEFT", "LOOKING_RIGHT", "LOOKING_UP", "LOOKING_DOWN"
}

OBJECT_CUE_LABELS = {
    "phone", "computer_device", "book_notes",
    "calculator", "watch", "audio_device"
}

PERSON_STATES = {"NO_PERSON", "ONE_PERSON", "MULTIPLE_PERSONS"}


@dataclass(frozen=True)
class EventRule:
    min_duration_s: float
    release_duration_s: float = 0.25
    cooldown_s: float = 0.0
    enabled: bool = True

    def validate(self) -> None:
        if self.min_duration_s < 0:
            raise ValueError("min_duration_s must be >= 0")
        if self.release_duration_s < 0:
            raise ValueError("release_duration_s must be >= 0")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")


@dataclass(frozen=True)
class ObjectEventEligibilityRule:
    """
    Extra event-eligibility gate for normalized YOLO object cues.

    This does NOT change 02's raw YOLO detections. It only decides whether a
    raw object cue is allowed to enter 03's temporal event tracker.
    """
    min_confidence: float = 0.25
    enabled: bool = True

    def validate(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1.")


@dataclass
class _CueTracker:
    first_seen_ts: Optional[float] = None
    last_seen_ts: Optional[float] = None
    absent_since_ts: Optional[float] = None
    active_since_ts: Optional[float] = None
    last_end_ts: Optional[float] = None
    is_active: bool = False
    latest_candidate: Optional[Dict[str, Any]] = None

    def reset_presence(self) -> None:
        self.first_seen_ts = None
        self.last_seen_ts = None
        self.absent_since_ts = None
        self.latest_candidate = None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _timestamp(
    head_gaze_output: Optional[Mapping[str, Any]],
    yolo_output: Optional[Mapping[str, Any]],
    timestamp: Optional[float],
) -> float:
    if timestamp is not None:
        return float(timestamp)
    if head_gaze_output is not None and "timestamp" in head_gaze_output:
        return float(head_gaze_output["timestamp"])
    if yolo_output is not None and "timestamp" in yolo_output:
        return float(yolo_output["timestamp"])
    return float(time.monotonic())


def extract_cue_candidates(
    head_gaze_output: Optional[Mapping[str, Any]] = None,
    yolo_output: Optional[Mapping[str, Any]] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Extract neutral cue candidates from the two frozen adapters."""

    ts = _timestamp(head_gaze_output, yolo_output, timestamp)
    candidates: Dict[str, Dict[str, Any]] = {}

    if head_gaze_output is not None:
        head = _mapping(head_gaze_output.get("head_pose"))
        gaze = _mapping(head_gaze_output.get("gaze"))
        combined = _mapping(head_gaze_output.get("combined"))

        head_label = str(head.get("label", ""))
        if (
            bool(head.get("is_valid", False))
            and bool(combined.get("head_can_trigger_event", False))
            and head_label in NON_FORWARD_HEAD_LABELS
        ):
            key = f"head:{head_label}"
            candidates[key] = {
                "cue_key": key,
                "cue_type": "head_pose",
                "label": head_label,
                "timestamp": ts,
                "confidence": head.get("confidence"),
                "relationship": gaze.get("relationship"),
                "source": "head_gaze_adapter",
            }

        gaze_label = str(gaze.get("label", ""))
        if (
            bool(gaze.get("is_valid", False))
            and bool(combined.get("gaze_deviation_event_eligible", False))
            and gaze_label in GAZE_DEVIATION_LABELS
        ):
            key = f"gaze:{gaze_label}"
            candidates[key] = {
                "cue_key": key,
                "cue_type": "gaze",
                "label": gaze_label,
                "timestamp": ts,
                "confidence": gaze.get("confidence"),
                "eye_status": gaze.get("eye_status"),
                "relationship": gaze.get("relationship"),
                "source": "head_gaze_adapter",
            }

    if yolo_output is not None:
        detections = yolo_output.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            label = str(detection.get("label", ""))
            if label in OBJECT_CUE_LABELS:
                grouped.setdefault(label, []).append(dict(detection))

        for label, items in grouped.items():
            confidences = [float(x.get("confidence", 0.0)) for x in items]
            key = f"object:{label}"
            candidates[key] = {
                "cue_key": key,
                "cue_type": "object",
                "label": label,
                "timestamp": ts,
                "count": len(items),
                "max_confidence": max(confidences) if confidences else 0.0,
                "detections": items,
                "source": "yolo_output_adapter",
            }

        person_state = str(yolo_output.get("person_state", "ONE_PERSON"))
        if person_state not in PERSON_STATES:
            raise ValueError(f"Unknown person_state={person_state!r}")

        if person_state in {"NO_PERSON", "MULTIPLE_PERSONS"}:
            key = f"person:{person_state}"
            candidates[key] = {
                "cue_key": key,
                "cue_type": "person_state",
                "label": person_state,
                "timestamp": ts,
                "person_count": int(yolo_output.get("person_count", 0)),
                "source": "yolo_output_adapter",
            }

    json.dumps(candidates)
    return candidates


def infer_head_gaze_face_evidence(
    head_gaze_output: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """
    Infer whether 01 provides CURRENT MediaPipe-based evidence that a face
    is present.

    Strong evidence is intentionally conservative:
    - a valid head-pose output, OR
    - an eye-status produced from current facial landmarks that is NOT
      NO_FACE / CALIBRATING / empty.

    L2CS face count is recorded for diagnostics but is NOT used as the
    deciding signal because an asynchronous/stale gaze result can outlive
    the current face observation.
    """
    if not isinstance(head_gaze_output, Mapping):
        return {
            "present": False,
            "head_pose_valid": False,
            "head_label": "",
            "eye_status": "",
            "l2cs_face_count": 0,
            "basis": [],
        }

    head = head_gaze_output.get("head_pose", {})
    gaze = head_gaze_output.get("gaze", {})

    if not isinstance(head, Mapping):
        head = {}
    if not isinstance(gaze, Mapping):
        gaze = {}

    head_pose_valid = bool(head.get("is_valid", False))
    head_label = str(head.get("label", ""))
    eye_status = str(gaze.get("eye_status", ""))
    l2cs_face_count = int(gaze.get("l2cs_face_count", 0) or 0)

    # These eye states require current facial-landmark evidence.
    eye_landmark_face_present = eye_status not in {
        "",
        "NO_FACE",
        "CALIBRATING",
    }

    basis = []
    if head_pose_valid:
        basis.append("valid_head_pose")
    if eye_landmark_face_present:
        basis.append(f"eye_status:{eye_status}")

    return {
        "present": bool(head_pose_valid or eye_landmark_face_present),
        "head_pose_valid": head_pose_valid,
        "head_label": head_label,
        "eye_status": eye_status,
        "l2cs_face_count": l2cs_face_count,
        "basis": basis,
    }


def qualify_event_candidates(
    raw_candidates: Mapping[str, Mapping[str, Any]],
    object_event_eligibility: Optional[
        Mapping[str, ObjectEventEligibilityRule]
    ] = None,
    *,
    head_gaze_face_evidence: Optional[Mapping[str, Any]] = None,
    suppress_no_person_if_face_present: bool = True,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Split raw cue candidates into event-eligible and rejected candidates.

    Object cues may have a class-specific minimum confidence for EVENT
    eligibility.

    person:NO_PERSON also has a cross-module validation rule:
    if 01 currently provides strong face-presence evidence, YOLO's
    NO_PERSON state remains visible as a raw candidate but is prevented from
    entering temporal event tracking.

    Other non-object cues pass through unchanged. Rejected candidates remain
    explicit for traceability instead of being silently discarded.
    """
    eligibility = dict(object_event_eligibility or {})
    qualified: Dict[str, Dict[str, Any]] = {}
    rejected: Dict[str, Dict[str, Any]] = {}

    face_evidence = dict(head_gaze_face_evidence or {})
    face_present = bool(face_evidence.get("present", False))

    for cue_key, candidate_mapping in raw_candidates.items():
        candidate = dict(candidate_mapping)

        cue_type = str(candidate.get("cue_type", ""))
        label = str(candidate.get("label", ""))

        if (
            cue_type == "person_state"
            and label == "NO_PERSON"
            and suppress_no_person_if_face_present
            and face_present
        ):
            rejected[cue_key] = {
                "cue_key": cue_key,
                "cue_type": "person_state",
                "label": label,
                "timestamp": candidate.get("timestamp"),
                "reason": "head_gaze_face_evidence_present",
                "source": candidate.get("source"),
                "cross_module_evidence_source": "head_gaze_adapter",
                "face_evidence": face_evidence,
                "raw_candidate": candidate,
            }
            continue

        if cue_type != "object":
            qualified[cue_key] = candidate
            continue

        rule = eligibility.get(label)

        if rule is None:
            qualified[cue_key] = candidate
            continue

        rule.validate()

        max_confidence = float(candidate.get("max_confidence", 0.0))
        if rule.enabled and max_confidence >= rule.min_confidence:
            candidate["event_min_confidence"] = float(rule.min_confidence)
            candidate["event_confidence_qualified"] = True
            qualified[cue_key] = candidate
            continue

        rejected[cue_key] = {
            "cue_key": cue_key,
            "cue_type": "object",
            "label": label,
            "timestamp": candidate.get("timestamp"),
            "max_confidence": max_confidence,
            "required_min_confidence": float(rule.min_confidence),
            "reason": (
                "event_confidence_gate_disabled"
                if not rule.enabled
                else "below_event_confidence_threshold"
            ),
            "source": candidate.get("source"),
            "raw_candidate": candidate,
        }

    json.dumps(qualified)
    json.dumps(rejected)
    return qualified, rejected


class EventManager:
    """Track cue persistence and emit START / ACTIVE / END lifecycle records."""

    def __init__(
        self,
        rules: Optional[Mapping[str, EventRule]] = None,
        object_event_eligibility: Optional[
            Mapping[str, ObjectEventEligibilityRule]
        ] = None,
        *,
        suppress_no_person_if_face_present: bool = True,
    ) -> None:
        self.rules = dict(rules or {})
        for rule in self.rules.values():
            rule.validate()

        self.object_event_eligibility = dict(object_event_eligibility or {})
        for rule in self.object_event_eligibility.values():
            rule.validate()

        self.suppress_no_person_if_face_present = bool(
            suppress_no_person_if_face_present
        )

        self._trackers: Dict[str, _CueTracker] = {}

        # Multi-rate YOLO support. The latest YOLO observation is cached so
        # the UI can keep showing the latest result between inferences, while
        # temporal trackers advance only when a NEW YOLO result arrives.
        self._last_yolo_raw_candidates: Dict[str, Dict[str, Any]] = {}
        self._last_yolo_candidates: Dict[str, Dict[str, Any]] = {}
        self._last_yolo_rejected: Dict[str, Dict[str, Any]] = {}
        self._last_yolo_observation_timestamp: Optional[float] = None
        self._last_yolo_result_sequence_id: Optional[int] = None
        self._no_person_blocked_until_next_yolo = False

    def reset(self) -> None:
        self._trackers.clear()
        self._last_yolo_raw_candidates.clear()
        self._last_yolo_candidates.clear()
        self._last_yolo_rejected.clear()
        self._last_yolo_observation_timestamp = None
        self._last_yolo_result_sequence_id = None
        self._no_person_blocked_until_next_yolo = False

    @staticmethod
    def _wildcard_prefix(cue_type: str) -> str:
        return {
            "head_pose": "head",
            "gaze": "gaze",
            "object": "object",
            "person_state": "person",
        }.get(cue_type, cue_type)

    def _rule_for(self, cue_key: str, cue_type: str) -> Optional[EventRule]:
        if cue_key in self.rules:
            return self.rules[cue_key]
        return self.rules.get(f"{self._wildcard_prefix(cue_type)}:*")

    @staticmethod
    def _cue_type_from_key(cue_key: str) -> str:
        prefix = cue_key.split(":", 1)[0]
        return {
            "head": "head_pose",
            "gaze": "gaze",
            "object": "object",
            "person": "person_state",
        }.get(prefix, prefix)

    @staticmethod
    def _event_record(
        lifecycle: str,
        cue_key: str,
        candidate: Mapping[str, Any],
        timestamp: float,
        duration_s: float,
        event_start_timestamp: Optional[float] = None,
        event_end_timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build one JSON-safe lifecycle record.

        timestamp:
            When the lifecycle record is emitted/confirmed.

        event_start_timestamp:
            When the cue-presence sequence first began.

        event_end_timestamp:
            For END records, the first timestamp at which the cue was observed
            absent. This keeps release-duration grace from inflating event duration.
        """
        return {
            "lifecycle": lifecycle,
            "cue_key": cue_key,
            "cue_type": str(candidate.get("cue_type", "")),
            "label": str(candidate.get("label", "")),
            "timestamp": float(timestamp),
            "event_start_timestamp": (
                float(event_start_timestamp)
                if event_start_timestamp is not None
                else None
            ),
            "event_end_timestamp": (
                float(event_end_timestamp)
                if event_end_timestamp is not None
                else None
            ),
            "duration_s": float(max(0.0, duration_s)),
            "source": candidate.get("source"),
            "context": dict(candidate),
        }

    @staticmethod
    def _is_yolo_cue_type(cue_type: str) -> bool:
        return cue_type in {"object", "person_state"}

    def process(
        self,
        head_gaze_output: Optional[Mapping[str, Any]] = None,
        yolo_output: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[float] = None,
        *,
        yolo_observation_is_new: Optional[bool] = None,
        yolo_result_sequence_id: Optional[int] = None,
        yolo_result_age_s: Optional[float] = None,
        yolo_stale_after_s: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Process one integration step.

        Synchronous/backwards-compatible use:
            Pass yolo_output normally and omit yolo_observation_is_new.
            Every supplied YOLO output is treated as a fresh observation.

        Asynchronous/multi-rate use:
            Pass yolo_output ONLY when a new worker result is consumed and set
            yolo_observation_is_new accordingly. Between YOLO results, pass
            yolo_output=None and yolo_observation_is_new=False.

        The manager caches the latest YOLO observation for current-status and
        multi-cue context, but object/person temporal trackers advance only on
        fresh YOLO observations. This prevents one old YOLO result from being
        counted repeatedly at the faster UI/head-pose loop rate.
        """
        ts = _timestamp(head_gaze_output, yolo_output, timestamp)

        if yolo_stale_after_s < 0:
            raise ValueError("yolo_stale_after_s must be >= 0")

        if yolo_observation_is_new is None:
            yolo_observation_is_new = yolo_output is not None
        yolo_observation_is_new = bool(yolo_observation_is_new)

        if yolo_observation_is_new and yolo_output is None:
            raise ValueError(
                "yolo_output must be provided when yolo_observation_is_new=True"
            )

        head_gaze_face_evidence = infer_head_gaze_face_evidence(
            head_gaze_output
        )

        # --------------------------------------------------------------
        # Head/gaze is evaluated at the main-loop rate.
        # --------------------------------------------------------------
        head_raw = extract_cue_candidates(
            head_gaze_output=head_gaze_output,
            yolo_output=None,
            timestamp=ts,
        )
        head_candidates, head_rejected = qualify_event_candidates(
            head_raw,
            self.object_event_eligibility,
            head_gaze_face_evidence=head_gaze_face_evidence,
            suppress_no_person_if_face_present=(
                self.suppress_no_person_if_face_present
            ),
        )

        fresh_yolo_raw: Dict[str, Dict[str, Any]] = {}
        fresh_yolo_candidates: Dict[str, Dict[str, Any]] = {}
        fresh_yolo_rejected: Dict[str, Dict[str, Any]] = {}
        yolo_observation_ts: Optional[float] = None

        # --------------------------------------------------------------
        # YOLO temporal evidence advances only on NEW inference results.
        # --------------------------------------------------------------
        if yolo_observation_is_new:
            assert yolo_output is not None
            yolo_observation_ts = float(yolo_output.get("timestamp", ts))

            fresh_yolo_raw = extract_cue_candidates(
                head_gaze_output=None,
                yolo_output=yolo_output,
                timestamp=yolo_observation_ts,
            )
            (
                fresh_yolo_candidates,
                fresh_yolo_rejected,
            ) = qualify_event_candidates(
                fresh_yolo_raw,
                self.object_event_eligibility,
                head_gaze_face_evidence=head_gaze_face_evidence,
                suppress_no_person_if_face_present=(
                    self.suppress_no_person_if_face_present
                ),
            )

            self._last_yolo_raw_candidates = {
                key: dict(value)
                for key, value in fresh_yolo_raw.items()
            }
            self._last_yolo_candidates = {
                key: dict(value)
                for key, value in fresh_yolo_candidates.items()
            }
            self._last_yolo_rejected = {
                key: dict(value)
                for key, value in fresh_yolo_rejected.items()
            }
            self._last_yolo_observation_timestamp = yolo_observation_ts
            self._last_yolo_result_sequence_id = (
                int(yolo_result_sequence_id)
                if yolo_result_sequence_id is not None
                else None
            )

            # If this fresh NO_PERSON result is contradicted by current face
            # evidence, keep that specific stale result blocked until another
            # fresh YOLO inference arrives. It must not become valid later just
            # because the face subsequently disappears.
            self._no_person_blocked_until_next_yolo = bool(
                "person:NO_PERSON" in fresh_yolo_raw
                and "person:NO_PERSON" in fresh_yolo_rejected
                and fresh_yolo_rejected["person:NO_PERSON"].get("reason")
                == "head_gaze_face_evidence_present"
            )

        observed_yolo_raw = {
            key: dict(value)
            for key, value in self._last_yolo_raw_candidates.items()
        }
        observed_yolo_candidates = {
            key: dict(value)
            for key, value in self._last_yolo_candidates.items()
        }
        observed_yolo_rejected = {
            key: dict(value)
            for key, value in self._last_yolo_rejected.items()
        }

        # Current face evidence may safely invalidate a previously observed
        # NO_PERSON state even before the next YOLO result arrives.
        forced_absent_keys = set()
        cached_no_person = "person:NO_PERSON" in observed_yolo_raw
        if cached_no_person and bool(head_gaze_face_evidence.get("present", False)):
            self._no_person_blocked_until_next_yolo = True

        if cached_no_person and self._no_person_blocked_until_next_yolo:
            if "person:NO_PERSON" in observed_yolo_candidates:
                observed_yolo_candidates.pop("person:NO_PERSON", None)
                forced_absent_keys.add("person:NO_PERSON")

            current_reason = (
                "head_gaze_face_evidence_present"
                if bool(head_gaze_face_evidence.get("present", False))
                else "stale_no_person_requires_fresh_yolo"
            )
            observed_yolo_rejected["person:NO_PERSON"] = {
                "cue_key": "person:NO_PERSON",
                "cue_type": "person_state",
                "label": "NO_PERSON",
                "timestamp": ts,
                "reason": current_reason,
                "source": "yolo_output_adapter",
                "cross_module_evidence_source": "head_gaze_adapter",
                "face_evidence": dict(head_gaze_face_evidence),
                "raw_candidate": dict(
                    observed_yolo_raw.get("person:NO_PERSON", {})
                ),
            }
            forced_absent_keys.add("person:NO_PERSON")

        # Result age is supplied by the async worker when available. For the
        # synchronous path it is effectively zero because every step is fresh.
        if yolo_result_age_s is not None:
            effective_yolo_age_s = max(0.0, float(yolo_result_age_s))
        elif yolo_observation_is_new:
            effective_yolo_age_s = 0.0
        elif self._last_yolo_observation_timestamp is not None:
            effective_yolo_age_s = max(
                0.0,
                float(ts) - float(self._last_yolo_observation_timestamp),
            )
        else:
            effective_yolo_age_s = None

        yolo_is_stale = bool(
            effective_yolo_age_s is not None
            and effective_yolo_age_s > float(yolo_stale_after_s)
        )

        # `cue_candidates` is intentionally the set of temporal observations
        # processed THIS step. For async YOLO, reused cached detections are not
        # repeated here. This also prevents 04 from sampling one confidence
        # value multiple times between actual YOLO inferences.
        step_candidates: Dict[str, Dict[str, Any]] = {
            **head_candidates,
            **fresh_yolo_candidates,
        }

        # `observed_cue_candidates` is the latest currently usable state for
        # UI/multi-cue context. Cached YOLO state is excluded once it is stale.
        observed_candidates: Dict[str, Dict[str, Any]] = dict(head_candidates)
        if not yolo_is_stale:
            observed_candidates.update(observed_yolo_candidates)

        raw_candidates: Dict[str, Dict[str, Any]] = {
            **head_raw,
            **observed_yolo_raw,
        }
        rejected_event_candidates: Dict[str, Dict[str, Any]] = {
            **head_rejected,
            **observed_yolo_rejected,
        }

        events: List[Dict[str, Any]] = []

        # --------------------------------------------------------------
        # Presence handling: only fresh temporal observations advance state.
        # --------------------------------------------------------------
        for cue_key, candidate in step_candidates.items():
            cue_type = str(candidate["cue_type"])
            rule = self._rule_for(cue_key, cue_type)

            if rule is None or not rule.enabled:
                continue

            tracker = self._trackers.setdefault(cue_key, _CueTracker())
            observation_ts = float(candidate.get("timestamp", ts))

            if (
                not tracker.is_active
                and tracker.last_end_ts is not None
                and (observation_ts - tracker.last_end_ts) < rule.cooldown_s
            ):
                tracker.latest_candidate = dict(candidate)
                tracker.last_seen_ts = observation_ts
                continue

            if tracker.first_seen_ts is None:
                tracker.first_seen_ts = observation_ts

            tracker.last_seen_ts = observation_ts
            tracker.absent_since_ts = None
            tracker.latest_candidate = dict(candidate)

            present_duration = observation_ts - tracker.first_seen_ts

            if not tracker.is_active and present_duration >= rule.min_duration_s:
                tracker.is_active = True
                tracker.active_since_ts = tracker.first_seen_ts
                events.append(
                    self._event_record(
                        "START",
                        cue_key,
                        candidate,
                        ts,
                        present_duration,
                        event_start_timestamp=tracker.active_since_ts,
                    )
                )

            elif tracker.is_active:
                active_since = (
                    tracker.active_since_ts
                    if tracker.active_since_ts is not None
                    else observation_ts
                )
                active_duration = observation_ts - float(active_since)
                events.append(
                    self._event_record(
                        "ACTIVE",
                        cue_key,
                        candidate,
                        ts,
                        active_duration,
                        event_start_timestamp=active_since,
                    )
                )

        # --------------------------------------------------------------
        # Absence/release handling is domain-aware.
        # Head/gaze can update every main loop; object/person absence is only
        # inferred from a fresh YOLO result (or current face evidence that
        # directly disproves cached NO_PERSON).
        # --------------------------------------------------------------
        head_current_keys = set(head_candidates)
        fresh_yolo_current_keys = set(fresh_yolo_candidates)

        for cue_key, tracker in list(self._trackers.items()):
            cue_type = self._cue_type_from_key(cue_key)
            rule = self._rule_for(cue_key, cue_type)
            if rule is None or not rule.enabled:
                continue

            is_yolo_cue = self._is_yolo_cue_type(cue_type)

            if is_yolo_cue:
                should_update_absence = bool(
                    yolo_observation_is_new
                    or cue_key in forced_absent_keys
                )
                if not should_update_absence:
                    continue

                if (
                    yolo_observation_is_new
                    and cue_key in fresh_yolo_current_keys
                    and cue_key not in forced_absent_keys
                ):
                    continue

                absence_observation_ts = (
                    float(yolo_observation_ts)
                    if yolo_observation_is_new
                    and yolo_observation_ts is not None
                    else float(ts)
                )
            else:
                if head_gaze_output is None:
                    continue
                if cue_key in head_current_keys:
                    continue
                absence_observation_ts = float(ts)

            if tracker.first_seen_ts is None and not tracker.is_active:
                continue

            if tracker.absent_since_ts is None:
                tracker.absent_since_ts = absence_observation_ts

            if (
                absence_observation_ts - tracker.absent_since_ts
            ) < rule.release_duration_s:
                continue

            if tracker.is_active:
                active_since = (
                    tracker.active_since_ts
                    if tracker.active_since_ts is not None
                    else absence_observation_ts
                )
                observed_end_ts = (
                    tracker.absent_since_ts
                    if tracker.absent_since_ts is not None
                    else absence_observation_ts
                )

                candidate = tracker.latest_candidate or {
                    "cue_key": cue_key,
                    "cue_type": cue_type,
                    "label": cue_key.split(":", 1)[-1],
                    "source": "event_manager",
                }
                events.append(
                    self._event_record(
                        "END",
                        cue_key,
                        candidate,
                        ts,
                        float(observed_end_ts) - float(active_since),
                        event_start_timestamp=active_since,
                        event_end_timestamp=observed_end_ts,
                    )
                )
                tracker.last_end_ts = absence_observation_ts
                tracker.is_active = False
                tracker.active_since_ts = None

            tracker.reset_presence()

        active_event_keys = sorted(
            key for key, tracker in self._trackers.items()
            if tracker.is_active
        )

        observed_current_keys = set(observed_candidates)

        stale_active_event_keys = sorted(
            key
            for key in active_event_keys
            if yolo_is_stale
            and self._is_yolo_cue_type(self._cue_type_from_key(key))
        )
        stale_active_set = set(stale_active_event_keys)

        observed_active_event_keys = sorted(
            key
            for key in active_event_keys
            if key in observed_current_keys
            and key not in stale_active_set
        )

        release_grace_event_keys = sorted(
            key
            for key in active_event_keys
            if key not in observed_current_keys
            and key not in stale_active_set
        )

        output = {
            "timestamp": float(ts),

            # Latest neutral state for debugging/current-status display.
            "raw_cue_candidates": raw_candidates,

            # Fresh temporal observations processed in THIS integration step.
            "cue_candidates": step_candidates,

            # Latest currently usable state, including cached non-stale YOLO.
            "observed_cue_candidates": observed_candidates,

            # Rejections are held until the next relevant observation so 04's
            # transition logger does not create false FILTER_END/START flicker.
            "rejected_event_candidates": rejected_event_candidates,

            "cross_module_validation": {
                "head_gaze_face_evidence": head_gaze_face_evidence,
                "suppress_no_person_if_face_present": bool(
                    self.suppress_no_person_if_face_present
                ),
                "no_person_suppressed": (
                    "person:NO_PERSON" in rejected_event_candidates
                    and rejected_event_candidates["person:NO_PERSON"].get(
                        "reason"
                    ) in {
                        "head_gaze_face_evidence_present",
                        "stale_no_person_requires_fresh_yolo",
                    }
                ),
            },

            "observation_state": {
                "yolo_observation_is_new": bool(yolo_observation_is_new),
                "yolo_result_sequence_id": self._last_yolo_result_sequence_id,
                "yolo_result_age_s": effective_yolo_age_s,
                "yolo_stale_after_s": float(yolo_stale_after_s),
                "yolo_is_stale": yolo_is_stale,
            },

            "events": events,
            "active_event_keys": active_event_keys,
            "observed_active_event_keys": observed_active_event_keys,
            "release_grace_event_keys": release_grace_event_keys,
            "stale_active_event_keys": stale_active_event_keys,
            "multi_cue_context": self._multi_cue_context(
                observed_active_event_keys
            ),
        }

        json.dumps(output)
        return output

    @staticmethod
    def _multi_cue_context(observed_active_event_keys: Iterable[str]) -> Dict[str, Any]:
        """
        Summarize simultaneous confirmed cues that are still observed NOW.

        Lifecycle-active events that are only being held by release grace are
        intentionally excluded so they cannot create false multi-cue overlap.
        """
        keys = list(observed_active_event_keys)
        return {
            "has_multiple_active_cues": len(keys) >= 2,
            "active_head_cues": [k for k in keys if k.startswith("head:")],
            "active_gaze_cues": [k for k in keys if k.startswith("gaze:")],
            "active_object_cues": [k for k in keys if k.startswith("object:")],
            "active_person_cues": [k for k in keys if k.startswith("person:")],
            "active_event_count": len(keys),
        }


def build_demo_rules() -> Dict[str, EventRule]:
    """DEMO-ONLY rules for the synthetic smoke test; not final research rules."""
    return {
        "head:*": EventRule(0.50, 0.20),
        "gaze:*": EventRule(0.50, 0.20),
        "object:phone": EventRule(0.50, 0.20),
        "person:MULTIPLE_PERSONS": EventRule(0.50, 0.20),
        # audio_device intentionally excluded: its rule is not frozen.
    }


def build_initial_integration_rules(
    *,
    enable_audio_device_event: bool = True,
) -> Dict[str, EventRule]:
    """
    Engineering defaults for the FIRST live 01+02+03 integration test.

    These values are provisional integration settings, NOT final research
    thresholds. They should be reviewed after live testing and later reported
    as tuned/selected parameters if retained.

    audio_device is enabled by default in the selected Week 8 integration
    configuration after dedicated offline-image, physical live-webcam, and
    recorded-video checkpoint evaluation. The event remains a visual-cue
    indicator and should not be interpreted as reliable small-earbud detection.
    """

    rules: Dict[str, EventRule] = {
        # Head pose: relatively stable compared with gaze.
        "head:*": EventRule(
            min_duration_s=0.50,
            release_duration_s=0.30,
            cooldown_s=0.30,
        ),

        # Independent gaze deviations are eligible only when 01 says so.
        "gaze:*": EventRule(
            min_duration_s=0.75,
            release_duration_s=0.30,
            cooldown_s=0.30,
        ),

        # Object cues. These are engineering defaults for live integration.
        "object:phone": EventRule(
            min_duration_s=0.50,
            release_duration_s=0.40,
            cooldown_s=0.50,
        ),
        "object:computer_device": EventRule(
            min_duration_s=0.75,
            release_duration_s=0.50,
            cooldown_s=0.50,
        ),
        "object:book_notes": EventRule(
            min_duration_s=0.75,
            release_duration_s=0.50,
            cooldown_s=0.50,
        ),
        "object:calculator": EventRule(
            min_duration_s=0.50,
            release_duration_s=0.40,
            cooldown_s=0.50,
        ),
        "object:watch": EventRule(
            min_duration_s=0.75,
            release_duration_s=0.50,
            cooldown_s=0.50,
        ),

        # Person state. ONE_PERSON remains the neutral expected state.
        "person:NO_PERSON": EventRule(
            min_duration_s=1.00,
            release_duration_s=0.50,
            cooldown_s=0.50,
        ),
        "person:MULTIPLE_PERSONS": EventRule(
            min_duration_s=0.50,
            release_duration_s=0.40,
            cooldown_s=0.50,
        ),
    }

    if enable_audio_device_event:
        # Selected engineering rule after the Week 8 audio-device evaluation.
        rules["object:audio_device"] = EventRule(
            min_duration_s=0.75,
            release_duration_s=0.60,
            cooldown_s=0.50,
        )

    return rules


def build_initial_object_event_eligibility(
    *,
    computer_device_min_confidence: float = 0.45,
    book_notes_min_confidence: float = 0.40,
    audio_device_min_confidence: float = 0.43,
) -> Dict[str, ObjectEventEligibilityRule]:
    """
    Provisional engineering gates for event eligibility.

    02 keeps its frozen raw inference threshold (0.25). These values only
    determine whether a normalized object cue may enter 03's temporal tracker.

    The two raised thresholds target observed live false positives:
    - computer_device: 0.45
    - book_notes: 0.40

    Other object classes currently keep the raw 0.25 eligibility floor.

    audio_device uses 0.43 as the selected event-eligibility threshold. This
    does NOT change 02's raw YOLO output threshold (0.25); lower-confidence
    audio_device detections remain available in raw detections but are rejected
    before temporal event tracking.
    """
    rules = {
        "phone": ObjectEventEligibilityRule(0.25),
        "computer_device": ObjectEventEligibilityRule(
            float(computer_device_min_confidence)
        ),
        "book_notes": ObjectEventEligibilityRule(
            float(book_notes_min_confidence)
        ),
        "calculator": ObjectEventEligibilityRule(0.25),
        "watch": ObjectEventEligibilityRule(0.25),
        "audio_device": ObjectEventEligibilityRule(
            float(audio_device_min_confidence)
        ),
    }

    for rule in rules.values():
        rule.validate()

    return rules


def describe_object_event_eligibility(
    rules: Mapping[str, ObjectEventEligibilityRule],
) -> Dict[str, Dict[str, Any]]:
    """Return JSON-safe object event-eligibility metadata."""
    return {
        label: {
            "min_confidence": float(rule.min_confidence),
            "enabled": bool(rule.enabled),
        }
        for label, rule in rules.items()
    }


def describe_rules(
    rules: Mapping[str, EventRule],
) -> Dict[str, Dict[str, Any]]:
    """Return JSON-safe rule metadata for console/session traceability."""
    return {
        key: {
            "min_duration_s": float(rule.min_duration_s),
            "release_duration_s": float(rule.release_duration_s),
            "cooldown_s": float(rule.cooldown_s),
            "enabled": bool(rule.enabled),
        }
        for key, rule in rules.items()
    }


def _hg(ts: float, head: str, gaze: str, gaze_eligible: bool) -> Dict[str, Any]:
    return {
        "timestamp": ts,
        "head_pose": {
            "label": head,
            "confidence": "HIGH",
            "is_valid": True,
        },
        "gaze": {
            "label": gaze,
            "confidence": "HIGH",
            "is_valid": True,
            "eye_status": "EYES_OPEN",
            "relationship": f"{head}__{gaze}",
        },
        "combined": {
            "head_can_trigger_event": True,
            "gaze_output_available": True,
            "gaze_deviation_event_eligible": gaze_eligible,
        },
    }


def _yolo(
    ts: float,
    object_labels: Optional[List[str]] = None,
    person_count: int = 1,
) -> Dict[str, Any]:
    object_labels = object_labels or []
    detections = [
        {
            "class_id": i,
            "model_label": "laptop" if label == "computer_device" else label,
            "label": label,
            "confidence": 0.80,
            "bbox_xyxy": [10.0, 20.0, 100.0, 150.0],
        }
        for i, label in enumerate(object_labels)
    ]

    if person_count == 0:
        state = "NO_PERSON"
    elif person_count == 1:
        state = "ONE_PERSON"
    else:
        state = "MULTIPLE_PERSONS"

    return {
        "timestamp": ts,
        "detections": detections,
        "person_count": person_count,
        "person_state": state,
    }


def run_synthetic_smoke_test() -> None:
    """Exercise candidate extraction, lifecycle, multi-cue and no-double-counting."""
    manager = EventManager(build_demo_rules())

    samples = [
        # neutral
        (0.00, _hg(0.00, "HEAD_FORWARD", "FORWARD", False), _yolo(0.00)),

        # head-left + phone: gaze follows head but is NOT independently eligible
        (0.20, _hg(0.20, "HEAD_LEFT", "LOOKING_LEFT", False), _yolo(0.20, ["phone"])),
        (0.75, _hg(0.75, "HEAD_LEFT", "LOOKING_LEFT", False), _yolo(0.75, ["phone"])),
        (1.00, _hg(1.00, "HEAD_LEFT", "LOOKING_LEFT", False), _yolo(1.00, ["phone"])),

        # release
        (1.20, _hg(1.20, "HEAD_FORWARD", "FORWARD", False), _yolo(1.20)),
        (1.45, _hg(1.45, "HEAD_FORWARD", "FORWARD", False), _yolo(1.45)),

        # independent forward-head gaze deviation
        (2.00, _hg(2.00, "HEAD_FORWARD", "LOOKING_RIGHT", True), _yolo(2.00)),
        (2.60, _hg(2.60, "HEAD_FORWARD", "LOOKING_RIGHT", True), _yolo(2.60)),

        # multiple persons starts while the previous gaze event enters release grace
        (3.00, _hg(3.00, "HEAD_FORWARD", "FORWARD", False), _yolo(3.00, person_count=2)),
        (3.25, _hg(3.25, "HEAD_FORWARD", "FORWARD", False), _yolo(3.25, person_count=2)),
        (3.60, _hg(3.60, "HEAD_FORWARD", "FORWARD", False), _yolo(3.60, person_count=2)),
    ]

    for ts, head_gaze, yolo in samples:
        output = manager.process(head_gaze, yolo, timestamp=ts)
        print("\n[EVENT_MANAGER_OUTPUT]")
        print(json.dumps(output, indent=2))

    print("\nSynthetic smoke test completed. Confirm active vs observed active semantics.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-test",
        action="store_true",
        help="Run the built-in synthetic test using demo-only thresholds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.synthetic_test:
        run_synthetic_smoke_test()
    else:
        print(
            "03_event_manager.py is a library module. "
            "Use --synthetic-test for the built-in smoke test."
        )
