"""
Experimental multi-cue review score for the TEEP CV prototype.

This module consumes the frozen Event Manager output. It does not inspect raw
frames and does not change the 01-04 integration rules. Only confirmed cues in
``observed_active_event_keys`` contribute to the current score; release-grace
and stale events are deliberately excluded.

The output is an interpretable review-priority indicator, not a calibrated
probability and not an automatic cheating verdict.

Formula
-------
For each currently observed confirmed cue i:

    cue_i = weight_i * confidence_factor_i * duration_factor_i

    duration_factor_i = floor + (1 - floor)
                        * min(duration_i / reference_duration, 1)

The live review score is:

    score = min(1, sum(cue_i) + concurrency_bonus(unique_domain_count))

The session peak is the maximum live score observed during the process. All
tunable constants are loadable from an external JSON configuration file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Set


SCORE_VERSION = "experimental_multi_cue_review_score_v1_1"
DISCLAIMER = (
    "Experimental review indicator; not a calibrated cheating probability "
    "and not an automatic cheating verdict."
)


DEFAULT_CUE_WEIGHTS: Dict[str, float] = {
    "head:*": 0.26,
    "gaze:*": 0.26,
    "object:phone": 0.36,
    "object:computer_device": 0.18,
    "object:book_notes": 0.32,
    "object:calculator": 0.20,
    "object:watch": 0.16,
    "object:audio_device": 0.22,
    "person:NO_PERSON": 0.42,
    "person:MULTIPLE_PERSONS": 0.50,
}

DEFAULT_CONFIDENCE_FACTORS: Dict[str, float] = {
    "HIGH": 1.00,
    "MEDIUM": 0.85,
    "LOW": 0.65,
    "UNRELIABLE": 0.00,
}

DEFAULT_CONCURRENCY_BONUSES: Dict[int, float] = {
    0: 0.00,
    1: 0.00,
    2: 0.10,
    3: 0.18,
    4: 0.25,
}

DEFAULT_REVIEW_LEVEL_THRESHOLDS: Dict[str, float] = {
    "MODERATE": 0.25,
    "HIGH": 0.50,
    "VERY_HIGH": 0.75,
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ReviewScoreConfig:
    """Transparent, configurable constants for the experimental score."""

    score_version: str = SCORE_VERSION
    disclaimer: str = DISCLAIMER

    cue_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CUE_WEIGHTS)
    )
    confidence_factors: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CONFIDENCE_FACTORS)
    )
    concurrency_bonuses: Mapping[int, float] = field(
        default_factory=lambda: dict(DEFAULT_CONCURRENCY_BONUSES)
    )
    review_level_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REVIEW_LEVEL_THRESHOLDS)
    )
    duration_reference_s: float = 5.0
    duration_floor: float = 0.80

    def __post_init__(self) -> None:
        if self.duration_reference_s <= 0.0:
            raise ValueError("duration_reference_s must be > 0")
        if not 0.0 <= self.duration_floor <= 1.0:
            raise ValueError("duration_floor must be between 0 and 1")

        for cue_key, weight in self.cue_weights.items():
            if not 0.0 <= float(weight) <= 1.0:
                raise ValueError(f"Invalid weight for {cue_key}: {weight}")

        for domain_count, bonus in self.concurrency_bonuses.items():
            if int(domain_count) < 0 or not 0.0 <= float(bonus) <= 1.0:
                raise ValueError(
                    f"Invalid concurrency bonus {domain_count}: {bonus}"
                )

        required_levels = {"MODERATE", "HIGH", "VERY_HIGH"}
        missing_levels = required_levels.difference(
            str(key).upper() for key in self.review_level_thresholds
        )
        if missing_levels:
            raise ValueError(
                "Missing review-level thresholds: "
                + ", ".join(sorted(missing_levels))
            )

        normalized_levels = {
            str(key).upper(): float(value)
            for key, value in self.review_level_thresholds.items()
        }
        moderate = normalized_levels["MODERATE"]
        high = normalized_levels["HIGH"]
        very_high = normalized_levels["VERY_HIGH"]
        if not (0.0 <= moderate < high < very_high <= 1.0):
            raise ValueError(
                "Review-level thresholds must satisfy "
                "0 <= MODERATE < HIGH < VERY_HIGH <= 1"
            )


def load_review_score_config(path: Path) -> ReviewScoreConfig:
    """Load all tunable score parameters from one external JSON file."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Review-score config file not found: {config_path}"
        )

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid review-score JSON in {config_path}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise ValueError("Review-score config root must be a JSON object")

    required_keys = {
        "cue_weights",
        "confidence_factors",
        "duration_factor",
        "concurrency_bonuses",
        "review_level_thresholds",
    }
    missing_keys = required_keys.difference(payload)
    if missing_keys:
        raise ValueError(
            "Missing review-score config keys: "
            + ", ".join(sorted(missing_keys))
        )

    duration = _as_mapping(payload["duration_factor"])
    raw_bonuses = _as_mapping(payload["concurrency_bonuses"])

    return ReviewScoreConfig(
        score_version=str(payload.get("score_version", SCORE_VERSION)),
        disclaimer=str(payload.get("disclaimer", DISCLAIMER)),
        cue_weights=dict(_as_mapping(payload["cue_weights"])),
        confidence_factors=dict(
            _as_mapping(payload["confidence_factors"])
        ),
        concurrency_bonuses={
            int(domain_count): float(bonus)
            for domain_count, bonus in raw_bonuses.items()
        },
        review_level_thresholds={
            str(level).upper(): float(threshold)
            for level, threshold in _as_mapping(
                payload["review_level_thresholds"]
            ).items()
        },
        duration_reference_s=float(duration.get("reference_s", 5.0)),
        duration_floor=float(duration.get("floor", 0.80)),
    )


@dataclass
class _CueState:
    start_timestamp: float
    duration_s: float
    context: Dict[str, Any]


class MultiCueReviewScorer:
    """Stateful live scorer driven by frozen Event Manager outputs."""

    def __init__(self, config: Optional[ReviewScoreConfig] = None) -> None:
        self.config = config or ReviewScoreConfig()
        self._cue_states: MutableMapping[str, _CueState] = {}
        self._session_peak_score = 0.0
        self._session_peak_timestamp: Optional[float] = None

    def reset(self) -> None:
        self._cue_states.clear()
        self._session_peak_score = 0.0
        self._session_peak_timestamp = None

    @staticmethod
    def _domain(cue_key: str) -> str:
        return cue_key.split(":", 1)[0]

    def _weight_for(self, cue_key: str) -> Optional[float]:
        if cue_key in self.config.cue_weights:
            return float(self.config.cue_weights[cue_key])

        domain = self._domain(cue_key)
        wildcard = f"{domain}:*"
        if wildcard in self.config.cue_weights:
            return float(self.config.cue_weights[wildcard])
        return None

    def _confidence_factor(
        self,
        cue_key: str,
        context: Mapping[str, Any],
    ) -> float:
        domain = self._domain(cue_key)

        if domain == "object":
            return _clamp(float(context.get("max_confidence", 0.0)))

        if domain in {"head", "gaze"}:
            confidence = context.get("confidence", "UNRELIABLE")
            if isinstance(confidence, (int, float)):
                return _clamp(float(confidence))
            return _clamp(
                float(
                    self.config.confidence_factors.get(
                        str(confidence).upper(),
                        0.0,
                    )
                )
            )

        if domain == "person":
            # Person-state candidates do not expose a comparable numeric
            # confidence. They contribute only after temporal confirmation.
            return 1.0

        return 0.0

    def _duration_factor(self, duration_s: float) -> float:
        progress = _clamp(
            max(0.0, float(duration_s)) / self.config.duration_reference_s
        )
        return self.config.duration_floor + (
            (1.0 - self.config.duration_floor) * progress
        )

    def _concurrency_bonus(self, domains: Set[str]) -> float:
        count = len(domains)
        if count in self.config.concurrency_bonuses:
            return float(self.config.concurrency_bonuses[count])

        largest_key = max(self.config.concurrency_bonuses)
        return float(self.config.concurrency_bonuses[largest_key])

    def _review_level(self, score: float) -> str:
        thresholds = {
            str(key).upper(): float(value)
            for key, value in self.config.review_level_thresholds.items()
        }
        if score < thresholds["MODERATE"]:
            return "LOW"
        if score < thresholds["HIGH"]:
            return "MODERATE"
        if score < thresholds["VERY_HIGH"]:
            return "HIGH"
        return "VERY_HIGH"

    @staticmethod
    def _explanation(contributions: Mapping[str, Mapping[str, Any]]) -> str:
        if not contributions:
            return "No currently observed confirmed visual cues."

        ranked = sorted(
            contributions.items(),
            key=lambda item: float(item[1]["contribution"]),
            reverse=True,
        )
        cue_text = ", ".join(key for key, _ in ranked[:3])
        if len(ranked) == 1:
            return f"Confirmed visual cue currently observed: {cue_text}."
        return f"Concurrent confirmed visual cues currently observed: {cue_text}."

    def _update_lifecycle_state(
        self,
        events: Sequence[Any],
        timestamp: float,
    ) -> None:
        for raw_event in events:
            event = _as_mapping(raw_event)
            cue_key = str(event.get("cue_key", ""))
            lifecycle = str(event.get("lifecycle", "")).upper()
            if not cue_key:
                continue

            if lifecycle == "END":
                self._cue_states.pop(cue_key, None)
                continue

            if lifecycle not in {"START", "ACTIVE"}:
                continue

            context = dict(_as_mapping(event.get("context")))
            start_timestamp = event.get("event_start_timestamp")
            if start_timestamp is None:
                start_timestamp = timestamp - float(event.get("duration_s", 0.0))

            self._cue_states[cue_key] = _CueState(
                start_timestamp=float(start_timestamp),
                duration_s=max(0.0, float(event.get("duration_s", 0.0))),
                context=context,
            )

    def process(self, event_output: Mapping[str, Any]) -> Dict[str, Any]:
        """Calculate the current score from one Event Manager output."""

        timestamp = float(event_output.get("timestamp", 0.0))
        events = event_output.get("events", [])
        if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
            events = []
        self._update_lifecycle_state(events, timestamp)

        active_keys = {
            str(key) for key in event_output.get("active_event_keys", [])
        }
        for cue_key in list(self._cue_states):
            if cue_key not in active_keys:
                self._cue_states.pop(cue_key, None)

        observed_keys = sorted(
            {
                str(key)
                for key in event_output.get("observed_active_event_keys", [])
            }
        )
        observed_candidates = _as_mapping(
            event_output.get("observed_cue_candidates")
        )

        contributions: Dict[str, Dict[str, Any]] = {}
        ignored_cues = []
        domains: Set[str] = set()

        for cue_key in observed_keys:
            weight = self._weight_for(cue_key)
            if weight is None:
                ignored_cues.append(cue_key)
                continue

            candidate = dict(_as_mapping(observed_candidates.get(cue_key)))
            state = self._cue_states.get(cue_key)

            if state is None:
                # Supports attaching the scorer after an event has already
                # started. The event is still trusted because it appears in
                # observed_active_event_keys.
                state = _CueState(timestamp, 0.0, candidate)
                self._cue_states[cue_key] = state
            elif candidate:
                state.context = candidate

            duration_s = max(
                state.duration_s,
                max(0.0, timestamp - state.start_timestamp),
            )
            state.duration_s = duration_s

            context = state.context
            confidence_factor = self._confidence_factor(cue_key, context)
            duration_factor = self._duration_factor(duration_s)
            contribution = weight * confidence_factor * duration_factor

            domains.add(self._domain(cue_key))
            contributions[cue_key] = {
                "weight": round(weight, 4),
                "confidence_factor": round(confidence_factor, 4),
                "duration_s": round(duration_s, 4),
                "duration_factor": round(duration_factor, 4),
                "contribution": round(contribution, 4),
                "source": context.get("source"),
            }

        concurrency_bonus = self._concurrency_bonus(domains)
        contribution_total = sum(
            float(item["contribution"]) for item in contributions.values()
        )
        score = _clamp(contribution_total + concurrency_bonus)

        if score > self._session_peak_score:
            self._session_peak_score = score
            self._session_peak_timestamp = timestamp

        result = {
            "score_version": self.config.score_version,
            "timestamp": timestamp,
            "score": round(score, 4),
            "review_level": self._review_level(score),
            "active_cues": list(contributions),
            "active_domains": sorted(domains),
            "contributions": contributions,
            "contribution_total": round(contribution_total, 4),
            "concurrency_bonus": round(concurrency_bonus, 4),
            "ignored_cues": ignored_cues,
            "session_peak_score": round(self._session_peak_score, 4),
            "session_peak_timestamp": self._session_peak_timestamp,
            "explanation": self._explanation(contributions),
            "disclaimer": self.config.disclaimer,
        }

        json.dumps(result)
        return result


def score_event_output(
    event_output: Mapping[str, Any],
    config: Optional[ReviewScoreConfig] = None,
) -> Dict[str, Any]:
    """Stateless convenience helper for one already-confirmed snapshot."""

    return MultiCueReviewScorer(config=config).process(event_output)
