"""Synthetic tests for the external-config review score v1.1."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


TEST_VERSION = "review_score_test_v1_1_external_config"

MODULE_CANDIDATES = [
    Path(__file__).with_name("05_multi_cue_review_score.py"),
    (
        Path(__file__).resolve().parents[2]
        / "src"
        / "integration"
        / "05_multi_cue_review_score.py"
    ),
]
MODULE_PATH = next(
    (path for path in MODULE_CANDIDATES if path.is_file()),
    MODULE_CANDIDATES[-1],
)

CONFIG_CANDIDATES = [
    Path(__file__).with_name("multi_cue_review_score_v1_1.json"),
    (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "multi_cue_review_score_v1_1.json"
    ),
]
CONFIG_PATH = next(
    (path for path in CONFIG_CANDIDATES if path.is_file()),
    CONFIG_CANDIDATES[-1],
)


def load_module():
    if not MODULE_PATH.is_file():
        raise FileNotFoundError(f"Scoring module not found: {MODULE_PATH}")

    spec = importlib.util.spec_from_file_location(
        "multi_cue_review_score_v1_1",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load scoring module: {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def event(
    timestamp,
    *,
    observed=(),
    active=None,
    lifecycle_events=(),
    candidates=None,
    release=(),
    stale=(),
):
    return {
        "timestamp": float(timestamp),
        "events": list(lifecycle_events),
        "active_event_keys": list(observed if active is None else active),
        "observed_active_event_keys": list(observed),
        "release_grace_event_keys": list(release),
        "stale_active_event_keys": list(stale),
        "observed_cue_candidates": dict(candidates or {}),
    }


def lifecycle(kind, cue_key, timestamp, duration, context):
    return {
        "lifecycle": kind,
        "cue_key": cue_key,
        "timestamp": float(timestamp),
        "event_start_timestamp": float(timestamp) - float(duration),
        "duration_s": float(duration),
        "context": dict(context),
    }


def run_tests():
    print(f"TEST VERSION: {TEST_VERSION}")

    module = load_module()
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"External config not found: {CONFIG_PATH}")

    config = module.load_review_score_config(CONFIG_PATH)
    scorer = module.MultiCueReviewScorer(config=config)

    assert config.score_version == "experimental_multi_cue_review_score_v1_1"
    assert config.cue_weights["head:*"] == 0.26
    assert config.cue_weights["gaze:*"] == 0.26
    assert config.cue_weights["object:phone"] == 0.36
    print("PASS: external JSON config loaded with v1.1 weights")

    neutral = scorer.process(event(0.0))
    assert neutral["score"] == 0.0
    assert neutral["review_level"] == "LOW"
    print("PASS: neutral score")

    phone_context = {
        "cue_type": "object",
        "label": "phone",
        "max_confidence": 0.85,
        "detections": [
            {"label": "phone", "confidence": 0.85},
            {"label": "phone", "confidence": 0.77},
        ],
        "source": "yolo_output_adapter",
    }
    phone = scorer.process(
        event(
            1.0,
            observed=("object:phone",),
            lifecycle_events=(
                lifecycle("START", "object:phone", 1.0, 0.5, phone_context),
            ),
            candidates={"object:phone": phone_context},
        )
    )
    assert 0.25 <= phone["score"] <= 0.26
    assert list(phone["contributions"]) == ["object:phone"]
    assert phone["contributions"]["object:phone"]["weight"] == 0.36
    print("PASS: duplicate phone boxes produce one v1.1 contribution")

    head_context = {
        "cue_type": "head_pose",
        "label": "HEAD_LEFT",
        "confidence": "HIGH",
        "source": "head_gaze_adapter",
    }
    phone_and_head = scorer.process(
        event(
            2.5,
            observed=("object:phone", "head:HEAD_LEFT"),
            lifecycle_events=(
                lifecycle("ACTIVE", "object:phone", 2.5, 2.0, phone_context),
                lifecycle("START", "head:HEAD_LEFT", 2.5, 0.5, head_context),
            ),
            candidates={
                "object:phone": phone_context,
                "head:HEAD_LEFT": head_context,
            },
        )
    )
    assert phone_and_head["concurrency_bonus"] == 0.10
    assert phone_and_head["active_domains"] == ["head", "object"]
    assert 0.58 <= phone_and_head["score"] <= 0.59
    print("PASS: v1.1 cross-domain concurrency score")

    grace_only = scorer.process(
        event(
            3.0,
            active=("object:phone",),
            release=("object:phone",),
        )
    )
    assert grace_only["score"] == 0.0

    stale_only = scorer.process(
        event(
            3.2,
            active=("object:phone",),
            stale=("object:phone",),
        )
    )
    assert stale_only["score"] == 0.0
    print("PASS: release-grace and stale cues excluded")

    ended = scorer.process(
        event(
            4.0,
            lifecycle_events=(
                lifecycle("END", "object:phone", 4.0, 3.5, phone_context),
            ),
        )
    )
    assert ended["score"] == 0.0
    assert ended["session_peak_score"] == phone_and_head["session_peak_score"]
    print("PASS: END removes current contribution")
    print("PASS: session peak retained")

    person_scorer = module.MultiCueReviewScorer(config=config)
    person_context = {
        "cue_type": "person_state",
        "label": "MULTIPLE_PERSONS",
        "source": "yolo_output_adapter",
    }
    multiple_people = person_scorer.process(
        event(
            5.0,
            observed=("person:MULTIPLE_PERSONS",),
            lifecycle_events=(
                lifecycle(
                    "START",
                    "person:MULTIPLE_PERSONS",
                    5.0,
                    5.0,
                    person_context,
                ),
            ),
            candidates={"person:MULTIPLE_PERSONS": person_context},
        )
    )
    assert multiple_people["score"] == 0.50
    assert multiple_people["review_level"] == "HIGH"
    print("PASS: JSON review-level thresholds applied")


if __name__ == "__main__":
    run_tests()
