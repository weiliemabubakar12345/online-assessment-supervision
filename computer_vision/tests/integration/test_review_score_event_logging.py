"""Synthetic compatibility test for Event Logger v3 review-score logging."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


TEST_VERSION = "event_logger_v3_review_score_audit"

LOGGER_CANDIDATES = [
    Path(__file__).with_name("04_event_logger.py"),
    Path(__file__).parent / "upload" / "04_event_logger(3).py",
    (
        Path(__file__).resolve().parents[2]
        / "src"
        / "integration"
        / "04_event_logger.py"
    ),
]
LOGGER_PATH = next(
    (path for path in LOGGER_CANDIDATES if path.is_file()),
    LOGGER_CANDIDATES[-1],
)


def load_logger_module():
    if not LOGGER_PATH.is_file():
        raise FileNotFoundError(f"Event Logger module not found: {LOGGER_PATH}")

    spec = importlib.util.spec_from_file_location(
        "teep_event_logger_v3_test",
        LOGGER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Event Logger: {LOGGER_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def event_output(timestamp, events=(), candidates=None):
    return {
        "timestamp": float(timestamp),
        "events": list(events),
        "cue_candidates": dict(candidates or {}),
        "observed_active_event_keys": sorted((candidates or {}).keys()),
        "rejected_event_candidates": {},
        "cross_module_validation": {},
        "multi_cue_context": {
            "has_multiple_active_cues": False,
            "active_event_count": len(candidates or {}),
            "active_head_cues": [],
            "active_gaze_cues": [],
            "active_object_cues": sorted((candidates or {}).keys()),
            "active_person_cues": [],
        },
    }


def lifecycle(kind, timestamp):
    return {
        "lifecycle": kind,
        "cue_key": "object:phone",
        "cue_type": "object",
        "label": "phone",
        "timestamp": float(timestamp),
        "event_start_timestamp": 1.0,
        "event_end_timestamp": 2.0 if kind == "END" else None,
        "duration_s": max(0.0, float(timestamp) - 1.0),
        "source": "yolo_output_adapter",
        "context": {
            "max_confidence": 0.90,
            "event_min_confidence": 0.25,
            "count": 1,
        },
    }


def review(timestamp, score, level, contributions, bonus=0.0, peak=None):
    peak_score = score if peak is None else peak
    return {
        "score_version": "experimental_multi_cue_review_score_v1_1",
        "timestamp": float(timestamp),
        "score": float(score),
        "review_level": level,
        "active_cues": list(contributions),
        "active_domains": sorted(
            {key.split(":", 1)[0] for key in contributions}
        ),
        "contributions": contributions,
        "contribution_total": round(float(score) - float(bonus), 4),
        "concurrency_bonus": float(bonus),
        "ignored_cues": [],
        "session_peak_score": float(peak_score),
        "session_peak_timestamp": float(timestamp),
        "explanation": "Synthetic review-score logging test.",
    }


def run_tests():
    print(f"TEST VERSION: {TEST_VERSION}")
    module = load_logger_module()

    phone_candidate = {
        "cue_key": "object:phone",
        "cue_type": "object",
        "label": "phone",
        "max_confidence": 0.90,
    }
    phone_contribution = {
        "object:phone": {
            "weight": 0.36,
            "confidence_factor": 0.90,
            "duration_s": 1.0,
            "duration_factor": 0.84,
            "contribution": 0.2722,
            "source": "yolo_output_adapter",
        }
    }
    multi_contributions = {
        **phone_contribution,
        "head:HEAD_LEFT": {
            "weight": 0.26,
            "confidence_factor": 1.0,
            "duration_s": 1.0,
            "duration_factor": 0.84,
            "contribution": 0.2184,
            "source": "head_gaze_adapter",
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        logger = module.EventLogger(
            base_dir=Path(temp_dir),
            session_metadata={"test": TEST_VERSION},
            session_id="synthetic_review_logging",
        )

        # Existing callers remain compatible when review_score_output is absent.
        logger.process(
            event_output=event_output(0.0),
            frame_timestamp=0.0,
        )

        logger.process(
            event_output=event_output(1.0),
            frame_timestamp=1.0,
            review_score_output=review(1.0, 0.0, "LOW", {}),
        )
        logger.process(
            event_output=event_output(
                1.5,
                events=(lifecycle("START", 1.5),),
                candidates={"object:phone": phone_candidate},
            ),
            frame_timestamp=1.5,
            review_score_output=review(
                1.5,
                0.2722,
                "MODERATE",
                phone_contribution,
                peak=0.2722,
            ),
        )
        logger.process(
            event_output=event_output(
                2.0,
                events=(lifecycle("END", 2.0),),
            ),
            frame_timestamp=2.0,
            review_score_output=review(
                2.0,
                0.5906,
                "HIGH",
                multi_contributions,
                bonus=0.10,
                peak=0.5906,
            ),
        )
        logger.close(final_frame_timestamp=2.0)

        paths = logger.get_paths()
        review_path = Path(paths["review_scores_jsonl"])
        records = [
            json.loads(line)
            for line in review_path.read_text(encoding="utf-8").splitlines()
        ]

        assert len(records) == 3
        assert all(record["record_type"] == "REVIEW_SCORE" for record in records)
        assert records[-1]["score"] == 0.5906
        assert records[-1]["concurrency_bonus"] == 0.10
        assert set(records[-1]["contributions"]) == {
            "object:phone",
            "head:HEAD_LEFT",
        }
        print("PASS: per-frame review-score breakdown persisted")

        metadata = json.loads(
            Path(paths["session_metadata"]).read_text(encoding="utf-8")
        )
        assert metadata["logger_version"] == "04_event_logger_v3"
        assert metadata["files"]["review_scores_jsonl"] == "review_scores.jsonl"
        print("PASS: metadata lists review_scores.jsonl")

        summary = json.loads(
            Path(paths["session_summary"]).read_text(encoding="utf-8")
        )
        assert summary["logger_version"] == "04_event_logger_v3"
        assert summary["review_score_samples"] == 3
        assert summary["review_level_sample_counts"] == {
            "LOW": 1,
            "MODERATE": 1,
            "HIGH": 1,
        }
        assert summary["final_review_score"] == 0.5906
        assert summary["session_peak_review_score"] == 0.5906
        print("PASS: session review-score summary retained")

        with Path(paths["completed_events_csv"]).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert len(rows) == 1
        assert rows[0]["cue_key"] == "object:phone"
        print("PASS: existing completed-events CSV remains compatible")


if __name__ == "__main__":
    run_tests()
