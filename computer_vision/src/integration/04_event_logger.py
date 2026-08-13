"""
04_event_logger.py

Structured event logging for the TEEP visual-cue integration pipeline.

Pipeline position
-----------------
01_head_gaze_adapter.py
    +
02_yolo_output_adapter.py
    ->
03_event_manager.py
    ->
04_event_logger.py

Purpose
-------
Persist event-manager outputs in a research-friendly format without changing
the event logic itself.

This module:
- logs START / END lifecycle records to JSONL;
- writes one CSV row for each completed event;
- logs transitions of rejected/filtered event candidates for debugging;
- writes session metadata and a final session summary;
- never invents an END event when the program quits.

Important
---------
04 is a logging layer only. It does not decide whether a cue is suspicious,
does not change thresholds, and does not modify 01/02/03 outputs.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _now_local_iso() -> str:
    """Return a timezone-aware local ISO-8601 timestamp."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _safe_json(value: Any) -> Any:
    """Round-trip through JSON so persisted values are JSON-safe."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _event_id(cue_key: str, event_start_timestamp: Any) -> str:
    try:
        start_ms = int(round(float(event_start_timestamp) * 1000.0))
    except Exception:
        start_ms = 0
    clean_key = str(cue_key).replace(":", "_").replace("/", "_")
    return f"{clean_key}__{start_ms:010d}"


def _context_scalar(
    context: Mapping[str, Any],
    key: str,
    default: Any = "",
) -> Any:
    value = context.get(key, default)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


class EventLogger:
    """
    Persist one integration session.

    Files created inside one session directory:
    - session_metadata.json
    - events.jsonl
    - completed_events.csv
    - filtered_candidates.jsonl
    - session_summary.json
    """

    CSV_FIELDS = [
        "session_id",
        "event_id",
        "cue_key",
        "cue_type",
        "label",
        "source",
        "event_start_timestamp",
        "event_end_timestamp",
        "duration_s",
        "start_wall_time",
        "end_wall_time",
        "start_observed_active_keys",
        "end_observed_active_keys",
        "start_multi_cue",
        "end_multi_cue",
        "confidence",
        "object_confidence_start",
        "object_confidence_peak",
        "object_confidence_mean",
        "object_confidence_last",
        "object_confidence_samples",
        "event_min_confidence",
        "count",
        "person_count",
        "eye_status",
        "relationship",
    ]

    def __init__(
        self,
        *,
        base_dir: Path,
        session_metadata: Mapping[str, Any],
        session_id: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(base_dir)

        if session_id is None:
            session_id = datetime.now().astimezone().strftime(
                "integration_%Y%m%d_%H%M%S_%f"
            )
        self.session_id = str(session_id)

        self.session_dir = self.base_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)

        self.events_jsonl_path = self.session_dir / "events.jsonl"
        self.completed_csv_path = self.session_dir / "completed_events.csv"
        self.filtered_jsonl_path = (
            self.session_dir / "filtered_candidates.jsonl"
        )
        self.metadata_path = self.session_dir / "session_metadata.json"
        self.summary_path = self.session_dir / "session_summary.json"

        self.session_start_wall_time = _now_local_iso()
        self.session_end_wall_time: Optional[str] = None

        self._events_fp = self.events_jsonl_path.open(
            "a",
            encoding="utf-8",
            newline="",
        )
        self._filtered_fp = self.filtered_jsonl_path.open(
            "a",
            encoding="utf-8",
            newline="",
        )
        self._csv_fp = self.completed_csv_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        )
        self._csv_writer = csv.DictWriter(
            self._csv_fp,
            fieldnames=self.CSV_FIELDS,
        )
        self._csv_writer.writeheader()
        self._csv_fp.flush()

        self._open_events: Dict[str, Dict[str, Any]] = {}
        self._active_filtered: Dict[str, Dict[str, Any]] = {}

        self._lifecycle_counts: Counter[str] = Counter()
        self._completed_by_cue: Counter[str] = Counter()
        self._filtered_episode_counts: Counter[str] = Counter()

        metadata = {
            "session_id": self.session_id,
            "session_start_wall_time": self.session_start_wall_time,
            "logger_version": "04_event_logger_v2",
            "files": {
                "events_jsonl": self.events_jsonl_path.name,
                "completed_events_csv": self.completed_csv_path.name,
                "filtered_candidates_jsonl": self.filtered_jsonl_path.name,
                "session_summary_json": self.summary_path.name,
            },
            "integration_session_metadata": _safe_json(
                dict(session_metadata)
            ),
        }
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._closed = False

    def _write_jsonl(self, fp: Any, record: Mapping[str, Any]) -> None:
        fp.write(
            json.dumps(
                _safe_json(dict(record)),
                ensure_ascii=False,
            )
            + "\n"
        )
        fp.flush()

    @staticmethod
    def _multi_cue_snapshot(
        event_output: Mapping[str, Any],
    ) -> Dict[str, Any]:
        context = event_output.get("multi_cue_context", {})
        if not isinstance(context, Mapping):
            context = {}

        observed = event_output.get(
            "observed_active_event_keys",
            [],
        )
        if not isinstance(observed, list):
            observed = list(observed) if observed else []

        return {
            "observed_active_event_keys": [
                str(key) for key in observed
            ],
            "has_multiple_active_cues": bool(
                context.get("has_multiple_active_cues", False)
            ),
            "active_event_count": int(
                context.get("active_event_count", 0) or 0
            ),
            "active_head_cues": _safe_json(
                context.get("active_head_cues", [])
            ),
            "active_gaze_cues": _safe_json(
                context.get("active_gaze_cues", [])
            ),
            "active_object_cues": _safe_json(
                context.get("active_object_cues", [])
            ),
            "active_person_cues": _safe_json(
                context.get("active_person_cues", [])
            ),
        }

    @staticmethod
    def _empty_object_confidence_stats() -> Dict[str, Any]:
        return {
            "start": None,
            "peak": None,
            "sum": 0.0,
            "last": None,
            "samples": 0,
        }

    @staticmethod
    def _finalize_object_confidence_stats(
        stats: Mapping[str, Any],
    ) -> Dict[str, Any]:
        samples = int(stats.get("samples", 0) or 0)
        total = float(stats.get("sum", 0.0) or 0.0)
        return {
            "start": stats.get("start"),
            "peak": stats.get("peak"),
            "mean": (total / samples) if samples else None,
            "last": stats.get("last"),
            "samples": samples,
        }

    def _update_open_object_confidence(
        self,
        *,
        event_output: Mapping[str, Any],
    ) -> None:
        """Sample confidence from currently qualified object candidates."""
        candidates = event_output.get("cue_candidates", {})
        if not isinstance(candidates, Mapping):
            return

        for cue_key, start_info in self._open_events.items():
            if (
                not isinstance(start_info, dict)
                or start_info.get("cue_type") != "object"
            ):
                continue

            candidate = candidates.get(cue_key)
            if not isinstance(candidate, Mapping):
                continue

            try:
                confidence = float(candidate.get("max_confidence"))
            except (TypeError, ValueError):
                continue

            stats = start_info.setdefault(
                "object_confidence_stats",
                self._empty_object_confidence_stats(),
            )

            if int(stats["samples"]) == 0:
                stats["start"] = confidence
                stats["peak"] = confidence
            else:
                stats["peak"] = max(float(stats["peak"]), confidence)

            stats["sum"] = float(stats["sum"]) + confidence
            stats["last"] = confidence
            stats["samples"] = int(stats["samples"]) + 1

    def _log_lifecycle_events(
        self,
        *,
        event_output: Mapping[str, Any],
        frame_timestamp: float,
    ) -> None:
        events = event_output.get("events", [])
        if not isinstance(events, list):
            return

        multi_snapshot = self._multi_cue_snapshot(event_output)

        for event in events:
            if not isinstance(event, Mapping):
                continue

            lifecycle = str(event.get("lifecycle", ""))
            if lifecycle not in {"START", "END"}:
                continue

            cue_key = str(event.get("cue_key", ""))
            event_start_timestamp = event.get(
                "event_start_timestamp",
                event.get("timestamp", frame_timestamp),
            )
            event_id = _event_id(
                cue_key,
                event_start_timestamp,
            )
            wall_time = _now_local_iso()

            start_info_for_record = (
                self._open_events.get(cue_key, {})
                if lifecycle == "END"
                else {}
            )
            object_confidence_summary = {}
            if isinstance(start_info_for_record, Mapping):
                stats = start_info_for_record.get(
                    "object_confidence_stats",
                    {},
                )
                if isinstance(stats, Mapping) and stats:
                    object_confidence_summary = (
                        self._finalize_object_confidence_stats(stats)
                    )

            record = {
                "record_type": "EVENT_LIFECYCLE",
                "session_id": self.session_id,
                "event_id": event_id,
                "wall_time": wall_time,
                "frame_timestamp": float(frame_timestamp),
                "lifecycle": lifecycle,
                "event": _safe_json(dict(event)),
                "object_confidence_summary": _safe_json(
                    object_confidence_summary
                ),
                "multi_cue_context": multi_snapshot,
                "rejected_event_candidates": _safe_json(
                    event_output.get(
                        "rejected_event_candidates",
                        {},
                    )
                ),
                "cross_module_validation": _safe_json(
                    event_output.get(
                        "cross_module_validation",
                        {},
                    )
                ),
            }
            self._write_jsonl(self._events_fp, record)
            self._lifecycle_counts[lifecycle] += 1

            if lifecycle == "START":
                cue_type = str(event.get("cue_type", ""))
                self._open_events[cue_key] = {
                    "event_id": event_id,
                    "cue_type": cue_type,
                    "start_wall_time": wall_time,
                    "start_event": _safe_json(dict(event)),
                    "start_multi_cue": multi_snapshot,
                    "object_confidence_stats": (
                        self._empty_object_confidence_stats()
                        if cue_type == "object"
                        else {}
                    ),
                }
                continue

            # END
            start_info = self._open_events.pop(cue_key, {})
            context = event.get("context", {})
            if not isinstance(context, Mapping):
                context = {}

            start_multi = start_info.get(
                "start_multi_cue",
                {},
            )
            if not isinstance(start_multi, Mapping):
                start_multi = {}

            end_multi = multi_snapshot

            row = {
                "session_id": self.session_id,
                "event_id": start_info.get(
                    "event_id",
                    event_id,
                ),
                "cue_key": cue_key,
                "cue_type": event.get("cue_type", ""),
                "label": event.get("label", ""),
                "source": event.get("source", ""),
                "event_start_timestamp": event.get(
                    "event_start_timestamp",
                    "",
                ),
                "event_end_timestamp": event.get(
                    "event_end_timestamp",
                    "",
                ),
                "duration_s": event.get("duration_s", ""),
                "start_wall_time": start_info.get(
                    "start_wall_time",
                    "",
                ),
                "end_wall_time": wall_time,
                "start_observed_active_keys": json.dumps(
                    start_multi.get(
                        "observed_active_event_keys",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "end_observed_active_keys": json.dumps(
                    end_multi.get(
                        "observed_active_event_keys",
                        [],
                    ),
                    ensure_ascii=False,
                ),
                "start_multi_cue": bool(
                    start_multi.get(
                        "has_multiple_active_cues",
                        False,
                    )
                ),
                "end_multi_cue": bool(
                    end_multi.get(
                        "has_multiple_active_cues",
                        False,
                    )
                ),
                "confidence": _context_scalar(
                    context,
                    "confidence",
                ),
                "object_confidence_start": (
                    object_confidence_summary.get("start", "")
                    if str(event.get("cue_type", "")) == "object"
                    else ""
                ),
                "object_confidence_peak": (
                    object_confidence_summary.get("peak", "")
                    if str(event.get("cue_type", "")) == "object"
                    else ""
                ),
                "object_confidence_mean": (
                    object_confidence_summary.get("mean", "")
                    if str(event.get("cue_type", "")) == "object"
                    else ""
                ),
                "object_confidence_last": (
                    object_confidence_summary.get("last", "")
                    if str(event.get("cue_type", "")) == "object"
                    else ""
                ),
                "object_confidence_samples": (
                    object_confidence_summary.get("samples", "")
                    if str(event.get("cue_type", "")) == "object"
                    else ""
                ),
                "event_min_confidence": _context_scalar(
                    context,
                    "event_min_confidence",
                ),
                "count": _context_scalar(
                    context,
                    "count",
                ),
                "person_count": _context_scalar(
                    context,
                    "person_count",
                ),
                "eye_status": _context_scalar(
                    context,
                    "eye_status",
                ),
                "relationship": _context_scalar(
                    context,
                    "relationship",
                ),
            }

            self._csv_writer.writerow(row)
            self._csv_fp.flush()
            self._completed_by_cue[cue_key] += 1

    def _log_filtered_transitions(
        self,
        *,
        event_output: Mapping[str, Any],
        frame_timestamp: float,
    ) -> None:
        rejected = event_output.get(
            "rejected_event_candidates",
            {},
        )
        if not isinstance(rejected, Mapping):
            rejected = {}

        current_keys = {
            str(key)
            for key, value in rejected.items()
            if isinstance(value, Mapping)
        }
        previous_keys = set(self._active_filtered.keys())

        # New filtered/rejected episode.
        for key in sorted(current_keys - previous_keys):
            detail = rejected.get(key, {})
            if not isinstance(detail, Mapping):
                continue

            wall_time = _now_local_iso()
            state = {
                "start_wall_time": wall_time,
                "start_frame_timestamp": float(frame_timestamp),
                "reason": str(detail.get("reason", "")),
                "detail": _safe_json(dict(detail)),
            }
            self._active_filtered[key] = state
            self._filtered_episode_counts[key] += 1

            self._write_jsonl(
                self._filtered_fp,
                {
                    "record_type": "FILTER_START",
                    "session_id": self.session_id,
                    "wall_time": wall_time,
                    "frame_timestamp": float(frame_timestamp),
                    "cue_key": key,
                    "reason": state["reason"],
                    "detail": state["detail"],
                },
            )

        # Rejected episode ended.
        for key in sorted(previous_keys - current_keys):
            state = self._active_filtered.pop(key)
            end_wall_time = _now_local_iso()
            start_ts = float(
                state.get(
                    "start_frame_timestamp",
                    frame_timestamp,
                )
            )

            self._write_jsonl(
                self._filtered_fp,
                {
                    "record_type": "FILTER_END",
                    "session_id": self.session_id,
                    "wall_time": end_wall_time,
                    "frame_timestamp": float(frame_timestamp),
                    "cue_key": key,
                    "reason": state.get("reason", ""),
                    "filter_start_wall_time": state.get(
                        "start_wall_time",
                        "",
                    ),
                    "filter_start_frame_timestamp": start_ts,
                    "duration_s": max(
                        0.0,
                        float(frame_timestamp) - start_ts,
                    ),
                },
            )

    def process(
        self,
        *,
        event_output: Mapping[str, Any],
        frame_timestamp: float,
    ) -> None:
        """Persist meaningful changes from one 03 event-manager output."""
        if self._closed:
            raise RuntimeError("EventLogger is already closed.")

        self._log_lifecycle_events(
            event_output=event_output,
            frame_timestamp=float(frame_timestamp),
        )
        self._update_open_object_confidence(
            event_output=event_output,
        )
        self._log_filtered_transitions(
            event_output=event_output,
            frame_timestamp=float(frame_timestamp),
        )

    def close(
        self,
        *,
        final_frame_timestamp: Optional[float] = None,
    ) -> None:
        """
        Finish the session and write a summary.

        Open events are reported as incomplete. No synthetic END is created.
        """
        if self._closed:
            return

        self.session_end_wall_time = _now_local_iso()

        # Close any currently active filtered episodes as SESSION_END records.
        for key, state in list(self._active_filtered.items()):
            self._write_jsonl(
                self._filtered_fp,
                {
                    "record_type": "FILTER_SESSION_END",
                    "session_id": self.session_id,
                    "wall_time": self.session_end_wall_time,
                    "frame_timestamp": final_frame_timestamp,
                    "cue_key": key,
                    "reason": state.get("reason", ""),
                    "filter_start_wall_time": state.get(
                        "start_wall_time",
                        "",
                    ),
                    "filter_start_frame_timestamp": state.get(
                        "start_frame_timestamp",
                        "",
                    ),
                },
            )
        self._active_filtered.clear()

        summary = {
            "session_id": self.session_id,
            "logger_version": "04_event_logger_v2",
            "session_start_wall_time": self.session_start_wall_time,
            "session_end_wall_time": self.session_end_wall_time,
            "final_frame_timestamp": final_frame_timestamp,
            "lifecycle_counts": dict(self._lifecycle_counts),
            "completed_event_counts_by_cue": dict(
                self._completed_by_cue
            ),
            "completed_event_total": int(
                sum(self._completed_by_cue.values())
            ),
            "filtered_episode_counts_by_cue": dict(
                self._filtered_episode_counts
            ),
            "open_events_at_session_end": _safe_json(
                self._open_events
            ),
            "note": (
                "Open events are not given a synthetic END. "
                "This preserves the observed event timestamps."
            ),
        }

        self.summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self._events_fp.flush()
        self._filtered_fp.flush()
        self._csv_fp.flush()

        self._events_fp.close()
        self._filtered_fp.close()
        self._csv_fp.close()

        self._closed = True

    def get_paths(self) -> Dict[str, str]:
        """Return JSON-safe session output paths."""
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "session_metadata": str(self.metadata_path),
            "events_jsonl": str(self.events_jsonl_path),
            "completed_events_csv": str(self.completed_csv_path),
            "filtered_candidates_jsonl": str(
                self.filtered_jsonl_path
            ),
            "session_summary": str(self.summary_path),
        }
