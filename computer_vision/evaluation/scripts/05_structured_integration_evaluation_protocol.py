"""Interactive helper for the frozen 30-scenario, 90-trial evaluation.

The CSV trial log remains the source of truth. This v2 helper preserves
earlier-session rows during resume, records wall-clock cue boundaries, and does
not score model results automatically.
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    import winsound
except ImportError:
    winsound = None


INSTRUCTIONS: Dict[str, str] = {
    "N01": "Remain seated normally with forward head, centre gaze, and no target object visible.",
    "H01": "Turn your head LEFT naturally and hold it. Keep your eyes natural.",
    "H02": "Turn your head RIGHT naturally and hold it. Keep your eyes natural.",
    "H03": "Raise/tilt your head UP naturally and hold it.",
    "H04": "Lower/tilt your head DOWN naturally and hold it.",
    "G01": "Keep your head forward and move only your eyes LEFT.",
    "G02": "Keep your head forward and move only your eyes RIGHT.",
    "G03": "Keep your head forward and move only your eyes UP.",
    "G04": "Keep your head forward and move only your eyes DOWN.",
    "O01": "Show the same PHONE in a natural assessment-relevant position.",
    "O02": "Show the OPEN LAPTOP in a natural, repeatable position.",
    "O03": "Show the same BOOK / NOTES / PAPER clearly.",
    "O04": "Show the same CALCULATOR clearly.",
    "O05": "Show the same WATCH clearly, preferably worn in a natural position.",
    "O06": "Show/wear an OVER-EAR HEADPHONE or HEADSET clearly. Defer if unavailable.",
    "P01": "Fully leave the webcam frame. Your face must not remain visible.",
    "P02": "Have a second person become clearly visible. Defer if unavailable.",
    "M01": "Turn your head LEFT while holding the PHONE visibly. Keep both cues present.",
    "M02": "Keep head forward, look RIGHT with your eyes, and show BOOK / NOTES at the same time.",
    "R01": "Remain neutral and blink naturally several times. Do not intentionally look away.",
    "R02": "Keep head forward and briefly close your eyes during the cue window.",
    "R03": "Make a large head turn so one eye may become unavailable; observe gaze reliability/suppression.",
    "R04": "Keep your face clearly visible while partially hiding/occluding your body so YOLO may lose the person box.",
    "K01": "Present the CLOSED / FLAT LAPTOP at the planned difficult angle.",
    "K02": "Present the PHONE in a thin side view.",
    "K03": "Wear/show SMALL EARBUDS normally. This is diagnostic; misses are a known limitation.",
    "K04": "Wear/show WIRED EARPHONES normally. This is diagnostic.",
    "K05": "Use the known background condition with only ONE real person visible; observe false extra-person detections.",
    "K06": "Place/show a person at the planned farther distance; observe person detection and any false NO_PERSON state.",
    "K07": "Perform the known large head-turn case; observe pose flips, direction reversal, and gaze reliability.",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def beep(freq: int = 1000, duration_ms: int = 180) -> None:
    if winsound is not None:
        try:
            winsound.Beep(freq, duration_ms)
            return
        except Exception:
            pass
    print("\a", end="", flush=True)


def countdown(seconds: int, label: str) -> None:
    if seconds <= 0:
        return
    print(f"{label}: {seconds}s")
    for remaining in range(seconds, 0, -1):
        print(f"\r  {remaining:2d} ", end="", flush=True)
        time.sleep(1)
    print("\r   0 ")


def load_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames or not rows:
        raise RuntimeError(f"Template is empty or invalid: {path}")
    return fieldnames, rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict, key: str, default: str = "") -> str:
    v = row.get(key, default)
    return "" if v is None else str(v).strip()


def int_value(row: dict, key: str, default: int = 0) -> int:
    v = value(row, key)
    if not v:
        return default
    return int(float(v))


def set_value_if_column(row: dict, key: str, new_value: Optional[str]) -> None:
    if key in row and new_value is not None:
        row[key] = new_value


def selected(row: dict, block: str, scenario_ids: set[str]) -> bool:
    sid = value(row, "scenario_id").upper()
    group = value(row, "scenario_group")

    if scenario_ids and sid not in scenario_ids:
        return False

    if block == "all":
        return True
    if block == "core":
        return group.startswith("core-")
    if block == "reliability":
        return group == "reliability"
    if block == "known-failure":
        return group == "known-failure"
    return False


def attach_current_session_metadata(row: dict, args, session_start: str) -> None:
    """
    IMPORTANT:
    Called ONLY for a row that is being handled in the CURRENT run.
    Existing VALID / INVALID / DEFERRED rows from earlier sessions are not touched.
    """
    set_value_if_column(row, "session_id", args.session_id or "")
    set_value_if_column(row, "participant_id", args.participant_id)
    set_value_if_column(row, "session_start_wall_iso", session_start)
    set_value_if_column(row, "runtime_commit", args.runtime_commit or "")
    set_value_if_column(row, "runtime_config_reference",
                        args.runtime_config_reference or "")
    set_value_if_column(row, "checkpoint_id", "5e")
    set_value_if_column(row, "runtime_mode", "async_latest_frame")
    set_value_if_column(row, "camera_id", args.camera_id)
    set_value_if_column(row, "camera_resolution", args.camera_resolution)
    set_value_if_column(row, "lighting_condition", args.lighting_condition or "")
    set_value_if_column(row, "background_condition", args.background_condition or "")
    set_value_if_column(row, "participant_distance", args.participant_distance or "")
    set_value_if_column(row, "glasses_or_occlusion", args.glasses_or_occlusion or "")
    set_value_if_column(row, "calibration_outcome", args.calibration_outcome or "")
    set_value_if_column(row, "output_directory",
                        args.integration_output_directory or "")

    # Never pretend wall-clock/runtime alignment exists.
    if "clock_alignment_available" in row:
        row["clock_alignment_available"] = "FALSE"
    for key in (
        "clock_alignment_method",
        "alignment_reference_wall_iso",
        "alignment_reference_runtime_s",
        "boundary_start_error_s",
        "boundary_end_error_s",
    ):
        if key in row:
            row[key] = ""


def clear_trial_execution_fields(row: dict) -> None:
    """
    Re-open a previously DEFERRED/SKIPPED/INVALID row as a new trial attempt.
    Preserve scenario definition columns, but clear old execution/session results.
    """
    clear_keys = [
        "session_id",
        "participant_id",
        "session_start_wall_iso",
        "runtime_commit",
        "runtime_config_reference",
        "checkpoint_id",
        "runtime_mode",
        "camera_id",
        "camera_resolution",
        "lighting_condition",
        "background_condition",
        "participant_distance",
        "glasses_or_occlusion",
        "calibration_outcome",
        "output_directory",
        "cue_onset_wall_iso",
        "cue_offset_wall_iso",
        "observed_cue_duration_s",
        "clock_alignment_available",
        "clock_alignment_method",
        "alignment_reference_wall_iso",
        "alignment_reference_runtime_s",
        "actual_event",
        "actual_event_family",
        "actual_event_label",
        "event_start_runtime_s",
        "event_end_runtime_s",
        "start_record_runtime_s",
        "end_record_runtime_s",
        "start_confirmation_delay_s",
        "end_emission_delay_s",
        "boundary_start_error_s",
        "boundary_end_error_s",
        "duration_error_s",
        "expected_event_detected",
        "missed_event",
        "false_alert",
        "duplicate_event",
        "lifecycle_failure",
        "logging_failure",
        "eligibility_failure",
        "deviation_code",
        "notes",
    ]
    for key in clear_keys:
        if key in row:
            row[key] = ""
    row["round_status"] = "PLANNED"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final end-to-end evaluation helper. "
            "The CSV trial log remains the source of truth. "
            "v2 preserves earlier-session rows during resume."
        )
    )
    parser.add_argument("--trial-log", type=Path, required=True,
                        help="Final end_to_end_trial_log.csv template.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Filled trial-log CSV.")
    parser.add_argument(
        "--block",
        choices=["core", "reliability", "known-failure", "all"],
        default="core",
    )
    parser.add_argument(
        "--scenario-ids",
        default="",
        help="Optional comma-separated scenario IDs, e.g. N01,H01,H02 or P02.",
    )
    parser.add_argument("--session-id", default="",
                        help="CURRENT integration logger session ID.")
    parser.add_argument("--participant-id", default="P01")
    parser.add_argument("--runtime-commit", default="")
    parser.add_argument("--runtime-config-reference", default="")
    parser.add_argument("--camera-id", default="0")
    parser.add_argument("--camera-resolution", default="1280x720")
    parser.add_argument("--lighting-condition", default="")
    parser.add_argument("--background-condition", default="")
    parser.add_argument("--participant-distance", default="")
    parser.add_argument("--glasses-or-occlusion", default="")
    parser.add_argument("--calibration-outcome", default="PASS")
    parser.add_argument("--integration-output-directory", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Load --output if it exists. Existing non-PLANNED rows are preserved "
            "and skipped; current-session metadata is written only to newly handled rows."
        ),
    )
    parser.add_argument(
        "--reopen-status",
        default="",
        help=(
            "Optional comma-separated statuses to reopen for the selected scenarios, "
            "e.g. DEFERRED. Use this only when intentionally supplementing a prior trial."
        ),
    )
    args = parser.parse_args()

    scenario_ids = {
        x.strip().upper()
        for x in args.scenario_ids.split(",")
        if x.strip()
    }
    reopen_statuses = {
        x.strip().upper()
        for x in args.reopen_status.split(",")
        if x.strip()
    }

    source = args.output if args.resume and args.output.exists() else args.trial_log
    fieldnames, rows = load_rows(source)

    required = {
        "trial_id", "scenario_id", "scenario_group", "scenario_name", "round",
        "round_status", "pre_neutral_s", "cue_duration_s", "recovery_s",
        "cue_onset_wall_iso", "cue_offset_wall_iso", "observed_cue_duration_s",
        "notes",
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Optional intentional reopening, scoped ONLY to selected rows.
    if reopen_statuses:
        for row in rows:
            if not selected(row, args.block, scenario_ids):
                continue
            status = value(row, "round_status", "PLANNED").upper()
            if status in reopen_statuses:
                clear_trial_execution_fields(row)

    pending_indices = [
        i for i, row in enumerate(rows)
        if selected(row, args.block, scenario_ids)
        and value(row, "round_status", "PLANNED").upper() == "PLANNED"
    ]

    print("=" * 78)
    print("FINAL END-TO-END STRUCTURED EVALUATION — v2")
    print("=" * 78)
    print(f"Loaded from      : {source}")
    print(f"Output log       : {args.output}")
    print(f"Block            : {args.block}")
    print(f"Scenario filter  : {', '.join(sorted(scenario_ids)) if scenario_ids else 'none'}")
    print(f"Current session  : {args.session_id or '[not supplied]'}")
    print(f"Pending trials   : {len(pending_indices)}")
    print()
    print("Resume safety:")
    print("  - Earlier completed/deferred/invalid rows are NOT overwritten.")
    print("  - Current session metadata is written only to rows handled now.")
    print("  - Use --reopen-status DEFERRED only for an intentional supplementary run.")
    print()

    if not pending_indices:
        print("No selected PLANNED trials remain.")
        write_rows(args.output, fieldnames, rows)
        return

    session_start = now_iso()

    print("Before continuing:")
    print("  1. Start canonical run_integrated_demo.py in another terminal.")
    print("  2. Complete calibration and confirm READY.")
    print("  3. Keep checkpoint=5e and async runtime fixed.")
    print("  4. Do not silently repeat a valid model failure.")
    input("\nPress Enter when the CURRENT integrated runtime is READY... ")

    for position, idx in enumerate(pending_indices, 1):
        row = rows[idx]
        sid = value(row, "scenario_id").upper()
        trial_id = value(row, "trial_id")
        scenario_name = value(row, "scenario_name")
        round_no = value(row, "round")
        expected = (
            value(row, "expected_event_label")
            or value(row, "expected_handling")
        )
        secondary = value(row, "expected_secondary_event_label")
        pre = int_value(row, "pre_neutral_s")
        cue = int_value(row, "cue_duration_s")
        recovery = int_value(row, "recovery_s")

        print("\n" + "-" * 78)
        print(f"[{position}/{len(pending_indices)}] {trial_id}")
        print(f"Scenario : {sid} — {scenario_name} | round {round_no}")
        print(f"Expected : {expected or 'See protocol'}")
        if secondary:
            print(f"Secondary: {secondary}")
        print(f"Action   : {INSTRUCTIONS.get(sid, scenario_name)}")
        print(f"Timing   : pre-neutral {pre}s | cue {cue}s | recovery {recovery}s")

        cmd = input(
            "Enter=run | D=defer | I=invalid | S=skip/administrative: "
        ).strip().lower()

        # Attach current-session metadata only NOW, once this row is actually handled.
        attach_current_session_metadata(row, args, session_start)

        if cmd in {"d", "i", "s"}:
            status = {"d": "DEFERRED", "i": "INVALID", "s": "SKIPPED"}[cmd]
            row["round_status"] = status
            note = input(f"Reason for {status}: ").strip()
            if note:
                row["notes"] = note
            write_rows(args.output, fieldnames, rows)
            print(f"{trial_id}: {status}")
            continue

        if pre > 0:
            print("\nReturn to the required neutral state.")
            countdown(pre, "PRE-NEUTRAL")

        beep(850, 250)
        onset = now_iso()
        row["cue_onset_wall_iso"] = onset
        print("\n>>> CUE START <<<")
        print(INSTRUCTIONS.get(sid, scenario_name))
        countdown(cue, "CUE")

        beep(1250, 300)
        offset = now_iso()
        row["cue_offset_wall_iso"] = offset

        try:
            t0 = datetime.fromisoformat(onset)
            t1 = datetime.fromisoformat(offset)
            row["observed_cue_duration_s"] = f"{(t1 - t0).total_seconds():.3f}"
        except Exception:
            row["observed_cue_duration_s"] = ""

        print(">>> CUE END — return to neutral <<<")

        if recovery > 0:
            countdown(recovery, "RECOVERY")

        row["round_status"] = "VALID"
        note = input(
            "Optional execution note (blank if normal; do NOT score model yet): "
        ).strip()
        if note:
            row["notes"] = note

        write_rows(args.output, fieldnames, rows)
        print(f"Saved {trial_id} as VALID.")

        if recovery > 0:
            input(
                "Confirm the previous event has ended in the UI/logger, "
                "then press Enter for the next trial... "
            )

    print("\n" + "=" * 78)
    print("SELECTED BLOCK COMPLETE")
    print("=" * 78)
    print(f"Filled trial log: {args.output}")
    print("Earlier-session rows were preserved.")
    print("Do not manually score actual_event / PASS / FAIL yet.")
    print("Preserve the matching CURRENT integration-session files for analysis.")


if __name__ == "__main__":
    main()
