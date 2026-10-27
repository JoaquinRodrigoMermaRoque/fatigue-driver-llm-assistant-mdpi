"""Batch H8 A-E replay for the accepted INTERNAL and USB captures."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import fmean

from run_h8_ablation_replay import (
    MODES,
    OUTPUT_FIELDS,
    load_rows,
    replay_mode_from_captured_temporals,
    verify_same_frames,
    write_output,
)

SCORE_VALIDATION_TOLERANCE = 1e-8

ACCEPTED_RUNS = (
    "H8_INT_NORMAL_01",
    "H8V2_INT_PERCLOS_01",
    "H8_INT_YAWN_01",
    "H8_INT_GAZE_01",
    "H8V2_INT_NOD_01",
    "H8V2_INT_COMBINED_01",
    "H8_USB_NORMAL_01",
    "H8_USB_PERCLOS_01",
    "H8_USB_YAWN_02",
    "H8_USB_GAZE_01",
    "H8_USB_NOD_01",
    "H8_USB_COMBINED_01",
)

SUMMARY_FIELDS = (
    "feature_run_id",
    "source",
    "clip_id",
    "config_id",
    "frames",
    "duration_s",
    "score_mean",
    "score_max",
    "normal_frames",
    "suspected_frames",
    "fatigue_frames",
    "critical_frames",
    "recovery_frames",
    "state_events",
    "e_validation_max_abs_score_diff",
    "e_validation_state_mismatches",
    "e_validation_event_mismatches",
    "validation_status",
)


def validate_e(
    replay_rows: list[dict[str, object]],
) -> tuple[float, int, int]:
    score_diffs = [
        abs(float(r["score"]) - float(r["capture_score"]))
        for r in replay_rows
    ]
    state_mm = sum(
        str(r["state"]) != str(r["capture_state"])
        for r in replay_rows
    )
    event_mm = sum(
        str(r["state_event"]) != str(r["capture_state_event"])
        for r in replay_rows
    )
    return max(score_diffs, default=0.0), state_mm, event_mm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    summary = []

    for run_id in ACCEPTED_RUNS:
        feature_file = args.features_dir / f"h8_features_{run_id}.csv"
        if not feature_file.is_file():
            missing.append(str(feature_file))
            continue

        rows = load_rows(feature_file)
        by_mode = {
            mode: replay_mode_from_captured_temporals(rows, mode)
            for mode in MODES
        }
        verify_same_frames(by_mode)

        output_file = args.output_dir / f"{run_id}_AE.csv"
        write_output(output_file, by_mode)

        e_score_diff, e_state_mm, e_event_mm = validate_e(by_mode["E"])

        validation_ok = (
            e_score_diff <= SCORE_VALIDATION_TOLERANCE
            and e_state_mm == 0
            and e_event_mm == 0
        )

        duration_s = int(rows[-1]["t_ms"]) / 1000.0

        for mode in MODES:
            mode_rows = by_mode[mode]
            states = Counter(str(r["state"]) for r in mode_rows)
            events = Counter(
                str(r["state_event"])
                for r in mode_rows
                if str(r["state_event"])
            )
            summary.append(
                {
                    "feature_run_id": run_id,
                    "source": rows[0]["source"],
                    "clip_id": rows[0]["clip_id"],
                    "config_id": mode,
                    "frames": len(mode_rows),
                    "duration_s": f"{duration_s:.3f}",
                    "score_mean": f"{fmean(float(r['score']) for r in mode_rows):.8f}",
                    "score_max": f"{max(float(r['score']) for r in mode_rows):.8f}",
                    "normal_frames": states["NORMAL"],
                    "suspected_frames": states["SUSPECTED"],
                    "fatigue_frames": states["FATIGUE"],
                    "critical_frames": states["CRITICAL"],
                    "recovery_frames": states["RECOVERY"],
                    "state_events": ";".join(
                        f"{name}:{count}"
                        for name, count in sorted(events.items())
                    ),
                    "e_validation_max_abs_score_diff": f"{e_score_diff:.8f}",
                    "e_validation_state_mismatches": e_state_mm,
                    "e_validation_event_mismatches": e_event_mm,
                    "validation_status": (
                        "VALIDATED_EXACT"
                        if validation_ok
                        else "REQUIRES_COMPLETE_FRAME_RECAPTURE"
                    ),
                }
            )

        status = "OK" if validation_ok else "FAIL"
        print(
            f"{status} {run_id}: frames={len(rows)} "
            f"E_diff={e_score_diff:.8f} "
            f"E_state_mm={e_state_mm} "
            f"E_event_mm={e_event_mm}"
        )

    if missing:
        print()
        print("FALTAN ARCHIVOS:")
        for item in missing:
            print(" -", item)
        raise SystemExit(2)

    summary_path = args.output_dir / "H8_AE_SUMMARY.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    failed_runs = sorted({
        row["feature_run_id"]
        for row in summary
        if row["validation_status"] != "VALIDATED_EXACT"
    })

    print()
    print("Runs procesados =", len(ACCEPTED_RUNS))
    print("Filas resumen =", len(summary))
    print("Runs VALIDATED_EXACT =", len(ACCEPTED_RUNS) - len(failed_runs))
    print("Runs a recapturar =", len(failed_runs))
    for run_id in failed_runs:
        print("RECAPTURE", run_id)
    print("Resumen =", summary_path)


if __name__ == "__main__":
    main()
