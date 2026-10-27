"""Descriptive H1 ablation summary from H8_AE_SUMMARY.csv.

This produces only confirmed descriptive outputs from the validated A-E replay.
It intentionally does NOT compute precision/recall/F1 because frame-level
ground-truth intervals have not yet been annotated.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

STATE_ORDER = {
    "NORMAL": 0,
    "SUSPECTED": 1,
    "RECOVERY": 2,
    "FATIGUE": 3,
    "CRITICAL": 4,
}

STATE_COLUMNS = {
    "NORMAL": "normal_frames",
    "SUSPECTED": "suspected_frames",
    "FATIGUE": "fatigue_frames",
    "CRITICAL": "critical_frames",
    "RECOVERY": "recovery_frames",
}

MODE_SUMMARY_FIELDS = (
    "config_id",
    "runs",
    "frames",
    "total_duration_s",
    "frame_weighted_score_mean",
    "max_score",
    "runs_with_non_normal_state",
    "normal_runs_with_false_alert",
    "normal_runs_total",
    "normal_frames",
    "suspected_frames",
    "fatigue_frames",
    "critical_frames",
    "recovery_frames",
)

RUN_MATRIX_FIELDS = (
    "feature_run_id",
    "source",
    "clip_id",
    "config_id",
    "frames",
    "duration_s",
    "score_mean",
    "score_max",
    "max_state",
    "state_events",
)


def max_state(row: dict[str, str]) -> str:
    present = []
    for state, column in STATE_COLUMNS.items():
        if int(row[column]) > 0:
            present.append(state)
    return max(present, key=lambda s: STATE_ORDER[s]) if present else "NORMAL"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    with args.summary.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 60:
        raise RuntimeError(f"Se esperaban 60 filas (12 runs x 5 modos), hay {len(rows)}")

    bad = [
        r for r in rows
        if r.get("validation_status") != "VALIDATED_EXACT"
    ]
    if bad:
        raise RuntimeError(
            "Hay filas no validadas en el resumen final: "
            + ", ".join(sorted({r["feature_run_id"] for r in bad}))
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_matrix = []
    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        by_mode[row["config_id"]].append(row)
        run_matrix.append(
            {
                "feature_run_id": row["feature_run_id"],
                "source": row["source"],
                "clip_id": row["clip_id"],
                "config_id": row["config_id"],
                "frames": row["frames"],
                "duration_s": row["duration_s"],
                "score_mean": row["score_mean"],
                "score_max": row["score_max"],
                "max_state": max_state(row),
                "state_events": row["state_events"],
            }
        )

    mode_summary = []

    for mode in "ABCDE":
        mrows = by_mode[mode]
        frames = sum(int(r["frames"]) for r in mrows)
        total_duration = sum(float(r["duration_s"]) for r in mrows)
        weighted_score = (
            sum(float(r["score_mean"]) * int(r["frames"]) for r in mrows)
            / frames
        )
        max_score_value = max(float(r["score_max"]) for r in mrows)
        non_normal_runs = sum(max_state(r) != "NORMAL" for r in mrows)

        normal_runs = [r for r in mrows if r["clip_id"].startswith("normal")]
        normal_false_alerts = sum(max_state(r) != "NORMAL" for r in normal_runs)

        state_totals = {
            state: sum(int(r[column]) for r in mrows)
            for state, column in STATE_COLUMNS.items()
        }

        mode_summary.append(
            {
                "config_id": mode,
                "runs": len(mrows),
                "frames": frames,
                "total_duration_s": f"{total_duration:.3f}",
                "frame_weighted_score_mean": f"{weighted_score:.6f}",
                "max_score": f"{max_score_value:.6f}",
                "runs_with_non_normal_state": non_normal_runs,
                "normal_runs_with_false_alert": normal_false_alerts,
                "normal_runs_total": len(normal_runs),
                "normal_frames": state_totals["NORMAL"],
                "suspected_frames": state_totals["SUSPECTED"],
                "fatigue_frames": state_totals["FATIGUE"],
                "critical_frames": state_totals["CRITICAL"],
                "recovery_frames": state_totals["RECOVERY"],
            }
        )

    mode_path = args.output_dir / "H8_H1_MODE_SUMMARY.csv"
    with mode_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MODE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(mode_summary)

    matrix_path = args.output_dir / "H8_H1_RUN_MODE_MATRIX.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_MATRIX_FIELDS)
        writer.writeheader()
        writer.writerows(run_matrix)

    print("H1 DESCRIPTIVE SUMMARY (validated replay only)")
    print()
    print(
        f"{'Mode':<5} {'Runs alert':>10} {'Normal FP':>10} "
        f"{'Score mean':>11} {'Score max':>10} "
        f"{'NORMAL':>8} {'SUSP':>8} {'FATIGUE':>8} {'CRIT':>8}"
    )
    for r in mode_summary:
        print(
            f"{r['config_id']:<5} "
            f"{r['runs_with_non_normal_state']:>10} "
            f"{r['normal_runs_with_false_alert']:>10} "
            f"{float(r['frame_weighted_score_mean']):>11.4f} "
            f"{float(r['max_score']):>10.4f} "
            f"{r['normal_frames']:>8} "
            f"{r['suspected_frames']:>8} "
            f"{r['fatigue_frames']:>8} "
            f"{r['critical_frames']:>8}"
        )

    print()
    print("RUN x MODE MAX STATE")
    for run_id in sorted({r["feature_run_id"] for r in run_matrix}):
        selected = [r for r in run_matrix if r["feature_run_id"] == run_id]
        states = " ".join(
            f"{r['config_id']}={r['max_state']}"
            for r in sorted(selected, key=lambda x: x["config_id"])
        )
        print(run_id, states)

    print()
    print("Mode summary =", mode_path)
    print("Run-mode matrix =", matrix_path)
    print()
    print(
        "NOTE: precision/recall/F1 are not reported here because "
        "frame-level ground-truth intervals have not been annotated."
    )


if __name__ == "__main__":
    main()
