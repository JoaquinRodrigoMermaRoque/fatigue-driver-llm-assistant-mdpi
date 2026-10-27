"""Audit whether existing H8 captures are suitable for H2 calibration replay.

H2 must recompute gaze/pose temporal cues from raw features, so captured temporal
outputs cannot be reused. This audit replays each final H1 run in personalized
mode E from raw cues and compares it against the Android-captured mode E output.

Runs that reproduce Android within CSV precision are eligible for H2 without
recapture. Runs that do not reproduce exactly are NOT automatically recaptured;
we first inspect whether enough NORMAL/GAZE/NOD/COMBINED evidence remains.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from run_h8_ablation_replay import load_rows, replay_mode

FINAL_RUNS = (
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

SCENARIO = {
    "H8_INT_NORMAL_01": "NORMAL",
    "H8V2_INT_PERCLOS_01": "PERCLOS",
    "H8_INT_YAWN_01": "YAWN",
    "H8_INT_GAZE_01": "GAZE",
    "H8V2_INT_NOD_01": "NOD",
    "H8V2_INT_COMBINED_01": "COMBINED",
    "H8_USB_NORMAL_01": "NORMAL",
    "H8_USB_PERCLOS_01": "PERCLOS",
    "H8_USB_YAWN_02": "YAWN",
    "H8_USB_GAZE_01": "GAZE",
    "H8_USB_NOD_01": "NOD",
    "H8_USB_COMBINED_01": "COMBINED",
}

TOL = 1e-8
FIELDS = (
    "run_id",
    "source",
    "scenario",
    "schema_version",
    "frames",
    "max_abs_score_diff",
    "state_mismatches",
    "event_mismatches",
    "max_logged_gap_ms",
    "gaps_gt_250ms",
    "status",
)


def validate(replay: list[dict[str, object]]) -> tuple[float, int, int]:
    diffs = [
        abs(float(r["score"]) - float(r["capture_score"]))
        for r in replay
    ]
    state_mm = sum(
        str(r["state"]) != str(r["capture_state"])
        for r in replay
    )
    event_mm = sum(
        str(r["state_event"]) != str(r["capture_state_event"])
        for r in replay
    )
    return max(diffs, default=0.0), state_mm, event_mm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    out = []
    missing = []

    for run_id in FINAL_RUNS:
        file = args.features_dir / f"h8_features_{run_id}.csv"
        if not file.is_file():
            missing.append(str(file))
            continue

        rows = load_rows(file)
        replay = replay_mode(rows, "E")
        score_diff, state_mm, event_mm = validate(replay)

        ts = [int(float(r["t_ms"])) for r in rows]
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        max_gap = max(gaps, default=0)
        gaps250 = sum(g > 250 for g in gaps)

        schema_values = sorted({
            str(r.get("schema_version", "")).strip() or "1"
            for r in rows
        })
        schema = ";".join(schema_values)

        ok = (
            score_diff <= TOL
            and state_mm == 0
            and event_mm == 0
        )
        status = "ELIGIBLE_H2_EXACT" if ok else "NOT_EXACT_RAW_REPLAY"

        out.append(
            {
                "run_id": run_id,
                "source": rows[0]["source"],
                "scenario": SCENARIO[run_id],
                "schema_version": schema,
                "frames": len(rows),
                "max_abs_score_diff": f"{score_diff:.8f}",
                "state_mismatches": state_mm,
                "event_mismatches": event_mm,
                "max_logged_gap_ms": max_gap,
                "gaps_gt_250ms": gaps250,
                "status": status,
            }
        )

        print(
            f"{'OK' if ok else 'FAIL'} {run_id}: "
            f"scenario={SCENARIO[run_id]} schema={schema} "
            f"score_diff={score_diff:.8f} "
            f"state_mm={state_mm} event_mm={event_mm} "
            f"max_gap_ms={max_gap} gaps_gt_250={gaps250}"
        )

    if missing:
        print()
        print("MISSING FILES:")
        for item in missing:
            print(" -", item)
        raise SystemExit(2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)

    eligible = [r for r in out if r["status"] == "ELIGIBLE_H2_EXACT"]
    print()
    print("Runs auditados =", len(out))
    print("H2 exact eligible =", len(eligible))
    print("H2 not exact =", len(out) - len(eligible))
    print("Por escenario elegible:")
    for scenario in ("NORMAL", "PERCLOS", "YAWN", "GAZE", "NOD", "COMBINED"):
        items = [
            r["run_id"] for r in eligible
            if r["scenario"] == scenario
        ]
        print(f"  {scenario}: {', '.join(items) if items else 'NONE'}")
    print("Audit CSV =", args.output)


if __name__ == "__main__":
    main()
