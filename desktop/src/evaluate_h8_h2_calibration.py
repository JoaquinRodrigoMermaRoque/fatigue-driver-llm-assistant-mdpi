"""Evaluate H8 H2: fixed global baseline vs personalized calibration.

Uses only runs marked ELIGIBLE_H2_EXACT by audit_h8_h2_raw_replay.py.
The personalized raw replay is revalidated against captured Android output
before any comparison is accepted.

Calibration affects gaze center and head-pose baselines only. EAR/MAR/PERCLOS
thresholds remain identical in both conditions, matching the Android build.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from run_h8_ablation_replay import load_rows, replay_mode

POSITIVE_STATES = {"SUSPECTED", "FATIGUE", "CRITICAL", "RECOVERY"}
TOL = 1e-8

RUN_FIELDS = (
    "run_id",
    "source",
    "scenario",
    "calibration_mode",
    "frames",
    "duration_s",
    "score_mean",
    "score_max",
    "max_state",
    "predicted_positive",
    "max_gaze_away_s",
    "gaze_detected",
    "nod_event_frames",
    "nod_detected",
    "personalized_validation_score_diff",
    "personalized_validation_state_mismatches",
    "personalized_validation_event_mismatches",
)

METRIC_FIELDS = (
    "evaluation_set",
    "calibration_mode",
    "runs",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1_positive",
    "f1_negative",
    "macro_f1",
    "balanced_accuracy",
    "missed_positive_scenarios",
    "false_positive_scenarios",
)

CUE_FIELDS = (
    "cue",
    "calibration_mode",
    "positive_runs",
    "detected_positive_runs",
    "missed_positive_runs",
    "normal_control_false_activation",
    "recall",
)


STATE_ORDER = {
    "NORMAL": 0,
    "SUSPECTED": 1,
    "RECOVERY": 2,
    "FATIGUE": 3,
    "CRITICAL": 4,
}


def safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def f1(p: float, r: float) -> float:
    return safe_div(2.0 * p * r, p + r)


def max_state(rows: list[dict[str, object]]) -> str:
    states = {str(r["state"]) for r in rows}
    return max(states, key=lambda s: STATE_ORDER[s])


def validate_personalized(
    rows: list[dict[str, object]],
) -> tuple[float, int, int]:
    score_diff = max(
        (
            abs(float(r["score"]) - float(r["capture_score"]))
            for r in rows
        ),
        default=0.0,
    )
    state_mm = sum(
        str(r["state"]) != str(r["capture_state"])
        for r in rows
    )
    event_mm = sum(
        str(r["state_event"]) != str(r["capture_state_event"])
        for r in rows
    )
    return score_diff, state_mm, event_mm


def scenario_metrics(
    records: list[dict[str, object]],
    calibration_mode: str,
    evaluation_set: str,
) -> dict[str, object]:
    m = [
        r for r in records
        if r["calibration_mode"] == calibration_mode
    ]
    tp = sum(
        int(r["ground_truth_positive"]) == 1
        and int(r["predicted_positive"]) == 1
        for r in m
    )
    tn = sum(
        int(r["ground_truth_positive"]) == 0
        and int(r["predicted_positive"]) == 0
        for r in m
    )
    fp = sum(
        int(r["ground_truth_positive"]) == 0
        and int(r["predicted_positive"]) == 1
        for r in m
    )
    fn = sum(
        int(r["ground_truth_positive"]) == 1
        and int(r["predicted_positive"]) == 0
        for r in m
    )
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1_pos = f1(precision, recall)
    p_neg = safe_div(tn, tn + fn)
    r_neg = specificity
    f1_neg = f1(p_neg, r_neg)

    return {
        "evaluation_set": evaluation_set,
        "calibration_mode": calibration_mode,
        "runs": len(m),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": f"{accuracy:.6f}",
        "precision": f"{precision:.6f}",
        "recall_sensitivity": f"{recall:.6f}",
        "specificity": f"{specificity:.6f}",
        "f1_positive": f"{f1_pos:.6f}",
        "f1_negative": f"{f1_neg:.6f}",
        "macro_f1": f"{((f1_pos + f1_neg) / 2.0):.6f}",
        "balanced_accuracy": f"{((recall + specificity) / 2.0):.6f}",
        "missed_positive_scenarios": fn,
        "false_positive_scenarios": fp,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    with args.audit.open(encoding="utf-8", newline="") as f:
        audit = list(csv.DictReader(f))
    eligible = [
        r for r in audit
        if r["status"] == "ELIGIBLE_H2_EXACT"
    ]
    if not eligible:
        raise RuntimeError("No ELIGIBLE_H2_EXACT runs in audit")

    with args.ground_truth.open(encoding="utf-8", newline="") as f:
        gt_rows = list(csv.DictReader(f))
    gt = {r["run_id"]: r for r in gt_rows}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    scenario_records: list[dict[str, object]] = []

    for item in eligible:
        run_id = item["run_id"]
        feature_path = args.features_dir / f"h8_features_{run_id}.csv"
        rows = load_rows(feature_path)

        personalized = replay_mode(
            rows, "E", calibration_mode="personalized"
        )
        global_replay = replay_mode(
            rows, "E", calibration_mode="global"
        )

        vdiff, vstate, vevent = validate_personalized(personalized)
        if vdiff > TOL or vstate != 0 or vevent != 0:
            raise RuntimeError(
                f"Personalized replay no longer exact for {run_id}: "
                f"diff={vdiff} state_mm={vstate} event_mm={vevent}"
            )

        for calibration_mode, replay in (
            ("personalized", personalized),
            ("global", global_replay),
        ):
            state = max_state(replay)
            pred = int(state in POSITIVE_STATES)
            duration_s = float(replay[-1]["t_ms"]) / 1000.0
            max_gaze = max(float(r["gaze_away_s"]) for r in replay)
            nod_event_frames = sum(
                int(r["nod_event"]) != 0 for r in replay
            )
            score_mean = fmean(float(r["score"]) for r in replay)
            score_max = max(float(r["score"]) for r in replay)

            run_rows.append(
                {
                    "run_id": run_id,
                    "source": item["source"],
                    "scenario": item["scenario"],
                    "calibration_mode": calibration_mode,
                    "frames": len(replay),
                    "duration_s": f"{duration_s:.3f}",
                    "score_mean": f"{score_mean:.6f}",
                    "score_max": f"{score_max:.6f}",
                    "max_state": state,
                    "predicted_positive": pred,
                    "max_gaze_away_s": f"{max_gaze:.6f}",
                    "gaze_detected": int(max_gaze >= 1.5),
                    "nod_event_frames": nod_event_frames,
                    "nod_detected": int(nod_event_frames > 0),
                    "personalized_validation_score_diff": f"{vdiff:.8f}",
                    "personalized_validation_state_mismatches": vstate,
                    "personalized_validation_event_mismatches": vevent,
                }
            )

            g = gt[run_id]
            scenario_records.append(
                {
                    "run_id": run_id,
                    "source": item["source"],
                    "scenario": item["scenario"],
                    "calibration_mode": calibration_mode,
                    "ground_truth_positive": int(
                        g["ground_truth_positive"]
                    ),
                    "predicted_positive": pred,
                    "gaze_detected": int(max_gaze >= 1.5),
                    "nod_detected": int(nod_event_frames > 0),
                }
            )

    # Full exact subset: all runs whose personalized raw replay is exact.
    metrics = []
    for mode in ("global", "personalized"):
        metrics.append(
            scenario_metrics(
                scenario_records,
                calibration_mode=mode,
                evaluation_set="all_exact_runs",
            )
        )

    # Calibration-sensitive subset. PERCLOS/YAWN are excluded because current
    # personalization does not modify EAR/MAR thresholds.
    sensitive_scenarios = {"NORMAL", "GAZE", "NOD", "COMBINED"}
    sensitive = [
        r for r in scenario_records
        if r["scenario"] in sensitive_scenarios
    ]
    for mode in ("global", "personalized"):
        metrics.append(
            scenario_metrics(
                sensitive,
                calibration_mode=mode,
                evaluation_set="calibration_sensitive",
            )
        )

    # Cue-level presence/absence: target scenario detection plus NORMAL-control
    # false activation. This evaluates what calibration directly changes.
    cue_metrics = []
    for cue, positive_scenarios, field in (
        ("gaze", {"GAZE", "COMBINED"}, "gaze_detected"),
        ("nod", {"NOD", "COMBINED"}, "nod_detected"),
    ):
        for mode in ("global", "personalized"):
            subset = [
                r for r in scenario_records
                if r["calibration_mode"] == mode
            ]
            positives = [
                r for r in subset
                if r["scenario"] in positive_scenarios
            ]
            detected = sum(int(r[field]) == 1 for r in positives)
            normal = [
                r for r in subset
                if r["scenario"] == "NORMAL"
            ]
            false_normal = sum(int(r[field]) == 1 for r in normal)
            cue_metrics.append(
                {
                    "cue": cue,
                    "calibration_mode": mode,
                    "positive_runs": len(positives),
                    "detected_positive_runs": detected,
                    "missed_positive_runs": len(positives) - detected,
                    "normal_control_false_activation": false_normal,
                    "recall": f"{safe_div(detected, len(positives)):.6f}",
                }
            )

    run_path = args.output_dir / "H8_H2_CALIBRATION_RUNS.csv"
    metric_path = args.output_dir / "H8_H2_CALIBRATION_METRICS.csv"
    cue_path = args.output_dir / "H8_H2_CALIBRATION_CUE_METRICS.csv"

    with run_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        w.writerows(run_rows)

    with metric_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        w.writerows(metrics)

    with cue_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CUE_FIELDS)
        w.writeheader()
        w.writerows(cue_metrics)

    print("H2 GLOBAL vs PERSONALIZED CALIBRATION")
    print()
    print("Eligible exact runs =", len(eligible))
    print()
    print(
        f"{'Set':<24} {'Mode':<12} {'TP':>3} {'TN':>3} "
        f"{'FP':>3} {'FN':>3} {'Recall':>7} {'Spec':>7} "
        f"{'MacroF1':>8}"
    )
    for r in metrics:
        print(
            f"{r['evaluation_set']:<24} "
            f"{r['calibration_mode']:<12} "
            f"{r['tp']:>3} {r['tn']:>3} {r['fp']:>3} {r['fn']:>3} "
            f"{float(r['recall_sensitivity']):>7.3f} "
            f"{float(r['specificity']):>7.3f} "
            f"{float(r['macro_f1']):>8.3f}"
        )

    print()
    print("Cue-level calibration-sensitive detection")
    print(
        f"{'Cue':<6} {'Mode':<12} {'Detected':>9} {'Positive':>9} "
        f"{'Missed':>7} {'NormalFP':>8} {'Recall':>7}"
    )
    for r in cue_metrics:
        print(
            f"{r['cue']:<6} {r['calibration_mode']:<12} "
            f"{r['detected_positive_runs']:>9} "
            f"{r['positive_runs']:>9} "
            f"{r['missed_positive_runs']:>7} "
            f"{r['normal_control_false_activation']:>8} "
            f"{float(r['recall']):>7.3f}"
        )

    print()
    print("Run details =", run_path)
    print("Scenario metrics =", metric_path)
    print("Cue metrics =", cue_path)
    print()
    print(
        "Scope: one participant / two camera geometries. These results test "
        "within-subject camera-specific calibration, not cross-subject "
        "generalization."
    )


if __name__ == "__main__":
    main()
