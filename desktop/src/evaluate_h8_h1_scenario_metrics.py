"""Evaluate H8 H1 at the controlled-scenario level.

Ground truth is intentionally scenario-level, not physiological fatigue truth:
NORMAL runs are negative controls; scripted PERCLOS/YAWN/GAZE/NOD/COMBINED
runs are positive controlled fatigue-cue conditions.

A mode predicts positive when its replay reaches any non-NORMAL state at least
once in that run. This directly answers H1 without inventing frame-level labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

POSITIVE_STATES = {"SUSPECTED", "FATIGUE", "CRITICAL", "RECOVERY"}

METRIC_FIELDS = (
    "config_id",
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
    "normal_false_alerts",
    "normal_exposure_hours",
    "false_alerts_per_hour",
)

PRED_FIELDS = (
    "feature_run_id",
    "source",
    "scenario",
    "config_id",
    "ground_truth_positive",
    "predicted_positive",
    "max_state",
    "correct",
)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def max_state(row: dict[str, str]) -> str:
    state_cols = (
        ("CRITICAL", "critical_frames"),
        ("FATIGUE", "fatigue_frames"),
        ("RECOVERY", "recovery_frames"),
        ("SUSPECTED", "suspected_frames"),
        ("NORMAL", "normal_frames"),
    )
    for state, col in state_cols:
        if int(row[col]) > 0:
            return state
    return "NORMAL"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    with args.summary.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with args.ground_truth.open(encoding="utf-8", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    if len(rows) != 60:
        raise RuntimeError(
            f"Expected 60 replay rows (12 runs x 5 modes), found {len(rows)}"
        )

    bad = [
        r for r in rows
        if r.get("validation_status") != "VALIDATED_EXACT"
    ]
    if bad:
        raise RuntimeError(
            "Final replay contains non-validated runs: "
            + ", ".join(sorted({r["feature_run_id"] for r in bad}))
        )

    gt = {r["run_id"]: r for r in gt_rows}
    replay_runs = {r["feature_run_id"] for r in rows}
    gt_runs = set(gt)

    missing_gt = sorted(replay_runs - gt_runs)
    extra_gt = sorted(gt_runs - replay_runs)
    if missing_gt or extra_gt:
        raise RuntimeError(
            f"Ground-truth mismatch. missing_gt={missing_gt}, extra_gt={extra_gt}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_mode: dict[str, list[dict[str, str]]] = defaultdict(list)
    predictions: list[dict[str, object]] = []

    # NORMAL exposure is identical across modes because all modes replay the
    # same two negative-control runs.
    normal_duration_s = 0.0
    for run_id, g in gt.items():
        if int(g["ground_truth_positive"]) == 0:
            matching = [
                r for r in rows
                if r["feature_run_id"] == run_id and r["config_id"] == "E"
            ]
            if len(matching) != 1:
                raise RuntimeError(f"Missing unique E row for negative run {run_id}")
            normal_duration_s += float(matching[0]["duration_s"])
    normal_exposure_h = normal_duration_s / 3600.0

    for row in rows:
        mode = row["config_id"]
        g = gt[row["feature_run_id"]]
        truth = int(g["ground_truth_positive"])
        state = max_state(row)
        pred = int(state in POSITIVE_STATES)
        record = {
            "feature_run_id": row["feature_run_id"],
            "source": row["source"],
            "scenario": g["scenario"],
            "config_id": mode,
            "ground_truth_positive": truth,
            "predicted_positive": pred,
            "max_state": state,
            "correct": int(truth == pred),
        }
        predictions.append(record)
        by_mode[mode].append(record)

    metrics = []
    for mode in "ABCDE":
        m = by_mode[mode]
        tp = sum(r["ground_truth_positive"] == 1 and r["predicted_positive"] == 1 for r in m)
        tn = sum(r["ground_truth_positive"] == 0 and r["predicted_positive"] == 0 for r in m)
        fp = sum(r["ground_truth_positive"] == 0 and r["predicted_positive"] == 1 for r in m)
        fn = sum(r["ground_truth_positive"] == 1 and r["predicted_positive"] == 0 for r in m)

        accuracy = safe_div(tp + tn, tp + tn + fp + fn)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        f1_pos = f1(precision, recall)

        # Negative class metrics for macro-F1.
        precision_neg = safe_div(tn, tn + fn)
        recall_neg = specificity
        f1_neg = f1(precision_neg, recall_neg)
        macro_f1 = (f1_pos + f1_neg) / 2.0
        balanced_accuracy = (recall + specificity) / 2.0

        metrics.append(
            {
                "config_id": mode,
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
                "macro_f1": f"{macro_f1:.6f}",
                "balanced_accuracy": f"{balanced_accuracy:.6f}",
                "missed_positive_scenarios": fn,
                "normal_false_alerts": fp,
                "normal_exposure_hours": f"{normal_exposure_h:.6f}",
                "false_alerts_per_hour": f"{safe_div(fp, normal_exposure_h):.6f}",
            }
        )

    metric_path = args.output_dir / "H8_H1_SCENARIO_METRICS.csv"
    pred_path = args.output_dir / "H8_H1_SCENARIO_PREDICTIONS.csv"

    with metric_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader()
        w.writerows(metrics)

    with pred_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PRED_FIELDS)
        w.writeheader()
        w.writerows(predictions)

    print("H1 CONTROLLED-SCENARIO METRICS")
    print()
    print(
        f"{'Mode':<5} {'TP':>3} {'TN':>3} {'FP':>3} {'FN':>3} "
        f"{'Acc':>7} {'Recall':>7} {'Spec':>7} {'F1':>7} {'MacroF1':>8} {'FA/h':>8}"
    )
    for r in metrics:
        print(
            f"{r['config_id']:<5} "
            f"{r['tp']:>3} {r['tn']:>3} {r['fp']:>3} {r['fn']:>3} "
            f"{float(r['accuracy']):>7.3f} "
            f"{float(r['recall_sensitivity']):>7.3f} "
            f"{float(r['specificity']):>7.3f} "
            f"{float(r['f1_positive']):>7.3f} "
            f"{float(r['macro_f1']):>8.3f} "
            f"{float(r['false_alerts_per_hour']):>8.3f}"
        )

    by_id = {r["config_id"]: r for r in metrics}
    a = by_id["A"]
    e = by_id["E"]
    delta_macro = float(e["macro_f1"]) - float(a["macro_f1"])
    delta_recall = float(e["recall_sensitivity"]) - float(a["recall_sensitivity"])
    reduced_misses = int(a["fn"]) - int(e["fn"])

    print()
    print("E vs A (primary H1 contrast)")
    print(f"delta_macro_f1 = {delta_macro:.6f}")
    print(f"delta_recall = {delta_recall:.6f}")
    print(f"missed_positive_scenarios_reduced_by = {reduced_misses}")
    print(
        "false_alerts: "
        f"A={a['normal_false_alerts']} E={e['normal_false_alerts']}"
    )
    print()
    print("Metrics =", metric_path)
    print("Predictions =", pred_path)
    print()
    print(
        "Interpretation constraint: these are controlled scenario-level "
        "metrics for scripted fatigue-associated cues, not clinical or "
        "physiological fatigue labels and not frame-level metrics."
    )


if __name__ == "__main__":
    main()
