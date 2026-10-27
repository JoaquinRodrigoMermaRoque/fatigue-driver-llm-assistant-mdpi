"""Final H8 H2 evaluation using two subjects.

Compares fixed global references with per-user calibration on the SAME frames.
Only calibration-sensitive scenarios are included: NORMAL, GAZE, NOD, COMBINED.

The personalized raw replay must reproduce the captured Android execution
(score/state/event) before a run is admitted. This prevents H2 results from
depending on an offline reconstruction that is not faithful to the phone.

Important scope:
- calibration currently changes gaze center and head-pose baselines only;
- EAR/MAR/PERCLOS thresholds are unchanged;
- labels are controlled cue-condition labels, not clinical fatigue labels.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from run_h8_ablation_replay import load_rows, replay_mode

TOL = 1e-8
POSITIVE_STATES = {"SUSPECTED", "FATIGUE", "CRITICAL", "RECOVERY"}
STATE_ORDER = {
    "NORMAL": 0,
    "SUSPECTED": 1,
    "RECOVERY": 2,
    "FATIGUE": 3,
    "CRITICAL": 4,
}

RUN_FIELDS = (
    "run_id","subject_id","source","scenario","calibration_mode",
    "frames","duration_s","score_mean","score_max","max_state",
    "predicted_positive","max_gaze_away_s","gaze_detected",
    "nod_event_frames","nod_detected",
    "personalized_validation_score_diff",
    "personalized_validation_state_mismatches",
    "personalized_validation_event_mismatches",
)

METRIC_FIELDS = (
    "scope","subject_id","calibration_mode","runs","tp","tn","fp","fn",
    "accuracy","precision","recall_sensitivity","specificity",
    "f1_positive","f1_negative","macro_f1","balanced_accuracy",
    "missed_positive_scenarios","false_positive_scenarios",
)

CUE_FIELDS = (
    "scope","subject_id","cue","calibration_mode",
    "positive_runs","detected_positive_runs","missed_positive_runs",
    "normal_runs","normal_false_activations","recall","specificity",
)


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


def classification_metrics(
    records: list[dict[str, object]],
    scope: str,
    subject_id: str,
    mode: str,
) -> dict[str, object]:
    m = [
        r for r in records
        if r["calibration_mode"] == mode
        and (subject_id == "ALL" or r["subject_id"] == subject_id)
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
    precision_neg = safe_div(tn, tn + fn)
    f1_neg = f1(precision_neg, specificity)

    return {
        "scope": scope,
        "subject_id": subject_id,
        "calibration_mode": mode,
        "runs": len(m),
        "tp": tp,"tn": tn,"fp": fp,"fn": fn,
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


def cue_metrics(
    records: list[dict[str, object]],
    scope: str,
    subject_id: str,
    cue: str,
    mode: str,
) -> dict[str, object]:
    field = "gaze_detected" if cue == "gaze" else "nod_detected"
    target = "gaze_target" if cue == "gaze" else "nod_target"
    m = [
        r for r in records
        if r["calibration_mode"] == mode
        and (subject_id == "ALL" or r["subject_id"] == subject_id)
    ]
    positives = [r for r in m if int(r[target]) == 1]
    negatives = [r for r in m if r["scenario"] == "NORMAL"]
    detected = sum(int(r[field]) == 1 for r in positives)
    false_pos = sum(int(r[field]) == 1 for r in negatives)

    return {
        "scope": scope,
        "subject_id": subject_id,
        "cue": cue,
        "calibration_mode": mode,
        "positive_runs": len(positives),
        "detected_positive_runs": detected,
        "missed_positive_runs": len(positives) - detected,
        "normal_runs": len(negatives),
        "normal_false_activations": false_pos,
        "recall": f"{safe_div(detected, len(positives)):.6f}",
        "specificity": f"{safe_div(len(negatives)-false_pos, len(negatives)):.6f}",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    with args.ground_truth.open(encoding="utf-8", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []

    for g in gt_rows:
        run_id = g["run_id"]
        path = args.features_dir / f"h8_features_{run_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)

        rows = load_rows(path)

        subjects = {r["subject_id"] for r in rows}
        if len(subjects) != 1:
            raise RuntimeError(f"{run_id}: multiple subject IDs: {subjects}")

        expected_subject = g["subject_id"]
        actual_subject = next(iter(subjects))
        # Historical H1 subject was stored as pilot_001; normalize it to
        # subject_001 for the final two-subject H2 table.
        allowed = (
            actual_subject == expected_subject
            or (
                expected_subject == "subject_001"
                and actual_subject == "pilot_001"
            )
        )
        if not allowed:
            raise RuntimeError(
                f"{run_id}: subject mismatch "
                f"expected={expected_subject} actual={actual_subject}"
            )

        personalized = replay_mode(
            rows, "E", calibration_mode="personalized"
        )
        global_replay = replay_mode(
            rows, "E", calibration_mode="global"
        )

        vdiff, vstate, vevent = validate_personalized(personalized)
        if vdiff > TOL or vstate != 0 or vevent != 0:
            raise RuntimeError(
                f"{run_id}: personalized raw replay not exact: "
                f"score_diff={vdiff:.10f} "
                f"state_mm={vstate} event_mm={vevent}"
            )

        for mode, replay in (
            ("global", global_replay),
            ("personalized", personalized),
        ):
            state = max_state(replay)
            pred = int(state in POSITIVE_STATES)
            max_gaze = max(float(r["gaze_away_s"]) for r in replay)
            nod_frames = sum(int(r["nod_event"]) != 0 for r in replay)

            detail = {
                "run_id": run_id,
                "subject_id": expected_subject,
                "source": g["source"],
                "scenario": g["scenario"],
                "calibration_mode": mode,
                "frames": len(replay),
                "duration_s": f"{float(replay[-1]['t_ms'])/1000.0:.3f}",
                "score_mean": f"{fmean(float(r['score']) for r in replay):.6f}",
                "score_max": f"{max(float(r['score']) for r in replay):.6f}",
                "max_state": state,
                "predicted_positive": pred,
                "max_gaze_away_s": f"{max_gaze:.6f}",
                "gaze_detected": int(max_gaze >= 1.5),
                "nod_event_frames": nod_frames,
                "nod_detected": int(nod_frames > 0),
                "personalized_validation_score_diff": f"{vdiff:.8f}",
                "personalized_validation_state_mismatches": vstate,
                "personalized_validation_event_mismatches": vevent,
            }
            run_rows.append(detail)

            records.append({
                **detail,
                "ground_truth_positive": int(g["ground_truth_positive"]),
                "gaze_target": int(g["gaze_target"]),
                "nod_target": int(g["nod_target"]),
            })

        print(
            f"OK {run_id}: subject={expected_subject} "
            f"scenario={g['scenario']} "
            f"personalized_diff={vdiff:.8f} "
            f"state_mm={vstate} event_mm={vevent}"
        )

    metrics = []
    cues = []
    scopes = [
        ("pooled_two_subjects", "ALL"),
        ("subject_001", "subject_001"),
        ("subject_002", "subject_002"),
    ]

    for scope, subject in scopes:
        for mode in ("global", "personalized"):
            metrics.append(
                classification_metrics(records, scope, subject, mode)
            )
            for cue in ("gaze", "nod"):
                cues.append(
                    cue_metrics(records, scope, subject, cue, mode)
                )

    run_path = args.output_dir / "H8_H2_TWO_SUBJECT_RUNS.csv"
    metric_path = args.output_dir / "H8_H2_TWO_SUBJECT_METRICS.csv"
    cue_path = args.output_dir / "H8_H2_TWO_SUBJECT_CUE_METRICS.csv"

    with run_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader(); w.writerows(run_rows)
    with metric_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        w.writeheader(); w.writerows(metrics)
    with cue_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CUE_FIELDS)
        w.writeheader(); w.writerows(cues)

    print()
    print("H2 TWO-SUBJECT STATE-LEVEL METRICS")
    print(
        f"{'Scope':<20} {'Mode':<12} {'TP':>3} {'TN':>3} "
        f"{'FP':>3} {'FN':>3} {'Recall':>7} {'Spec':>7} {'MacroF1':>8}"
    )
    for r in metrics:
        print(
            f"{r['scope']:<20} {r['calibration_mode']:<12} "
            f"{r['tp']:>3} {r['tn']:>3} {r['fp']:>3} {r['fn']:>3} "
            f"{float(r['recall_sensitivity']):>7.3f} "
            f"{float(r['specificity']):>7.3f} "
            f"{float(r['macro_f1']):>8.3f}"
        )

    print()
    print("H2 TWO-SUBJECT CUE-LEVEL METRICS")
    print(
        f"{'Scope':<20} {'Cue':<6} {'Mode':<12} "
        f"{'Detected':>8} {'Pos':>4} {'Miss':>5} "
        f"{'NormalFP':>8} {'Recall':>7} {'Spec':>7}"
    )
    for r in cues:
        print(
            f"{r['scope']:<20} {r['cue']:<6} "
            f"{r['calibration_mode']:<12} "
            f"{r['detected_positive_runs']:>8} "
            f"{r['positive_runs']:>4} "
            f"{r['missed_positive_runs']:>5} "
            f"{r['normal_false_activations']:>8} "
            f"{float(r['recall']):>7.3f} "
            f"{float(r['specificity']):>7.3f}"
        )

    pooled = {
        (r["calibration_mode"]): r
        for r in metrics if r["scope"] == "pooled_two_subjects"
    }
    g = pooled["global"]; pz = pooled["personalized"]
    print()
    print("PERSONALIZED minus GLOBAL (pooled state-level)")
    print(
        "delta_macro_f1 = "
        f"{float(pz['macro_f1'])-float(g['macro_f1']):.6f}"
    )
    print(
        "delta_recall = "
        f"{float(pz['recall_sensitivity'])-float(g['recall_sensitivity']):.6f}"
    )
    print(
        "delta_false_positives = "
        f"{int(pz['fp'])-int(g['fp'])}"
    )
    print()
    print("Run details =", run_path)
    print("State metrics =", metric_path)
    print("Cue metrics =", cue_path)
    print()
    print(
        "Interpretation constraint: N=2 subjects; this is an exploratory "
        "controlled validation of personalization, not a population-level "
        "generalization claim."
    )


if __name__ == "__main__":
    main()
