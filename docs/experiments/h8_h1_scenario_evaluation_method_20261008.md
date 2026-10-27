# H8 H1 scenario-level evaluation method — 2026-10-08

## Purpose

Close H1 with the quantitative evidence required by the project without inventing frame-level ground truth that was not manually annotated during capture.

## Ground truth

The unit of evaluation is the **controlled run/scenario**. The two NORMAL runs are negative controls. PERCLOS, YAWN, GAZE, NOD, and COMBINED runs are positive controlled conditions because the operator intentionally executed the corresponding fatigue-associated facial/head cues.

This is deliberately described as **controlled fatigue-cue ground truth**, not physiological or clinical fatigue ground truth.

## Prediction rule

For each ablation mode A-E and each run, prediction is positive when the replay reaches any non-NORMAL state at least once (SUSPECTED, FATIGUE, CRITICAL, or RECOVERY). Prediction is negative when the entire run remains NORMAL.

All modes use the already validated same replay evidence. The final H1 replay contains 12 runs x 5 modes = 60 run-mode combinations, with mode E matching the captured Android execution for all 12 runs with zero state/event mismatches.

## Metrics

The evaluator reports TP, TN, FP, FN, accuracy, precision, recall/sensitivity, specificity, positive-class F1, negative-class F1, macro-F1, balanced accuracy, missed positive scenarios, normal false alerts, and false alerts/hour over the two NORMAL-control durations.

No frame-level precision/recall/F1 is claimed because frame-level event onset/offset annotations were not created independently during capture. This avoids circularly deriving ground truth from the detector outputs.

## Confirmed results

| Mode | TP | TN | FP | FN | Accuracy | Recall | Specificity | F1 | Macro-F1 | Missed positive scenarios | Normal false alerts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 2 | 0 | 9 | 0.250 | 0.100 | 1.000 | 0.182 | 0.245 | 9 | 0 |
| B | 5 | 2 | 0 | 5 | 0.583 | 0.500 | 1.000 | 0.667 | 0.556 | 5 | 0 |
| C | 5 | 2 | 0 | 5 | 0.583 | 0.500 | 1.000 | 0.667 | 0.556 | 5 | 0 |
| D | 3 | 2 | 0 | 7 | 0.417 | 0.300 | 1.000 | 0.462 | 0.413 | 7 | 0 |
| E | 4 | 2 | 0 | 6 | 0.500 | 0.400 | 1.000 | 0.571 | 0.486 | 6 | 0 |

## Primary H1 contrast: E vs A

- Macro-F1: 0.485714 vs 0.244755; absolute improvement **+0.240959**.
- Recall/sensitivity: 0.400000 vs 0.100000; absolute improvement **+0.300000**.
- Missed positive scenarios: 6 vs 9; **3 fewer misses**.
- Normal false alerts: 0 vs 0; no increase in false alarms in the two negative-control runs.
- Specificity: 1.0 for both under this controlled sample.

Therefore H1 is supported for the planned full multimodal-vs-EAR-only contrast on this controlled scenario dataset: multimodal mode E improved macro-F1 and recall without increasing observed false alarms. The ablation also shows that B and C obtained higher scenario-level macro-F1 than E under the current renormalized weights, so the data do not support claiming universal superiority of the five-cue configuration over every partial configuration.
