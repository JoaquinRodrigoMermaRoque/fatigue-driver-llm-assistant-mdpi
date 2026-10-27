# Publication Evidence Map

This file maps the manuscript hypotheses to the exact publication-level artifacts retained in the repository. It does not modify or recompute any experimental result.

| Hypothesis / objective | Evidence files |
|---|---|
| H1 — multi-cue ablation | `docs/experiments/h8_h1_scenario_ground_truth_20261008.csv`; `h8_h1_scenario_metrics_20261008.csv`; `h8_h1_scenario_evaluation_method_20261008.md`; `h8_h1_interpretation_20261007.md` |
| H2 — personalization | `docs/experiments/h8_h2_two_subject_ground_truth_20261009.csv`; `docs/experiments/h8_final_h2/H8_H2_TWO_SUBJECT_RUNS.csv`; `H8_H2_TWO_SUBJECT_METRICS.csv`; `H8_H2_TWO_SUBJECT_CUE_METRICS.csv`; `h8_final_camera_calibrations_20261007.csv` |
| H3 — mobile/LLM performance | `docs/experiments/h7_9_mediapipe_benchmarks.csv`; `h7_9_model_benchmarks.csv`; `android/FatigueMonitor/scripts/README_H7_9_LLM_BENCHMARK.md` |
| H4 — offline contextual assistance | privacy-reviewed public H4 summary in `docs/experiments/`; archival unredacted note remains in the private source branch |

## Integrity rule

The numerical CSV files above are copied from the validated source commit and must not be edited for the public release. Any future derived table or figure should cite the relevant source file and be generated without replacing these source artifacts.
