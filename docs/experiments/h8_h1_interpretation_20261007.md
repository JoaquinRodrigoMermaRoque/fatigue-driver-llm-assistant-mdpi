# H8 H1 validated descriptive interpretation — 2026-10-07

## Validation status

The final H1 replay used 12 accepted runs (6 INTERNAL, 6 USB), 5 ablation modes (A-E), for 60 run-mode combinations. Mode E reproduced the Android capture for every run with zero state/event mismatches and score differences at CSV precision only (<=1e-8).

## Ablation definitions

- A: closure
- B: closure + yawn
- C: closure + yawn + PERCLOS
- D: closure + yawn + PERCLOS + gaze
- E: closure + yawn + PERCLOS + gaze + nod

Weights are renormalized over the active cues.

## Confirmed descriptive results

| Mode | Runs with non-normal state | Normal-run false alerts | Frame-weighted mean score | Max score | NORMAL frames | SUSPECTED frames | FATIGUE frames | CRITICAL frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1 | 0 | 0.0047 | 1.0000 | 6715 | 39 | 0 | 0 |
| B | 5 | 0 | 0.0728 | 0.6667 | 5296 | 1458 | 0 | 0 |
| C | 5 | 0 | 0.0938 | 0.7857 | 5131 | 880 | 743 | 0 |
| D | 3 | 0 | 0.0827 | 0.6875 | 5631 | 809 | 314 | 0 |
| E | 4 | 0 | 0.1090 | 0.5645 | 5022 | 1317 | 415 | 0 |

## Interpretation supported by these data

1. EAR/closure-only mode A was the least responsive overall: only one of 12 runs reached a non-normal state and its frame-weighted mean score was 0.0047.
2. Adding temporal/multimodal cues increased aggregate fatigue evidence. Mode E had the highest frame-weighted mean score (0.1090), about 23 times mode A.
3. No mode produced a non-normal state in either NORMAL run; descriptive normal-run false-alert count was 0/2 for A-E.
4. Full multimodal mode E captured cases unavailable to partial modes. In H8_USB_NOD_01, only E reached SUSPECTED. In H8_USB_COMBINED_01, E reached SUSPECTED while C and D remained NORMAL.
5. More cues did not monotonically increase alert severity. C produced FATIGUE in H8V2_INT_PERCLOS_01 and H8V2_INT_COMBINED_01, while E was only SUSPECTED in the former. Because fusion weights are renormalized over enabled cues, adding cues can dilute strong isolated evidence if the newly added cues are inactive.
6. Gaze alone did not trigger a non-normal state in the isolated gaze runs. Under the current full/partial fusion weights and state threshold, gaze contributes contextual evidence but is not designed to dominate the fatigue decision by itself.
7. No mode reached CRITICAL in the selected H1 captures because prolonged-eye-closure contamination (>=3 s) was intentionally avoided and sustained fused scores did not satisfy the CRITICAL transition.

## What H1 does not yet establish

These descriptive replay results do not by themselves establish precision, recall, specificity, F1, sensitivity, or statistically significant superiority. Those metrics require explicit ground-truth event/state intervals. Therefore the confirmed H1 conclusion should be framed as evidence enrichment and cue-specific detection behavior, not as a claim that E is universally more accurate than every partial mode.

## Recommended manuscript conclusion for H1

"The ablation study showed that the closure-only configuration was substantially less responsive than multimodal configurations, while the complete five-cue model produced the highest aggregate fatigue-evidence score and uniquely responded to nod-dominant and multi-cue scenarios. No false non-normal states were observed in the two normal-condition runs. However, the response was not monotonic with the number of enabled cues because active-cue weight renormalization can dilute isolated strong evidence; therefore, these results support complementary multimodal evidence rather than universal dominance of the full model in every scenario."
