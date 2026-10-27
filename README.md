# Driver Fatigue Detection with Offline Contextual Voice Assistance

Publication snapshot prepared for the manuscript submitted to **MDPI Technologies**.

This repository contains the Android prototype, analysis scripts, and non-identifying experimental artifacts used to support the reported results. The publication branch is derived from the validated experimental commit:

- Source branch: `h8.3-experiment-logging`
- Source commit: `878f38fe1e88387311659ea8213222a6458adc2b`
- Publication branch: `publication-mdpi`

## Scope

The prototype implements an on-device driver-fatigue monitoring pipeline based on pre-trained MediaPipe Face Landmarker facial landmarks. Fatigue decisions are deterministic and use temporal facial cues. The local LLM is an optional contextual language layer and **does not determine fatigue state, urgency, or mandatory safety action**.

Main components:

- Android camera input: internal camera and USB UVC webcam.
- MediaPipe Face Landmarker (478 landmarks).
- EAR, MAR, PERCLOS, geometric gaze, and head-pose cues.
- Deterministic multi-cue fusion and hysteretic state machine.
- Camera/user-specific calibration for gaze center and head-pose reference.
- Local SQLite event history.
- Offline Android text-to-speech.
- Optional local LFM2.5 contextual generation through llama.cpp/JNI.

## Experimental evidence used in the manuscript

The publication-level evidence is stored under `docs/experiments/`.

### H1 — Multi-cue ablation

- `h8_h1_scenario_ground_truth_20261008.csv`
- `h8_h1_scenario_metrics_20261008.csv`
- `h8_h1_scenario_evaluation_method_20261008.md`
- `h8_h1_interpretation_20261007.md`

### H2 — Personalized calibration

- `h8_h2_two_subject_ground_truth_20261009.csv`
- `h8_final_h2/H8_H2_TWO_SUBJECT_RUNS.csv`
- `h8_final_h2/H8_H2_TWO_SUBJECT_METRICS.csv`
- `h8_final_h2/H8_H2_TWO_SUBJECT_CUE_METRICS.csv`
- `h8_final_camera_calibrations_20261007.csv`

Participants are represented only by pseudonymous identifiers (`subject_001`, `subject_002`) in the public experimental tables.

### H3 — Android performance and local LLM benchmarks

- `h7_9_mediapipe_benchmarks.csv`
- `h7_9_model_benchmarks.csv`
- `android/FatigueMonitor/scripts/README_H7_9_LLM_BENCHMARK.md`

### H4 — Offline contextual assistance

A privacy-reviewed public summary of the final airplane-mode trial is provided in the publication snapshot. The original private experimental branch remains the archival source of the unredacted internal log note.

## Important interpretation constraints

- The H1/H2 recordings were controlled validation scenarios with the vehicle parked; they are not clinical fatigue ground truth.
- H2 is an exploratory two-subject feasibility test and is not population-level validation.
- Battery temperature is used only as a thermal-state proxy.
- The final local LLM is outside the safety-critical path.
- In the final H4 trial, local LLM inference completed offline, but the generated candidate did not pass the grounding/format validator; the spoken contextual response came from the deterministic contextual fallback.

## Reproducibility

The repository intentionally contains publication-level summaries rather than every large transient raw artifact. Each manuscript-level benchmark is tied to a run identifier where applicable. The detailed experiment registry in `docs/experiments/README.md` distinguishes accepted manuscript-level runs from diagnostic or excluded runs.

## Privacy

Identifiable facial screenshots and other unnecessary participant-identifying media are intentionally excluded from the publication snapshot. Experimental tables use pseudonymous subject identifiers. Do not add names, raw identifiable images, or other personally identifying participant information to the public release.

## Third-party software and models

See `THIRD_PARTY.md`. Third-party models and dependencies remain subject to their original licenses and terms.

## Intended use

This repository is a research prototype. It is **not a medical device**, does not provide a clinical diagnosis of fatigue, and should not be used as the sole basis for safety-critical driving decisions.

## Manuscript

Working title:

**A face-landmarks personalized driver fatigue detector with contextualized LLM voice assistant framework for resource-limited Android devices**

The final article citation and DOI should be added here after publication.
