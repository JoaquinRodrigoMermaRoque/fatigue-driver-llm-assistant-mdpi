# Data dictionary and limitations

The file data/h1_mode_confusion.csv has one row per cue-fusion mode A-E
with total scenario run count, TP/TN/FP/FN and derived classification metrics.
The source 60 per-run rows are retained ONLY in the private source.

The file data/h2_pooled_confusion.csv has just two pooled mode rows,
global and personalized, each representing all 13 controlled scenarios.
No subject code, individual result, sample time, camera choice, or source
run identifier is shared. The data/h2_pooled_cues.csv file holds pooled
gaze and nod recall/specificity counts in these two calibration modes.

The file data/h3_window_summaries.csv has 21 operation-only window rows
(8 CPU reference, 7 GPU reference, 6 GPU plus local LLM), with sequential
indices, callback sample count, E2E mean and window p95, MediaPipe mean,
and callback throughput. Original run IDs, start/end times and device
temperature information are intentionally omitted.

The file data/table10_na_aggregate.csv counts records in three non-
concurrent memory test batches with application RSS unmeasured (NA).

For each class, precision = TP/(TP+FP), recall = TP/(TP+FN),
specificity = TN/(TN+FP), and F1 = 2*precision*recall/(precision+recall).
Macro-F1 is the arithmetic average of positive-class and negative-class
F1. Zero denominators yield zero.

H1/H2 labels are deliberate controlled fatigue-associated actions, not
clinical fatigue diagnoses. Scores are based on entire runs, not frames.
The verifier recalculates aggregate arithmetic and reported numerical
summaries. It CANNOT independently reproduce the confusion counts from
individual participant predictions because these were deliberately omitted.
The original private, processed-data dataset passed a stricter per-run
reproducibility check before these aggregated values were exported.

The H3 source tables contained window-level values and do not enable
recalculation of the overall callback latency distribution or pooled p95.
Individual recordings, privacy-sensitive calibration, raw model logs, and
participant context are outside this release.

The study team must separately document participation consent or a valid
ethics waiver, confirm publication rights, and assess potential
re-identification through external data before any public distribution.
