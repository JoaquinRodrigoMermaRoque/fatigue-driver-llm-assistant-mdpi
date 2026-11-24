# Aggregate experiment tables: numerical reproducibility

Only POOLED controlled-experiment counts and benchmark window summaries
are provided. No participant names, pseudonyms, individualized runs, facial
features, images, recordings, contact details, geographic locations,
collection dates/times, or Git history are included.

With Python 3.10+ from this package root:

    python3 scripts/verify_aggregate_only.py
    sha256sum -c SHA256SUMS.txt

H1: 12 controlled scenarios, five cue-fusion configurations A-E. Stored
confusion counts produce macro-F1 of 0.245 for A and 0.486 for E; some
partial configurations outperform E.

H2: 13 controlled scenario runs performed by two participants, compared
under global and personalized calibration. ONLY pooled confusion matrices
and cue-level counts remain. Macro-F1 was 0.291 global vs 0.606 personalized;
the sample does not support population generalization.

H3: eight CPU-only and seven GPU-only baseline 30-callback windows, plus
six 30-callback concurrent GPU and local language model windows (180
callbacks). Concurrent E2E mean 286.72 ms, arithmetic mean of window p95
394.33 ms, callback throughput 6.68 callbacks/s. A mean of window p95
is NOT the pooled p95 of 180 callback timings.

Memory: where the Android application was not running concurrently,
application RSS is NA (not applicable), NOT zero.

Positive scenario labels refer to deliberately enacted fatigue-associated
facial/head cues, not independently verified physiological fatigue. H1/H2
classification metrics use scenario runs, not individual video frames.
Only arithmetic checks of retained aggregate counts are possible; complete
data acquisition, inference replay and verification of per-run outcomes
cannot be repeated from these public-candidate tables.

A technical data minimization check does not establish consent, irreversible
anonymity or an ethics exemption. Confirm the original participation consent,
journal/ethics policy and legal release permissions before making public.
No reuse license is granted automatically.
