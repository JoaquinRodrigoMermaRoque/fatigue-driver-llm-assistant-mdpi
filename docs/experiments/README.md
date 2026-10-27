# H7.9 Experimental Registry

This directory contains publication-oriented summaries of the local-LLM
experiments performed on the Android prototype.

## Files

- `h7_9_model_benchmarks.csv`: one row per measured inference/request.
- The detailed chronological notes remain in
  `android/FatigueMonitor/scripts/README_H7_9_LLM_BENCHMARK.md`.

Raw benchmark artifacts remain under
`android/FatigueMonitor/.local/h7_9_llm/benchmarks_extended/<RUN_ID>/` and are
intentionally not committed because they can contain large transient files.
Every result used in a manuscript should retain its `RUN_ID`.

## Interpretation of quality fields

- `output_valid=1`: the output passed the structural/current-format criterion
  used by that experiment.
- `semantic_correct=1`: the selected/generated content matched the supplied
  scenario and did not change its intended semantic category.
- `NA`: the historical run was not assessed with that criterion.

A structurally valid JSON is not necessarily semantically correct. This was
observed in RUN_ID `20260924T193706Z`, repetition 3.

## Important methodological cautions

1. Battery temperature is not SoC temperature; it is retained only as a
   reproducible thermal-state proxy.
2. MediaPipe CPU/GPU comparisons made before direct per-frame telemetry are
   exploratory. Delegate labels marked as manually selected should not be used
   alone to claim a causal GPU speedup/slowdown.
3. RUN_ID `20260924T194641Z` confirms the APK was concurrently active, but
   the benchmark did not yet capture the effective MediaPipe delegate
   automatically; therefore it is recorded as `unverified`.
4. Static warm-cache measurements reuse an identical prompt and are an
   optimistic upper bound. Dynamic-context measurements are more
   representative.
5. Cross-model comparisons should use the same scenario set, context size,
   thread count, and equivalent output task. Free generation, semantic
   selection, and style selection are different tasks and should not be
   presented as if they were identical workloads.

## Current architecture under evaluation

The safety-critical path is deterministic. The local LLM is not allowed to
decide fatigue state, urgency, trigger, or mandatory action.

Current candidate flow:

```text
MediaPipe + temporal indicators
    -> FatigueEngine
    -> ContextualAlertPlanner
    -> deterministic semantic template
    -> optional LFM2.5 style variant selection
    -> reviewed template renderer
    -> validation/fallback
    -> TTS
```

This design separates safety semantics from linguistic variation and makes the
contextual layer auditable and suitable for multilingual expansion.


## MediaPipe telemetry fields

Starting with H7.9.9, concurrent runs may include a
`mediapipe_telemetry.tsv` file generated from app logcat telemetry. Each row
is a non-overlapping 30-callback window.

The fields distinguish:

- `pre_*`: Android image preprocessing before `detectAsync`;
- `mp_*`: callback latency after submission to MediaPipe;
- `e2e_*`: total frame-to-callback latency;
- `callback_fps`: observed result callback throughput;
- `delegate`: delegate successfully instantiated by Face Landmarker.

For CPU/GPU claims in the manuscript, use only runs where the telemetry file
contains the expected verified delegate. Older manually selected runs remain
exploratory.


## Controlled MediaPipe-only protocol

Use `android/FatigueMonitor/scripts/h7_9_benchmark_mediapipe_only.sh` before
concurrent LLM runs. CPU and GPU are measured independently for 30 seconds with
the same camera path and 100 ms analysis throttle. The app telemetry verifies
the delegate that was actually instantiated.

The primary comparison variables are the 30-callback window distributions of
`mp_mean_ms`, `mp_p95_ms`, `e2e_mean_ms`, `e2e_p95_ms`, and
`callback_fps`. Preprocessing metrics are retained to distinguish delegate
effects from Android image-conversion costs.


## H7.9.9 CPU telemetry smoke test

After enabling the Android log tags explicitly, six consecutive 30-callback
windows were observed with the verified CPU delegate.

Preliminary window-level averages across those six windows:

- end-to-end mean: approximately 87.56 ms;
- Android preprocessing mean: approximately 16.29 ms;
- MediaPipe callback latency mean: approximately 71.27 ms;
- callback throughput: approximately 8.78 FPS.

MediaPipe accounted for roughly 81 % of the measured end-to-end latency in
this short smoke test. This is diagnostic evidence only, not the controlled
CPU baseline used for the manuscript. A separate 30 s MediaPipe-only run must
be collected with the benchmark script after the log-tag fix.


## Excluded diagnostic run: 20260924T201422Z

The first 30 s CPU-only baseline produced eight telemetry windows, but the
printed TSV showed an ambiguous concatenation between `pre_max_ms` and
`mp_mean_ms` (for example `4572.57`, consistent with `45` + `72.57`).
Although the underlying values can be reconstructed plausibly and yield
approximately 91.5 ms end-to-end, 18.25 ms preprocessing, 73.27 ms MediaPipe
latency, and 8.66 callback FPS, this run is excluded from manuscript-level
analysis because no published result should depend on manual reconstruction.

H7.9.9 now uses comma-delimited log fields and a strict parser that rejects
rows whose percentile ordering is invalid or whose timing components do not
sum consistently. The controlled CPU baseline must be repeated after this
fix.


## H7.9.9 telemetry-window isolation revision

Absolute filtering by comparing Android `elapsedRealtime()` against
`/proc/uptime` was removed from the publication pipeline after a valid CPU
run was rejected despite containing correctly formatted H7_9_PERF rows.

The measurement protocol now clears logcat immediately before acquisition and
discards the first complete 30-callback telemetry window. That first window is
the only one that can contain samples accumulated in the app before logcat was
cleared. Every later non-overlapping window must have been completed after the
measurement began.

This avoids cross-clock assumptions while preserving all subsequent complete
windows. Raw `logcat_after.txt` remains the source-of-truth artifact and can
be reparsed if parser logic changes.


## H7.9.9 parser hardening after CPU-only attempt

A subsequent CPU-only acquisition produced correctly comma-delimited raw
H7_9_PERF rows, but the host parser was still using its earlier
whitespace-delimited regular expression. The raw telemetry itself was valid;
the parser therefore rejected all rows.

The parser now extracts values with `([^,\\s]+)`, the benchmark inserts a
unique `H7_9_START run_id=<RUN_ID>` marker into logcat, parsing begins only
after that marker, and the first complete 30-callback window is discarded to
avoid any partially pre-existing in-memory telemetry window.

Both the MediaPipe-only and concurrent LLM benchmark scripts were also checked
for and corrected from literal `\\n` text accidentally introduced between
shell commands.

No manuscript result should be taken from the failed host-parsing attempt.
The raw visible CPU rows remain diagnostic only until a clean marker-isolated
run completes.


## H7.9.9 final host-isolation rule

A 30 s CPU run produced nine correctly formatted H7_9_PERF windows after
`adb logcat -c`, but the optional RUN_ID marker was not present in the
captured buffer. Because parsing was configured to wait for that marker, all
otherwise valid rows were skipped.

The marker requirement has therefore been removed. The final host protocol is:

1. explicitly enable the H7_9_PERF log tag;
2. clear logcat immediately before acquisition;
3. acquire the requested interval;
4. parse every structurally valid H7_9_PERF row captured after the clear;
5. discard the first complete 30-callback window to remove a possible
   in-memory window that began before acquisition;
6. validate percentile ordering and the equality
   `e2e_mean ~= pre_mean + mp_mean`.

The observed nine-window count in 30 s is consistent with the measured
approximately 8.5–9 callback FPS and confirms that logcat clearing is now
behaving as expected. The failed host-filtering run remains diagnostic rather
than manuscript-level data.


## Official H7.9.9 CPU baseline — RUN_ID 20260924T203231Z

This run is accepted for manuscript-level analysis.

Conditions:

- verified delegate: CPU;
- duration: 30 s;
- retained telemetry windows: 8;
- callbacks represented: 240;
- battery temperature: 31.8 °C -> 31.6 °C;
- each telemetry row contained the expected 18 columns and passed the parser
  integrity checks.

Window-level aggregate statistics:

| Metric | Mean across 8 windows |
|---|---:|
| End-to-end mean latency | 77.05 ms |
| Mean of window E2E p50 values | 66.25 ms |
| Mean of window E2E p95 values | 132.00 ms |
| Android preprocessing mean | 13.66 ms |
| MediaPipe callback latency mean | 63.39 ms |
| Mean of window MediaPipe p50 values | 54.13 ms |
| Mean of window MediaPipe p95 values | 114.13 ms |
| Callback throughput | 8.83 FPS |

MediaPipe represented approximately 82.3 % of the mean measured end-to-end
latency, while Android preprocessing represented approximately 17.7 %.

Important: the p50/p95 values above are means of per-window percentiles, not
global percentiles calculated over all 240 individual callbacks. The raw
30-callback windows are retained in
`docs/experiments/h7_9_mediapipe_benchmarks.csv` and the device-side source
artifact remains under the local RUN_ID directory.

The next controlled comparison must use the same protocol with the verified
GPU delegate and similar starting thermal conditions.


## Official H7.9.9 GPU baseline — RUN_ID 20260924T203839Z

This run is accepted for manuscript-level analysis.

Conditions:

- verified delegate: GPU;
- duration: 30 s;
- retained telemetry windows: 7;
- callbacks represented: 210;
- battery temperature: 31.8 °C -> 31.8 °C;
- same camera path and 100 ms analysis throttle as the accepted CPU baseline.

Window-level aggregate statistics:

| Metric | GPU mean across 7 windows | CPU baseline |
|---|---:|---:|
| End-to-end mean latency | 91.08 ms | 77.05 ms |
| Mean of window E2E p50 values | 92.86 ms | 66.25 ms |
| Mean of window E2E p95 values | 105.29 ms | 132.00 ms |
| Android preprocessing mean | 28.07 ms | 13.66 ms |
| MediaPipe callback latency mean | 63.00 ms | 63.39 ms |
| Mean of window MediaPipe p50 values | 61.86 ms | 54.13 ms |
| Mean of window MediaPipe p95 values | 71.00 ms | 114.13 ms |
| Callback throughput | 8.63 FPS | 8.83 FPS |

Compared with CPU, GPU changed the window-level metrics approximately as
follows:

- end-to-end mean latency: +18.2 %;
- preprocessing mean: +105.5 %;
- MediaPipe mean latency: -0.6 %;
- callback throughput: -2.2 %;
- MediaPipe p95 (mean of per-window p95 values): -37.8 %;
- end-to-end p95 (mean of per-window p95 values): -20.2 %.

Interpretation: GPU did not improve mean Face Landmarker callback latency in
this protocol, but it substantially reduced tail latency/variability inside the
MediaPipe stage. However, the measured preprocessing/submission portion was
about twice as large during the GPU run, causing worse mean end-to-end latency
and slightly lower callback throughput. Because preprocessing code is nominally
the same for both delegates, this increase must not be attributed directly to
the GPU without further instrumentation; scheduling, synchronization, image
handling, or run-to-run effects may contribute.

For the current real-time prototype, CPU therefore has the better observed
mean end-to-end latency under this controlled pair, while GPU has the better
observed tail latency. The next decision should be based on concurrent
LFM2.5+MediaPipe measurements rather than this isolated comparison alone.


## Concurrent H7.9.9 run — GPU + LFM2.5, 4 threads

RUN_ID `20260924T204559Z` is accepted for the LFM-side measurements and has a
verified GPU MediaPipe delegate.

LLM conditions:

- model: LFM2.5-1.2B-Instruct Q4_K_M;
- output mode: style-selector;
- dynamic context;
- 4 CPU threads;
- three repetitions;
- concurrent FatigueMonitor APK;
- all three outputs structurally valid and semantically correct.

LLM results:

| Repeat | Host latency | Peak LLM RSS | Peak APK RSS | Battery temp |
|---|---:|---:|---:|---:|
| 1 | 11.430 s | 1,497,228 KiB | 256,928 KiB | 30.8 -> 30.8 °C |
| 2 | 12.230 s | 1,500,960 KiB | 263,732 KiB | 30.8 -> 30.9 °C |
| 3 | 13.100 s | 1,508,764 KiB | 267,972 KiB | 30.9 -> 31.3 °C |

Mean LLM host latency: approximately 12.25 s.

Six MediaPipe telemetry windows were retained by the benchmark. The terminal
summary displayed only the last four windows, so a manuscript-level aggregate
for all six MediaPipe windows is intentionally deferred until the complete
`mediapipe_telemetry.tsv` is recorded in the experiment registry.

The four visible final windows already show strong concurrent degradation:
their mean end-to-end latency is approximately 305.75 ms, mean preprocessing
is 62.59 ms, mean MediaPipe callback latency is 243.16 ms, and callback
throughput is approximately 6.49 FPS. These are preliminary partial-window
statistics and must not be substituted for the six-window aggregate.

The isolated GPU baseline was 91.08 ms end-to-end, 28.07 ms preprocessing,
63.00 ms MediaPipe, and 8.63 FPS. Therefore the visible concurrent windows
indicate substantial resource contention despite the Face Landmarker delegate
being GPU. This does not establish a GPU-specific cause because the Android
pipeline, callback handling and LFM runtime still share CPU and memory
resources.


## Concurrent H7.9.9 run — GPU + LFM2.5, 6 threads

RUN_ID `20260924T205348Z` is accepted for the LFM-side measurements and has a
verified GPU MediaPipe delegate.

LLM conditions:

- model: LFM2.5-1.2B-Instruct Q4_K_M;
- output mode: style-selector;
- dynamic context;
- 6 CPU threads;
- three repetitions;
- concurrent FatigueMonitor APK;
- all three outputs structurally valid and semantically correct.

LLM host latencies were 12.160 s, 10.890 s, and 11.320 s, for a mean of
approximately 11.46 s. Relative to the GPU + LFM2.5 4-thread run
(approximately 12.25 s), 6 threads reduced mean LLM latency by approximately
6.5 %.

Seven MediaPipe telemetry windows were retained. The terminal summary displayed
only the final four windows, whose preliminary partial-window means are:

- end-to-end latency: approximately 364.18 ms;
- preprocessing: approximately 69.12 ms;
- MediaPipe callback latency: approximately 295.06 ms;
- callback throughput: approximately 5.46 FPS.

These partial-window MediaPipe values must not be treated as the final
seven-window aggregate. They nevertheless indicate stronger contention than
the visible final windows of the 4-thread run.

Battery temperature remained at 30.4 °C during the three LLM requests, so the
observed difference is unlikely to be explained by increasing battery
temperature alone.


## Official concurrent aggregate — GPU + LFM2.5, 6 threads

RUN_ID `20260924T205348Z` contains seven retained 30-callback MediaPipe
windows (210 callbacks) with the GPU delegate verified.

Aggregate window-level means:

| Metric | GPU + LFM 6T |
|---|---:|
| End-to-end mean latency | 331.45 ms |
| Mean of window E2E p50 values | 327.29 ms |
| Mean of window E2E p95 values | 516.71 ms |
| Android preprocessing mean | 67.69 ms |
| MediaPipe callback latency mean | 263.76 ms |
| Mean of window MediaPipe p50 values | 259.14 ms |
| Mean of window MediaPipe p95 values | 427.00 ms |
| Callback throughput | 5.76 FPS |

The same run produced an LFM2.5 mean host latency of approximately 11.46 s
with 3/3 structurally valid and semantically correct style selections.

Using the complete six-window aggregate previously reconstructed from the
GPU + LFM 4-thread run (RUN_ID `20260924T204559Z`), the 6-thread
configuration changes the MediaPipe-side metrics approximately as follows:

- end-to-end mean: 286.72 -> 331.45 ms (+15.6 %);
- preprocessing mean: 60.05 -> 67.69 ms (+12.7 %);
- MediaPipe mean: 226.67 -> 263.76 ms (+16.4 %);
- mean of MediaPipe window p95: 345.67 -> 427.00 ms (+23.5 %);
- callback throughput: 6.68 -> 5.76 FPS (-13.8 %).

In exchange, LFM mean host latency improved from approximately 12.25 s to
11.46 s (-6.5 %). Under the tested conditions, the additional two LFM threads
therefore provide a modest language-side speedup but a substantially larger
degradation in the real-time vision pipeline.

This supports testing CPU + LFM configurations next before selecting the final
runtime allocation.


## Concurrent H7.9.9 run — CPU + LFM2.5, 4 threads

RUN_ID `20260924T205726Z` is accepted for the LFM-side measurements and has a
verified CPU MediaPipe delegate.

LLM host latencies were 12.540 s, 13.340 s, and 14.000 s, for a mean of
approximately 13.29 s. All three style-selector outputs were structurally valid
and semantically correct.

Compared with GPU + LFM2.5 at 4 threads (mean LLM host latency approximately
12.25 s), CPU + LFM2.5 at 4 threads is approximately 8.5 % slower on the
language side under these runs.

Ten MediaPipe telemetry windows were retained. The terminal summary displayed
only the final four, so the full MediaPipe aggregate is intentionally deferred
until all ten rows are added to the experiment registry. The visible final
windows indicate substantial contention, with callback throughput around
6.4–7.1 FPS.

Battery temperature was approximately 32.0–32.1 °C during the three LLM
requests, close enough to the previous controlled runs for exploratory
comparison, but final CPU-vs-GPU concurrent conclusions should use the complete
window aggregates.


## Official concurrent aggregate — CPU + LFM2.5, 4 threads

RUN_ID `20260924T205726Z` contains ten retained 30-callback MediaPipe
windows (300 callbacks) with the CPU delegate verified.

Aggregate window-level means:

| Metric | CPU + LFM 4T |
|---|---:|
| End-to-end mean latency | 258.45 ms |
| Mean of window E2E p50 values | 255.60 ms |
| Mean of window E2E p95 values | 386.30 ms |
| Android preprocessing mean | 47.86 ms |
| MediaPipe callback latency mean | 210.59 ms |
| Mean of window MediaPipe p50 values | 207.50 ms |
| Mean of window MediaPipe p95 values | 340.70 ms |
| Callback throughput | 6.88 FPS |

The same run produced an LFM2.5 mean host latency of approximately 13.29 s
with 3/3 structurally valid and semantically correct style selections.

Compared with GPU + LFM2.5 at 4 threads:

- end-to-end mean: 286.72 -> 258.45 ms for CPU (-9.9 %);
- preprocessing mean: 60.05 -> 47.86 ms (-20.3 %);
- MediaPipe mean: 226.67 -> 210.59 ms (-7.1 %);
- mean of MediaPipe window p95: 345.67 -> 340.70 ms (-1.4 %);
- callback throughput: 6.68 -> 6.88 FPS (+2.9 %);
- LFM mean host latency: 12.25 -> 13.29 s (+8.5 %).

Thus, under the tested 4-thread concurrent condition, CPU provides better
vision-side mean latency and slightly better throughput, while GPU allows a
faster LFM response. MediaPipe tail latency is nearly tied. The final runtime
allocation should not be selected until the CPU + LFM 6-thread condition is
measured.

Note: these session aggregates include the whole benchmark interval rather than
only token-generation-active windows. Because the same protocol is used across
conditions, they are useful for comparative system-level evaluation, but they
should be described as session-level concurrent performance rather than pure
kernel inference benchmarks.


## Concurrent H7.9.9 run — CPU + LFM2.5, 6 threads

RUN_ID `20260924T210214Z` is accepted for the LFM-side measurements and has a
verified CPU MediaPipe delegate.

LLM host latencies were 13.370 s, 13.250 s, and 13.890 s, for a mean of
approximately 13.50 s. All three style-selector outputs were structurally valid
and semantically correct.

Compared with CPU + LFM2.5 at 4 threads (approximately 13.29 s), 6 threads did
not improve the observed LFM host latency; it was approximately 1.6 % slower in
this run.

Eight MediaPipe telemetry windows were retained. The terminal summary displayed
only the final four, whose preliminary means are approximately:

- end-to-end latency: 372.02 ms;
- preprocessing: 68.38 ms;
- MediaPipe callback latency: 303.64 ms;
- callback throughput: 5.33 FPS.

These are not the final eight-window aggregate.

Battery temperature during the LLM repetitions was 33.2–33.6 °C, which is
higher than the CPU + LFM 4-thread run (about 32.0–32.1 °C) and the GPU
concurrent runs (about 30.4–31.3 °C). Therefore thermal state is a plausible
confounder and must be reported when interpreting this run. The full
eight-window aggregate is still useful for the system-level matrix, but any
claim that 6 threads intrinsically caused the entire degradation should remain
qualified unless repeated at matched starting temperature.


## Official concurrent aggregate — CPU + LFM2.5, 6 threads

RUN_ID `20260924T210214Z` contains eight retained 30-callback MediaPipe
windows (240 callbacks) with the CPU delegate verified.

Aggregate window-level means:

| Metric | CPU + LFM 6T |
|---|---:|
| End-to-end mean latency | 316.58 ms |
| Mean of window E2E p50 values | 313.38 ms |
| Mean of window E2E p95 values | 497.25 ms |
| Android preprocessing mean | 55.05 ms |
| MediaPipe callback latency mean | 261.54 ms |
| Mean of window MediaPipe p50 values | 257.38 ms |
| Mean of window MediaPipe p95 values | 419.00 ms |
| Callback throughput | 5.80 FPS |

The same run produced an LFM2.5 mean host latency of approximately 13.50 s
with 3/3 structurally valid and semantically correct style selections.

Relative to CPU + LFM 4T, CPU + LFM 6T changed the observed metrics as follows:

- LFM host latency: 13.29 -> 13.50 s (+1.6 %);
- end-to-end vision latency: 258.45 -> 316.58 ms (+22.5 %);
- preprocessing: 47.86 -> 55.05 ms (+15.0 %);
- MediaPipe mean: 210.59 -> 261.54 ms (+24.2 %);
- mean of MediaPipe window p95: 340.70 -> 419.00 ms (+23.0 %);
- callback throughput: 6.88 -> 5.80 FPS (-15.8 %).

The 6-thread CPU run was warmer (approximately 33.2–33.6 °C) than the 4-thread
CPU run (approximately 32.0–32.1 °C), so this comparison is system-level
evidence rather than proof that thread count alone caused all degradation.

## H7.9.9 concurrent configuration matrix

| MediaPipe delegate | LFM threads | LFM mean host latency | Vision E2E mean | MediaPipe mean | MediaPipe p95* | Callback FPS |
|---|---:|---:|---:|---:|---:|---:|
| GPU | 4 | 12.25 s | 286.72 ms | 226.67 ms | 345.67 ms | 6.68 |
| GPU | 6 | 11.46 s | 331.45 ms | 263.76 ms | 427.00 ms | 5.76 |
| CPU | 4 | 13.29 s | 258.45 ms | 210.59 ms | 340.70 ms | 6.88 |
| CPU | 6 | 13.50 s | 316.58 ms | 261.54 ms | 419.00 ms | 5.80 |

`* MediaPipe p95` is the mean of per-window p95 values, not a global p95 over
all callbacks.

### Engineering selection from the current matrix

For a safety-oriented architecture in which deterministic perception and alert
logic remain authoritative and the LLM is optional/non-critical, the observed
matrix favors **MediaPipe CPU + LFM2.5 with 4 threads** as the current runtime
allocation. It has the best concurrent vision E2E mean, best MediaPipe mean,
best callback throughput, and essentially tied best MediaPipe tail latency.
GPU + 4 threads makes the LFM response about 8.5 % faster, but at the cost of
about 9.9 % worse vision E2E latency and about 2.9 % lower callback throughput.

Neither 6-thread condition is attractive under the observed runs: GPU+6T
provides only a modest LFM speedup while degrading vision substantially, and
CPU+6T does not improve LFM latency at all while also degrading vision.

### Manuscript caution and next measurement refinement

The concurrent values above are session-level aggregates. They include the
whole benchmark interval, including windows before, between, and during LFM
generation. This is appropriate for system-level runtime selection, but it is
not equivalent to measuring only callbacks that overlap active token
generation.

Before using these values for a strong causal manuscript claim about LFM
contention, repeat the selected comparisons with explicit inference
start/stop markers or equivalent overlap labeling, matched starting thermal
conditions, and multiple independent runs. The present matrix is retained as
the engineering-selection dataset and as preliminary system-level experimental
evidence.


## H7.9.10 — GPU-offload feasibility check

The currently deployed Android llama.cpp runtime at
`/data/local/tmp/fatigue-llm/runtime/b11073/llama-b11073/llama-cli` was
queried with `--list-devices`. It returned:

```text
Available devices:
  (none)
```

This establishes that the deployed b11073 runtime has no accelerator backend
registered and is therefore unsuitable for GPU-offload experiments as-is. It
does **not** establish that the phone GPU is unsupported.

The CPU runtime must be retained unchanged as the reproducible baseline. A
separate Android Vulkan build will be prepared and deployed under a different
runtime directory, then validated first with `--list-devices` before any
`--gpu-layers/-ngl` benchmark is attempted.

If Vulkan detects the Adreno GPU, the planned heterogeneous allocation is:

- MediaPipe: CPU;
- LFM2.5: Vulkan GPU offload;
- minimal CPU threads for llama.cpp coordination.

This configuration will be compared against the accepted CPU-only concurrent
baseline using the same H7.9 telemetry.

## H7.9.10 Vulkan runtime result on VIVO Y20 Lite

A reproducible Android ARM64 Vulkan build of llama.cpp tag `b11073`
(commit `1aa2954bde90b1cb4d2dca96f90b07d7b155124b`) completed successfully
and produced `libggml-vulkan.so`, `llama-cli`, and `llama-server`.

When deployed separately to the phone and queried with `--list-devices`,
the runtime returned:

```text
ggml_vulkan: Error: Vulkan 1.2 required.
Available devices:
  (none)
```

This is a runtime capability failure, not a build failure. The b11073 Vulkan
backend explicitly rejects Vulkan API versions below 1.2. The Snapdragon 460 /
Adreno 610 platform is specified for Vulkan 1.1, so this phone cannot use the
b11073 Vulkan backend without changing backend requirements or using a
different backend/revision.

The accepted CPU runtime remains unchanged. The next GPU feasibility path is
OpenCL, which is supported by the Snapdragon 460 platform specification and
has a llama.cpp backend targeting Qualcomm Adreno. Because llama.cpp documents
limited support for older A6xx phone drivers, OpenCL must first be validated
against the actual vendor runtime on the device before any benchmark claim.

## H7.9.10 OpenCL runtime probe on V2058

A minimal ARM64 Android probe loaded `/vendor/lib64/libOpenCL.so` successfully
and enumerated one OpenCL platform and one device:

```text
Platform: ARM Platform
Vendor: ARM
Platform version: OpenCL 2.1 v1.r26p0-01eac0.455662e55e7c7fb95a4b1db7e7af49a8
Device: Mali-G52 MC2 r1p0
Device vendor: ARM
Device version: OpenCL 2.1 v1.r26p0-01eac0.455662e55e7c7fb95a4b1db7e7af49a8
Driver: 2.1
OpenCL C: OpenCL C 2.0 v1.r26p0-01eac0.455662e55e7c7fb95a4b1db7e7af49a8
```

Therefore OpenCL is genuinely usable at the runtime-enumeration level on this
device. The GPU exposed by the phone is ARM Mali-G52 MC2, not Qualcomm Adreno.
Any OpenCL llama.cpp experiment on this device must disable Adreno-specific
kernels and use the generic OpenCL path first.

The next feasibility test is a separate b11073 OpenCL build with
`GGML_OPENCL=ON`, `GGML_OPENCL_USE_ADRENO_KERNELS=OFF`, embedded kernels,
and an OpenCL target version compatible with the device's OpenCL C 2.0 runtime.

## H7.9.10 OpenCL build result

The Android ARM64 OpenCL build of llama.cpp tag `b11073`
(commit `1aa2954bde90b1cb4d2dca96f90b07d7b155124b`) completed successfully.

Configuration highlights:

```text
GGML_OPENCL=ON
GGML_OPENCL_PROFILING=OFF
GGML_OPENCL_EMBED_KERNELS=ON
GGML_OPENCL_USE_ADRENO_KERNELS=OFF
GGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF
GGML_OPENCL_TARGET_VERSION=200
```

The build linked against the device vendor `libOpenCL.so` only for
cross-linking and produced:

```text
bin/llama-cli
bin/llama-server
lib/libggml-opencl.so
lib/libggml-cpu.so
lib/libggml.so
lib/libllama.so
```

The backend embedded generic OpenCL kernels including Q4_K kernels.
The next step is runtime device enumeration on the phone using this separate
OpenCL runtime. The accepted CPU-only runtime remains unchanged.

## H7.9.10 OpenCL llama.cpp runtime result on Mali-G52

The separate Android ARM64 OpenCL runtime was deployed successfully and queried
with `llama-cli --list-devices`.

Observed output:

```text
ggml_opencl: unsupported GPU 'Mali-G52 MC2 r1p0'.
ggml_opencl: drop unsupported device 'Mali-G52 MC2 r1p0'.
Available devices:
  (none)
```

This confirms that the vendor OpenCL runtime itself is functional and exposes
the Mali-G52, but llama.cpp b11073 intentionally rejects this GPU family.
Therefore this is not an OpenCL loader or cross-build failure.

Combined GPU feasibility result for this phone:

- Vulkan: unavailable to b11073 because the phone exposes Vulkan 1.1 while the
  backend requires Vulkan 1.2.
- OpenCL: vendor runtime works and enumerates Mali-G52 MC2, but the b11073
  backend does not support this GPU family.

Do not bypass the GPU-family guard for production evaluation: multiple OpenCL
execution paths in this backend select implementation parameters for supported
GPU families and may assert or behave incorrectly for an unknown family.

The supported baseline remains the CPU runtime. For the current phone, further
work should focus on CPU scheduling/model latency and asynchronous contextual
LLM assistance rather than llama.cpp GPU offload.

## H7.10.2 JNI + llama.cpp backend smoke test

The Android app now links llama.cpp tag `b11073`
(commit `1aa2954bde90b1cb4d2dca96f90b07d7b155124b`) directly into the JNI
library using a CPU-only native build.

On-device verification returned:

```text
H7.10.1 JNI_OK arm64-v8a
H7.10.2 LLAMA_BACKEND_OK b11073 CPU
```

The corresponding logcat entry was emitted from the app process, confirming
the chain Kotlin -> JNI -> llama.cpp backend init/free -> Kotlin. No
`llama-cli`, ADB-launched inference process, Vulkan, or OpenCL backend is
involved in this path.

H7.10.2 is therefore accepted. The next step is H7.10.3: asynchronously load
the real LFM2.5 GGUF from app-accessible storage, inspect model metadata, and
unload it without running text generation.

## H7.10.3 real LFM2.5 GGUF load inside Android app

The app loaded and unloaded the real
`LFM2.5-1.2B-Instruct-Q4_K_M.gguf` through JNI using
`llama_model_load_from_file()`, CPU-only (`n_gpu_layers=0`).

Observed on device:

```text
H7.10.3 MODEL_LOAD_OK
desc=lfm2 1.2B Q4_K - Medium
size_bytes=728509440
params=1170340608
backend=CPU
ngl=0
elapsed_ms=1651
file_bytes=730895168
```

The model was stored in the app-specific external files directory and loaded
from the Android app process on a background thread. The model object was then
released with `llama_model_free()`.

H7.10.3 is accepted. H7.10.4 proceeds to context creation, chat-template
formatting, tokenization, decode/sampling, and a short generation fully inside
the app process.

## H7.10.5 first persistent-engine attempt: callback ordering crash

The first persistent-model integration attempt successfully initialized LFM2.5 and
completed one generation:

```text
STEP_INIT H7.10.5 INIT_OK threads=4 n_ctx=512 elapsed_ms=1558
STEP_GEN_1 H7.10.5 GENERATE_OK prompt_tokens=17 generated_tokens=2 threads=4 elapsed_ms=52892 response=UNO
```

The Android process then exited with `reason=5 (APP CRASH(NATIVE))`. Logcat
showed that the root cause was not llama.cpp or memory pressure. A MediaPipe
LIVE_STREAM callback reached `HeadPoseTemporalAnalyzer.update()` with a
non-increasing timestamp, which raised:

```text
java.lang.IllegalArgumentException:
Los timestamps deben ser estrictamente crecientes.
```

Because the exception remained pending while MediaPipe JNI continued, ART
aborted with SIGABRT. The llama.cpp thread was still inside CPU matrix
multiplication when the process was aborted.

The fix serializes MediaPipe LIVE_STREAM result delivery and drops stale or
duplicate result timestamps before forwarding callbacks to the temporal
analyzers. Their strict monotonic-timestamp invariant remains unchanged.

The same investigation also found that the embedded llama.cpp/ggml native code
inside the debug APK had no `-O2/-O3` optimization flags. Native compilation is
now forced to `-O3` with `NDEBUG` while the APK itself remains debuggable.
H7.10.5 must be rerun before acceptance.

## H7.10.5 persistent LFM2.5 engine accepted

After enabling optimized native compilation (`-O3`, `NDEBUG`) and serializing
MediaPipe LIVE_STREAM callback delivery, the persistent Android LLM engine
completed two consecutive generations with the same loaded model and then
released all native state cleanly.

Observed on device:

```text
H7.10.5 ENGINE_OK
INIT=H7.10.5 INIT_OK threads=4 n_ctx=512 elapsed_ms=440
GEN1=H7.10.5 GENERATE_OK prompt_tokens=17 generated_tokens=2 threads=4 elapsed_ms=2470 response=UNO
GEN2=H7.10.5 GENERATE_OK prompt_tokens=17 generated_tokens=2 threads=4 elapsed_ms=3026 response=DOS
RELEASE=H7.10.5 RELEASE_OK
```

The Android process remained alive after the test.

During concurrent MediaPipe + LLM execution, two duplicate/stale
LIVE_STREAM callbacks were detected and safely discarded:

```text
Dropping stale LIVE_STREAM callback:
timestampMs=499538431, lastDeliveredTimestampMs=499538431
```

This validates that the temporal analyzers can preserve their strict
monotonic-timestamp invariant under CPU contention without allowing callback
exceptions to escape into MediaPipe JNI.

Compared with the first unoptimized persistent attempt
(`elapsed_ms=52892` for a 2-token response), optimized embedded llama.cpp
reduced short-response generation latency to about 2.5-3.0 s while keeping
the model resident between requests.

H7.10.5 is accepted. The validated JNI lifecycle is now:

```text
initializeModel(path)
generate(prompt)
generate(prompt)
cancelGeneration()
releaseModel()
```

Next milestone: H7.11, where the persistent LLM engine must be integrated as a
fully asynchronous secondary contextual path. The deterministic fatigue alert
and offline TTS remain immediate and authoritative and must never wait for the
LLM.

## H7.11 asynchronous contextual path accepted

The Android app now runs deterministic safety speech and the local LFM2.5
contextual path independently.

Observed with the real `face_not_visible` trigger:

```text
ContextualAlert: Emitir MEDIUM ...
H7_11_ASYNC: DETERMINISTIC_TTS accepted=true
H7_11_ASYNC: SUBMIT trigger=face_not_visible urgency=MEDIUM
H7_11_ASYNC: LLM_SUBMITTED=true
```

While a generation was active, a later alert was not allowed to start another
LLM request:

```text
H7_11_ASYNC: SKIP ready=true closed=false in_flight=true
H7_11_ASYNC: LLM_SUBMITTED=false
```

A cancelled generation returned `H7.10.5 GENERATE_CANCELLED` and was converted
to a safe fallback without contextual TTS. Another completed generation was
classified `FALLBACK_UNSAFE`, also preventing unsafe model text from reaching
speech.

This validates the H7.11 safety architecture: deterministic alerts remain
immediate and authoritative, LLM work is secondary/asynchronous, concurrent
generations are suppressed, and rejected/cancelled output cannot override the
deterministic path.

The main remaining issue is contextual latency: the current verbose prompt and
48-token output budget required roughly 38-50 seconds on CPU. This is not a
failure of the asynchronous architecture, but it is unacceptable for the final
assistant. H7.12 will replace the verbose free-form prompt with a compact,
structured grounded prompt/output contract and a smaller output budget.
