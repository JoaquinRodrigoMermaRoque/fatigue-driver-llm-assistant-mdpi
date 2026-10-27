# H8 / H4 — Final offline contextual-assistance trial (public privacy-reviewed copy, 2026-10-09)

> Privacy note: this public copy changes only the direct personal name in the model-output quotation to `[name redacted]`. Numerical values, timestamps, runtime measurements, decisions, and interpretation are unchanged from the validated private source artifact.

## Device and runtime

- Device: Vivo V2058, Android 12.
- Local model: LFM2.5-1.2B-Instruct-Q4_K_M via llama.cpp/JNI, CPU.
- Persistent context: 4 threads, n_ctx=512.
- Network condition during trial: airplane mode enabled.
- The fatigue-state decision and immediate safety recommendation remained deterministic; the LLM was used only as a deferred contextual layer.

## User context stored locally

At 17:09:47 the app stored:

```text
sleep_last_night_h=5.0
sleep_avg_3d_h=6.0
sleepiness_1_9=5
work_shift_h=8.0
night_shift=false
physical_demand=light
mental_demand=light
```

At the critical alert the context layer applied:

```text
sleep_last_night_h=5.0
sleep_avg_3d_h=6.0
hours_awake=10.2
sleepiness_1_9=5
work_shift_h=8.0
night_shift=false
physical_demand=light
mental_demand=light
```

## Critical event and deterministic alert

At 17:09:56.107 the planner emitted a CRITICAL STOP_SAFELY decision for `prolonged_eye_closure`.

The optional LLM was explicitly deferred until eye reopening:

```text
LLM_DEFERRED trigger=prolonged_eye_closure until=eyes_reopened
```

After reopening, at closure_s=0.00:

```text
LLM_DEFERRED_SUBMIT=true trigger=prolonged_eye_closure closure_s=0.00
```

The immediate deterministic TTS was accepted:

```text
DETERMINISTIC_TTS accepted=true trigger=prolonged_eye_closure urgency=CRITICAL
```

## Local LLM result

Prompt/runtime:

```text
PROMPT chars=191 max_new=16
prompt_tokens=72
generated_tokens=16
threads=4
prefill_ms=7333
elapsed_ms=13025
```

The local model completed generation (`GENERATE_OK`) and produced:

```text
[name redacted] cierra los ojos; contexto: reportaste 5.
```

This candidate did not satisfy the contextual safety/grounding validator, so it was not used as the spoken LLM message:

```text
RESULT status=FALLBACK_CONTEXTUAL elapsed_ms=13135
```

The system instead used the local deterministic contextual fallback. Contextual TTS was accepted and completed:

```text
CONTEXTUAL_TTS accepted=true status=FALLBACK_CONTEXTUAL
```

The spoken fallback included the stored sleep context ("reportaste 5 horas de sueño").

## Offline evidence

The device reported:

```text
airplane_mode_on=1
```

The connectivity check returned:

```text
connect: Network is unreachable
```

## Interpretation

The exact H4 condition is supported in this controlled prototype trial: local context retrieval, deterministic recommendation selection, and offline TTS continued operating with airplane mode enabled and no reachable network.

The local LLM execution path was also demonstrated offline: the model initialized locally and completed a 16-token generation in 13.025 s after being deferred until eye reopening. However, the generated candidate was incomplete/malformed for the required contextual wording and was rejected by the safety/grounding layer. Therefore, the final contextual spoken message in this trial must be reported as a deterministic contextual fallback, not as LLM-generated speech.

This distinction should be preserved in the Results/Discussion. The trial validates the safety architecture (deterministic immediate alert + deferred optional local generation + guarded fallback), while showing a practical limitation of the 1.2B local model on this handset for concise, reliably grounded Spanish contextual generation under the selected token budget.
