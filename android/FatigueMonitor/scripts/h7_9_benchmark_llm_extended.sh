#!/usr/bin/env bash
set -Eeuo pipefail

# H7.9.4/H7.9.5
# Benchmark extendido del LLM local SIN JNI.
#
# Distingue:
#   - process-cold: un proceso nuevo por inferencia;
#   - warm-server: mismo llama-server, modelo residente;
#   - concurrent: APK en primer plano mientras se mide el LLM.
#
# No modifica la APK ni sustituye el benchmark validado H7.9.2.

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly ENV_PATH="${LOCAL_ROOT}/runtime.env"

readonly MODE="${MODE:-all}"                 # process-cold | warm-server | all
readonly CONTEXT_SIZE="${CONTEXT_SIZE:-1024}"
readonly MAX_TOKENS="${MAX_TOKENS:-48}"
readonly THREADS_LIST="${THREADS_LIST:-4 6}"
readonly TEMPERATURES="${TEMPERATURES:-0.0 0.3 0.7}"
readonly REPEATS="${REPEATS:-5}"
readonly SEED="${SEED:-42}"
readonly TIMEOUT_S="${TIMEOUT_S:-180}"
readonly SAMPLE_INTERVAL_S="${SAMPLE_INTERVAL_S:-0.25}"
readonly MAX_BATTERY_TEMP_TENTHS="${MAX_BATTERY_TEMP_TENTHS:-420}"
readonly EXPECTED_MODEL_PRESET="${EXPECTED_MODEL_PRESET:-1.7b-q4_k_m}"
readonly WARM_CONTEXT_MODE="${WARM_CONTEXT_MODE:-static}"
readonly OUTPUT_MODE="${OUTPUT_MODE:-free}" # free | template-selector | style-selector
readonly TEMPLATE_HELPER="${PROJECT_DIR}/scripts/h7_9_template_selector.py"
readonly MEDIAPIPE_TELEMETRY_PARSER="${PROJECT_DIR}/scripts/h7_9_parse_mediapipe_telemetry.py"

readonly CONCURRENT_APP="${CONCURRENT_APP:-0}"
readonly LAUNCH_APP="${LAUNCH_APP:-0}"
readonly APP_PACKAGE="${APP_PACKAGE:-com.google.mediapipe.examples.facelandmarker}"
readonly EXPECTED_MEDIAPIPE_DELEGATE="${EXPECTED_MEDIAPIPE_DELEGATE:-}" # CPU | GPU | empty
readonly APP_ACTIVITY="${APP_ACTIVITY:-.MainActivity}"

readonly REMOTE_SERVER_PORT="${REMOTE_SERVER_PORT:-8080}"
readonly HOST_SERVER_PORT="${HOST_SERVER_PORT:-18080}"

if [[ ! -f "${ENV_PATH}" ]]; then
    echo "No existe ${ENV_PATH}" >&2
    echo "Ejecuta primero: MODEL_PRESET=1.7b-q4_k_m bash scripts/h7_9_prepare_llm_android.sh" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${ENV_PATH}"

case "${MODEL_FAMILY:-qwen3}" in
    lfm2.5)
        readonly TOP_K="${TOP_K:-50}"
        readonly TOP_P="${TOP_P:-0.95}"
        readonly REPEAT_PENALTY="${REPEAT_PENALTY:-1.05}"
        readonly PRESENCE_PENALTY="${PRESENCE_PENALTY:-0.0}"
        ;;
    *)
        readonly TOP_K="${TOP_K:-20}"
        readonly TOP_P="${TOP_P:-0.8}"
        readonly REPEAT_PENALTY="${REPEAT_PENALTY:-1.0}"
        readonly PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
        ;;
esac

if [[ -n "${EXPECTED_MODEL_PRESET}" ]] &&
   [[ "${MODEL_PRESET:-unknown}" != "${EXPECTED_MODEL_PRESET}" ]]; then
    echo "Modelo activo incorrecto." >&2
    echo "Esperado: ${EXPECTED_MODEL_PRESET}" >&2
    echo "Activo: ${MODEL_PRESET:-unknown} (${MODEL_FILE:-unknown})" >&2
    exit 1
fi

for variable_name in REMOTE_ROOT REMOTE_BINARY REMOTE_LIBRARY_PATH REMOTE_MODEL; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Falta ${variable_name} en ${ENV_PATH}" >&2
        exit 1
    fi
done

for command_name in adb awk curl date grep paste python3 sort timeout tr; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Falta el comando requerido: ${command_name}" >&2
        exit 1
    fi
done

case "${MODE}" in
    process-cold|warm-server|all) ;;
    *)
        echo "MODE inválido: ${MODE}" >&2
        echo "Usa process-cold, warm-server o all." >&2
        exit 1
        ;;
esac

if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "REPEATS inválido: ${REPEATS}" >&2
    exit 1
fi

case "${WARM_CONTEXT_MODE}" in
    static|dynamic) ;;
    *)
        echo "WARM_CONTEXT_MODE inválido: ${WARM_CONTEXT_MODE}" >&2
        exit 1
        ;;
esac

case "${EXPECTED_MEDIAPIPE_DELEGATE}" in
    ""|CPU|GPU) ;;
    *)
        echo "EXPECTED_MEDIAPIPE_DELEGATE debe ser CPU, GPU o vacío." >&2
        exit 1
        ;;
esac

case "${OUTPUT_MODE}" in
    free|template-selector|style-selector) ;;
    *)
        echo "OUTPUT_MODE inválido: ${OUTPUT_MODE}" >&2
        echo "Usa free, template-selector o style-selector." >&2
        exit 1
        ;;
esac

if [[ "${OUTPUT_MODE}" == "template-selector" || "${OUTPUT_MODE}" == "style-selector" ]]; then
    if [[ ! -f "${TEMPLATE_HELPER}" ]]; then
        echo "No existe ${TEMPLATE_HELPER}" >&2
        exit 1
    fi
    if [[ "${MODE}" == "process-cold" ]]; then
        echo "Los selectores se evalúan mediante warm-server/json_schema." >&2
        exit 1
    fi
fi

adb get-state >/dev/null

readonly REMOTE_BIN_DIR="${REMOTE_BINARY%/*}"
readonly REMOTE_COMPLETION="${REMOTE_BIN_DIR}/llama-completion"
readonly REMOTE_SERVER="${REMOTE_BIN_DIR}/llama-server"

if adb shell test -f "${REMOTE_COMPLETION}" >/dev/null 2>&1; then
    adb shell chmod 755 "${REMOTE_COMPLETION}"
    readonly INFERENCE_BINARY="${REMOTE_COMPLETION}"
    readonly INFERENCE_PROCESS_NAME="llama-completion"
else
    readonly INFERENCE_BINARY="${REMOTE_BINARY}"
    readonly INFERENCE_PROCESS_NAME="llama-cli"
fi

readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_DIR="${LOCAL_ROOT}/benchmarks_extended/${RUN_ID}"
readonly SUMMARY_PATH="${RESULT_DIR}/summary.tsv"
readonly LOCAL_PROMPT="${RESULT_DIR}/prompt.txt"
readonly REMOTE_PROMPT="${REMOTE_ROOT}/prompt_h7_9_extended.txt"
readonly RESOURCE_HEADER=$'elapsed_ms\tllm_rss_kb\tapp_rss_kb\tmem_available_kb\tswap_free_kb\tbattery_temp_tenths'

mkdir -p "${RESULT_DIR}"

write_static_prompt() {
    case "${MODEL_FAMILY:-qwen3}" in
        lfm2.5)
            cat > "${LOCAL_PROMPT}" <<'PROMPT'
<|im_start|>system
Eres el módulo lingüístico local de un asistente de seguridad para conducción.
Solo puedes describir señales y antecedentes incluidos en el contexto.
No diagnostiques, no inventes causas y no contradigas al motor determinista.
No repitas ni reformules la acción obligatoria.
Usa español claro, tono calmado y no moralizante.
<|im_end|>
<|im_start|>user
snapshot_id: bench-critical-repeat-001
estado actual: FATIGA GRAVE
disparador principal: cierre ocular prolongado
eventos recientes: dos bostezos y un cabeceo
historial: segundo episodio crítico de la sesión
acción reservada para otro módulo: detenerse en un lugar seguro y descansar antes de continuar

Redacta solamente un comentario contextual de una oración y entre 10 y 24 palabras.
Menciona el cierre ocular y que la señal se ha repetido.
No des instrucciones.
No escribas la acción reservada.
No inventes sueño, ejercicio, enfermedades, causas ni recomendaciones.
Devuelve únicamente la oración.
<|im_end|>
<|im_start|>assistant
PROMPT
            ;;
        *)
            cat > "${LOCAL_PROMPT}" <<'PROMPT'
<|im_start|>system
Eres el módulo lingüístico local de un asistente de seguridad para conducción.
Solo puedes describir señales y antecedentes incluidos en el contexto.
No diagnostiques, no inventes causas y no contradigas al motor determinista.
No repitas ni reformules la acción obligatoria.
Usa español claro, tono calmado y no moralizante.
<|im_end|>
<|im_start|>user
snapshot_id: bench-critical-repeat-001
estado actual: FATIGA GRAVE
disparador principal: cierre ocular prolongado
eventos recientes: dos bostezos y un cabeceo
historial: segundo episodio crítico de la sesión
acción reservada para otro módulo: detenerse en un lugar seguro y descansar antes de continuar

Redacta solamente un comentario contextual de una oración y entre 10 y 24 palabras.
Menciona el cierre ocular y que la señal se ha repetido.
No des instrucciones.
No escribas la acción reservada.
No inventes sueño, ejercicio, enfermedades, causas ni recomendaciones.
Devuelve únicamente la oración.
/no_think
<|im_end|>
<|im_start|>assistant
<think>

</think>

PROMPT
            ;;
    esac
}

write_static_prompt

adb push "${LOCAL_PROMPT}" "${REMOTE_PROMPT}" >/dev/null

monotonic_milliseconds() {
    awk '{printf "%.0f\n", $1 * 1000}' /proc/uptime
}

device_uptime_milliseconds() {
    adb shell "awk '{printf \"%.0f\\n\", \$1 * 1000}' /proc/uptime" |
        tr -d '\r'
}

battery_temperature() {
    adb shell dumpsys battery |
        awk '/temperature:/ {print $2; exit}' |
        tr -d '\r'
}

remote_first_pid() {
    local process_name="$1"
    adb shell "pidof ${process_name} 2>/dev/null || true" |
        tr -d '\r' |
        awk '{print $1}'
}

remote_rss_kb() {
    local pid="$1"
    if [[ -z "${pid}" ]]; then
        echo "NA"
        return
    fi

    adb shell "awk '/^VmRSS:/ {print \$2; exit}' /proc/${pid}/status 2>/dev/null" |
        tr -d '\r' |
        awk 'NF {print; found=1} END {if (!found) print "NA"}'
}

meminfo_value() {
    local key="$1"
    adb shell "awk '/^${key}:/ {print \$2; exit}' /proc/meminfo" |
        tr -d '\r' |
        awk 'NF {print; found=1} END {if (!found) print "NA"}'
}

app_pid() {
    adb shell "pidof ${APP_PACKAGE} 2>/dev/null || true" |
        tr -d '\r' |
        awk '{print $1}'
}

sample_resources() {
    local start_ms="$1"
    local llm_process="$2"
    local destination="$3"
    local now_ms
    local elapsed_ms
    local llm_pid
    local current_app_pid
    local llm_rss
    local app_rss
    local mem_available
    local swap_free
    local temp

    now_ms="$(monotonic_milliseconds)"
    elapsed_ms=$(( now_ms - start_ms ))
    llm_pid="$(remote_first_pid "${llm_process}")"
    current_app_pid="$(app_pid)"
    llm_rss="$(remote_rss_kb "${llm_pid}")"
    app_rss="$(remote_rss_kb "${current_app_pid}")"
    mem_available="$(meminfo_value MemAvailable)"
    swap_free="$(meminfo_value SwapFree)"
    temp="$(battery_temperature)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\n'         "${elapsed_ms}"         "${llm_rss}"         "${app_rss}"         "${mem_available}"         "${swap_free}"         "${temp:-NA}"         >> "${destination}"
}

metric_max() {
    local file="$1"
    local column="$2"
    awk -F '\t' -v c="${column}" '
        NR > 1 && $c ~ /^[0-9]+$/ && $c + 0 > maximum {
            maximum = $c + 0
        }
        END {
            if (maximum > 0) print maximum
            else print "NA"
        }
    ' "${file}"
}

metric_min() {
    local file="$1"
    local column="$2"
    awk -F '\t' -v c="${column}" '
        NR > 1 && $c ~ /^[0-9]+$/ {
            if (!found || $c + 0 < minimum) minimum = $c + 0
            found = 1
        }
        END {
            if (found) print minimum
            else print "NA"
        }
    ' "${file}"
}

assert_temperature_safe() {
    local temperature
    temperature="$(battery_temperature)"

    if [[ "${temperature}" =~ ^[0-9]+$ ]] &&
       (( temperature >= MAX_BATTERY_TEMP_TENTHS )); then
        echo "Prueba detenida: batería a ${temperature} décimas de °C." >&2
        exit 1
    fi
}

stop_remote_llm() {
    local process_name
    local ids
    local pid

    for process_name in llama-cli llama-completion llama-server; do
        ids="$(adb shell "pidof ${process_name} 2>/dev/null || true" | tr -d '\r')"
        for pid in ${ids}; do
            adb shell kill "${pid}" >/dev/null 2>&1 || true
        done
    done
}

prepare_concurrent_app() {
    if [[ "${CONCURRENT_APP}" != "1" ]]; then
        return
    fi

    if [[ "${LAUNCH_APP}" == "1" ]]; then
        adb shell am start -n "${APP_PACKAGE}/${APP_ACTIVITY}" >/dev/null
        sleep 2
    fi

    if [[ -z "$(app_pid)" ]]; then
        echo "CONCURRENT_APP=1 pero la APK no está ejecutándose." >&2
        echo "Abre FatigueMonitor y deja la cámara visible, o usa LAUNCH_APP=1." >&2
        exit 1
    fi

    echo "APK concurrente detectada: ${APP_PACKAGE} pid=$(app_pid)"
    echo "Mantén la pantalla desbloqueada y la vista de cámara visible durante la prueba."
}

write_device_snapshot() {
    local destination="$1"
    {
        echo "=== run ==="
        echo "run_id=${RUN_ID}"
        echo "mode=${MODE}"
        echo "model_preset=${MODEL_PRESET:-unknown}"
        echo "model_file=${MODEL_FILE:-unknown}"
        echo "model_family=${MODEL_FAMILY:-unknown}"
        echo "output_mode=${OUTPUT_MODE}"
        echo "top_k=${TOP_K}"
        echo "top_p=${TOP_P}"
        echo "repeat_penalty=${REPEAT_PENALTY}"
        echo "presence_penalty=${PRESENCE_PENALTY}"
        echo "concurrent_app=${CONCURRENT_APP}"
        echo "=== device ==="
        adb shell getprop ro.product.manufacturer
        adb shell getprop ro.product.model
        adb shell getprop ro.build.version.release
        adb shell getprop ro.build.version.sdk
        printf 'boot_id='
        adb shell cat /proc/sys/kernel/random/boot_id
        echo "=== battery ==="
        adb shell dumpsys battery | grep -E 'level:|temperature:'
        echo "=== memory ==="
        adb shell "grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo"
        echo "=== app ==="
        echo "package=${APP_PACKAGE}"
        echo "pid=$(app_pid)"
    } > "${destination}" 2>&1
}

record_summary() {
    local phase="$1"
    local threads="$2"
    local temperature="$3"
    local repeat="$4"
    local elapsed_ms="$5"
    local resource_path="$6"
    local exit_code="$7"
    local temp_before="$8"
    local temp_after="$9"

    local peak_llm_rss
    local peak_app_rss
    local min_mem_available
    local min_swap_free

    peak_llm_rss="$(metric_max "${resource_path}" 2)"
    peak_app_rss="$(metric_max "${resource_path}" 3)"
    min_mem_available="$(metric_min "${resource_path}" 4)"
    min_swap_free="$(metric_min "${resource_path}" 5)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'         "${phase}"         "${threads}"         "${temperature}"         "${repeat}"         "${elapsed_ms}"         "${peak_llm_rss}"         "${peak_app_rss}"         "${min_mem_available}"         "${min_swap_free}"         "${temp_before:-NA}"         "${temp_after:-NA}"         "${exit_code}"         >> "${SUMMARY_PATH}"
}

run_process_cold() {
    local threads="$1"
    local temperature="$2"
    local repetition="$3"
    local prefix="process_cold_t${threads}_temp${temperature}_r${repetition}"
    local output_path="${RESULT_DIR}/${prefix}.log"
    local resource_path="${RESULT_DIR}/${prefix}_resources.tsv"
    local start_ms
    local end_ms
    local elapsed_ms
    local host_pid
    local exit_code
    local temp_before
    local temp_after

    assert_temperature_safe
    stop_remote_llm
    temp_before="$(battery_temperature)"
    printf '%s\n' "${RESOURCE_HEADER}" > "${resource_path}"

    echo "Ejecutando ${prefix}..."
    start_ms="$(monotonic_milliseconds)"

    set +e
    timeout "${TIMEOUT_S}" adb shell         env "LD_LIBRARY_PATH=${REMOTE_LIBRARY_PATH}"         "${INFERENCE_BINARY}"         -m "${REMOTE_MODEL}"         -f "${REMOTE_PROMPT}"         -c "${CONTEXT_SIZE}"         -n "${MAX_TOKENS}"         -t "${threads}"         -ngl 0         --seed "${SEED}"         --temp "${temperature}"         --top-k "${TOP_K}"         --top-p "${TOP_P}"         --min-p 0.0         --repeat-penalty "${REPEAT_PENALTY}"         --presence-penalty "${PRESENCE_PENALTY}"         --no-display-prompt         --no-context-shift         -no-cnv         > "${output_path}" 2>&1 &
    host_pid=$!

    while kill -0 "${host_pid}" 2>/dev/null; do
        sample_resources "${start_ms}" "${INFERENCE_PROCESS_NAME}" "${resource_path}"
        sleep "${SAMPLE_INTERVAL_S}"
    done

    wait "${host_pid}"
    exit_code=$?
    set -e

    end_ms="$(monotonic_milliseconds)"
    elapsed_ms=$(( end_ms - start_ms ))
    temp_after="$(battery_temperature)"

    if (( exit_code != 0 )); then
        stop_remote_llm
    fi

    record_summary         "process-cold"         "${threads}"         "${temperature}"         "${repetition}"         "${elapsed_ms}"         "${resource_path}"         "${exit_code}"         "${temp_before}"         "${temp_after}"

    grep -E 'prompt eval time|eval time|total time|tokens per second' "${output_path}" |
        tail -n 10 || true

    if (( exit_code != 0 )); then
        echo "Falló ${prefix} con código ${exit_code}." >&2
        tail -n 30 "${output_path}" >&2
        return "${exit_code}"
    fi
}

server_available() {
    adb shell test -f "${REMOTE_SERVER}" >/dev/null 2>&1
}

wait_server_ready() {
    local attempts=0
    while (( attempts < 120 )); do
        if curl --silent --fail "http://127.0.0.1:${HOST_SERVER_PORT}/health" >/dev/null 2>&1 ||
           curl --silent --fail "http://127.0.0.1:${HOST_SERVER_PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        attempts=$(( attempts + 1 ))
        sleep 0.5
    done
    return 1
}

build_server_payload() {
    local temperature="$1"
    local destination="$2"
    local repetition="$3"

    python3 - "${LOCAL_PROMPT}" "${temperature}" "${MAX_TOKENS}" "${SEED}" "${WARM_CONTEXT_MODE}" "${repetition}" "${MODEL_FAMILY:-qwen3}" "${TOP_K}" "${TOP_P}" "${REPEAT_PENALTY}" "${PRESENCE_PENALTY}" "${OUTPUT_MODE}" "${TEMPLATE_HELPER}" > "${destination}" <<'PY'
import json
import pathlib
import subprocess
import sys

(
    prompt_path,
    temperature,
    max_tokens,
    seed,
    context_mode,
    repetition,
    model_family,
    top_k,
    top_p,
    repeat_penalty,
    presence_penalty,
    output_mode,
    template_helper,
) = sys.argv[1:]
repetition = int(repetition)

scenarios = [
    """snapshot_id: warm-dynamic-001
estado actual: FATIGA GRAVE
disparador principal: cierre ocular prolongado
eventos recientes: dos bostezos y un cabeceo
historial: segundo episodio crítico de la sesión
acción reservada para otro módulo: detenerse en un lugar seguro y descansar antes de continuar""",
    """snapshot_id: warm-dynamic-002
estado actual: FATIGA
disparador principal: bostezos repetidos
eventos recientes: tres bostezos y una desviación prolongada de mirada
historial: antes solo se había observado somnolencia leve
acción reservada para otro módulo: planificar una pausa segura lo antes posible""",
    """snapshot_id: warm-dynamic-003
estado actual: RECUPERACIÓN
disparador principal: recuperación sostenida
eventos recientes: sin nuevos bostezos ni cabeceos
historial: hubo un episodio crítico previo en la sesión
acción reservada para otro módulo: mantener la atención y continuar solo si la recuperación se sostiene"""
]

semantic_templates = [
    "repeated_primary_with_recent_events",
    "recent_events_changed_from_history",
    "recovery_after_critical",
]
semantic_template = semantic_templates[(repetition - 1) % len(semantic_templates)]

if output_mode == "template-selector":
    scenario = scenarios[(repetition - 1) % len(scenarios)]
    system_text = """Eres un selector contextual local para un sistema de seguridad de conducción.
No redactes frases para el usuario.
Selecciona únicamente identificadores aprobados a partir de los hechos proporcionados.
No inventes hechos, causas, diagnósticos ni recomendaciones."""
    user_text = (
        scenario
        + "\n\nCombinaciones aprobadas:"
        + "\n1) cierre ocular repetido + bostezos/cabeceo => "
          "template=repeated_primary_with_recent_events, "
          "primary=eye_closure, secondary=yawns_and_nod"
        + "\n2) bostezos + mirada desviada con cambio respecto al historial => "
          "template=recent_events_changed_from_history, "
          "primary=yawns, secondary=gaze_away"
        + "\n3) recuperación después de episodio crítico => "
          "template=recovery_after_critical, "
          "primary=recovery, secondary=none"
        + "\nDevuelve solamente el objeto JSON permitido."
    )
elif output_mode == "style-selector":
    scenario = scenarios[(repetition - 1) % len(scenarios)]
    system_text = """Eres un selector de estilo para un asistente de conducción.
La lógica determinista ya fijó el significado del mensaje.
No puedes cambiar hechos, gravedad ni acción. Solo elige una variante de redacción aprobada."""
    user_text = (
        scenario
        + "\nplantilla semántica ya fijada: " + semantic_template
        + "\nElige únicamente una variante:"
        + "\na = natural y equilibrada"
        + "\nb = directa y compacta"
        + "\nc = conversacional y factual"
        + "\nDevuelve solamente el objeto JSON permitido."
    )
else:
    if context_mode == "static":
        prompt = pathlib.Path(prompt_path).read_text(encoding="utf-8")
    else:
        system_text = """Eres el módulo lingüístico local de un asistente de seguridad para conducción.
Solo puedes describir señales y antecedentes incluidos en el contexto.
No diagnostiques, no inventes causas y no contradigas al motor determinista.
No repitas ni reformules la acción obligatoria.
Usa español claro, tono calmado y no moralizante."""
        scenario = scenarios[(repetition - 1) % len(scenarios)]
        instruction = [
            "Menciona el cierre ocular y que la señal se ha repetido.",
            "Menciona los bostezos y el cambio respecto al historial.",
            "Menciona la recuperación y el episodio previo sin afirmar que el riesgo desapareció.",
        ][(repetition - 1) % len(scenarios)]
        user_text = (
            scenario
            + "\n" + instruction
            + "\nRedacta únicamente un comentario contextual de una oración y entre 10 y 24 palabras."
            + "\nNo des instrucciones ni escribas la acción reservada."
            + "\nNo inventes causas ni recomendaciones."
            + "\nDevuelve únicamente la oración."
        )

if output_mode in ("template-selector", "style-selector") or context_mode != "static":
    if model_family == "lfm2.5":
        prompt = (
            "<|im_start|>system\n"
            + system_text
            + "<|im_end|>\n<|im_start|>user\n"
            + user_text
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        prompt = (
            "<|im_start|>system\n"
            + system_text
            + "<|im_end|>\n<|im_start|>user\n"
            + user_text
            + "\n/no_think\n<|im_end|>\n<|im_start|>assistant\n"
            + "<think>\n\n</think>\n\n"
        )

payload = {
    "prompt": prompt,
    "n_predict": int(max_tokens),
    "temperature": float(temperature),
    "top_k": int(top_k),
    "top_p": float(top_p),
    "min_p": 0.0,
    "repeat_penalty": float(repeat_penalty),
    "presence_penalty": float(presence_penalty),
    "seed": int(seed),
    "cache_prompt": True,
    "stream": False,
}

if output_mode == "template-selector":
    schema_text = subprocess.check_output(
        [sys.executable, template_helper, "schema"],
        text=True,
    ).strip()
    payload["json_schema"] = json.loads(schema_text)
elif output_mode == "style-selector":
    payload["n_predict"] = min(int(max_tokens), 16)
    schema_text = subprocess.check_output(
        [sys.executable, template_helper, "style-schema"],
        text=True,
    ).strip()
    payload["json_schema"] = json.loads(schema_text)

print(json.dumps(payload, ensure_ascii=False))
PY
}

run_warm_server_for_threads() {
    local threads="$1"
    local server_log="${RESULT_DIR}/warm_server_t${threads}.log"
    local server_host_pid
    local temperature
    local repetition
    local prefix
    local payload_path
    local output_path
    local resource_path
    local start_ms
    local end_ms
    local elapsed_ms
    local request_pid
    local exit_code
    local temp_before
    local temp_after

    if ! server_available; then
        echo "SKIPPED warm-server: el runtime no contiene llama-server."
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n'             "warm-server-SKIPPED" "${threads}" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "NA" "0"             >> "${SUMMARY_PATH}"
        return 0
    fi

    stop_remote_llm
    adb shell chmod 755 "${REMOTE_SERVER}"
    adb forward --remove "tcp:${HOST_SERVER_PORT}" >/dev/null 2>&1 || true
    adb forward "tcp:${HOST_SERVER_PORT}" "tcp:${REMOTE_SERVER_PORT}" >/dev/null

    echo "Cargando warm server con ${threads} hilos..."
    adb shell         env "LD_LIBRARY_PATH=${REMOTE_LIBRARY_PATH}"         "${REMOTE_SERVER}"         -m "${REMOTE_MODEL}"         -c "${CONTEXT_SIZE}"         -t "${threads}"         -ngl 0         --host 127.0.0.1         --port "${REMOTE_SERVER_PORT}"         > "${server_log}" 2>&1 &
    server_host_pid=$!

    if ! wait_server_ready; then
        echo "llama-server no quedó listo." >&2
        tail -n 40 "${server_log}" >&2 || true
        kill "${server_host_pid}" >/dev/null 2>&1 || true
        stop_remote_llm
        return 1
    fi

    for temperature in ${TEMPERATURES}; do
        for (( repetition = 1; repetition <= REPEATS; repetition += 1 )); do
            assert_temperature_safe
            prefix="warm_server_t${threads}_temp${temperature}_r${repetition}"
            payload_path="${RESULT_DIR}/${prefix}_payload.json"
            output_path="${RESULT_DIR}/${prefix}.json"
            resource_path="${RESULT_DIR}/${prefix}_resources.tsv"
            printf '%s\n' "${RESOURCE_HEADER}" > "${resource_path}"
            build_server_payload "${temperature}" "${payload_path}" "${repetition}"

            temp_before="$(battery_temperature)"
            start_ms="$(monotonic_milliseconds)"

            set +e
            timeout "${TIMEOUT_S}" curl                 --silent                 --show-error                 --fail                 --header 'Content-Type: application/json'                 --request POST                 --data-binary "@${payload_path}"                 "http://127.0.0.1:${HOST_SERVER_PORT}/completion"                 > "${output_path}" 2>&1 &
            request_pid=$!

            while kill -0 "${request_pid}" 2>/dev/null; do
                sample_resources "${start_ms}" "llama-server" "${resource_path}"
                sleep "${SAMPLE_INTERVAL_S}"
            done

            wait "${request_pid}"
            exit_code=$?
            set -e

            end_ms="$(monotonic_milliseconds)"
            elapsed_ms=$(( end_ms - start_ms ))
            temp_after="$(battery_temperature)"

            record_summary                 "warm-server-${WARM_CONTEXT_MODE}"                 "${threads}"                 "${temperature}"                 "${repetition}"                 "${elapsed_ms}"                 "${resource_path}"                 "${exit_code}"                 "${temp_before}"                 "${temp_after}"

            if (( exit_code != 0 )); then
                echo "Falló ${prefix} con código ${exit_code}." >&2
                tail -n 20 "${output_path}" >&2 || true
                kill "${server_host_pid}" >/dev/null 2>&1 || true
                stop_remote_llm
                return "${exit_code}"
            fi

            if [[ "${OUTPUT_MODE}" == "template-selector" ]]; then
                local rendered_path="${RESULT_DIR}/${prefix}_rendered.json"
                python3 "${TEMPLATE_HELPER}" render-response "${output_path}" \
                    > "${rendered_path}"
                echo "Selector r${repetition}: $(cat "${rendered_path}")"
            elif [[ "${OUTPUT_MODE}" == "style-selector" ]]; then
                local rendered_path="${RESULT_DIR}/${prefix}_rendered.json"
                local semantic_template
                case $(( (repetition - 1) % 3 )) in
                    0) semantic_template="repeated_primary_with_recent_events" ;;
                    1) semantic_template="recent_events_changed_from_history" ;;
                    2) semantic_template="recovery_after_critical" ;;
                esac
                python3 "${TEMPLATE_HELPER}" render-style-response \
                    "${semantic_template}" "${output_path}" > "${rendered_path}"
                echo "Estilo r${repetition}: $(cat "${rendered_path}")"
            fi
        done
    done

    kill "${server_host_pid}" >/dev/null 2>&1 || true
    stop_remote_llm
    adb forward --remove "tcp:${HOST_SERVER_PORT}" >/dev/null 2>&1 || true
}

cleanup() {
    stop_remote_llm
    adb forward --remove "tcp:${HOST_SERVER_PORT}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

printf '%s\n'     $'phase\tthreads\ttemperature\trepeat\telapsed_ms\tpeak_llm_rss_kb\tpeak_app_rss_kb\tmin_mem_available_kb\tmin_swap_free_kb\ttemp_before_tenths\ttemp_after_tenths\texit_code'     > "${SUMMARY_PATH}"

prepare_concurrent_app
write_device_snapshot "${RESULT_DIR}/device_before.txt"

adb shell setprop log.tag.H7_9_PERF VERBOSE >/dev/null 2>&1 || true
adb shell setprop log.tag.FaceLandmarkerHelper VERBOSE >/dev/null 2>&1 || true
adb logcat -c >/dev/null 2>&1 || true
DEVICE_TELEMETRY_START_MS="$(device_uptime_milliseconds)"
printf '%s\n' "${DEVICE_TELEMETRY_START_MS}" \
    > "${RESULT_DIR}/mediapipe_measurement_start_uptime_ms.txt"

if [[ "${MODE}" == "process-cold" || "${MODE}" == "all" ]]; then
    for threads in ${THREADS_LIST}; do
        if ! [[ "${threads}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Cantidad de hilos inválida: ${threads}" >&2
            exit 1
        fi

        for temperature in ${TEMPERATURES}; do
            for (( repetition = 1; repetition <= REPEATS; repetition += 1 )); do
                run_process_cold "${threads}" "${temperature}" "${repetition}"
            done
        done
    done
fi

if [[ "${MODE}" == "warm-server" || "${MODE}" == "all" ]]; then
    for threads in ${THREADS_LIST}; do
        run_warm_server_for_threads "${threads}"
    done
fi

write_device_snapshot "${RESULT_DIR}/device_after.txt"
adb logcat -d > "${RESULT_DIR}/logcat_after.txt" 2>&1 || true
adb shell dumpsys meminfo "${APP_PACKAGE}" > "${RESULT_DIR}/app_meminfo_after.txt" 2>&1 || true

if [[ -f "${MEDIAPIPE_TELEMETRY_PARSER}" ]]; then
    python3 "${MEDIAPIPE_TELEMETRY_PARSER}" \
        "${RESULT_DIR}/logcat_after.txt" \
        --drop-first-window \
        > "${RESULT_DIR}/mediapipe_telemetry.tsv"
fi

echo
echo "Benchmark extendido completado: ${RUN_ID}"
echo "Modelo: ${MODEL_FILE} | preset=${MODEL_PRESET:-unknown} | family=${MODEL_FAMILY:-unknown}"
echo "Concurrente con APK: ${CONCURRENT_APP}"
echo "Contexto warm: ${WARM_CONTEXT_MODE}"
echo "Modo de salida: ${OUTPUT_MODE}"
echo "Resumen:"
cat "${SUMMARY_PATH}"
echo
echo "Resultados: ${RESULT_DIR}"
if [[ -f "${RESULT_DIR}/mediapipe_telemetry.tsv" ]]; then
    telemetry_rows="$(awk 'END {print NR - 1}' "${RESULT_DIR}/mediapipe_telemetry.tsv")"
    echo "Ventanas MediaPipe registradas: ${telemetry_rows}"
    if (( telemetry_rows > 0 )); then
        delegates="$(awk -F '\t' 'NR > 1 {print $1}'             "${RESULT_DIR}/mediapipe_telemetry.tsv" | sort -u | paste -sd, -)"
        echo "Delegate(s) MediaPipe verificado(s): ${delegates}"
        echo "Últimas ventanas MediaPipe:"
        tail -n 4 "${RESULT_DIR}/mediapipe_telemetry.tsv"
    fi
fi
echo
echo "Interpretación:"
echo "  process-cold = proceso nuevo y recarga del modelo."
echo "  warm-server = mismo proceso con el modelo residente."
echo "  Si warm-server figura SKIPPED, no existe una medición warm real."
