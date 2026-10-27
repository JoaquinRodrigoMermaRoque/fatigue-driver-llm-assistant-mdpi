#!/usr/bin/env bash
set -Eeuo pipefail

# H7.9.2: compara inferencia CPU con 4 y 6 hilos en el teléfono.
# El LLM solo redacta una introducción. No decide el estado ni la acción.

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly ENV_PATH="${LOCAL_ROOT}/runtime.env"

readonly CONTEXT_SIZE="${CONTEXT_SIZE:-1024}"
readonly MAX_TOKENS="${MAX_TOKENS:-48}"
readonly THREADS_LIST="${THREADS_LIST:-4 6}"
readonly REPEATS="${REPEATS:-1}"
readonly SEED="${SEED:-42}"
readonly TIMEOUT_S="${TIMEOUT_S:-180}"
readonly MAX_BATTERY_TEMP_TENTHS="${MAX_BATTERY_TEMP_TENTHS:-420}"
readonly EXPECTED_MODEL_PRESET="${EXPECTED_MODEL_PRESET:-}"

if [[ ! -f "${ENV_PATH}" ]]; then
    echo "No existe ${ENV_PATH}" >&2
    echo "Ejecuta primero: bash scripts/h7_9_prepare_llm_android.sh" >&2
    exit 1
fi

# Contiene solamente rutas creadas por el script de preparación.
# shellcheck disable=SC1090
source "${ENV_PATH}"

if [[ -n "${EXPECTED_MODEL_PRESET}" ]] &&
    [[ "${MODEL_PRESET:-unknown}" != "${EXPECTED_MODEL_PRESET}" ]]; then
    echo "Modelo activo incorrecto." >&2
    echo "Esperado: ${EXPECTED_MODEL_PRESET}" >&2
    echo "Activo: ${MODEL_PRESET:-unknown} (${MODEL_FILE:-unknown})" >&2
    echo "Ejecuta primero el script de preparación con el preset esperado." >&2
    exit 1
fi

for variable_name in \
    REMOTE_ROOT \
    REMOTE_BINARY \
    REMOTE_LIBRARY_PATH \
    REMOTE_MODEL; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Falta ${variable_name} en ${ENV_PATH}" >&2
        exit 1
    fi
done

for command_name in adb awk date grep sort timeout tr; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Falta el comando requerido: ${command_name}" >&2
        exit 1
    fi
done

adb get-state >/dev/null

readonly REMOTE_COMPLETION="${REMOTE_BINARY%/*}/llama-completion"

if adb shell test -f "${REMOTE_COMPLETION}" >/dev/null 2>&1; then
    adb shell chmod 755 "${REMOTE_COMPLETION}"
    readonly INFERENCE_BINARY="${REMOTE_COMPLETION}"
    readonly INFERENCE_PROCESS_NAME="llama-completion"
else
    readonly INFERENCE_BINARY="${REMOTE_BINARY}"
    readonly INFERENCE_PROCESS_NAME="llama-cli"
fi

# El prompt ya incluye la plantilla ChatML de Qwen. Se desactiva el modo
# conversación automático para realizar una sola inferencia reproducible.
readonly -a INFERENCE_MODE_ARGS=(-no-cnv)

stop_remote_inference() {
    local process_name
    local process_ids
    local process_id

    for process_name in llama-cli llama-completion; do
        process_ids="$(
            adb shell "pidof ${process_name} 2>/dev/null || true" |
                tr -d '\r'
        )"

        for process_id in ${process_ids}; do
            adb shell kill "${process_id}" >/dev/null 2>&1 || true
        done
    done
}

# El intento anterior pudo dejar el proceso remoto vivo después del timeout.
stop_remote_inference
trap stop_remote_inference EXIT INT TERM

readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_DIR="${LOCAL_ROOT}/benchmarks/${RUN_ID}"
readonly SUMMARY_PATH="${RESULT_DIR}/summary.tsv"
readonly LOCAL_PROMPT="${RESULT_DIR}/prompt.txt"
readonly REMOTE_PROMPT="${REMOTE_ROOT}/prompt_h7_9.txt"

mkdir -p "${RESULT_DIR}"

cat > "${LOCAL_PROMPT}" <<'PROMPT'
<|im_start|>system
Eres el módulo lingüístico local de un asistente de seguridad para conducción.
Redacta únicamente a partir de señales observadas por el sistema.
No diagnostiques, no inventes causas y no contradigas la decisión determinista.
Usa un tono calmado, empático, directo y no moralizante.
<|im_end|>
<|im_start|>user
Contexto actual:
- estado: FATIGA GRAVE
- disparador principal: cierre ocular prolongado
- eventos recientes: dos bostezos y un cabeceo
- historial local: segundo episodio crítico de la sesión
- acción obligatoria, que otro módulo añadirá sin cambios: detenerse en un lugar seguro y descansar antes de continuar

Redacta una introducción de una sola oración y entre 12 y 22 palabras.
Habla directamente a la persona.
Menciona el cierre ocular y que la señal se ha repetido.
No des instrucciones ni escribas la acción obligatoria.
No uses culpa, responsable, irresponsable, enfermedad, diagnóstico ni problemas futuros.
Devuelve únicamente la introducción.
/no_think
<|im_end|>
<|im_start|>assistant
<think>

</think>

PROMPT

adb push "${LOCAL_PROMPT}" "${REMOTE_PROMPT}" >/dev/null

monotonic_milliseconds() {
    awk '{printf "%.0f\n", $1 * 1000}' /proc/uptime
}

battery_temperature() {
    adb shell dumpsys battery |
        awk '/temperature:/ {print $2; exit}' |
        tr -d '\r'
}

record_device_snapshot() {
    local destination="$1"

    {
        echo "=== device ==="
        adb shell getprop ro.product.manufacturer
        adb shell getprop ro.product.model
        adb shell getprop ro.product.cpu.abilist
        adb shell getprop ro.build.version.release
        adb shell getprop ro.build.version.sdk
        echo "=== inference ==="
        echo "binary=${INFERENCE_BINARY}"
        printf 'mode_args='
        printf '%q ' "${INFERENCE_MODE_ARGS[@]}"
        echo
        echo "=== battery ==="
        adb shell dumpsys battery |
            grep -E 'level:|temperature:'
        echo "=== memory ==="
        adb shell "grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo"
        echo "=== storage ==="
        adb shell df -h /data
    } > "${destination}" 2>&1
}

printf '%s\n' \
    $'threads\trepeat\telapsed_ms\tpeak_rss_kb\ttemp_before_tenths\ttemp_after_tenths\texit_code' \
    > "${SUMMARY_PATH}"

record_device_snapshot "${RESULT_DIR}/device_before.txt"

echo "Ejecutable de inferencia: ${INFERENCE_BINARY}"
echo "Modelo activo: ${MODEL_FILE} | preset=${MODEL_PRESET:-unknown}"
if (( ${#INFERENCE_MODE_ARGS[@]} > 0 )); then
    echo "Modo adicional: ${INFERENCE_MODE_ARGS[*]}"
fi

for threads in ${THREADS_LIST}; do
    if ! [[ "${threads}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Cantidad de hilos inválida: ${threads}" >&2
        exit 1
    fi

    for (( repetition = 1; repetition <= REPEATS; repetition += 1 )); do
        temperature_before="$(battery_temperature)"

        if [[ "${temperature_before}" =~ ^[0-9]+$ ]] &&
            (( temperature_before >= MAX_BATTERY_TEMP_TENTHS )); then
            echo "Prueba detenida: batería a ${temperature_before} décimas de °C." >&2
            echo "Espera a que el teléfono se enfríe antes de continuar." >&2
            exit 1
        fi

        run_prefix="threads_${threads}_repeat_${repetition}"
        output_path="${RESULT_DIR}/${run_prefix}.log"
        memory_path="${RESULT_DIR}/${run_prefix}_memory.tsv"

        printf 'elapsed_ms\tvm_rss_kb\n' > "${memory_path}"

        echo
        echo "Ejecutando ${run_prefix}..."

        benchmark_start_ms="$(monotonic_milliseconds)"

        set +e
        timeout "${TIMEOUT_S}" adb shell \
            env "LD_LIBRARY_PATH=${REMOTE_LIBRARY_PATH}" \
            "${INFERENCE_BINARY}" \
            -m "${REMOTE_MODEL}" \
            -f "${REMOTE_PROMPT}" \
            -c "${CONTEXT_SIZE}" \
            -n "${MAX_TOKENS}" \
            -t "${threads}" \
            -ngl 0 \
            --seed "${SEED}" \
            --temp 0.7 \
            --top-k 20 \
            --top-p 0.8 \
            --min-p 0.0 \
            --presence-penalty 1.5 \
            --no-display-prompt \
            --no-context-shift \
            "${INFERENCE_MODE_ARGS[@]}" \
            > "${output_path}" 2>&1 &
        host_process_id=$!

        while kill -0 "${host_process_id}" 2>/dev/null; do
            sample_ms="$(monotonic_milliseconds)"
            elapsed_sample_ms=$(( sample_ms - benchmark_start_ms ))
            remote_process_ids="$(
                adb shell "pidof ${INFERENCE_PROCESS_NAME} 2>/dev/null || true" |
                    tr -d '\r'
            )"

            for remote_process_id in ${remote_process_ids}; do
                vm_rss_kb="$(
                    adb shell \
                        "awk '/^VmRSS:/ {print \$2; exit}' /proc/${remote_process_id}/status 2>/dev/null" |
                        tr -d '\r'
                )"

                if [[ "${vm_rss_kb}" =~ ^[0-9]+$ ]]; then
                    printf '%s\t%s\n' \
                        "${elapsed_sample_ms}" \
                        "${vm_rss_kb}" \
                        >> "${memory_path}"
                fi
            done

            sleep 0.25
        done

        wait "${host_process_id}"
        exit_code=$?
        set -e

        if (( exit_code != 0 )); then
            stop_remote_inference
        fi

        benchmark_end_ms="$(monotonic_milliseconds)"
        elapsed_ms=$(( benchmark_end_ms - benchmark_start_ms ))
        temperature_after="$(battery_temperature)"

        peak_rss_kb="$(
            awk -F '\t' '
                NR > 1 && $2 + 0 > maximum {
                    maximum = $2 + 0
                }
                END {
                    if (maximum > 0) {
                        print maximum
                    } else {
                        print "NA"
                    }
                }
            ' "${memory_path}"
        )"

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${threads}" \
            "${repetition}" \
            "${elapsed_ms}" \
            "${peak_rss_kb}" \
            "${temperature_before:-NA}" \
            "${temperature_after:-NA}" \
            "${exit_code}" \
            >> "${SUMMARY_PATH}"

        if (( exit_code != 0 )); then
            echo "Falló ${run_prefix} con código ${exit_code}." >&2
            tail -n 30 "${output_path}" >&2
            exit "${exit_code}"
        fi

        grep -E \
            'prompt eval time|eval time|total time|tokens per second' \
            "${output_path}" |
            tail -n 10 || true
    done
done

record_device_snapshot "${RESULT_DIR}/device_after.txt"

echo
echo "Benchmark completado: ${RUN_ID}"
echo "Resumen:"
column -t -s $'\t' "${SUMMARY_PATH}" 2>/dev/null ||
    cat "${SUMMARY_PATH}"
echo
echo "Resultados locales: ${RESULT_DIR}"
echo "Para revisar las respuestas:"
echo "  sed -n '1,180p' ${RESULT_DIR}/threads_*_repeat_*.log"
