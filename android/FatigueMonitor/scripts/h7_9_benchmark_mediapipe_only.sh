#!/usr/bin/env bash
set -Eeuo pipefail

# H7.9.9
# Benchmark controlado de MediaPipe sin ejecutar el LLM.
# La APK debe estar abierta con la cámara visible y el delegate deseado
# seleccionado manualmente. La telemetría de la app verifica el delegate
# realmente instanciado.

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly PARSER="${PROJECT_DIR}/scripts/h7_9_parse_mediapipe_telemetry.py"
readonly APP_PACKAGE="${APP_PACKAGE:-com.google.mediapipe.examples.facelandmarker}"
readonly SAMPLE_DURATION_S="${SAMPLE_DURATION_S:-30}"
readonly EXPECTED_DELEGATE="${EXPECTED_DELEGATE:-}" # CPU | GPU | empty
readonly RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
readonly RESULT_DIR="${LOCAL_ROOT}/mediapipe_only/${RUN_ID}"

for command_name in adb awk date python3 sleep sort paste tr; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Falta el comando requerido: ${command_name}" >&2
        exit 1
    fi
done

if [[ ! -f "${PARSER}" ]]; then
    echo "No existe ${PARSER}" >&2
    exit 1
fi

if ! [[ "${SAMPLE_DURATION_S}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SAMPLE_DURATION_S inválido: ${SAMPLE_DURATION_S}" >&2
    exit 1
fi

case "${EXPECTED_DELEGATE}" in
    ""|CPU|GPU) ;;
    *)
        echo "EXPECTED_DELEGATE debe ser CPU, GPU o vacío." >&2
        exit 1
        ;;
esac

adb get-state >/dev/null

app_pid="$(
    adb shell "pidof ${APP_PACKAGE} 2>/dev/null || true" |
        tr -d '\r' |
        awk '{print $1}'
)"

if [[ -z "${app_pid}" ]]; then
    echo "La APK no está ejecutándose." >&2
    echo "Abre FatigueMonitor y deja la vista de cámara visible." >&2
    exit 1
fi

mkdir -p "${RESULT_DIR}"

device_uptime_milliseconds() {
    adb shell "awk '{printf \"%.0f\\n\", \$1 * 1000}' /proc/uptime" |
        tr -d '\r'
}

battery_temperature() {
    adb shell dumpsys battery |
        awk '/temperature:/ {print $2; exit}' |
        tr -d '\r'
}

temp_before="$(battery_temperature)"

adb shell setprop log.tag.H7_9_PERF VERBOSE >/dev/null 2>&1 || true
adb shell setprop log.tag.FaceLandmarkerHelper VERBOSE >/dev/null 2>&1 || true
adb logcat -c >/dev/null 2>&1 || true
measurement_start_ms="$(device_uptime_milliseconds)"
printf '%s\n' "${measurement_start_ms}" \
    > "${RESULT_DIR}/measurement_start_uptime_ms.txt"

{
    echo "run_id=${RUN_ID}"
    echo "app_package=${APP_PACKAGE}"
    echo "app_pid=${app_pid}"
    echo "sample_duration_s=${SAMPLE_DURATION_S}"
    echo "expected_delegate=${EXPECTED_DELEGATE:-UNSPECIFIED}"
    echo "temp_before_tenths=${temp_before:-NA}"
} > "${RESULT_DIR}/run_metadata.txt"

echo "Midiendo MediaPipe durante ${SAMPLE_DURATION_S} s..."
echo "Mantén la cámara visible y el rostro en condiciones similares."
sleep "${SAMPLE_DURATION_S}"

temp_after="$(battery_temperature)"
adb logcat -d > "${RESULT_DIR}/logcat_after.txt" 2>&1 || true

python3 "${PARSER}" \
    "${RESULT_DIR}/logcat_after.txt" \
    --drop-first-window \
    > "${RESULT_DIR}/mediapipe_telemetry.tsv"

telemetry_rows="$(
    awk 'END {print NR - 1}' "${RESULT_DIR}/mediapipe_telemetry.tsv"
)"
raw_telemetry_lines="$(
    grep -c 'H7_9_PERF.*delegate=' "${RESULT_DIR}/logcat_after.txt" || true
)"

if (( telemetry_rows <= 0 )); then
    echo "No se registraron ventanas válidas de MediaPipe." >&2
    echo "Líneas H7_9_PERF crudas encontradas: ${raw_telemetry_lines}" >&2
    if (( raw_telemetry_lines > 0 )); then
        echo "Hay telemetría en logcat, pero no quedaron ventanas completas válidas después del descarte inicial." >&2
        grep 'H7_9_PERF' "${RESULT_DIR}/logcat_after.txt" | tail -n 5 >&2 || true
    else
        echo "No aparece telemetría H7_9_PERF en logcat." >&2
        echo "Verifica que instalaste la APK instrumentada y que la cámara estaba activa." >&2
    fi
    exit 1
fi

delegates="$(
    awk -F '\t' 'NR > 1 {print $1}' \
        "${RESULT_DIR}/mediapipe_telemetry.tsv" |
        sort -u |
        paste -sd, -
)"

if [[ -n "${EXPECTED_DELEGATE}" ]] &&
   [[ "${delegates}" != "${EXPECTED_DELEGATE}" ]]; then
    echo "Delegate efectivo inesperado." >&2
    echo "Esperado: ${EXPECTED_DELEGATE}" >&2
    echo "Verificado: ${delegates}" >&2
    exit 1
fi

{
    echo "temp_after_tenths=${temp_after:-NA}"
    echo "telemetry_windows=${telemetry_rows}"
    echo "verified_delegates=${delegates}"
} >> "${RESULT_DIR}/run_metadata.txt"

echo
echo "Benchmark MediaPipe completado: ${RUN_ID}"
echo "Delegate verificado: ${delegates}"
echo "Temperatura: ${temp_before:-NA} -> ${temp_after:-NA} décimas de °C"
echo "Ventanas completas: ${telemetry_rows}"
echo "Resultados: ${RESULT_DIR}"
echo
cat "${RESULT_DIR}/mediapipe_telemetry.tsv"
