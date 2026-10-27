#!/usr/bin/env bash
set -Eeuo pipefail

# H7.9: prepara un runtime llama.cpp ARM64 y un modelo candidato
# para pruebas reproducibles en un dispositivo Android conectado por ADB.

readonly LLAMA_TAG="${LLAMA_TAG:-b11073}"
readonly LLAMA_ARCHIVE_NAME="llama-${LLAMA_TAG}-bin-android-arm64.tar.gz"
readonly LLAMA_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${LLAMA_ARCHIVE_NAME}"

readonly MODEL_PRESET="${MODEL_PRESET:-0.6b-q4_0}"

case "${MODEL_PRESET}" in
    0.6b-q4_0)
        default_model_file="Qwen3-0.6B-Q4_0.gguf"
        default_model_url="https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF/resolve/main/${default_model_file}?download=true"
        default_model_family="qwen3"
        default_expected_sha256=""
        ;;
    1.7b-q4_k_m)
        default_model_file="Qwen3-1.7B-Q4_K_M.gguf"
        default_model_url="https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/main/${default_model_file}?download=true"
        default_model_family="qwen3"
        default_expected_sha256=""
        ;;
    lfm2.5-1.2b-q4_k_m)
        default_model_file="LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
        default_model_url="https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/${default_model_file}?download=true"
        default_model_family="lfm2.5"
        default_expected_sha256="b1b3de114215d9507409a662a501a631095a479a419584e8a2ded6304b19b4f5"
        ;;
    custom)
        if [[ -z "${MODEL_FILE:-}" || -z "${MODEL_URL:-}" ]]; then
            echo "El preset custom requiere MODEL_FILE y MODEL_URL." >&2
            exit 1
        fi
        default_model_file="${MODEL_FILE}"
        default_model_url="${MODEL_URL}"
        default_model_family="${MODEL_FAMILY:-custom}"
        default_expected_sha256="${EXPECTED_MODEL_SHA256:-}"
        ;;
    *)
        echo "MODEL_PRESET no reconocido: ${MODEL_PRESET}" >&2
        echo "Opciones: 0.6b-q4_0, 1.7b-q4_k_m, lfm2.5-1.2b-q4_k_m o custom" >&2
        exit 1
        ;;
esac

readonly MODEL_FILE="${MODEL_FILE:-${default_model_file}}"
readonly MODEL_URL="${MODEL_URL:-${default_model_url}}"
readonly MODEL_FAMILY="${MODEL_FAMILY:-${default_model_family}}"
readonly EXPECTED_MODEL_SHA256="${EXPECTED_MODEL_SHA256:-${default_expected_sha256}}"

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly DOWNLOAD_DIR="${LOCAL_ROOT}/downloads"
readonly RUNTIME_DIR="${LOCAL_ROOT}/runtime/${LLAMA_TAG}"
readonly MODEL_DIR="${LOCAL_ROOT}/models"
readonly ARCHIVE_PATH="${DOWNLOAD_DIR}/${LLAMA_ARCHIVE_NAME}"
readonly MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
readonly MANIFEST_PATH="${LOCAL_ROOT}/artifacts.sha256"
readonly ENV_PATH="${LOCAL_ROOT}/runtime.env"

readonly REMOTE_ROOT="/data/local/tmp/fatigue-llm"
readonly REMOTE_RUNTIME="${REMOTE_ROOT}/runtime/${LLAMA_TAG}"
readonly REMOTE_MODEL="${REMOTE_ROOT}/models/${MODEL_FILE}"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Falta el comando requerido: $1" >&2
        exit 1
    fi
}

download_if_missing() {
    local url="$1"
    local destination="$2"

    if [[ -s "${destination}" ]]; then
        echo "Ya existe: ${destination}"
        return
    fi

    echo "Descargando: ${url}"
    curl         --location         --fail         --retry 3         --retry-delay 2         --output "${destination}.part"         "${url}"
    mv "${destination}.part" "${destination}"
}

for command_name in adb awk curl find realpath sha256sum tar; do
    require_command "${command_name}"
done

adb get-state >/dev/null

device_abis="$(adb shell getprop ro.product.cpu.abilist | tr -d '\r')"
device_sdk="$(adb shell getprop ro.build.version.sdk | tr -d '\r')"

if [[ "${device_abis}" != *"arm64-v8a"* ]]; then
    echo "El dispositivo no declara arm64-v8a: ${device_abis}" >&2
    exit 1
fi

if ! [[ "${device_sdk}" =~ ^[0-9]+$ ]] || (( device_sdk < 28 )); then
    echo "Se requiere Android API 28 o superior; detectado: ${device_sdk}" >&2
    exit 1
fi

mkdir -p "${DOWNLOAD_DIR}" "${RUNTIME_DIR}" "${MODEL_DIR}"

download_if_missing "${LLAMA_URL}" "${ARCHIVE_PATH}"
download_if_missing "${MODEL_URL}" "${MODEL_PATH}"

if [[ -n "${EXPECTED_MODEL_SHA256}" ]]; then
    actual_model_sha256="$(sha256sum "${MODEL_PATH}" | awk '{print $1}')"
    if [[ "${actual_model_sha256}" != "${EXPECTED_MODEL_SHA256}" ]]; then
        echo "SHA-256 del modelo no coincide." >&2
        echo "Esperado: ${EXPECTED_MODEL_SHA256}" >&2
        echo "Actual:   ${actual_model_sha256}" >&2
        exit 1
    fi
    echo "SHA-256 del modelo verificado."
fi

if ! find "${RUNTIME_DIR}" -type f -name llama-cli -print -quit | grep -q .; then
    echo "Extrayendo runtime llama.cpp ${LLAMA_TAG}..."
    tar -xzf "${ARCHIVE_PATH}" -C "${RUNTIME_DIR}"
fi

local_binary="$(find "${RUNTIME_DIR}" -type f -name llama-cli -print -quit)"

if [[ -z "${local_binary}" ]]; then
    echo "No se encontró llama-cli dentro del paquete oficial." >&2
    exit 1
fi

relative_binary="$(realpath --relative-to="${RUNTIME_DIR}" "${local_binary}")"
remote_binary="${REMOTE_RUNTIME}/${relative_binary}"

mapfile -t local_library_dirs < <(
    find "${RUNTIME_DIR}" -type f -name '*.so*' -printf '%h\n' |
        sort -u
)

remote_library_path=""

for local_library_dir in "${local_library_dirs[@]}"; do
    relative_library_dir="$(
        realpath --relative-to="${RUNTIME_DIR}" "${local_library_dir}"
    )"
    remote_library_dir="${REMOTE_RUNTIME}/${relative_library_dir}"

    if [[ -z "${remote_library_path}" ]]; then
        remote_library_path="${remote_library_dir}"
    else
        remote_library_path="${remote_library_path}:${remote_library_dir}"
    fi
done

if [[ -z "${remote_library_path}" ]]; then
    remote_library_path="$(dirname "${remote_binary}")"
fi

(
    cd "${LOCAL_ROOT}"
    sha256sum         "downloads/${LLAMA_ARCHIVE_NAME}"         "models/${MODEL_FILE}"         > "${MANIFEST_PATH}"
)

echo "Copiando runtime al teléfono..."
adb shell mkdir -p "${REMOTE_RUNTIME}" "${REMOTE_ROOT}/models"
adb push "${RUNTIME_DIR}/." "${REMOTE_RUNTIME}/" >/dev/null
adb push "${MODEL_PATH}" "${REMOTE_MODEL}" >/dev/null
adb shell chmod 755 "${remote_binary}"

{
    printf 'LLAMA_TAG=%q\n' "${LLAMA_TAG}"
    printf 'MODEL_PRESET=%q\n' "${MODEL_PRESET}"
    printf 'MODEL_FAMILY=%q\n' "${MODEL_FAMILY}"
    printf 'MODEL_FILE=%q\n' "${MODEL_FILE}"
    printf 'REMOTE_ROOT=%q\n' "${REMOTE_ROOT}"
    printf 'REMOTE_BINARY=%q\n' "${remote_binary}"
    printf 'REMOTE_LIBRARY_PATH=%q\n' "${remote_library_path}"
    printf 'REMOTE_MODEL=%q\n' "${REMOTE_MODEL}"
} > "${ENV_PATH}"

echo
echo "Runtime instalado temporalmente:"
adb shell "LD_LIBRARY_PATH='${remote_library_path}' '${remote_binary}' --version"

echo
echo "Dispositivo: ABI=${device_abis} | API=${device_sdk}"
echo "Modelo: ${MODEL_FILE} | preset=${MODEL_PRESET} | family=${MODEL_FAMILY}"
echo "Manifest SHA-256: ${MANIFEST_PATH}"
echo "Configuración local: ${ENV_PATH}"
echo
echo "Siguiente paso:"
echo "  EXPECTED_MODEL_PRESET=${MODEL_PRESET} MODE=process-cold THREADS_LIST=\"6\" TEMPERATURES=\"0.1\" REPEATS=1 bash scripts/h7_9_benchmark_llm_extended.sh"
