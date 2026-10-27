#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PACKAGE="com.google.mediapipe.examples.facelandmarker"
readonly MODEL_FILE="LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
readonly LOCAL_MODEL="${PROJECT_DIR}/.local/h7_9_llm/models/${MODEL_FILE}"
readonly REMOTE_DIR="/sdcard/Android/data/${PACKAGE}/files/models"
readonly REMOTE_MODEL="${REMOTE_DIR}/${MODEL_FILE}"

command -v adb >/dev/null 2>&1 || {
    echo "Missing required command: adb" >&2
    exit 1
}

[[ -s "${LOCAL_MODEL}" ]] || {
    echo "Local model not found: ${LOCAL_MODEL}" >&2
    exit 1
}

adb get-state >/dev/null

echo "Preparing app-specific external model directory..."
adb shell mkdir -p "${REMOTE_DIR}"

echo "Copying model to app storage..."
adb push "${LOCAL_MODEL}" "${REMOTE_MODEL}"

echo
echo "Remote model:"
adb shell ls -lh "${REMOTE_MODEL}"

echo
echo "Path used by Android:"
echo "/storage/emulated/0/Android/data/${PACKAGE}/files/models/${MODEL_FILE}"
