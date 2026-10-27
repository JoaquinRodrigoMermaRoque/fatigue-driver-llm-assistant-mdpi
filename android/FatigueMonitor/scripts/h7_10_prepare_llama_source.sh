#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly SRC_DIR="${LOCAL_ROOT}/src/llama.cpp-b11073"
readonly LLAMA_TAG="b11073"
readonly EXPECTED_COMMIT="1aa2954bde90b1cb4d2dca96f90b07d7b155124b"

command -v git >/dev/null 2>&1 || {
    echo "Missing required command: git" >&2
    exit 1
}

mkdir -p "${LOCAL_ROOT}/src"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    echo "Cloning llama.cpp ${LLAMA_TAG}..."
    git clone         --branch "${LLAMA_TAG}"         --depth 1         https://github.com/ggml-org/llama.cpp.git         "${SRC_DIR}"
else
    echo "Reusing existing llama.cpp source checkout."
    git -C "${SRC_DIR}" fetch         --depth 1         origin         "refs/tags/${LLAMA_TAG}:refs/tags/${LLAMA_TAG}"
fi

git -C "${SRC_DIR}" checkout --detach "${LLAMA_TAG}"
git -C "${SRC_DIR}" reset --hard "${LLAMA_TAG}"
git -C "${SRC_DIR}" clean -fd

actual_commit="$(git -C "${SRC_DIR}" rev-parse HEAD)"

if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Unexpected llama.cpp commit." >&2
    echo "Expected: ${EXPECTED_COMMIT}" >&2
    echo "Actual:   ${actual_commit}" >&2
    exit 1
fi

echo
echo "H7.10 llama.cpp source ready."
echo "Tag:    ${LLAMA_TAG}"
echo "Commit: ${actual_commit}"
echo "Path:   ${SRC_DIR}"
