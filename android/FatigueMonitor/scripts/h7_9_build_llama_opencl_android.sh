#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly SRC_DIR="${LOCAL_ROOT}/src/llama.cpp-b11073"
readonly BUILD_DIR="${LOCAL_ROOT}/build/llama.cpp-b11073-opencl-android"
readonly INSTALL_DIR="${LOCAL_ROOT}/runtime-builds/b11073-opencl"
readonly OPENCL_ROOT="${LOCAL_ROOT}/toolchain/opencl"
readonly OPENCL_INCLUDE_DIR="${OPENCL_ROOT}/include"
readonly OPENCL_LIBRARY="${OPENCL_ROOT}/lib/libOpenCL.so"
readonly LLAMA_TAG="b11073"
readonly EXPECTED_COMMIT="1aa2954bde90b1cb4d2dca96f90b07d7b155124b"
readonly NDK_DEFAULT="${HOME}/Android/Sdk/ndk/29.0.14206865"
readonly ANDROID_NDK="${ANDROID_NDK:-${NDK_DEFAULT}}"
readonly TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"

for cmd in git cmake ninja adb; do
    command -v "${cmd}" >/dev/null 2>&1 || {
        echo "Missing required command: ${cmd}" >&2
        exit 1
    }
done

[[ -f "${TOOLCHAIN}" ]] || {
    echo "Android toolchain not found: ${TOOLCHAIN}" >&2
    exit 1
}

if [[ ! -f /usr/include/CL/cl.h ]]; then
    echo "OpenCL headers not found at /usr/include/CL/cl.h" >&2
    echo "Install them with: sudo apt install opencl-headers" >&2
    exit 1
fi

rm -rf "${OPENCL_ROOT}"
mkdir -p "${OPENCL_INCLUDE_DIR}" "$(dirname "${OPENCL_LIBRARY}")"
cp -a /usr/include/CL "${OPENCL_INCLUDE_DIR}/"

echo "Pulling vendor OpenCL library from the phone for link-time only..."
adb pull /vendor/lib64/libOpenCL.so "${OPENCL_LIBRARY}" >/dev/null

[[ -s "${OPENCL_LIBRARY}" ]] || {
    echo "Failed to pull /vendor/lib64/libOpenCL.so" >&2
    exit 1
}

mkdir -p "${LOCAL_ROOT}/src" "${LOCAL_ROOT}/build" "${LOCAL_ROOT}/runtime-builds"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
    git clone --branch "${LLAMA_TAG}" --depth 1 \
        https://github.com/ggml-org/llama.cpp.git "${SRC_DIR}"
fi

git -C "${SRC_DIR}" fetch --depth 1 origin "refs/tags/${LLAMA_TAG}:refs/tags/${LLAMA_TAG}"
git -C "${SRC_DIR}" checkout --detach "${LLAMA_TAG}"
git -C "${SRC_DIR}" reset --hard "${LLAMA_TAG}"

actual_commit="$(git -C "${SRC_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "Unexpected commit for ${LLAMA_TAG}" >&2
    echo "Expected: ${EXPECTED_COMMIT}" >&2
    echo "Actual:   ${actual_commit}" >&2
    exit 1
fi

rm -rf "${BUILD_DIR}" "${INSTALL_DIR}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_LLAMAFILE=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_PROFILING=OFF \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
    -DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF \
    -DGGML_OPENCL_TARGET_VERSION=200 \
    -DOpenCL_INCLUDE_DIR="${OPENCL_INCLUDE_DIR}" \
    -DOpenCL_LIBRARY="${OPENCL_LIBRARY}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"
cmake --install "${BUILD_DIR}" --config Release

echo
echo "Android OpenCL build completed."
echo "Tag:        ${LLAMA_TAG}"
echo "Commit:     ${actual_commit}"
echo "NDK:        ${ANDROID_NDK}"
echo "OpenCL hdr: ${OPENCL_INCLUDE_DIR}"
echo "OpenCL lib: ${OPENCL_LIBRARY}"
echo "Install:    ${INSTALL_DIR}"

echo
echo "Binaries:"
find "${INSTALL_DIR}" -maxdepth 3 -type f \
    \( -name 'llama-cli' -o -name 'llama-server' -o -name 'llama-completion' \) \
    -print

echo
echo "Relevant shared libraries:"
find "${INSTALL_DIR}" -maxdepth 3 -type f \
    \( -name 'libggml*.so' -o -name 'libllama*.so' \) \
    -print
