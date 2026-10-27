#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly SRC_DIR="${LOCAL_ROOT}/src/llama.cpp-b11073"
readonly BUILD_DIR="${LOCAL_ROOT}/build/llama.cpp-b11073-vulkan-android"
readonly INSTALL_DIR="${LOCAL_ROOT}/runtime-builds/b11073-vulkan"
readonly VULKAN_HEADER_ROOT="${LOCAL_ROOT}/toolchain/vulkan-headers"
readonly VULKAN_INCLUDE_DIR="${VULKAN_HEADER_ROOT}/include"
readonly LLAMA_TAG="b11073"
readonly EXPECTED_COMMIT="1aa2954bde90b1cb4d2dca96f90b07d7b155124b"
readonly NDK_DEFAULT="${HOME}/Android/Sdk/ndk/29.0.14206865"
readonly ANDROID_NDK="${ANDROID_NDK:-${NDK_DEFAULT}}"
readonly GLSLC="${GLSLC:-$(command -v glslc || true)}"
readonly TOOLCHAIN="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
readonly CXX="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++"
readonly SYSROOT="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot"

for command_name in git cmake ninja python3 dpkg awk; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
    fi
done

if [[ -z "${GLSLC}" || ! -x "${GLSLC}" ]]; then
    echo "glslc was not found." >&2
    exit 1
fi

if [[ ! -f "${TOOLCHAIN}" ]]; then
    echo "Android toolchain not found: ${TOOLCHAIN}" >&2
    exit 1
fi

if [[ ! -x "${CXX}" ]]; then
    echo "NDK clang++ not found: ${CXX}" >&2
    exit 1
fi

SPIRV_HEADERS_CONFIG="$(dpkg -L spirv-headers 2>/dev/null | awk '/\/SPIRV-HeadersConfig\.cmake$/ {print; exit}')"
if [[ -z "${SPIRV_HEADERS_CONFIG}" || ! -f "${SPIRV_HEADERS_CONFIG}" ]]; then
    echo "SPIRV-HeadersConfig.cmake was not found." >&2
    exit 1
fi
readonly SPIRV_HEADERS_DIR="$(dirname "${SPIRV_HEADERS_CONFIG}")"

if [[ ! -f /usr/include/vulkan/vulkan.hpp ]]; then
    echo "/usr/include/vulkan/vulkan.hpp was not found. Install libvulkan-dev." >&2
    exit 1
fi

rm -rf "${VULKAN_HEADER_ROOT}"
mkdir -p "${VULKAN_INCLUDE_DIR}"
cp -a /usr/include/vulkan "${VULKAN_INCLUDE_DIR}/"
if [[ -d /usr/include/vk_video ]]; then
    cp -a /usr/include/vk_video "${VULKAN_INCLUDE_DIR}/"
fi
if [[ -d /usr/include/spirv ]]; then
    cp -a /usr/include/spirv "${VULKAN_INCLUDE_DIR}/"
fi

if [[ ! -f "${VULKAN_INCLUDE_DIR}/vulkan/vulkan.hpp" ]]; then
    echo "Failed to prepare Vulkan-Hpp headers." >&2
    exit 1
fi

if [[ ! -f "${VULKAN_INCLUDE_DIR}/spirv/unified1/spirv.hpp" ]]; then
    echo "Failed to prepare SPIR-V C++ headers." >&2
    echo "Expected: /usr/include/spirv/unified1/spirv.hpp" >&2
    exit 1
fi

echo "Checking Vulkan-Hpp with Android NDK clang++..."
printf '#include <vulkan/vulkan.hpp>\n#include <spirv/unified1/spirv.hpp>\nint main(){return 0;}\n' | \
    "${CXX}" \
        --target=aarch64-none-linux-android28 \
        --sysroot="${SYSROOT}" \
        -I"${VULKAN_INCLUDE_DIR}" \
        -std=c++17 \
        -fsyntax-only \
        -x c++ -
echo "VULKAN_HPP_AND_SPIRV_OK"

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
    echo "Unexpected commit for ${LLAMA_TAG}." >&2
    echo "Expected: ${EXPECTED_COMMIT}" >&2
    echo "Actual:   ${actual_commit}" >&2
    exit 1
fi

readonly VULKAN_CMAKE="${SRC_DIR}/ggml/src/ggml-vulkan/CMakeLists.txt"
python3 - "${VULKAN_CMAKE}" "${VULKAN_INCLUDE_DIR}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
include_dir = sys.argv[2]
text = path.read_text()
needle = "    target_include_directories(ggml-vulkan PRIVATE ${CMAKE_CURRENT_BINARY_DIR})\n"
addition = needle + f'    target_include_directories(ggml-vulkan PRIVATE "{include_dir}")\n'
if addition not in text:
    if needle not in text:
        raise SystemExit("ggml-vulkan include insertion point was not found")
    text = text.replace(needle, addition, 1)
    path.write_text(text)
PY

echo "Injected Vulkan-HPp include into ggml-vulkan:"
grep -n -A1 'target_include_directories(ggml-vulkan PRIVATE ${CMAKE_CURRENT_BINARY_DIR})' "${VULKAN_CMAKE}"

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
    -DGGML_VULKAN=ON \
    -DVulkan_GLSLC_EXECUTABLE="${GLSLC}" \
    -DSPIRV-Headers_DIR="${SPIRV_HEADERS_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"
cmake --install "${BUILD_DIR}" --config Release

echo
echo "Android Vulkan build completed."
echo "Tag:        ${LLAMA_TAG}"
echo "Commit:     ${actual_commit}"
echo "NDK:        ${ANDROID_NDK}"
echo "glslc:      ${GLSLC}"
echo "SPIR-V:     ${SPIRV_HEADERS_DIR}"
echo "Vulkan-Hpp: ${VULKAN_INCLUDE_DIR}"
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
