cmake_minimum_required(VERSION 3.22.1)

project(fatigue_llm_jni LANGUAGES C CXX)

# The Android app is installed as a debuggable APK during H7.10, but llama.cpp
# must still use optimized native code. Without this, AGP's Debug native build
# compiles ggml without -O2/-O3 and inference becomes tens of seconds slower.
if (ANDROID)
    add_compile_options(-O3)
    add_compile_definitions(NDEBUG)
endif()

get_filename_component(
    FATIGUE_PROJECT_ROOT
    "${CMAKE_CURRENT_LIST_DIR}/../../../.."
    ABSOLUTE
)

set(
    LLAMA_SRC_DIR
    "${FATIGUE_PROJECT_ROOT}/.local/h7_9_llm/src/llama.cpp-b11073"
)

if (NOT EXISTS "${LLAMA_SRC_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp b11073 source not found at ${LLAMA_SRC_DIR}. "
        "Run scripts/h7_10_prepare_llama_source.sh first."
    )
endif()

# H7.10 Android/JNI baseline: CPU-only llama.cpp.
# GPU paths were evaluated separately in H7.9.10 and are intentionally disabled.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

set(GGML_NATIVE OFF CACHE BOOL "" FORCE)

# Target handset H8.12: the previously validated llama.cpp CLI build reports
# ARMv8.2 FP16 vector arithmetic and DOTPROD. The JNI build had been using the
# generic arm64 baseline because GGML_NATIVE is intentionally OFF for Android
# cross-compilation. Compile the CPU backend for the validated target features
# so Q4 prompt evaluation uses the optimized ARM kernels.
set(
    GGML_CPU_ARM_ARCH
    "armv8.2-a+dotprod+fp16"
    CACHE STRING
    ""
    FORCE
)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "" FORCE)

add_subdirectory(
    "${LLAMA_SRC_DIR}"
    "${CMAKE_BINARY_DIR}/llama.cpp"
    EXCLUDE_FROM_ALL
)

add_library(
    fatigue_llm_jni
    SHARED
    llm_jni.cpp
)

target_compile_features(fatigue_llm_jni PRIVATE cxx_std_17)

target_include_directories(
    fatigue_llm_jni
    PRIVATE
    "${LLAMA_SRC_DIR}/include"
)

find_library(log-lib log)

target_link_libraries(
    fatigue_llm_jni
    PRIVATE
    llama
    ${log-lib}
)
