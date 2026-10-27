#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly LOCAL_ROOT="${PROJECT_DIR}/.local/h7_9_llm"
readonly PROBE_DIR="${LOCAL_ROOT}/opencl-probe"
readonly SRC="${PROBE_DIR}/opencl_probe.c"
readonly BIN="${PROBE_DIR}/opencl_probe"
readonly REMOTE_DIR="/data/local/tmp/fatigue-llm/opencl-probe"
readonly REMOTE_BIN="${REMOTE_DIR}/opencl_probe"
readonly NDK_DEFAULT="${HOME}/Android/Sdk/ndk/29.0.14206865"
readonly ANDROID_NDK="${ANDROID_NDK:-${NDK_DEFAULT}}"
readonly CC="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang"
readonly SYSROOT="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot"

for cmd in adb; do
    command -v "${cmd}" >/dev/null 2>&1 || {
        echo "Missing required command: ${cmd}" >&2
        exit 1
    }
done

[[ -x "${CC}" ]] || {
    echo "NDK clang not found: ${CC}" >&2
    exit 1
}

mkdir -p "${PROBE_DIR}"

cat > "${SRC}" <<'EOF'
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef uint32_t cl_platform_info;
typedef uint32_t cl_device_info;
typedef struct _cl_platform_id * cl_platform_id;
typedef struct _cl_device_id   * cl_device_id;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFFu

#define CL_PLATFORM_VERSION 0x0901
#define CL_PLATFORM_NAME    0x0902
#define CL_PLATFORM_VENDOR  0x0903

#define CL_DEVICE_TYPE              0x1000
#define CL_DEVICE_NAME              0x102B
#define CL_DEVICE_VENDOR            0x102C
#define CL_DRIVER_VERSION           0x102D
#define CL_DEVICE_VERSION           0x102F
#define CL_DEVICE_OPENCL_C_VERSION  0x103D

typedef cl_int (*PFN_clGetPlatformIDs)(
    cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*PFN_clGetPlatformInfo)(
    cl_platform_id, cl_platform_info, size_t, void *, size_t *);
typedef cl_int (*PFN_clGetDeviceIDs)(
    cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (*PFN_clGetDeviceInfo)(
    cl_device_id, cl_device_info, size_t, void *, size_t *);

static void print_platform_string(
    PFN_clGetPlatformInfo fn,
    cl_platform_id platform,
    cl_platform_info param,
    const char *label) {

    size_t n = 0;
    if (fn(platform, param, 0, NULL, &n) != CL_SUCCESS || n == 0) {
        printf("%s: <unavailable>\n", label);
        return;
    }

    char *buf = (char *) calloc(1, n + 1);
    if (!buf) {
        printf("%s: <allocation failed>\n", label);
        return;
    }

    if (fn(platform, param, n, buf, NULL) == CL_SUCCESS) {
        printf("%s: %s\n", label, buf);
    } else {
        printf("%s: <query failed>\n", label);
    }
    free(buf);
}

static void print_device_string(
    PFN_clGetDeviceInfo fn,
    cl_device_id device,
    cl_device_info param,
    const char *label) {

    size_t n = 0;
    if (fn(device, param, 0, NULL, &n) != CL_SUCCESS || n == 0) {
        printf("    %s: <unavailable>\n", label);
        return;
    }

    char *buf = (char *) calloc(1, n + 1);
    if (!buf) {
        printf("    %s: <allocation failed>\n", label);
        return;
    }

    if (fn(device, param, n, buf, NULL) == CL_SUCCESS) {
        printf("    %s: %s\n", label, buf);
    } else {
        printf("    %s: <query failed>\n", label);
    }
    free(buf);
}

int main(void) {
    const char *candidates[] = {
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "libOpenCL.so"
    };

    void *handle = NULL;
    const char *loaded = NULL;

    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        handle = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
        if (handle) {
            loaded = candidates[i];
            break;
        }
    }

    if (!handle) {
        fprintf(stderr, "OPENCL_DLOPEN_FAILED: %s\n", dlerror());
        return 2;
    }

    PFN_clGetPlatformIDs clGetPlatformIDs =
        (PFN_clGetPlatformIDs) dlsym(handle, "clGetPlatformIDs");
    PFN_clGetPlatformInfo clGetPlatformInfo =
        (PFN_clGetPlatformInfo) dlsym(handle, "clGetPlatformInfo");
    PFN_clGetDeviceIDs clGetDeviceIDs =
        (PFN_clGetDeviceIDs) dlsym(handle, "clGetDeviceIDs");
    PFN_clGetDeviceInfo clGetDeviceInfo =
        (PFN_clGetDeviceInfo) dlsym(handle, "clGetDeviceInfo");

    if (!clGetPlatformIDs || !clGetPlatformInfo ||
        !clGetDeviceIDs || !clGetDeviceInfo) {
        fprintf(stderr, "OPENCL_REQUIRED_SYMBOLS_MISSING\n");
        dlclose(handle);
        return 3;
    }

    printf("OPENCL_LIBRARY=%s\n", loaded);

    cl_uint platform_count = 0;
    cl_int rc = clGetPlatformIDs(0, NULL, &platform_count);
    printf("clGetPlatformIDs rc=%d platforms=%u\n", rc, platform_count);

    if (rc != CL_SUCCESS || platform_count == 0) {
        dlclose(handle);
        return 4;
    }

    cl_platform_id *platforms =
        (cl_platform_id *) calloc(platform_count, sizeof(*platforms));
    if (!platforms) {
        dlclose(handle);
        return 5;
    }

    rc = clGetPlatformIDs(platform_count, platforms, NULL);
    if (rc != CL_SUCCESS) {
        fprintf(stderr, "clGetPlatformIDs(list) rc=%d\n", rc);
        free(platforms);
        dlclose(handle);
        return 6;
    }

    for (cl_uint p = 0; p < platform_count; ++p) {
        printf("\nPlatform %u\n", p);
        print_platform_string(
            clGetPlatformInfo, platforms[p], CL_PLATFORM_NAME, "  Name");
        print_platform_string(
            clGetPlatformInfo, platforms[p], CL_PLATFORM_VENDOR, "  Vendor");
        print_platform_string(
            clGetPlatformInfo, platforms[p], CL_PLATFORM_VERSION, "  Version");

        cl_uint device_count = 0;
        rc = clGetDeviceIDs(
            platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
        printf("  clGetDeviceIDs rc=%d devices=%u\n", rc, device_count);

        if (rc != CL_SUCCESS || device_count == 0) {
            continue;
        }

        cl_device_id *devices =
            (cl_device_id *) calloc(device_count, sizeof(*devices));
        if (!devices) {
            continue;
        }

        rc = clGetDeviceIDs(
            platforms[p], CL_DEVICE_TYPE_ALL,
            device_count, devices, NULL);

        if (rc != CL_SUCCESS) {
            printf("  clGetDeviceIDs(list) rc=%d\n", rc);
            free(devices);
            continue;
        }

        for (cl_uint d = 0; d < device_count; ++d) {
            printf("  Device %u\n", d);
            print_device_string(
                clGetDeviceInfo, devices[d], CL_DEVICE_NAME, "Name");
            print_device_string(
                clGetDeviceInfo, devices[d], CL_DEVICE_VENDOR, "Vendor");
            print_device_string(
                clGetDeviceInfo, devices[d], CL_DEVICE_VERSION, "Version");
            print_device_string(
                clGetDeviceInfo, devices[d], CL_DRIVER_VERSION, "Driver");
            print_device_string(
                clGetDeviceInfo, devices[d],
                CL_DEVICE_OPENCL_C_VERSION, "OpenCL C");
        }

        free(devices);
    }

    free(platforms);
    dlclose(handle);
    return 0;
}
EOF

echo "Compiling OpenCL runtime probe..."
"${CC}" \
    --target=aarch64-none-linux-android28 \
    --sysroot="${SYSROOT}" \
    -O2 -fPIE -pie \
    "${SRC}" \
    -ldl \
    -o "${BIN}"

echo "Deploying probe..."
adb shell mkdir -p "${REMOTE_DIR}"
adb push "${BIN}" "${REMOTE_BIN}" >/dev/null
adb shell chmod 755 "${REMOTE_BIN}"

echo
echo "=== OpenCL runtime probe ==="
adb shell "${REMOTE_BIN}"
