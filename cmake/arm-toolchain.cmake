# ARM64 Cross-Compilation Toolchain
# For Raspberry Pi 4/5 and other ARM64 devices

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compiler prefix
set(CROSS_COMPILE aarch64-linux-gnu-)

# Compilers
set(CMAKE_C_COMPILER ${CROSS_COMPILE}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_COMPILE}g++)

# Sysroot (optional, set if cross-compiling with full sysroot)
# set(CMAKE_SYSROOT /path/to/sysroot)

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ONNX Runtime for ARM64
# Download from: https://github.com/microsoft/onnxruntime/releases
# Look for: onnxruntime-linux-aarch64-*.tgz
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-arm64" CACHE PATH "ONNX Runtime ARM64 installation")

# Disable GPU for ARM (unless Jetson)
set(ENABLE_CUDA OFF CACHE BOOL "Disable CUDA for ARM")
set(ENABLE_TENSORRT OFF CACHE BOOL "Disable TensorRT for ARM")

# Build for smaller binary size
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Os -ffunction-sections -fdata-sections")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

# Static linking for portability
set(BUILD_STATIC ON CACHE BOOL "Build static binary")
