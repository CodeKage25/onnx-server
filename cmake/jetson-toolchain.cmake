# NVIDIA Jetson Cross-Compilation Toolchain
# For Jetson Nano, Xavier, Orin

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Use NVIDIA's L4T toolchain
# Install from: https://developer.nvidia.com/embedded/jetson-linux
set(L4T_TOOLCHAIN_ROOT "/opt/nvidia/jetson" CACHE PATH "L4T toolchain root")

# Cross-compiler
set(CMAKE_C_COMPILER ${L4T_TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${L4T_TOOLCHAIN_ROOT}/bin/aarch64-linux-gnu-g++)

# Sysroot with CUDA libraries
set(CMAKE_SYSROOT ${L4T_TOOLCHAIN_ROOT}/sysroot)

# Search paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ONNX Runtime for Jetson with CUDA
# Build from source or use Jetson-specific release
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-jetson" CACHE PATH "ONNX Runtime Jetson installation")

# Enable CUDA for Jetson
set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA for Jetson")
set(ENABLE_TENSORRT ON CACHE BOOL "Enable TensorRT for Jetson")

# CUDA paths on Jetson (adjust version as needed)
set(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_SYSROOT}/usr/local/cuda-11.4")
set(CUDNN_ROOT "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")

# TensorRT paths
set(TENSORRT_ROOT "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")

# Compiler flags for Jetson
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a")

# Link against Jetson libraries
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")

# Jetson-specific defines
add_definitions(-D__JETSON__)
