cmake_minimum_required(VERSION 3.20)
project(onnx-server 
    VERSION 1.0.0
    DESCRIPTION "Lightweight high-performance ONNX inference server"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build example clients" OFF)
option(ENABLE_CUDA "Enable CUDA execution provider" ON)
option(ENABLE_TENSORRT "Enable TensorRT execution provider" ON)
option(ENABLE_SSL "Enable SSL/TLS support" OFF)
option(BUILD_STATIC "Build static binary for edge deployment" OFF)

# ============================================================================
# C++ Standard and Compiler Settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
    if(BUILD_STATIC)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
    endif()
endif()

# ============================================================================
# Dependencies
# ============================================================================

# ONNX Runtime
find_package(onnxruntime QUIET)
if(NOT onnxruntime_FOUND)
    # Try to find ONNX Runtime manually
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS 
            /usr/local/include/onnxruntime
            /opt/onnxruntime/include
            ${ONNXRUNTIME_ROOT}/include
    )
    find_library(ONNXRUNTIME_LIBRARY onnxruntime
        PATHS
            /usr/local/lib
            /opt/onnxruntime/lib
            ${ONNXRUNTIME_ROOT}/lib
    )
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
    else()
        message(STATUS "ONNX Runtime not found - will use FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            onnxruntime
            URL https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
        )
        FetchContent_MakeAvailable(onnxruntime)
        set(ONNXRUNTIME_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
        set(ONNXRUNTIME_LIBRARY ${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so)
    endif()
endif()

# Threads
find_package(Threads REQUIRED)

# ============================================================================
# Header-only Dependencies (vendored in third_party/)
# ============================================================================
set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)

# cpp-httplib
if(NOT EXISTS ${THIRD_PARTY_DIR}/httplib.h)
    file(DOWNLOAD 
        https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.14.3/httplib.h
        ${THIRD_PARTY_DIR}/httplib.h
        SHOW_PROGRESS
    )
endif()

# nlohmann/json
if(NOT EXISTS ${THIRD_PARTY_DIR}/json.hpp)
    file(DOWNLOAD
        https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp
        ${THIRD_PARTY_DIR}/json.hpp
        SHOW_PROGRESS
    )
endif()

# yaml-cpp for configuration (lightweight)
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found, configuration will use JSON only")
endif()

# ============================================================================
# Source Files
# ============================================================================
set(ONNX_SERVER_SOURCES
    src/main.cpp
    src/server/http_server.cpp
    src/server/router.cpp
    src/server/handlers.cpp
    src/inference/session_manager.cpp
    src/inference/model_registry.cpp
    src/inference/batch_executor.cpp
    src/metrics/collector.cpp
    src/metrics/prometheus.cpp
    src/utils/config.cpp
    src/utils/logging.cpp
    src/utils/thread_pool.cpp
)

set(ONNX_SERVER_HEADERS
    src/server/http_server.hpp
    src/server/router.hpp
    src/server/handlers.hpp
    src/inference/session_manager.hpp
    src/inference/model_registry.hpp
    src/inference/batch_executor.hpp
    src/metrics/collector.hpp
    src/metrics/prometheus.hpp
    src/utils/config.hpp
    src/utils/logging.hpp
    src/utils/thread_pool.hpp
)

# ============================================================================
# Main Executable
# ============================================================================
add_executable(onnx-server ${ONNX_SERVER_SOURCES})

target_include_directories(onnx-server PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${THIRD_PARTY_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(onnx-server PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    Threads::Threads
)

# Conditional linking
if(yaml-cpp_FOUND)
    target_link_libraries(onnx-server PRIVATE yaml-cpp)
    target_compile_definitions(onnx-server PRIVATE YAML_CONFIG_SUPPORT)
endif()

if(ENABLE_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(onnx-server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(onnx-server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()

# Compile definitions for providers
if(ENABLE_CUDA)
    target_compile_definitions(onnx-server PRIVATE ENABLE_CUDA)
endif()

if(ENABLE_TENSORRT)
    target_compile_definitions(onnx-server PRIVATE ENABLE_TENSORRT)
endif()

# ============================================================================
# Installation
# ============================================================================
install(TARGETS onnx-server RUNTIME DESTINATION bin)
install(DIRECTORY models/ DESTINATION share/onnx-server/models)
install(FILES config.example.yaml DESTINATION etc/onnx-server)

# ============================================================================
# Testing
# ============================================================================
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "======== ONNX Server Configuration ========")
message(STATUS "Version:        ${PROJECT_VERSION}")
message(STATUS "Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA support:   ${ENABLE_CUDA}")
message(STATUS "TensorRT:       ${ENABLE_TENSORRT}")
message(STATUS "SSL/TLS:        ${ENABLE_SSL}")
message(STATUS "Static build:   ${BUILD_STATIC}")
message(STATUS "Tests:          ${BUILD_TESTS}")
message(STATUS "============================================")
message(STATUS "")
