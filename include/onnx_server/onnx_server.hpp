#pragma once

/**
 * ONNX Inference Server - Public API
 *
 * Include this header for embedding the ONNX server as a library.
 */

#include "../src/inference/batch_executor.hpp"
#include "../src/inference/model_registry.hpp"
#include "../src/inference/session_manager.hpp"
#include "../src/metrics/collector.hpp"
#include "../src/server/handlers.hpp"
#include "../src/server/http_server.hpp"
#include "../src/server/router.hpp"
#include "../src/utils/config.hpp"
#include "../src/utils/logging.hpp"

namespace onnx_server {

/**
 * Server version information
 */
constexpr const char *VERSION = "1.0.0";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace onnx_server
