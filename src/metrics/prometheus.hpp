#pragma once

#include "collector.hpp"

// The prometheus exporter functionality is integrated into the MetricsCollector
// class This header exists for organizational purposes and potential future
// extensions

namespace onnx_server {

/**
 * Prometheus format constants
 */
namespace prometheus {
constexpr const char *CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8";

/**
 * Helper to format labels for Prometheus metrics
 */
inline std::string
format_labels(const std::vector<std::pair<std::string, std::string>> &labels) {
  if (labels.empty())
    return "";

  std::stringstream ss;
  ss << "{";
  bool first = true;
  for (const auto &[key, value] : labels) {
    if (!first)
      ss << ",";
    ss << key << "=\"" << value << "\"";
    first = false;
  }
  ss << "}";
  return ss.str();
}
} // namespace prometheus

} // namespace onnx_server
