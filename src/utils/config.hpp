#pragma once

#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "logging.hpp"

namespace onnx_server {

using json = nlohmann::json;

/**
 * Server configuration for HTTP settings
 */
struct ServerConfig {
  std::string host = "0.0.0.0";
  int port = 8080;
  int threads = 4;
};

/**
 * Inference configuration for ONNX Runtime
 */
struct InferenceConfig {
  std::vector<std::string> providers = {"cuda", "cpu"};
  int gpu_device_id = 0;
  size_t memory_limit_mb = 4096;
  int intra_op_threads = 0;
  int inter_op_threads = 0;
  std::string graph_optimization = "all";
};

/**
 * Dynamic batching configuration
 */
struct BatchingConfig {
  bool enabled = true;
  size_t max_batch_size = 32;
  size_t min_batch_size = 1;
  uint32_t max_wait_ms = 10;
  bool adaptive_sizing = true;
};

/**
 * Model loading configuration
 */
struct ModelsConfig {
  std::string directory = "./models";
  bool hot_reload = true;
  uint32_t watch_interval_ms = 5000;
  std::vector<std::string> preload;
};

/**
 * Prometheus metrics configuration
 */
struct MetricsConfig {
  bool enabled = true;
  std::string path = "/metrics";
  std::vector<double> latency_buckets = {0.001, 0.005, 0.01, 0.025, 0.05,
                                         0.1,   0.25,  0.5,  1.0};
};

/**
 * Logging configuration
 */
struct LoggingConfig {
  std::string level = "info";
  std::string format = "json";
  bool timestamp = true;
};

/**
 * Complete server configuration
 */
struct Config {
  ServerConfig server;
  InferenceConfig inference;
  BatchingConfig batching;
  ModelsConfig models;
  MetricsConfig metrics;
  LoggingConfig logging;

  /**
   * Load configuration from JSON file
   */
  static Config load_from_file(const std::string &path) {
    Config config;

    std::ifstream file(path);
    if (!file.is_open()) {
      LOG_WARN("Config file not found: {}, using defaults", path);
      return config;
    }

    try {
      json j;
      file >> j;
      config = parse_json(j);
      LOG_INFO("Loaded configuration from: {}", path);
    } catch (const std::exception &e) {
      LOG_ERROR("Failed to parse config file: {}", e.what());
      throw;
    }

    return config;
  }

  /**
   * Load configuration from environment variables (takes precedence)
   */
  void load_from_env() {
    // Server
    if (const char *val = std::getenv("ONNX_SERVER_HOST")) {
      server.host = val;
    }
    if (const char *val = std::getenv("ONNX_SERVER_PORT")) {
      server.port = std::stoi(val);
    }
    if (const char *val = std::getenv("ONNX_SERVER_THREADS")) {
      server.threads = std::stoi(val);
    }

    // Inference
    if (const char *val = std::getenv("ONNX_GPU_DEVICE_ID")) {
      inference.gpu_device_id = std::stoi(val);
    }
    if (const char *val = std::getenv("ONNX_MEMORY_LIMIT_MB")) {
      inference.memory_limit_mb = std::stoull(val);
    }

    // Batching
    if (const char *val = std::getenv("ONNX_BATCHING_ENABLED")) {
      batching.enabled =
          (std::string(val) == "true" || std::string(val) == "1");
    }
    if (const char *val = std::getenv("ONNX_MAX_BATCH_SIZE")) {
      batching.max_batch_size = std::stoull(val);
    }
    if (const char *val = std::getenv("ONNX_MAX_WAIT_MS")) {
      batching.max_wait_ms = std::stoul(val);
    }

    // Models
    if (const char *val = std::getenv("ONNX_MODELS_DIR")) {
      models.directory = val;
    }
    if (const char *val = std::getenv("ONNX_HOT_RELOAD")) {
      models.hot_reload =
          (std::string(val) == "true" || std::string(val) == "1");
    }

    // Metrics
    if (const char *val = std::getenv("ONNX_METRICS_ENABLED")) {
      metrics.enabled = (std::string(val) == "true" || std::string(val) == "1");
    }

    // Logging
    if (const char *val = std::getenv("ONNX_LOG_LEVEL")) {
      logging.level = val;
    }
  }

  /**
   * Convert config to JSON for debugging/introspection
   */
  json to_json() const {
    return json{
        {"server",
         {{"host", server.host},
          {"port", server.port},
          {"threads", server.threads}}},
        {"inference",
         {{"providers", inference.providers},
          {"gpu_device_id", inference.gpu_device_id},
          {"memory_limit_mb", inference.memory_limit_mb}}},
        {"batching",
         {{"enabled", batching.enabled},
          {"max_batch_size", batching.max_batch_size},
          {"max_wait_ms", batching.max_wait_ms}}},
        {"models",
         {{"directory", models.directory}, {"hot_reload", models.hot_reload}}},
        {"metrics", {{"enabled", metrics.enabled}, {"path", metrics.path}}}};
  }

private:
  static Config parse_json(const json &j) {
    Config config;

    if (j.contains("server")) {
      auto &s = j["server"];
      if (s.contains("host"))
        config.server.host = s["host"];
      if (s.contains("port"))
        config.server.port = s["port"];
      if (s.contains("threads"))
        config.server.threads = s["threads"];
    }

    if (j.contains("inference")) {
      auto &i = j["inference"];
      if (i.contains("providers"))
        config.inference.providers =
            i["providers"].get<std::vector<std::string>>();
      if (i.contains("gpu_device_id"))
        config.inference.gpu_device_id = i["gpu_device_id"];
      if (i.contains("memory_limit_mb"))
        config.inference.memory_limit_mb = i["memory_limit_mb"];
      if (i.contains("intra_op_threads"))
        config.inference.intra_op_threads = i["intra_op_threads"];
      if (i.contains("inter_op_threads"))
        config.inference.inter_op_threads = i["inter_op_threads"];
      if (i.contains("graph_optimization"))
        config.inference.graph_optimization = i["graph_optimization"];
    }

    if (j.contains("batching")) {
      auto &b = j["batching"];
      if (b.contains("enabled"))
        config.batching.enabled = b["enabled"];
      if (b.contains("max_batch_size"))
        config.batching.max_batch_size = b["max_batch_size"];
      if (b.contains("min_batch_size"))
        config.batching.min_batch_size = b["min_batch_size"];
      if (b.contains("max_wait_ms"))
        config.batching.max_wait_ms = b["max_wait_ms"];
      if (b.contains("adaptive_sizing"))
        config.batching.adaptive_sizing = b["adaptive_sizing"];
    }

    if (j.contains("models")) {
      auto &m = j["models"];
      if (m.contains("directory"))
        config.models.directory = m["directory"];
      if (m.contains("hot_reload"))
        config.models.hot_reload = m["hot_reload"];
      if (m.contains("watch_interval_ms"))
        config.models.watch_interval_ms = m["watch_interval_ms"];
      if (m.contains("preload"))
        config.models.preload = m["preload"].get<std::vector<std::string>>();
    }

    if (j.contains("metrics")) {
      auto &met = j["metrics"];
      if (met.contains("enabled"))
        config.metrics.enabled = met["enabled"];
      if (met.contains("path"))
        config.metrics.path = met["path"];
      if (met.contains("latency_buckets"))
        config.metrics.latency_buckets =
            met["latency_buckets"].get<std::vector<double>>();
    }

    if (j.contains("logging")) {
      auto &l = j["logging"];
      if (l.contains("level"))
        config.logging.level = l["level"];
      if (l.contains("format"))
        config.logging.format = l["format"];
      if (l.contains("timestamp"))
        config.logging.timestamp = l["timestamp"];
    }

    return config;
  }
};

} // namespace onnx_server
