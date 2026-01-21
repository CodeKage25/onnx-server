#pragma once

#include <memory>
#include <string>

#include "httplib.h"
#include "inference/batch_executor.hpp"
#include "inference/model_registry.hpp"
#include "inference/session_manager.hpp"
#include "json.hpp"
#include "metrics/collector.hpp"
#include "router.hpp"
#include "utils/config.hpp"
#include "utils/logging.hpp"

namespace onnx_server {

using json = nlohmann::json;

/**
 * API Handlers for the ONNX inference server
 */
class Handlers {
public:
  Handlers(ModelRegistry &model_registry, BatchExecutor &batch_executor,
           MetricsCollector &metrics, const Config &config)
      : model_registry_(model_registry), batch_executor_(batch_executor),
        metrics_(metrics), config_(config),
        start_time_(std::chrono::steady_clock::now()) {}

  /**
   * Register all API routes
   */
  void register_routes(Router &router) {
    // Health and status endpoints
    router.get("/health", [this](auto &req, auto &res, auto &ctx) {
      handle_health(req, res, ctx);
    });
    router.get("/ready", [this](auto &req, auto &res, auto &ctx) {
      handle_ready(req, res, ctx);
    });
    router.get("/", [this](auto &req, auto &res, auto &ctx) {
      handle_info(req, res, ctx);
    });

    // Model management endpoints
    router.get("/v1/models", [this](auto &req, auto &res, auto &ctx) {
      handle_list_models(req, res, ctx);
    });
    router.get(R"(/v1/models/([^/]+))",
               [this](auto &req, auto &res, auto &ctx) {
                 handle_get_model(req, res, ctx);
               });
    router.post(R"(/v1/models/([^/]+)/reload)",
                [this](auto &req, auto &res, auto &ctx) {
                  handle_reload_model(req, res, ctx);
                });

    // Inference endpoint
    router.post(R"(/v1/models/([^/]+)/infer)",
                [this](auto &req, auto &res, auto &ctx) {
                  handle_infer(req, res, ctx);
                });

    // Metrics endpoint
    router.get(config_.metrics.path, [this](auto &req, auto &res, auto &ctx) {
      handle_metrics(req, res, ctx);
    });

    LOG_INFO("Registered API routes");
  }

private:
  ModelRegistry &model_registry_;
  BatchExecutor &batch_executor_;
  MetricsCollector &metrics_;
  const Config &config_;
  std::chrono::steady_clock::time_point start_time_;

  /**
   * GET /health - Liveness probe
   */
  void handle_health(const httplib::Request &req, httplib::Response &res,
                     RequestContext &ctx) {
    json response = {{"status", "healthy"}, {"timestamp", get_iso_timestamp()}};
    res.status = 200;
    res.set_content(response.dump(), "application/json");
  }

  /**
   * GET /ready - Readiness probe
   */
  void handle_ready(const httplib::Request &req, httplib::Response &res,
                    RequestContext &ctx) {
    bool models_ready = model_registry_.has_models();

    json response = {{"status", models_ready ? "ready" : "not_ready"},
                     {"models_loaded", model_registry_.count()},
                     {"timestamp", get_iso_timestamp()}};

    res.status = models_ready ? 200 : 503;
    res.set_content(response.dump(), "application/json");
  }

  /**
   * GET / - Server info
   */
  void handle_info(const httplib::Request &req, httplib::Response &res,
                   RequestContext &ctx) {
    auto uptime = std::chrono::steady_clock::now() - start_time_;
    auto uptime_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(uptime).count();

    json response = {{"name", "onnx-server"},
                     {"version", "1.0.0"},
                     {"uptime_seconds", uptime_seconds},
                     {"models_loaded", model_registry_.count()},
                     {"batching_enabled", config_.batching.enabled},
                     {"providers", config_.inference.providers}};

    res.status = 200;
    res.set_content(response.dump(), "application/json");
  }

  /**
   * GET /v1/models - List all loaded models
   */
  void handle_list_models(const httplib::Request &req, httplib::Response &res,
                          RequestContext &ctx) {
    auto models = model_registry_.list();

    json response = {{"models", json::array()}};

    for (const auto &model : models) {
      response["models"].push_back({{"name", model.name},
                                    {"version", model.version},
                                    {"path", model.path},
                                    {"loaded_at", model.loaded_at},
                                    {"input_names", model.input_names},
                                    {"output_names", model.output_names}});
    }

    res.status = 200;
    res.set_content(response.dump(), "application/json");
  }

  /**
   * GET /v1/models/:name - Get model details
   */
  void handle_get_model(const httplib::Request &req, httplib::Response &res,
                        RequestContext &ctx) {
    std::string model_name = req.matches[1].str();

    auto model_opt = model_registry_.get(model_name);
    if (!model_opt) {
      res.status = 404;
      json error = {
          {"error",
           {{"code", 404}, {"message", "Model not found: " + model_name}}}};
      res.set_content(error.dump(), "application/json");
      return;
    }

    const auto &model = *model_opt;

    json response = {{"name", model.name},      {"version", model.version},
                     {"path", model.path},      {"loaded_at", model.loaded_at},
                     {"inputs", json::array()}, {"outputs", json::array()}};

    // Add input tensor info
    for (size_t i = 0; i < model.input_names.size(); ++i) {
      response["inputs"].push_back(
          {{"name", model.input_names[i]},
           {"shape", model.input_shapes.size() > i ? model.input_shapes[i]
                                                   : std::vector<int64_t>{}},
           {"dtype",
            model.input_types.size() > i ? model.input_types[i] : "unknown"}});
    }

    // Add output tensor info
    for (size_t i = 0; i < model.output_names.size(); ++i) {
      response["outputs"].push_back(
          {{"name", model.output_names[i]},
           {"shape", model.output_shapes.size() > i ? model.output_shapes[i]
                                                    : std::vector<int64_t>{}},
           {"dtype", model.output_types.size() > i ? model.output_types[i]
                                                   : "unknown"}});
    }

    res.status = 200;
    res.set_content(response.dump(), "application/json");
  }

  /**
   * POST /v1/models/:name/reload - Hot-reload a model
   */
  void handle_reload_model(const httplib::Request &req, httplib::Response &res,
                           RequestContext &ctx) {
    std::string model_name = req.matches[1].str();

    LOG_INFO("Reloading model: {}", model_name);

    try {
      bool success = model_registry_.reload(model_name);

      if (success) {
        json response = {{"status", "reloaded"},
                         {"model", model_name},
                         {"timestamp", get_iso_timestamp()}};
        res.status = 200;
        res.set_content(response.dump(), "application/json");
      } else {
        res.status = 404;
        json error = {
            {"error",
             {{"code", 404}, {"message", "Model not found: " + model_name}}}};
        res.set_content(error.dump(), "application/json");
      }
    } catch (const std::exception &e) {
      LOG_ERROR("Failed to reload model {}: {}", model_name, e.what());
      res.status = 500;
      json error = {{"error",
                     {{"code", 500},
                      {"message", "Failed to reload model"},
                      {"detail", e.what()}}}};
      res.set_content(error.dump(), "application/json");
    }
  }

  /**
   * POST /v1/models/:name/infer - Run inference
   */
  void handle_infer(const httplib::Request &req, httplib::Response &res,
                    RequestContext &ctx) {
    std::string model_name = req.matches[1].str();

    // Parse request body
    json request_body;
    try {
      request_body = json::parse(req.body);
    } catch (const std::exception &e) {
      res.status = 400;
      json error = {{"error",
                     {{"code", 400},
                      {"message", "Invalid JSON body"},
                      {"detail", e.what()}}}};
      res.set_content(error.dump(), "application/json");
      return;
    }

    // Validate inputs
    if (!request_body.contains("inputs")) {
      res.status = 400;
      json error = {
          {"error", {{"code", 400}, {"message", "Missing 'inputs' field"}}}};
      res.set_content(error.dump(), "application/json");
      return;
    }

    // Check model exists
    if (!model_registry_.has(model_name)) {
      res.status = 404;
      json error = {
          {"error",
           {{"code", 404}, {"message", "Model not found: " + model_name}}}};
      res.set_content(error.dump(), "application/json");
      return;
    }

    try {
      // Create inference request
      InferenceRequest infer_req;
      infer_req.model_name = model_name;
      infer_req.request_id = ctx.request_id;

      // Parse inputs
      for (auto &[name, tensor] : request_body["inputs"].items()) {
        TensorData input;
        input.name = name;

        if (tensor.contains("shape")) {
          input.shape = tensor["shape"].get<std::vector<int64_t>>();
        }

        if (tensor.contains("data")) {
          // Handle different data types
          if (tensor["data"].is_array()) {
            parse_tensor_data(tensor["data"], input);
          }
        }

        if (tensor.contains("dtype")) {
          input.dtype = tensor["dtype"];
        }

        infer_req.inputs.push_back(std::move(input));
      }

      // Run inference (through batch executor if enabled)
      InferenceResponse infer_res;
      if (config_.batching.enabled) {
        infer_res = batch_executor_.submit(std::move(infer_req)).get();
      } else {
        infer_res = model_registry_.run_inference(infer_req);
      }

      // Build response
      json response = {{"model_name", model_name}, {"outputs", json::object()}};

      for (const auto &output : infer_res.outputs) {
        response["outputs"][output.name] = {
            {"shape", output.shape},
            {"data", output.float_data.empty() ? json(output.int_data)
                                               : json(output.float_data)}};
      }

      // Include timing info if available
      if (infer_res.inference_time_ms > 0) {
        response["timing"] = {{"inference_ms", infer_res.inference_time_ms},
                              {"queue_ms", infer_res.queue_time_ms}};
      }

      res.status = 200;
      res.set_content(response.dump(), "application/json");

      // Record inference metrics
      metrics_.record_inference(model_name,
                                infer_res.inference_time_ms / 1000.0);

    } catch (const std::exception &e) {
      LOG_ERROR("Inference error for model {}: {}", model_name, e.what());
      res.status = 500;
      json error = {{"error",
                     {{"code", 500},
                      {"message", "Inference failed"},
                      {"detail", e.what()}}}};
      res.set_content(error.dump(), "application/json");
    }
  }

  /**
   * GET /metrics - Prometheus metrics
   */
  void handle_metrics(const httplib::Request &req, httplib::Response &res,
                      RequestContext &ctx) {
    std::string output = metrics_.export_prometheus();
    res.status = 200;
    res.set_content(output, "text/plain; version=0.0.4; charset=utf-8");
  }

  /**
   * Helper to get ISO timestamp
   */
  static std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
  }

  /**
   * Helper to parse tensor data from JSON
   */
  static void parse_tensor_data(const json &data, TensorData &tensor) {
    // Flatten nested arrays and store as float data by default
    std::function<void(const json &)> flatten = [&](const json &arr) {
      if (arr.is_array()) {
        for (const auto &item : arr) {
          flatten(item);
        }
      } else if (arr.is_number_float()) {
        tensor.float_data.push_back(arr.get<float>());
      } else if (arr.is_number_integer()) {
        tensor.float_data.push_back(static_cast<float>(arr.get<int64_t>()));
      }
    };

    flatten(data);
  }
};

} // namespace onnx_server
