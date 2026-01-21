#pragma once

#include <chrono>
#include <functional>
#include <regex>
#include <string>
#include <unordered_map>

#include "http_server.hpp"
#include "httplib.h"
#include "json.hpp"
#include "metrics/collector.hpp"
#include "utils/logging.hpp"

namespace onnx_server {

using json = nlohmann::json;

/**
 * Request context with parsed parameters and metadata
 */
struct RequestContext {
  std::unordered_map<std::string, std::string> path_params;
  std::chrono::steady_clock::time_point start_time;
  std::string request_id;

  RequestContext()
      : start_time(std::chrono::steady_clock::now()),
        request_id(generate_request_id()) {}

private:
  static std::string generate_request_id() {
    static std::atomic<uint64_t> counter{0};
    return "req-" + std::to_string(++counter);
  }
};

/**
 * Router for handling API endpoints with path parameter extraction
 */
class Router {
public:
  using RouteHandler = std::function<void(
      const httplib::Request &, httplib::Response &, RequestContext &)>;

  explicit Router(HttpServer &server, MetricsCollector *metrics = nullptr)
      : server_(server), metrics_(metrics) {}

  /**
   * Register GET route with path parameter support
   * Pattern example: "/v1/models/:name/infer"
   */
  void get(const std::string &pattern, RouteHandler handler) {
    auto regex_pattern = build_regex_pattern(pattern);
    server_.get(regex_pattern, wrap_handler(pattern, "GET", handler));
  }

  /**
   * Register POST route with path parameter support
   */
  void post(const std::string &pattern, RouteHandler handler) {
    auto regex_pattern = build_regex_pattern(pattern);
    server_.post(regex_pattern, wrap_handler(pattern, "POST", handler));
  }

  /**
   * Register PUT route with path parameter support
   */
  void put(const std::string &pattern, RouteHandler handler) {
    auto regex_pattern = build_regex_pattern(pattern);
    server_.put(regex_pattern, wrap_handler(pattern, "PUT", handler));
  }

  /**
   * Register DELETE route with path parameter support
   */
  void del(const std::string &pattern, RouteHandler handler) {
    auto regex_pattern = build_regex_pattern(pattern);
    server_.del(regex_pattern, wrap_handler(pattern, "DELETE", handler));
  }

  /**
   * Setup global error handling
   */
  void setup_error_handling() {
    server_.set_error_handler(
        [](const httplib::Request &req, httplib::Response &res) {
          json error = {{"error",
                         {{"code", res.status},
                          {"message", get_status_message(res.status)}}}};
          res.set_content(error.dump(), "application/json");
        });

    server_.set_exception_handler([](const httplib::Request &req,
                                     httplib::Response &res,
                                     std::exception_ptr ep) {
      try {
        if (ep)
          std::rethrow_exception(ep);
      } catch (const std::exception &e) {
        LOG_ERROR("Unhandled exception: {}", e.what());
        json error = {{"error",
                       {{"code", 500},
                        {"message", "Internal server error"},
                        {"detail", e.what()}}}};
        res.status = 500;
        res.set_content(error.dump(), "application/json");
      }
    });
  }

  /**
   * Setup request logging middleware
   */
  void setup_request_logging() {
    server_.set_pre_routing_handler(
        [this](const httplib::Request &req, httplib::Response &res) {
          LOG_DEBUG("{} {} from {}", req.method, req.path, req.remote_addr);
          return httplib::Server::HandlerResponse::Unhandled;
        });
  }

private:
  HttpServer &server_;
  MetricsCollector *metrics_;

  /**
   * Convert route pattern with :param to regex pattern
   */
  std::string build_regex_pattern(const std::string &pattern) {
    // Store param names for extraction later
    std::vector<std::string> params;

    std::string result;
    bool in_param = false;
    std::string param_name;

    for (size_t i = 0; i < pattern.size(); ++i) {
      char c = pattern[i];

      if (c == ':') {
        in_param = true;
        param_name.clear();
      } else if (in_param) {
        if (std::isalnum(c) || c == '_') {
          param_name += c;
        } else {
          params.push_back(param_name);
          result += "([^/]+)";
          result += c;
          in_param = false;
        }
      } else {
        result += c;
      }
    }

    if (in_param && !param_name.empty()) {
      params.push_back(param_name);
      result += "([^/]+)";
    }

    // Store pattern metadata
    pattern_params_[result] = params;

    return result;
  }

  /**
   * Extract path parameters from request using stored pattern info
   */
  std::unordered_map<std::string, std::string>
  extract_params(const httplib::Request &req, const std::string &pattern) {

    std::unordered_map<std::string, std::string> params;

    auto it = pattern_params_.find(pattern);
    if (it == pattern_params_.end())
      return params;

    const auto &param_names = it->second;
    for (size_t i = 0; i < param_names.size() && i < req.matches.size() - 1;
         ++i) {
      params[param_names[i]] = req.matches[i + 1].str();
    }

    return params;
  }

  /**
   * Wrap handler with context, logging, and metrics
   */
  HttpServer::Handler wrap_handler(const std::string &pattern,
                                   const std::string &method,
                                   RouteHandler handler) {
    std::string regex_pattern =
        pattern_params_.count(pattern) ? pattern : build_regex_pattern(pattern);

    // Find the regex pattern if we were given the original pattern
    for (const auto &[regex, _] : pattern_params_) {
      if (regex.find(pattern.substr(0, 5)) != std::string::npos) {
        regex_pattern = regex;
        break;
      }
    }

    return [this, regex_pattern, method, handler,
            pattern](const httplib::Request &req, httplib::Response &res) {
      RequestContext ctx;

      // Extract path parameters
      auto it = pattern_params_.find(regex_pattern);
      if (it != pattern_params_.end()) {
        const auto &param_names = it->second;
        for (size_t i = 0; i < param_names.size() && i < req.matches.size() - 1;
             ++i) {
          ctx.path_params[param_names[i]] = req.matches[i + 1].str();
        }
      }

      // Call the actual handler
      try {
        handler(req, res, ctx);
      } catch (const std::exception &e) {
        LOG_ERROR("Handler exception for {} {}: {}", method, pattern, e.what());
        res.status = 500;
        json error = {{"error", {{"code", 500}, {"message", e.what()}}}};
        res.set_content(error.dump(), "application/json");
      }

      // Record metrics
      auto duration = std::chrono::steady_clock::now() - ctx.start_time;
      double latency_ms =
          std::chrono::duration<double, std::milli>(duration).count();

      if (metrics_) {
        metrics_->record_request(pattern, method, res.status,
                                 latency_ms / 1000.0);
      }

      LOG_INFO("{} {} {} - {}ms", method, req.path, res.status, latency_ms);
    };
  }

  static std::string get_status_message(int status) {
    switch (status) {
    case 400:
      return "Bad Request";
    case 401:
      return "Unauthorized";
    case 403:
      return "Forbidden";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 422:
      return "Unprocessable Entity";
    case 500:
      return "Internal Server Error";
    case 503:
      return "Service Unavailable";
    default:
      return "Unknown Error";
    }
  }

  // Map from regex pattern to list of param names
  std::unordered_map<std::string, std::vector<std::string>> pattern_params_;
};

} // namespace onnx_server
