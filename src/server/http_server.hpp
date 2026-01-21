#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>

#include "httplib.h"
#include "utils/config.hpp"
#include "utils/logging.hpp"
#include "utils/thread_pool.hpp"

namespace onnx_server {

/**
 * HTTP Server wrapper around cpp-httplib
 * Provides a clean interface for the ONNX server with graceful shutdown
 */
class HttpServer {
public:
  using Handler =
      std::function<void(const httplib::Request &, httplib::Response &)>;

  explicit HttpServer(const ServerConfig &config)
      : config_(config), running_(false), thread_pool_(config.threads) {}

  ~HttpServer() { stop(); }

  // Non-copyable
  HttpServer(const HttpServer &) = delete;
  HttpServer &operator=(const HttpServer &) = delete;

  /**
   * Register a GET handler
   */
  void get(const std::string &pattern, Handler handler) {
    server_.Get(pattern,
                [handler](const httplib::Request &req, httplib::Response &res) {
                  handler(req, res);
                });
  }

  /**
   * Register a POST handler
   */
  void post(const std::string &pattern, Handler handler) {
    server_.Post(pattern,
                 [handler](const httplib::Request &req,
                           httplib::Response &res) { handler(req, res); });
  }

  /**
   * Register a PUT handler
   */
  void put(const std::string &pattern, Handler handler) {
    server_.Put(pattern,
                [handler](const httplib::Request &req, httplib::Response &res) {
                  handler(req, res);
                });
  }

  /**
   * Register a DELETE handler
   */
  void del(const std::string &pattern, Handler handler) {
    server_.Delete(pattern,
                   [handler](const httplib::Request &req,
                             httplib::Response &res) { handler(req, res); });
  }

  /**
   * Set error handler for uncaught exceptions
   */
  void set_error_handler(Handler handler) {
    server_.set_error_handler(
        [handler](const httplib::Request &req, httplib::Response &res) {
          handler(req, res);
        });
  }

  /**
   * Set exception handler
   */
  void set_exception_handler(
      std::function<void(const httplib::Request &, httplib::Response &,
                         std::exception_ptr)>
          handler) {
    server_.set_exception_handler(handler);
  }

  /**
   * Set pre-routing handler (middleware)
   */
  void set_pre_routing_handler(
      std::function<httplib::Server::HandlerResponse(const httplib::Request &,
                                                     httplib::Response &)>
          handler) {
    server_.set_pre_routing_handler(handler);
  }

  /**
   * Start the server (blocking)
   */
  bool start() {
    if (running_) {
      LOG_WARN("Server already running");
      return false;
    }

    running_ = true;

    // Configure server settings
    server_.set_keep_alive_max_count(100);
    server_.set_keep_alive_timeout(30);
    server_.set_read_timeout(30);
    server_.set_write_timeout(30);
    server_.set_payload_max_length(1024 * 1024 * 100); // 100MB max payload

    LOG_INFO("Starting HTTP server on {}:{}", config_.host, config_.port);

    bool success = server_.listen(config_.host, config_.port);

    if (!success) {
      LOG_ERROR("Failed to start server on {}:{}", config_.host, config_.port);
      running_ = false;
    }

    return success;
  }

  /**
   * Start the server in a background thread
   */
  void start_async() {
    server_thread_ = std::thread([this]() { start(); });
  }

  /**
   * Stop the server gracefully
   */
  void stop() {
    if (!running_)
      return;

    LOG_INFO("Shutting down HTTP server...");
    running_ = false;
    server_.stop();

    if (server_thread_.joinable()) {
      server_thread_.join();
    }

    LOG_INFO("HTTP server stopped");
  }

  /**
   * Check if server is running
   */
  bool is_running() const { return running_; }

  /**
   * Get the underlying httplib server for advanced configuration
   */
  httplib::Server &raw() { return server_; }

  /**
   * Get thread pool reference
   */
  ThreadPool &thread_pool() { return thread_pool_; }

private:
  ServerConfig config_;
  httplib::Server server_;
  std::atomic<bool> running_;
  std::thread server_thread_;
  ThreadPool thread_pool_;
};

} // namespace onnx_server
