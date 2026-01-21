/**
 * ONNX Inference Server
 *
 * A lightweight, high-performance inference server for deploying ONNX models
 * via REST API with dynamic batching and GPU acceleration.
 *
 * Usage:
 *   onnx-server [options]
 *
 * Options:
 *   --config <path>      Path to configuration file (default: config.yaml)
 *   --models <path>      Path to models directory (overrides config)
 *   --port <port>        Server port (overrides config)
 *   --help               Show this help message
 */

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>

#include "inference/batch_executor.hpp"
#include "inference/model_registry.hpp"
#include "inference/session_manager.hpp"
#include "metrics/collector.hpp"
#include "server/handlers.hpp"
#include "server/http_server.hpp"
#include "server/router.hpp"
#include "utils/config.hpp"
#include "utils/logging.hpp"

using namespace onnx_server;

// Global shutdown flag
std::atomic<bool> g_shutdown_requested{false};

// Signal handler for graceful shutdown
void signal_handler(int signal) {
  LOG_INFO("Received signal {}, initiating shutdown...", signal);
  g_shutdown_requested = true;
}

/**
 * Parse command line arguments
 */
struct CommandLineArgs {
  std::string config_path = "config.yaml";
  std::string models_path;
  int port = -1;
  bool help = false;

  static CommandLineArgs parse(int argc, char *argv[]) {
    CommandLineArgs args;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];

      if (arg == "--help" || arg == "-h") {
        args.help = true;
      } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
        args.config_path = argv[++i];
      } else if ((arg == "--models" || arg == "-m") && i + 1 < argc) {
        args.models_path = argv[++i];
      } else if ((arg == "--port" || arg == "-p") && i + 1 < argc) {
        args.port = std::stoi(argv[++i]);
      }
    }

    return args;
  }

  static void print_usage() {
    std::cout << R"(
ONNX Inference Server

Usage: onnx-server [options]

Options:
  -c, --config <path>   Path to configuration file (default: config.yaml)
  -m, --models <path>   Path to models directory (overrides config)
  -p, --port <port>     Server port (overrides config)
  -h, --help            Show this help message

Examples:
  onnx-server --config /etc/onnx-server/config.yaml
  onnx-server --models /models --port 8080
  
Environment Variables:
  ONNX_SERVER_HOST      Server bind address
  ONNX_SERVER_PORT      Server port
  ONNX_MODELS_DIR       Models directory
  ONNX_LOG_LEVEL        Log level (debug, info, warn, error)

)" << std::endl;
  }
};

int main(int argc, char *argv[]) {
  // Parse command line
  auto args = CommandLineArgs::parse(argc, argv);

  if (args.help) {
    CommandLineArgs::print_usage();
    return 0;
  }

  // Load configuration
  Config config;
  try {
    config = Config::load_from_file(args.config_path);
  } catch (const std::exception &e) {
    // Continue with defaults if config file not found
  }

  // Apply environment variable overrides
  config.load_from_env();

  // Apply command line overrides
  if (!args.models_path.empty()) {
    config.models.directory = args.models_path;
  }
  if (args.port > 0) {
    config.server.port = args.port;
  }

  // Initialize logging
  Logger::instance().set_level(config.logging.level);
  Logger::instance().set_json_format(config.logging.format == "json");

  LOG_INFO("Starting ONNX Inference Server v1.0.0");
  LOG_INFO("Configuration: {}", config.to_json().dump());

  // Setup signal handlers
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  try {
    // Initialize components
    MetricsCollector metrics(config.metrics);
    SessionManager session_manager(config.inference);
    ModelRegistry model_registry(session_manager, config.models);
    BatchExecutor batch_executor(model_registry, metrics, config.batching);

    // Initialize model registry
    model_registry.initialize();
    metrics.set_loaded_models(static_cast<int>(model_registry.count()));

    // Start batch executor
    batch_executor.start();

    // Create HTTP server
    HttpServer http_server(config.server);
    Router router(http_server, &metrics);

    // Setup handlers
    router.setup_error_handling();
    router.setup_request_logging();

    Handlers handlers(model_registry, batch_executor, metrics, config);
    handlers.register_routes(router);

    // Start server in async mode
    http_server.start_async();

    LOG_INFO("Server listening on {}:{}", config.server.host,
             config.server.port);
    LOG_INFO("Models directory: {}", config.models.directory);
    LOG_INFO("Loaded {} model(s)", model_registry.count());

    // Main loop - wait for shutdown signal
    while (!g_shutdown_requested && http_server.is_running()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));

      // Update metrics periodically
      metrics.set_loaded_models(static_cast<int>(model_registry.count()));
    }

    // Graceful shutdown
    LOG_INFO("Shutting down...");

    batch_executor.stop();
    model_registry.stop_watcher();
    http_server.stop();

    LOG_INFO("Server stopped successfully");
    return 0;

  } catch (const std::exception &e) {
    LOG_ERROR("Fatal error: {}", e.what());
    return 1;
  }
}
