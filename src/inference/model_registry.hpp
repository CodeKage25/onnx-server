#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "session_manager.hpp"
#include "utils/config.hpp"
#include "utils/logging.hpp"

namespace onnx_server {

namespace fs = std::filesystem;

/**
 * Model Registry - manages multiple models with hot-reload capability
 */
class ModelRegistry {
public:
  ModelRegistry(SessionManager &session_manager, const ModelsConfig &config)
      : session_manager_(session_manager), config_(config), running_(false) {}

  ~ModelRegistry() { stop_watcher(); }

  /**
   * Initialize the registry and load models from directory
   */
  void initialize() {
    LOG_INFO("Initializing model registry from: {}", config_.directory);

    if (!fs::exists(config_.directory)) {
      LOG_WARN("Models directory does not exist: {}", config_.directory);
      fs::create_directories(config_.directory);
      return;
    }

    scan_and_load_models();

    if (config_.hot_reload) {
      start_watcher();
    }
  }

  /**
   * Check if a model exists
   */
  bool has(const std::string &name) const {
    std::shared_lock lock(mutex_);
    return models_.count(name) > 0;
  }

  /**
   * Check if any models are loaded
   */
  bool has_models() const {
    std::shared_lock lock(mutex_);
    return !models_.empty();
  }

  /**
   * Get model count
   */
  size_t count() const {
    std::shared_lock lock(mutex_);
    return models_.size();
  }

  /**
   * Get model info by name
   */
  std::optional<ModelInfo> get(const std::string &name) const {
    std::shared_lock lock(mutex_);
    auto it = models_.find(name);
    if (it != models_.end()) {
      return it->second.info;
    }
    return std::nullopt;
  }

  /**
   * List all loaded models
   */
  std::vector<ModelInfo> list() const {
    std::shared_lock lock(mutex_);
    std::vector<ModelInfo> result;
    result.reserve(models_.size());
    for (const auto &[name, entry] : models_) {
      result.push_back(entry.info);
    }
    return result;
  }

  /**
   * Reload a specific model
   */
  bool reload(const std::string &name) {
    std::shared_lock read_lock(mutex_);
    auto it = models_.find(name);
    if (it == models_.end()) {
      return false;
    }
    std::string path = it->second.info.path;
    read_lock.unlock();

    return load_model(path, name);
  }

  /**
   * Run inference on a model
   */
  InferenceResponse run_inference(const InferenceRequest &request) {
    std::shared_lock lock(mutex_);

    auto it = models_.find(request.model_name);
    if (it == models_.end()) {
      InferenceResponse response;
      response.success = false;
      response.error = "Model not found: " + request.model_name;
      return response;
    }

    auto &entry = it->second;
    return session_manager_.run_inference(*entry.session, request, entry.info);
  }

  /**
   * Stop file watcher
   */
  void stop_watcher() {
    if (running_) {
      running_ = false;
      if (watcher_thread_.joinable()) {
        watcher_thread_.join();
      }
    }
  }

private:
  struct ModelEntry {
    std::unique_ptr<Ort::Session> session;
    ModelInfo info;
    fs::file_time_type last_modified;
  };

  SessionManager &session_manager_;
  ModelsConfig config_;

  mutable std::shared_mutex mutex_;
  std::unordered_map<std::string, ModelEntry> models_;

  std::atomic<bool> running_;
  std::thread watcher_thread_;

  /**
   * Scan directory and load all ONNX models
   */
  void scan_and_load_models() {
    for (const auto &entry : fs::directory_iterator(config_.directory)) {
      if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
        std::string name = entry.path().stem().string();
        load_model(entry.path().string(), name);
      }
    }
  }

  /**
   * Load a single model
   */
  bool load_model(const std::string &path, const std::string &name) {
    try {
      auto [session, info] = session_manager_.load_model(path, name);

      std::unique_lock lock(mutex_);

      ModelEntry entry;
      entry.session = std::move(session);
      entry.info = std::move(info);
      entry.last_modified = fs::last_write_time(path);

      models_[name] = std::move(entry);

      LOG_INFO("Model '{}' loaded successfully", name);
      return true;

    } catch (const std::exception &e) {
      LOG_ERROR("Failed to load model '{}': {}", name, e.what());
      return false;
    }
  }

  /**
   * Start the file watcher for hot-reload
   */
  void start_watcher() {
    running_ = true;
    watcher_thread_ = std::thread([this]() {
      LOG_INFO("Starting model file watcher (interval: {}ms)",
               config_.watch_interval_ms);

      while (running_) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.watch_interval_ms));

        if (!running_)
          break;

        check_for_changes();
      }
    });
  }

  /**
   * Check for model file changes
   */
  void check_for_changes() {
    if (!fs::exists(config_.directory))
      return;

    // Check for new/modified models
    for (const auto &entry : fs::directory_iterator(config_.directory)) {
      if (!entry.is_regular_file() || entry.path().extension() != ".onnx") {
        continue;
      }

      std::string name = entry.path().stem().string();
      std::string path = entry.path().string();
      auto mod_time = fs::last_write_time(path);

      std::shared_lock read_lock(mutex_);
      auto it = models_.find(name);

      if (it == models_.end()) {
        // New model
        read_lock.unlock();
        LOG_INFO("Detected new model: {}", name);
        load_model(path, name);
      } else if (it->second.last_modified != mod_time) {
        // Modified model
        read_lock.unlock();
        LOG_INFO("Detected model change: {}", name);
        load_model(path, name);
      }
    }

    // Check for removed models
    std::vector<std::string> to_remove;
    {
      std::shared_lock lock(mutex_);
      for (const auto &[name, entry] : models_) {
        if (!fs::exists(entry.info.path)) {
          to_remove.push_back(name);
        }
      }
    }

    if (!to_remove.empty()) {
      std::unique_lock lock(mutex_);
      for (const auto &name : to_remove) {
        LOG_INFO("Removing unloaded model: {}", name);
        models_.erase(name);
      }
    }
  }
};

} // namespace onnx_server
