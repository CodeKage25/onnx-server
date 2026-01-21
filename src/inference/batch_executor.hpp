#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

#include "metrics/collector.hpp"
#include "model_registry.hpp"
#include "session_manager.hpp"
#include "utils/config.hpp"
#include "utils/logging.hpp"

namespace onnx_server {

/**
 * Pending request in the batch queue
 */
struct PendingRequest {
  InferenceRequest request;
  std::promise<InferenceResponse> promise;
  std::chrono::steady_clock::time_point enqueue_time;
};

/**
 * Dynamic Request Batching Executor
 * Accumulates concurrent requests and executes them in batches for GPU
 * throughput
 */
class BatchExecutor {
public:
  BatchExecutor(ModelRegistry &model_registry, MetricsCollector &metrics,
                const BatchingConfig &config)
      : model_registry_(model_registry), metrics_(metrics), config_(config),
        running_(false) {}

  ~BatchExecutor() { stop(); }

  /**
   * Start the batch executor
   */
  void start() {
    if (!config_.enabled) {
      LOG_INFO("Batching disabled, requests will be processed individually");
      return;
    }

    running_ = true;
    executor_thread_ = std::thread([this]() { executor_loop(); });

    LOG_INFO("Batch executor started (max_batch_size: {}, max_wait_ms: {})",
             config_.max_batch_size, config_.max_wait_ms);
  }

  /**
   * Stop the batch executor
   */
  void stop() {
    if (!running_)
      return;

    running_ = false;
    queue_cv_.notify_all();

    if (executor_thread_.joinable()) {
      executor_thread_.join();
    }

    LOG_INFO("Batch executor stopped");
  }

  /**
   * Submit a request and get a future for the response
   */
  std::future<InferenceResponse> submit(InferenceRequest request) {
    auto pending = std::make_shared<PendingRequest>();
    pending->request = std::move(request);
    pending->enqueue_time = std::chrono::steady_clock::now();

    auto future = pending->promise.get_future();

    if (!config_.enabled) {
      // Process immediately without batching
      auto response = model_registry_.run_inference(pending->request);
      pending->promise.set_value(std::move(response));
      return future;
    }

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      pending_requests_.push(std::move(pending));
    }

    queue_cv_.notify_one();
    return future;
  }

  /**
   * Get current queue size
   */
  size_t queue_size() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return pending_requests_.size();
  }

  /**
   * Check if executor is running
   */
  bool is_running() const { return running_; }

private:
  ModelRegistry &model_registry_;
  MetricsCollector &metrics_;
  BatchingConfig config_;

  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::queue<std::shared_ptr<PendingRequest>> pending_requests_;

  std::atomic<bool> running_;
  std::thread executor_thread_;

  /**
   * Main executor loop
   */
  void executor_loop() {
    while (running_) {
      std::vector<std::shared_ptr<PendingRequest>> batch;

      {
        std::unique_lock<std::mutex> lock(queue_mutex_);

        // Wait for requests or timeout
        auto timeout = std::chrono::milliseconds(config_.max_wait_ms);
        queue_cv_.wait_for(lock, timeout, [this]() {
          return !pending_requests_.empty() || !running_;
        });

        if (!running_ && pending_requests_.empty()) {
          break;
        }

        // If we have minimum requests or timeout reached, collect batch
        if (pending_requests_.size() >= config_.min_batch_size ||
            should_flush_batch()) {

          size_t batch_size =
              std::min(pending_requests_.size(), config_.max_batch_size);
          batch.reserve(batch_size);

          for (size_t i = 0; i < batch_size && !pending_requests_.empty();
               ++i) {
            batch.push_back(std::move(pending_requests_.front()));
            pending_requests_.pop();
          }
        }
      }

      if (!batch.empty()) {
        process_batch(std::move(batch));
      }
    }

    // Process remaining requests on shutdown
    drain_remaining();
  }

  /**
   * Check if we should flush based on oldest request age
   */
  bool should_flush_batch() {
    if (pending_requests_.empty())
      return false;

    auto &oldest = pending_requests_.front();
    auto age = std::chrono::steady_clock::now() - oldest->enqueue_time;
    auto age_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(age).count();

    return age_ms >= static_cast<long>(config_.max_wait_ms);
  }

  /**
   * Process a batch of requests
   *
   * Note: For simplicity, this currently processes requests sequentially.
   * A production implementation would:
   * 1. Group requests by model
   * 2. Pad/reshape tensors to create true batched inputs
   * 3. Run single batched inference call
   * 4. Demultiplex outputs
   */
  void process_batch(std::vector<std::shared_ptr<PendingRequest>> batch) {
    LOG_DEBUG("Processing batch of {} requests", batch.size());

    auto batch_start = std::chrono::steady_clock::now();

    // Group by model name for efficient processing
    std::unordered_map<std::string,
                       std::vector<std::shared_ptr<PendingRequest>>>
        by_model;
    for (auto &req : batch) {
      by_model[req->request.model_name].push_back(std::move(req));
    }

    // Process each model group
    for (auto &[model_name, requests] : by_model) {
      // For now, process individually but in sequence
      // TODO: Implement true batched inference with tensor concatenation
      for (auto &pending : requests) {
        auto queue_time =
            std::chrono::steady_clock::now() - pending->enqueue_time;
        double queue_ms =
            std::chrono::duration<double, std::milli>(queue_time).count();

        try {
          auto response = model_registry_.run_inference(pending->request);
          response.queue_time_ms = queue_ms;
          pending->promise.set_value(std::move(response));
        } catch (const std::exception &e) {
          InferenceResponse error_response;
          error_response.success = false;
          error_response.error = e.what();
          error_response.queue_time_ms = queue_ms;
          pending->promise.set_value(std::move(error_response));
        }
      }
    }

    auto batch_duration = std::chrono::steady_clock::now() - batch_start;
    double batch_time_ms =
        std::chrono::duration<double, std::milli>(batch_duration).count();

    // Record batch metrics
    metrics_.record_batch(batch.size(), batch_time_ms / 1000.0);

    LOG_DEBUG("Batch of {} requests completed in {:.2f}ms", batch.size(),
              batch_time_ms);
  }

  /**
   * Drain and process remaining requests on shutdown
   */
  void drain_remaining() {
    std::vector<std::shared_ptr<PendingRequest>> remaining;

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      while (!pending_requests_.empty()) {
        remaining.push_back(std::move(pending_requests_.front()));
        pending_requests_.pop();
      }
    }

    if (!remaining.empty()) {
      LOG_INFO("Draining {} remaining requests", remaining.size());
      process_batch(std::move(remaining));
    }
  }
};

} // namespace onnx_server
