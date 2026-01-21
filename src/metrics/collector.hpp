#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/config.hpp"
#include "utils/logging.hpp"

namespace onnx_server {

/**
 * Histogram bucket for latency tracking
 * Uses pointer to atomic to allow vector storage
 */
struct HistogramBucket {
  double upper_bound;
  mutable std::unique_ptr<std::atomic<uint64_t>> count;

  HistogramBucket(double bound)
      : upper_bound(bound), count(std::make_unique<std::atomic<uint64_t>>(0)) {}
  HistogramBucket(HistogramBucket &&other) noexcept
      : upper_bound(other.upper_bound), count(std::move(other.count)) {}
  HistogramBucket &operator=(HistogramBucket &&other) noexcept {
    upper_bound = other.upper_bound;
    count = std::move(other.count);
    return *this;
  }
  HistogramBucket(const HistogramBucket &) = delete;
  HistogramBucket &operator=(const HistogramBucket &) = delete;
};

/**
 * Histogram for latency metrics with percentile support
 */
class Histogram {
public:
  explicit Histogram(const std::vector<double> &buckets = {0.001, 0.005, 0.01,
                                                           0.025, 0.05, 0.1,
                                                           0.25, 0.5, 1.0}) {
    buckets_.reserve(buckets.size() + 1);
    for (double bound : buckets) {
      buckets_.emplace_back(bound);
    }
    // Add +Inf bucket
    buckets_.emplace_back(std::numeric_limits<double>::infinity());
  }

  void observe(double value) {
    sum_.fetch_add(static_cast<uint64_t>(value * 1e9),
                   std::memory_order_relaxed);
    count_.fetch_add(1, std::memory_order_relaxed);

    for (auto &bucket : buckets_) {
      if (value <= bucket.upper_bound) {
        bucket.count->fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  uint64_t count() const { return count_.load(std::memory_order_relaxed); }
  double sum() const {
    return static_cast<double>(sum_.load(std::memory_order_relaxed)) / 1e9;
  }

  const std::vector<HistogramBucket> &buckets() const { return buckets_; }

private:
  std::vector<HistogramBucket> buckets_;
  std::atomic<uint64_t> count_{0};
  std::atomic<uint64_t> sum_{0}; // Stored as nanoseconds for atomic operations
};

/**
 * Counter metric
 */
class Counter {
public:
  void inc(uint64_t delta = 1) {
    value_.fetch_add(delta, std::memory_order_relaxed);
  }

  uint64_t value() const { return value_.load(std::memory_order_relaxed); }

private:
  std::atomic<uint64_t> value_{0};
};

/**
 * Gauge metric
 */
class Gauge {
public:
  void set(double value) { value_.store(value, std::memory_order_relaxed); }

  void inc(double delta = 1.0) {
    double old_val, new_val;
    do {
      old_val = value_.load(std::memory_order_relaxed);
      new_val = old_val + delta;
    } while (!value_.compare_exchange_weak(old_val, new_val));
  }

  void dec(double delta = 1.0) { inc(-delta); }

  double value() const { return value_.load(std::memory_order_relaxed); }

private:
  std::atomic<double> value_{0.0};
};

/**
 * Metrics Collector for Prometheus-compatible metrics
 */
class MetricsCollector {
public:
  explicit MetricsCollector(const MetricsConfig &config)
      : config_(config), request_latency_(config.latency_buckets),
        inference_latency_(config.latency_buckets),
        batch_latency_(config.latency_buckets),
        start_time_(std::chrono::steady_clock::now()) {}

  /**
   * Record an HTTP request
   */
  void record_request(const std::string &endpoint, const std::string &method,
                      int status, double latency_seconds) {
    std::string key = method + "_" + endpoint + "_" + std::to_string(status);

    {
      std::lock_guard<std::mutex> lock(mutex_);
      request_counts_[key].inc();
    }

    requests_total_.inc();
    request_latency_.observe(latency_seconds);

    if (status >= 400) {
      request_errors_.inc();
    }
  }

  /**
   * Record an inference operation
   */
  void record_inference(const std::string &model, double latency_seconds) {
    inference_total_.inc();
    inference_latency_.observe(latency_seconds);

    std::lock_guard<std::mutex> lock(mutex_);
    model_inference_counts_[model].inc();
  }

  /**
   * Record a batch execution
   */
  void record_batch(size_t batch_size, double latency_seconds) {
    batches_total_.inc();
    batch_latency_.observe(latency_seconds);

    std::lock_guard<std::mutex> lock(mutex_);
    batch_sizes_.push_back(batch_size);
    if (batch_sizes_.size() > 1000) {
      batch_sizes_.erase(batch_sizes_.begin());
    }
  }

  /**
   * Record model load time
   */
  void record_model_load(const std::string &model, double load_time_seconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_load_times_[model] = load_time_seconds;
  }

  /**
   * Set number of active sessions
   */
  void set_active_sessions(int count) { active_sessions_.set(count); }

  /**
   * Set loaded models count
   */
  void set_loaded_models(int count) { loaded_models_.set(count); }

  /**
   * Export metrics in Prometheus format
   */
  std::string export_prometheus() const {
    std::stringstream ss;

    // Server info
    auto uptime = std::chrono::steady_clock::now() - start_time_;
    auto uptime_seconds = std::chrono::duration<double>(uptime).count();

    ss << "# HELP onnx_server_uptime_seconds Time since server started\n";
    ss << "# TYPE onnx_server_uptime_seconds gauge\n";
    ss << "onnx_server_uptime_seconds " << uptime_seconds << "\n\n";

    // Request metrics
    ss << "# HELP onnx_requests_total Total number of HTTP requests\n";
    ss << "# TYPE onnx_requests_total counter\n";
    ss << "onnx_requests_total " << requests_total_.value() << "\n\n";

    ss << "# HELP onnx_request_errors_total Total number of HTTP error "
          "responses\n";
    ss << "# TYPE onnx_request_errors_total counter\n";
    ss << "onnx_request_errors_total " << request_errors_.value() << "\n\n";

    // Request latency histogram
    ss << "# HELP onnx_request_duration_seconds HTTP request latency\n";
    ss << "# TYPE onnx_request_duration_seconds histogram\n";
    export_histogram(ss, "onnx_request_duration_seconds", request_latency_);
    ss << "\n";

    // Inference metrics
    ss << "# HELP onnx_inference_total Total number of inference requests\n";
    ss << "# TYPE onnx_inference_total counter\n";
    ss << "onnx_inference_total " << inference_total_.value() << "\n\n";

    ss << "# HELP onnx_inference_duration_seconds Inference latency\n";
    ss << "# TYPE onnx_inference_duration_seconds histogram\n";
    export_histogram(ss, "onnx_inference_duration_seconds", inference_latency_);
    ss << "\n";

    // Per-model inference counts
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!model_inference_counts_.empty()) {
        ss << "# HELP onnx_model_inference_total Inference requests per "
              "model\n";
        ss << "# TYPE onnx_model_inference_total counter\n";
        for (const auto &[model, counter] : model_inference_counts_) {
          ss << "onnx_model_inference_total{model=\"" << model << "\"} "
             << counter.value() << "\n";
        }
        ss << "\n";
      }
    }

    // Batch metrics
    ss << "# HELP onnx_batches_total Total number of batch executions\n";
    ss << "# TYPE onnx_batches_total counter\n";
    ss << "onnx_batches_total " << batches_total_.value() << "\n\n";

    ss << "# HELP onnx_batch_duration_seconds Batch execution latency\n";
    ss << "# TYPE onnx_batch_duration_seconds histogram\n";
    export_histogram(ss, "onnx_batch_duration_seconds", batch_latency_);
    ss << "\n";

    // Average batch size
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!batch_sizes_.empty()) {
        double avg_batch = 0;
        for (auto size : batch_sizes_) {
          avg_batch += size;
        }
        avg_batch /= batch_sizes_.size();

        ss << "# HELP onnx_average_batch_size Average batch size\n";
        ss << "# TYPE onnx_average_batch_size gauge\n";
        ss << "onnx_average_batch_size " << avg_batch << "\n\n";
      }
    }

    // Gauges
    ss << "# HELP onnx_active_sessions Currently active inference sessions\n";
    ss << "# TYPE onnx_active_sessions gauge\n";
    ss << "onnx_active_sessions " << active_sessions_.value() << "\n\n";

    ss << "# HELP onnx_loaded_models Number of loaded models\n";
    ss << "# TYPE onnx_loaded_models gauge\n";
    ss << "onnx_loaded_models " << loaded_models_.value() << "\n";

    return ss.str();
  }

private:
  MetricsConfig config_;
  mutable std::mutex mutex_;

  // Counters
  Counter requests_total_;
  Counter request_errors_;
  Counter inference_total_;
  Counter batches_total_;

  // Histograms
  Histogram request_latency_;
  Histogram inference_latency_;
  Histogram batch_latency_;

  // Gauges
  Gauge active_sessions_;
  Gauge loaded_models_;

  // Per-model/endpoint metrics
  std::unordered_map<std::string, Counter> request_counts_;
  std::unordered_map<std::string, Counter> model_inference_counts_;
  std::unordered_map<std::string, double> model_load_times_;
  std::vector<size_t> batch_sizes_;

  std::chrono::steady_clock::time_point start_time_;

  void export_histogram(std::stringstream &ss, const std::string &name,
                        const Histogram &hist) const {
    for (const auto &bucket : hist.buckets()) {
      ss << name << "_bucket{le=\"";
      if (std::isinf(bucket.upper_bound)) {
        ss << "+Inf";
      } else {
        ss << bucket.upper_bound;
      }
      ss << "\"} " << bucket.count->load(std::memory_order_relaxed) << "\n";
    }
    ss << name << "_sum " << hist.sum() << "\n";
    ss << name << "_count " << hist.count() << "\n";
  }
};

} // namespace onnx_server
