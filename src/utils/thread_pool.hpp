#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace onnx_server {

/**
 * A simple thread pool for async task execution
 */
class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
      : stop_(false) {

    if (num_threads == 0) {
      num_threads = 1;
    }

    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] { worker_loop(); });
    }
  }

  ~ThreadPool() { shutdown(); }

  // Non-copyable, non-movable
  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;
  ThreadPool(ThreadPool &&) = delete;
  ThreadPool &operator=(ThreadPool &&) = delete;

  /**
   * Submit a task and get a future for the result
   */
  template <typename F, typename... Args>
  auto submit(F &&f, Args &&...args)
      -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> result = task->get_future();

    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("ThreadPool has been stopped");
      }
      tasks_.emplace([task]() { (*task)(); });
    }

    condition_.notify_one();
    return result;
  }

  /**
   * Submit a task without getting a future (fire-and-forget)
   */
  template <typename F> void enqueue(F &&f) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (stop_) {
        throw std::runtime_error("ThreadPool has been stopped");
      }
      tasks_.emplace(std::forward<F>(f));
    }
    condition_.notify_one();
  }

  /**
   * Get the number of pending tasks
   */
  size_t pending() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return tasks_.size();
  }

  /**
   * Get the number of worker threads
   */
  size_t size() const { return workers_.size(); }

  /**
   * Gracefully shutdown the pool
   */
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (stop_)
        return;
      stop_ = true;
    }

    condition_.notify_all();

    for (std::thread &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;

      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

        if (stop_ && tasks_.empty()) {
          return;
        }

        task = std::move(tasks_.front());
        tasks_.pop();
      }

      task();
    }
  }

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  mutable std::mutex queue_mutex_;
  std::condition_variable condition_;
  std::atomic<bool> stop_;
};

} // namespace onnx_server
