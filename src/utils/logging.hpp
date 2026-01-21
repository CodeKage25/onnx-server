#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace onnx_server {

/**
 * Log levels
 */
enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

/**
 * Lightweight logger with JSON and text output formats
 */
class Logger {
public:
  static Logger &instance() {
    static Logger logger;
    return logger;
  }

  void set_level(LogLevel level) { level_ = level; }
  void set_level(const std::string &level) {
    if (level == "debug")
      level_ = LogLevel::DEBUG;
    else if (level == "info")
      level_ = LogLevel::INFO;
    else if (level == "warn")
      level_ = LogLevel::WARN;
    else if (level == "error")
      level_ = LogLevel::ERROR;
  }

  void set_json_format(bool json) { json_format_ = json; }

  template <typename... Args>
  void log(LogLevel level, const char *file, int line, const std::string &fmt,
           Args &&...args) {
    if (level < level_)
      return;

    std::lock_guard<std::mutex> lock(mutex_);

    std::string message = format_string(fmt, std::forward<Args>(args)...);

    if (json_format_) {
      output_json(level, file, line, message);
    } else {
      output_text(level, file, line, message);
    }
  }

private:
  Logger() = default;

  LogLevel level_ = LogLevel::INFO;
  bool json_format_ = false;
  std::mutex mutex_;

  static std::string level_string(LogLevel level) {
    switch (level) {
    case LogLevel::DEBUG:
      return "DEBUG";
    case LogLevel::INFO:
      return "INFO";
    case LogLevel::WARN:
      return "WARN";
    case LogLevel::ERROR:
      return "ERROR";
    default:
      return "UNKNOWN";
    }
  }

  static std::string level_color(LogLevel level) {
    switch (level) {
    case LogLevel::DEBUG:
      return "\033[36m"; // Cyan
    case LogLevel::INFO:
      return "\033[32m"; // Green
    case LogLevel::WARN:
      return "\033[33m"; // Yellow
    case LogLevel::ERROR:
      return "\033[31m"; // Red
    default:
      return "\033[0m";
    }
  }

  static std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
  }

  void output_text(LogLevel level, const char *file, int line,
                   const std::string &message) {
    std::cout << level_color(level) << "[" << timestamp() << "] "
              << "[" << level_string(level) << "] "
              << "\033[0m" << message << " (" << file << ":" << line << ")"
              << std::endl;
  }

  void output_json(LogLevel level, const char *file, int line,
                   const std::string &message) {
    std::cout << "{"
              << "\"timestamp\":\"" << timestamp() << "\","
              << "\"level\":\"" << level_string(level) << "\","
              << "\"message\":\"" << escape_json(message) << "\","
              << "\"file\":\"" << file << "\","
              << "\"line\":" << line << "}" << std::endl;
  }

  static std::string escape_json(const std::string &s) {
    std::string result;
    result.reserve(s.size());
    for (char c : s) {
      switch (c) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        result += c;
      }
    }
    return result;
  }

  // Simple string formatting (replacement for fmt::format)
  template <typename... Args>
  static std::string format_string(const std::string &fmt, Args &&...args) {
    if constexpr (sizeof...(args) == 0) {
      return fmt;
    } else {
      return format_impl(fmt, std::forward<Args>(args)...);
    }
  }

  template <typename T, typename... Rest>
  static std::string format_impl(const std::string &fmt, T &&first,
                                 Rest &&...rest) {
    std::string result = fmt;
    size_t pos = result.find("{}");
    if (pos != std::string::npos) {
      std::stringstream ss;
      ss << first;
      result.replace(pos, 2, ss.str());
    }
    if constexpr (sizeof...(rest) > 0) {
      return format_impl(result, std::forward<Rest>(rest)...);
    }
    return result;
  }
};

// Convenience macros
#define LOG_DEBUG(...)                                                         \
  onnx_server::Logger::instance().log(onnx_server::LogLevel::DEBUG, __FILE__,  \
                                      __LINE__, __VA_ARGS__)
#define LOG_INFO(...)                                                          \
  onnx_server::Logger::instance().log(onnx_server::LogLevel::INFO, __FILE__,   \
                                      __LINE__, __VA_ARGS__)
#define LOG_WARN(...)                                                          \
  onnx_server::Logger::instance().log(onnx_server::LogLevel::WARN, __FILE__,   \
                                      __LINE__, __VA_ARGS__)
#define LOG_ERROR(...)                                                         \
  onnx_server::Logger::instance().log(onnx_server::LogLevel::ERROR, __FILE__,  \
                                      __LINE__, __VA_ARGS__)

} // namespace onnx_server
