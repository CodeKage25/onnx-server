#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/config.hpp"
#include "utils/logging.hpp"
#include <onnxruntime_cxx_api.h>

namespace onnx_server {

/**
 * Tensor data structures for inference
 */
struct TensorData {
  std::string name;
  std::string dtype = "float32";
  std::vector<int64_t> shape;
  std::vector<float> float_data;
  std::vector<int64_t> int_data;
  std::vector<uint8_t> raw_data;
};

/**
 * Inference request container
 */
struct InferenceRequest {
  std::string model_name;
  std::string request_id;
  std::vector<TensorData> inputs;

  // Timing metadata
  std::chrono::steady_clock::time_point enqueue_time;
};

/**
 * Inference response container
 */
struct InferenceResponse {
  std::vector<TensorData> outputs;
  double inference_time_ms = 0;
  double queue_time_ms = 0;
  std::string error;
  bool success = true;
};

/**
 * Model metadata
 */
struct ModelInfo {
  std::string name;
  std::string version = "1";
  std::string path;
  std::string loaded_at;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
  std::vector<std::string> input_types;
  std::vector<std::string> output_types;
};

/**
 * ONNX Runtime session manager
 * Handles session lifecycle, GPU/CPU provider selection, and inference
 * execution
 */
class SessionManager {
public:
  explicit SessionManager(const InferenceConfig &config)
      : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "onnx-server") {

    initialize_session_options();
  }

  ~SessionManager() = default;

  // Non-copyable
  SessionManager(const SessionManager &) = delete;
  SessionManager &operator=(const SessionManager &) = delete;

  /**
   * Load a model from file and create a session
   */
  std::pair<std::unique_ptr<Ort::Session>, ModelInfo>
  load_model(const std::string &path, const std::string &name) {
    LOG_INFO("Loading model: {} from {}", name, path);

    auto start = std::chrono::steady_clock::now();

    try {
      auto session =
          std::make_unique<Ort::Session>(env_, path.c_str(), session_options_);

      // Extract model info
      ModelInfo info;
      info.name = name;
      info.path = path;
      info.loaded_at = get_iso_timestamp();

      Ort::AllocatorWithDefaultOptions allocator;

      // Get input info
      size_t num_inputs = session->GetInputCount();
      for (size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session->GetInputNameAllocated(i, allocator);
        info.input_names.push_back(name_ptr.get());

        auto type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        info.input_shapes.push_back(tensor_info.GetShape());
        info.input_types.push_back(
            onnx_type_to_string(tensor_info.GetElementType()));
      }

      // Get output info
      size_t num_outputs = session->GetOutputCount();
      for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session->GetOutputNameAllocated(i, allocator);
        info.output_names.push_back(name_ptr.get());

        auto type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        info.output_shapes.push_back(tensor_info.GetShape());
        info.output_types.push_back(
            onnx_type_to_string(tensor_info.GetElementType()));
      }

      auto duration = std::chrono::steady_clock::now() - start;
      auto load_time_ms =
          std::chrono::duration<double, std::milli>(duration).count();

      LOG_INFO("Model {} loaded in {:.2f}ms with {} inputs and {} outputs",
               name, load_time_ms, num_inputs, num_outputs);

      return {std::move(session), info};

    } catch (const Ort::Exception &e) {
      LOG_ERROR("Failed to load model {}: {}", name, e.what());
      throw;
    }
  }

  /**
   * Run inference on a session
   */
  InferenceResponse run_inference(Ort::Session &session,
                                  const InferenceRequest &request,
                                  const ModelInfo &info) {
    InferenceResponse response;
    auto start = std::chrono::steady_clock::now();

    try {
      Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

      // Prepare input tensors
      std::vector<Ort::Value> input_tensors;
      std::vector<const char *> input_names;

      for (const auto &input : request.inputs) {
        input_names.push_back(input.name.c_str());

        // Create tensor from data
        if (!input.float_data.empty()) {
          auto tensor = Ort::Value::CreateTensor<float>(
              memory_info, const_cast<float *>(input.float_data.data()),
              input.float_data.size(), input.shape.data(), input.shape.size());
          input_tensors.push_back(std::move(tensor));
        } else if (!input.int_data.empty()) {
          auto tensor = Ort::Value::CreateTensor<int64_t>(
              memory_info, const_cast<int64_t *>(input.int_data.data()),
              input.int_data.size(), input.shape.data(), input.shape.size());
          input_tensors.push_back(std::move(tensor));
        }
      }

      // Prepare output names
      std::vector<const char *> output_names;
      for (const auto &name : info.output_names) {
        output_names.push_back(name.c_str());
      }

      // Run inference
      auto output_tensors = session.Run(
          Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
          input_tensors.size(), output_names.data(), output_names.size());

      // Extract outputs
      for (size_t i = 0; i < output_tensors.size(); ++i) {
        TensorData output;
        output.name = info.output_names[i];

        auto &tensor = output_tensors[i];
        auto type_info = tensor.GetTensorTypeAndShapeInfo();

        // Get shape
        output.shape = type_info.GetShape();

        // Get data based on type
        auto element_type = type_info.GetElementType();
        size_t element_count = type_info.GetElementCount();

        if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
          const float *data = tensor.GetTensorData<float>();
          output.float_data.assign(data, data + element_count);
          output.dtype = "float32";
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          const int64_t *data = tensor.GetTensorData<int64_t>();
          output.int_data.assign(data, data + element_count);
          output.dtype = "int64";
        } else if (element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
          const int32_t *data = tensor.GetTensorData<int32_t>();
          for (size_t j = 0; j < element_count; ++j) {
            output.int_data.push_back(data[j]);
          }
          output.dtype = "int32";
        }

        response.outputs.push_back(std::move(output));
      }

      response.success = true;

    } catch (const Ort::Exception &e) {
      response.success = false;
      response.error = e.what();
      LOG_ERROR("Inference error: {}", e.what());
    }

    auto end = std::chrono::steady_clock::now();
    response.inference_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return response;
  }

  /**
   * Get the ONNX Runtime environment
   */
  Ort::Env &env() { return env_; }

  /**
   * Get active providers list
   */
  std::vector<std::string> get_available_providers() const {
    return Ort::GetAvailableProviders();
  }

private:
  InferenceConfig config_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;

  void initialize_session_options() {
    // Graph optimization
    if (config_.graph_optimization == "all") {
      session_options_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_ALL);
    } else if (config_.graph_optimization == "extended") {
      session_options_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    } else if (config_.graph_optimization == "basic") {
      session_options_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_ENABLE_BASIC);
    } else {
      session_options_.SetGraphOptimizationLevel(
          GraphOptimizationLevel::ORT_DISABLE_ALL);
    }

    // Thread settings
    if (config_.intra_op_threads > 0) {
      session_options_.SetIntraOpNumThreads(config_.intra_op_threads);
    }
    if (config_.inter_op_threads > 0) {
      session_options_.SetInterOpNumThreads(config_.inter_op_threads);
    }

    // Add execution providers in priority order
    for (const auto &provider : config_.providers) {
      try {
        if (provider == "tensorrt") {
#ifdef ENABLE_TENSORRT
          OrtTensorRTProviderOptions trt_options{};
          trt_options.device_id = config_.gpu_device_id;
          trt_options.trt_fp16_enable = 1;
          session_options_.AppendExecutionProvider_TensorRT(trt_options);
          LOG_INFO("Added TensorRT execution provider");
#else
          LOG_DEBUG("TensorRT provider requested but not compiled in");
#endif
        } else if (provider == "cuda") {
#ifdef ENABLE_CUDA
          OrtCUDAProviderOptions cuda_options{};
          cuda_options.device_id = config_.gpu_device_id;
          cuda_options.arena_extend_strategy = 1;
          if (config_.memory_limit_mb > 0) {
            cuda_options.gpu_mem_limit = config_.memory_limit_mb * 1024 * 1024;
          }
          session_options_.AppendExecutionProvider_CUDA(cuda_options);
          LOG_INFO("Added CUDA execution provider");
#else
          LOG_DEBUG("CUDA provider requested but not compiled in");
#endif
        } else if (provider == "cpu") {
          // CPU is always available as fallback
          LOG_DEBUG("Using CPU execution provider");
        }
      } catch (const Ort::Exception &e) {
        LOG_WARN("Failed to add {} provider: {}", provider, e.what());
      }
    }
  }

  static std::string onnx_type_to_string(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      return "string";
    default:
      return "unknown";
    }
  }

  static std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
  }
};

} // namespace onnx_server
