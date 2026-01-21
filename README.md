# ONNX Inference Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)

A lightweight, high-performance inference server for deploying ONNX models via REST API. Designed for production workloads with GPU acceleration and edge deployment capabilities.

## Features

- 🚀 **High Performance**: Native C++ implementation with ONNX Runtime
- 📦 **Dynamic Batching**: Accumulates concurrent requests for GPU throughput
- 🔄 **Hot Reload**: Seamless model updates without downtime
- 🎮 **GPU Acceleration**: CUDA and TensorRT backends with CPU fallback
- 📊 **Prometheus Metrics**: Built-in monitoring with latency percentiles
- 🪶 **Edge Optimized**: ~15MB static binary for embedded devices

## Hardware Requirements

| Tier | CPU | RAM | Disk | GPU | Best For |
|------|-----|-----|------|-----|----------|
| **Minimum (Edge)** | 2 Cores (ARM64/x64) | 4GB | 1GB + Models | N/A | MobileNet, ResNet-18, Quantized Models |
| **Recommended** | 4+ Cores | 16GB | 5GB + Models | NVIDIA T4 / A10G | ResNet-50, BERT, SDXL (fp16) |
| **High Performance** | 8+ Cores | 32GB+ | NVMe SSD | NVIDIA A100 / H100 | LLMs (Llama, Mixtral), High Throughput |

> **Note**: Memory requirements are highly dependent on model size and batch size. Example: A loaded SDXL model requires ~16GB VRAM for decent batch sizes.

## Quick Start

### Using Docker

```bash
# Full-featured image with GPU support
docker run -p 8080:8080 -v /path/to/models:/models onnx-server:latest

# Edge-optimized image
docker run -p 8080:8080 -v /path/to/models:/models onnx-server:edge
```

### Building from Source

```bash
# Prerequisites: CMake 3.20+, C++17 compiler, ONNX Runtime

# Clone and build
git clone https://github.com/CodeKage25/onnx-server.git
cd onnx-server
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run
./build/onnx-server --config config.yaml
```

### Cross-Compile for Edge

```bash
# Raspberry Pi
cmake -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/arm-toolchain.cmake -DBUILD_STATIC=ON
cmake --build build-arm

# NVIDIA Jetson
cmake -B build-jetson -DCMAKE_TOOLCHAIN_FILE=cmake/jetson-toolchain.cmake -DENABLE_CUDA=ON
cmake --build build-jetson
```

## API Reference

### Inference

```bash
# Run inference on a model
curl -X POST http://localhost:8080/v1/models/resnet/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "input": {
        "shape": [1, 3, 224, 224],
        "data": [...]
      }
    }
  }'
```

### Model Management

```bash
# List models
curl http://localhost:8080/v1/models

# Get model info
curl http://localhost:8080/v1/models/resnet

# Hot-reload model
curl -X POST http://localhost:8080/v1/models/resnet/reload
```

### Health & Metrics

```bash
# Health check
curl http://localhost:8080/health

# Readiness check
curl http://localhost:8080/ready

# Prometheus metrics
curl http://localhost:8080/metrics
```

## Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  threads: 4

inference:
  providers: ["tensorrt", "cuda", "cpu"]
  gpu_device_id: 0
  memory_limit_mb: 4096

batching:
  enabled: true
  max_batch_size: 32
  max_wait_ms: 10

models:
  directory: "/models"
  hot_reload: true
  watch_interval_ms: 5000

metrics:
  enabled: true
  path: "/metrics"
```

## Performance

| Model | Batch Size | Latency (p99) | Throughput |
|-------|------------|---------------|------------|
| ResNet-50 | 1 | 2.3ms | 435 req/s |
| ResNet-50 | 32 | 18ms | 1,780 req/s |
| BERT-Base | 1 | 8.5ms | 118 req/s |
| BERT-Base | 16 | 42ms | 381 req/s |

*Benchmarks on NVIDIA A100, CUDA 12.1, TensorRT 8.6*

## Supported Platforms

- **Desktop/Server**: Linux (x86_64), Windows, macOS
- **Edge Devices**: Raspberry Pi 4/5, NVIDIA Jetson (Nano/Xavier/Orin), ARM64 SBCs
- **Cloud**: AWS, GCP, Azure (GPU instances)

## License

MIT License - see [LICENSE](LICENSE) for details.
