# Deployment Guide

This guide covers various deployment scenarios for the ONNX inference server.

## Quick Start

### Local Development

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run with local models
./build/onnx-server --models ./models --port 8080
```

### Docker Deployment

```bash
# Build image
docker build -t onnx-server:latest -f docker/Dockerfile.full .

# Run container
docker run -d \
  --name onnx-server \
  -p 8080:8080 \
  -v /path/to/models:/models \
  onnx-server:latest
```

---

## GPU Deployment

### NVIDIA GPU with CUDA

```bash
# Build CUDA image
docker build -t onnx-server:cuda -f docker/Dockerfile.cuda .

# Run with GPU access
docker run -d \
  --gpus all \
  --name onnx-server-gpu \
  -p 8080:8080 \
  -v /path/to/models:/models \
  onnx-server:cuda
```

### NVIDIA Jetson

```bash
# Cross-compile for Jetson
cmake -B build-jetson \
  -DCMAKE_TOOLCHAIN_FILE=cmake/jetson-toolchain.cmake \
  -DENABLE_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-jetson

# Copy to Jetson device
scp build-jetson/onnx-server jetson@device:/opt/onnx-server/
```

---

## Edge Deployment

### Raspberry Pi

```bash
# Cross-compile for ARM64
cmake -B build-arm \
  -DCMAKE_TOOLCHAIN_FILE=cmake/arm-toolchain.cmake \
  -DBUILD_STATIC=ON \
  -DENABLE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-arm

# Binary size should be ~15MB
ls -lh build-arm/onnx-server

# Copy to Pi
scp build-arm/onnx-server pi@raspberry:/opt/onnx-server/
```

### Docker Edge Image

```bash
# Build minimal edge image
docker build -t onnx-server:edge -f docker/Dockerfile.edge .

# Image size should be ~20-25MB
docker images onnx-server:edge
```

---

## Kubernetes Deployment

### Deployment Manifest

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: onnx-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: onnx-server
  template:
    metadata:
      labels:
        app: onnx-server
    spec:
      containers:
      - name: onnx-server
        image: onnx-server:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        volumeMounts:
        - name: models
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: onnx-server
spec:
  selector:
    app: onnx-server
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### GPU Deployment on Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: onnx-server-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: onnx-server-gpu
  template:
    spec:
      containers:
      - name: onnx-server
        image: onnx-server:cuda
        resources:
          limits:
            nvidia.com/gpu: 1
        # ... rest of config
```

---

## Configuration for Production

### Recommended Settings

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  threads: 8          # Match CPU cores

inference:
  providers: ["tensorrt", "cuda", "cpu"]
  memory_limit_mb: 8192
  intra_op_threads: 4
  inter_op_threads: 2
  graph_optimization: "all"

batching:
  enabled: true
  max_batch_size: 64
  max_wait_ms: 5      # Low latency
  adaptive_sizing: true

models:
  hot_reload: true    # Enable for A/B testing
  watch_interval_ms: 10000

metrics:
  enabled: true

logging:
  level: "info"
  format: "json"      # For log aggregation
```

### Environment Variables

```bash
# Server
export ONNX_SERVER_HOST=0.0.0.0
export ONNX_SERVER_PORT=8080
export ONNX_SERVER_THREADS=8

# Inference
export ONNX_GPU_DEVICE_ID=0
export ONNX_MEMORY_LIMIT_MB=8192

# Batching
export ONNX_BATCHING_ENABLED=true
export ONNX_MAX_BATCH_SIZE=64
export ONNX_MAX_WAIT_MS=5

# Models
export ONNX_MODELS_DIR=/models
export ONNX_HOT_RELOAD=true

# Logging
export ONNX_LOG_LEVEL=info
```

---

## Monitoring

### Prometheus Setup

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'onnx-server'
    static_configs:
      - targets: ['onnx-server:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Import the provided dashboard from `docs/grafana-dashboard.json` or create one with:

- Request rate panel: `rate(onnx_requests_total[5m])`
- Latency percentiles: `histogram_quantile(0.99, rate(onnx_inference_duration_seconds_bucket[5m]))`
- Model inference distribution: `sum by (model) (rate(onnx_model_inference_total[5m]))`

---

## Load Balancing

### nginx Configuration

```nginx
upstream onnx_servers {
    least_conn;
    server onnx-server-1:8080;
    server onnx-server-2:8080;
    server onnx-server-3:8080;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://onnx_servers;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_connect_timeout 5s;
        proxy_read_timeout 60s;
    }
    
    location /health {
        proxy_pass http://onnx_servers;
        proxy_connect_timeout 1s;
        proxy_read_timeout 1s;
    }
}
```

---

## Troubleshooting

### Common Issues

1. **Model not loading**
   - Check file permissions
   - Verify ONNX model validity with `onnx.checker.check_model()`
   - Check logs for specific error

2. **CUDA not detected**
   - Verify NVIDIA drivers: `nvidia-smi`
   - Check CUDA libraries: `ldconfig -p | grep cuda`
   - Ensure container has GPU access: `--gpus all`

3. **High latency**
   - Enable batching for throughput
   - Use TensorRT provider for NVIDIA GPUs
   - Check graph optimization level
   - Profile with ONNX Runtime Profiler

4. **Out of memory**
   - Reduce `max_batch_size`
   - Set `memory_limit_mb`
   - Use smaller model or quantization
