# ONNX Inference Server API Documentation

## Overview

The ONNX inference server provides a REST API for deploying and running ONNX models. All responses are in JSON format unless otherwise specified.

## Base URL

```
http://localhost:8080
```

## Authentication

Currently, the server does not implement authentication. For production deployments, consider placing the server behind a reverse proxy with authentication.

---

## Health & Status Endpoints

### Health Check

Check if the server is alive (liveness probe).

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Readiness Check

Check if the server is ready to accept requests (readiness probe).

```http
GET /ready
```

**Response (Ready):**
```json
{
  "status": "ready",
  "models_loaded": 3,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response (Not Ready):**
```json
{
  "status": "not_ready",
  "models_loaded": 0,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200` - Server is ready
- `503` - Server is not ready

### Server Info

Get general server information.

```http
GET /
```

**Response:**
```json
{
  "name": "onnx-server",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "models_loaded": 3,
  "batching_enabled": true,
  "providers": ["tensorrt", "cuda", "cpu"]
}
```

---

## Model Management Endpoints

### List Models

Get a list of all loaded models.

```http
GET /v1/models
```

**Response:**
```json
{
  "models": [
    {
      "name": "resnet50",
      "version": "1",
      "path": "/models/resnet50.onnx",
      "loaded_at": "2024-01-15T10:00:00Z",
      "input_names": ["input"],
      "output_names": ["output"]
    },
    {
      "name": "bert-base",
      "version": "1",
      "path": "/models/bert-base.onnx",
      "loaded_at": "2024-01-15T10:00:00Z",
      "input_names": ["input_ids", "attention_mask"],
      "output_names": ["logits"]
    }
  ]
}
```

### Get Model Details

Get detailed information about a specific model.

```http
GET /v1/models/{model_name}
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_name` | string | Name of the model (filename without .onnx extension) |

**Response:**
```json
{
  "name": "resnet50",
  "version": "1",
  "path": "/models/resnet50.onnx",
  "loaded_at": "2024-01-15T10:00:00Z",
  "inputs": [
    {
      "name": "input",
      "shape": [1, 3, 224, 224],
      "dtype": "float32"
    }
  ],
  "outputs": [
    {
      "name": "output",
      "shape": [1, 1000],
      "dtype": "float32"
    }
  ]
}
```

**Status Codes:**
- `200` - Success
- `404` - Model not found

### Reload Model

Hot-reload a model from disk without restarting the server.

```http
POST /v1/models/{model_name}/reload
```

**Response:**
```json
{
  "status": "reloaded",
  "model": "resnet50",
  "timestamp": "2024-01-15T10:35:00Z"
}
```

**Status Codes:**
- `200` - Model reloaded successfully
- `404` - Model not found
- `500` - Reload failed

---

## Inference Endpoint

### Run Inference

Execute inference on a model.

```http
POST /v1/models/{model_name}/infer
```

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "inputs": {
    "input_name": {
      "shape": [1, 3, 224, 224],
      "data": [0.1, 0.2, 0.3, ...],
      "dtype": "float32"
    }
  }
}
```

**Input Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `shape` | array[int] | Yes | Tensor dimensions |
| `data` | array[number] | Yes | Flattened tensor data |
| `dtype` | string | No | Data type (default: "float32") |

**Supported dtypes:**
- `float32` (default)
- `float64`
- `int32`
- `int64`
- `int8`
- `uint8`

**Response:**
```json
{
  "model_name": "resnet50",
  "outputs": {
    "output": {
      "shape": [1, 1000],
      "data": [0.001, 0.002, 0.95, ...]
    }
  },
  "timing": {
    "inference_ms": 2.5,
    "queue_ms": 0.3
  }
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid request body
- `404` - Model not found
- `500` - Inference failed

**Example with curl:**
```bash
curl -X POST http://localhost:8080/v1/models/resnet50/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "input": {
        "shape": [1, 3, 224, 224],
        "data": [0.0, 0.1, 0.2, ...]
      }
    }
  }'
```

---

## Metrics Endpoint

### Prometheus Metrics

Export metrics in Prometheus format.

```http
GET /metrics
```

**Response (text/plain):**
```
# HELP onnx_server_uptime_seconds Time since server started
# TYPE onnx_server_uptime_seconds gauge
onnx_server_uptime_seconds 3600.5

# HELP onnx_requests_total Total number of HTTP requests
# TYPE onnx_requests_total counter
onnx_requests_total 15243

# HELP onnx_inference_duration_seconds Inference latency
# TYPE onnx_inference_duration_seconds histogram
onnx_inference_duration_seconds_bucket{le="0.001"} 100
onnx_inference_duration_seconds_bucket{le="0.005"} 850
onnx_inference_duration_seconds_bucket{le="0.01"} 1200
onnx_inference_duration_seconds_bucket{le="+Inf"} 1500
onnx_inference_duration_seconds_sum 12.5
onnx_inference_duration_seconds_count 1500

# HELP onnx_loaded_models Number of loaded models
# TYPE onnx_loaded_models gauge
onnx_loaded_models 3
```

**Available Metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `onnx_server_uptime_seconds` | gauge | Server uptime |
| `onnx_requests_total` | counter | Total HTTP requests |
| `onnx_request_errors_total` | counter | HTTP error responses |
| `onnx_request_duration_seconds` | histogram | HTTP request latency |
| `onnx_inference_total` | counter | Total inference requests |
| `onnx_inference_duration_seconds` | histogram | Inference latency |
| `onnx_model_inference_total` | counter | Inference per model |
| `onnx_batches_total` | counter | Batch executions |
| `onnx_batch_duration_seconds` | histogram | Batch latency |
| `onnx_average_batch_size` | gauge | Average batch size |
| `onnx_active_sessions` | gauge | Active sessions |
| `onnx_loaded_models` | gauge | Loaded models count |

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": 404,
    "message": "Model not found: unknown_model",
    "detail": "Optional additional details"
  }
}
```

**Common Error Codes:**
| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid JSON or missing fields |
| 404 | Not Found - Model doesn't exist |
| 422 | Unprocessable Entity - Valid JSON but invalid data |
| 500 | Internal Server Error - Inference or server failure |
| 503 | Service Unavailable - Server not ready |

---

## Rate Limiting

The server does not implement rate limiting. For production use, consider:
- Using a reverse proxy (nginx, envoy) with rate limiting
- Implementing client-side throttling
- Using the batching feature to handle high concurrency

---

## WebSocket (Future)

WebSocket support for streaming inference is planned for a future release.
