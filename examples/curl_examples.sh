#!/bin/bash
# ONNX Server curl Examples
# 
# These examples demonstrate how to interact with the ONNX inference server
# using curl commands.

BASE_URL="${ONNX_SERVER_URL:-http://localhost:8080}"

echo "=== ONNX Server curl Examples ==="
echo "Server: $BASE_URL"
echo ""

# Health check
echo "=== Health Check ==="
curl -s "$BASE_URL/health" | jq .
echo ""

# Readiness check
echo "=== Readiness Check ==="
curl -s "$BASE_URL/ready" | jq .
echo ""

# Server info
echo "=== Server Info ==="
curl -s "$BASE_URL/" | jq .
echo ""

# List models
echo "=== List Models ==="
curl -s "$BASE_URL/v1/models" | jq .
echo ""

# Get first model name (requires jq)
MODEL_NAME=$(curl -s "$BASE_URL/v1/models" | jq -r '.models[0].name // empty')

if [ -n "$MODEL_NAME" ]; then
    echo "=== Model Details: $MODEL_NAME ==="
    curl -s "$BASE_URL/v1/models/$MODEL_NAME" | jq .
    echo ""
    
    # Run inference (example with dummy data)
    echo "=== Run Inference ==="
    echo "Note: Adjust input shape and data for your specific model"
    
    curl -s -X POST "$BASE_URL/v1/models/$MODEL_NAME/infer" \
        -H "Content-Type: application/json" \
        -d '{
            "inputs": {
                "input": {
                    "shape": [1, 3, 224, 224],
                    "data": '"$(python3 -c 'import json; print(json.dumps([0.0] * (1*3*224*224)))')"'
                }
            }
        }' | jq .
    echo ""
    
    # Hot-reload model
    echo "=== Hot-Reload Model ==="
    curl -s -X POST "$BASE_URL/v1/models/$MODEL_NAME/reload" | jq .
    echo ""
else
    echo "No models loaded. Add .onnx files to the models directory."
fi

# Prometheus metrics
echo "=== Prometheus Metrics ==="
curl -s "$BASE_URL/metrics" | head -50
echo ""
echo "... (truncated)"
