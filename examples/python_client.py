#!/usr/bin/env python3
"""
ONNX Server Python Client Example

This script demonstrates how to interact with the ONNX inference server
using Python's requests library.

Usage:
    pip install requests numpy
    python python_client.py
"""

import requests
import numpy as np
import json
import time
from typing import Dict, List, Any, Optional

class ONNXServerClient:
    """Client for interacting with the ONNX inference server."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health(self) -> Dict[str, Any]:
        """Check server health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def ready(self) -> Dict[str, Any]:
        """Check server readiness."""
        response = self.session.get(f"{self.base_url}/ready")
        return response.json()
    
    def info(self) -> Dict[str, Any]:
        """Get server information."""
        response = self.session.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json().get("models", [])
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get model details."""
        response = self.session.get(f"{self.base_url}/v1/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def reload_model(self, model_name: str) -> Dict[str, Any]:
        """Hot-reload a model."""
        response = self.session.post(f"{self.base_url}/v1/models/{model_name}/reload")
        response.raise_for_status()
        return response.json()
    
    def infer(
        self, 
        model_name: str, 
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on a model.
        
        Args:
            model_name: Name of the model
            inputs: Dictionary mapping input names to numpy arrays
            
        Returns:
            Dictionary mapping output names to numpy arrays
        """
        # Convert numpy arrays to JSON-serializable format
        payload = {
            "inputs": {}
        }
        
        for name, array in inputs.items():
            payload["inputs"][name] = {
                "shape": list(array.shape),
                "data": array.flatten().tolist(),
                "dtype": str(array.dtype)
            }
        
        response = self.session.post(
            f"{self.base_url}/v1/models/{model_name}/infer",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Convert outputs back to numpy arrays
        outputs = {}
        for name, tensor in result.get("outputs", {}).items():
            shape = tensor.get("shape", [])
            data = tensor.get("data", [])
            outputs[name] = np.array(data).reshape(shape)
        
        return outputs
    
    def metrics(self) -> str:
        """Get Prometheus metrics."""
        response = self.session.get(f"{self.base_url}/metrics")
        response.raise_for_status()
        return response.text


def benchmark_inference(
    client: ONNXServerClient,
    model_name: str,
    input_shape: tuple,
    num_requests: int = 100
) -> Dict[str, float]:
    """Run a simple benchmark."""
    
    # Generate random input
    inputs = {"input": np.random.randn(*input_shape).astype(np.float32)}
    
    latencies = []
    
    print(f"Running {num_requests} inference requests...")
    
    for i in range(num_requests):
        start = time.time()
        try:
            _ = client.infer(model_name, inputs)
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)
        except Exception as e:
            print(f"Request {i} failed: {e}")
    
    if not latencies:
        return {}
    
    latencies.sort()
    
    return {
        "total_requests": num_requests,
        "successful_requests": len(latencies),
        "avg_latency_ms": np.mean(latencies),
        "p50_latency_ms": latencies[int(len(latencies) * 0.5)],
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)],
        "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
        "throughput_rps": len(latencies) / (sum(latencies) / 1000)
    }


def main():
    """Example usage of the ONNX Server client."""
    
    client = ONNXServerClient("http://localhost:8080")
    
    # Check server health
    print("=== Server Health ===")
    try:
        health = client.health()
        print(f"Status: {health.get('status')}")
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to server. Is it running?")
        return
    
    # Get server info
    print("\n=== Server Info ===")
    info = client.info()
    print(f"Version: {info.get('version')}")
    print(f"Uptime: {info.get('uptime_seconds')}s")
    print(f"Models loaded: {info.get('models_loaded')}")
    
    # List models
    print("\n=== Loaded Models ===")
    models = client.list_models()
    if not models:
        print("No models loaded. Add .onnx files to the models directory.")
        return
    
    for model in models:
        print(f"  - {model.get('name')}: {len(model.get('input_names', []))} inputs, {len(model.get('output_names', []))} outputs")
    
    # Get first model details
    model_name = models[0]["name"]
    print(f"\n=== Model Details: {model_name} ===")
    details = client.get_model(model_name)
    
    print("Inputs:")
    for inp in details.get("inputs", []):
        print(f"  - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
    
    print("Outputs:")
    for out in details.get("outputs", []):
        print(f"  - {out['name']}: shape={out['shape']}, dtype={out['dtype']}")
    
    # Run inference example
    print(f"\n=== Running Inference ===")
    
    # Determine input shape from model
    input_info = details.get("inputs", [{}])[0]
    input_name = input_info.get("name", "input")
    input_shape = input_info.get("shape", [1, 3, 224, 224])
    
    # Replace dynamic dims (-1) with 1
    input_shape = [1 if d < 0 else d for d in input_shape]
    
    # Create random input
    test_input = np.random.randn(*input_shape).astype(np.float32)
    
    print(f"Input shape: {test_input.shape}")
    
    start = time.time()
    outputs = client.infer(model_name, {input_name: test_input})
    latency = (time.time() - start) * 1000
    
    print(f"Inference completed in {latency:.2f}ms")
    
    for name, data in outputs.items():
        print(f"Output '{name}': shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
    
    # Run benchmark
    print(f"\n=== Benchmark ===")
    results = benchmark_inference(client, model_name, tuple(input_shape), num_requests=50)
    
    print(f"Throughput: {results.get('throughput_rps', 0):.1f} req/s")
    print(f"Avg latency: {results.get('avg_latency_ms', 0):.2f}ms")
    print(f"P50 latency: {results.get('p50_latency_ms', 0):.2f}ms")
    print(f"P95 latency: {results.get('p95_latency_ms', 0):.2f}ms")
    print(f"P99 latency: {results.get('p99_latency_ms', 0):.2f}ms")


if __name__ == "__main__":
    main()
