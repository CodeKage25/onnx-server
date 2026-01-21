#!/usr/bin/env python3
"""
Create a sample ONNX model for testing.

This script generates a simple ResNet-like model that can be used
to test the ONNX inference server.

Usage:
    pip install onnx numpy
    python create_sample_model.py
"""

import numpy as np

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
except ImportError:
    print("Please install onnx: pip install onnx")
    exit(1)


def create_simple_model():
    """Create a simple convolution + ReLU model."""
    
    # Input: [batch, channels, height, width]
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    
    # Output
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 64, 112, 112])
    
    # Create a convolution weight (64 filters, 3 input channels, 7x7 kernel)
    conv_weight = numpy_helper.from_array(
        np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01,
        name='conv.weight'
    )
    
    # Convolution node
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'conv.weight'],
        outputs=['conv_out'],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3],
        name='conv1'
    )
    
    # ReLU node
    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_out'],
        outputs=['output'],
        name='relu1'
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[conv_node, relu_node],
        name='simple_conv_relu',
        inputs=[X],
        outputs=[Y],
        initializer=[conv_weight]
    )
    
    # Create model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 13)]
    )
    
    # Validate
    onnx.checker.check_model(model)
    
    return model


def create_classification_model():
    """Create a simple classification model (conv + global avg pool + fc)."""
    
    # Input
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    
    # Output (1000 classes like ImageNet)
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1000])
    
    # Weights
    conv_weight = numpy_helper.from_array(
        np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01,
        name='conv.weight'
    )
    
    fc_weight = numpy_helper.from_array(
        np.random.randn(1000, 64).astype(np.float32) * 0.01,
        name='fc.weight'
    )
    
    fc_bias = numpy_helper.from_array(
        np.zeros(1000).astype(np.float32),
        name='fc.bias'
    )
    
    # Nodes
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'conv.weight'],
        outputs=['conv_out'],
        kernel_shape=[7, 7],
        strides=[2, 2],
        pads=[3, 3, 3, 3]
    )
    
    relu_node = helper.make_node(
        'Relu',
        inputs=['conv_out'],
        outputs=['relu_out']
    )
    
    gap_node = helper.make_node(
        'GlobalAveragePool',
        inputs=['relu_out'],
        outputs=['gap_out']
    )
    
    flatten_node = helper.make_node(
        'Flatten',
        inputs=['gap_out'],
        outputs=['flat_out'],
        axis=1
    )
    
    fc_node = helper.make_node(
        'Gemm',
        inputs=['flat_out', 'fc.weight', 'fc.bias'],
        outputs=['fc_out'],
        alpha=1.0,
        beta=1.0,
        transB=1
    )
    
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['fc_out'],
        outputs=['output'],
        axis=1
    )
    
    # Create graph
    graph = helper.make_graph(
        nodes=[conv_node, relu_node, gap_node, flatten_node, fc_node, softmax_node],
        name='simple_classifier',
        inputs=[X],
        outputs=[Y],
        initializer=[conv_weight, fc_weight, fc_bias]
    )
    
    # Create model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid('', 13)]
    )
    
    onnx.checker.check_model(model)
    
    return model


def main():
    import os
    
    # Create models directory
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create simple model
    print("Creating simple_conv.onnx...")
    simple_model = create_simple_model()
    onnx.save(simple_model, os.path.join(models_dir, "simple_conv.onnx"))
    print(f"  Saved to {models_dir}/simple_conv.onnx")
    
    # Create classification model
    print("Creating classifier.onnx...")
    classifier_model = create_classification_model()
    onnx.save(classifier_model, os.path.join(models_dir, "classifier.onnx"))
    print(f"  Saved to {models_dir}/classifier.onnx")
    
    print("\nDone! Models are ready for testing.")


if __name__ == "__main__":
    main()
