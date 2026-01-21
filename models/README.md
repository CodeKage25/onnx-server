# Add sample models here
#
# The ONNX server will automatically load all .onnx files from this directory.
# You can generate sample models using: python examples/create_sample_model.py
#
# Supported models:
# - Any valid ONNX model (opset 7-19)
# - Models with float32, float64, int32, int64 inputs/outputs
# - Static and dynamic input shapes
#
# Hot-reload:
# When hot_reload is enabled, modifying or replacing model files will
# automatically reload them without restarting the server.
