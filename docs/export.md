# ONNX Export, Parity Verification, and Dynamic Quantization

The `plate_vision.export` package converts trained detector and recognizer checkpoints into ONNX artifacts for production serving, validates numerical parity between training checkpoints and ONNX Runtime inference, and supports dynamic int8 weight quantization.

## Public APIs

- `export_detector_to_onnx(checkpoint_path, output_path, *, opset_version=17, dynamic_quantize=False) -> OnnxModelMetadata`
- `export_recognizer_to_onnx(checkpoint_path, output_path, *, opset_version=17, dynamic_quantize=False) -> OnnxModelMetadata`
- `quantize_onnx_model(input_onnx_path, output_onnx_path) -> OnnxModelMetadata`
- `OnnxDetectorRuntime`: Implements `PlateDetectorProtocol` backed by an exported detector ONNX artifact.
- `OnnxRecognizerRuntime`: Implements `PlateRecognizerProtocol` backed by an exported recognizer ONNX artifact.
- `verify_detector_parity`: Measures `max_abs_diff` and `mean_abs_diff` between checkpoint predictions and `OnnxDetectorRuntime`.
- `verify_recognizer_parity`: Measures `max_abs_diff` and `mean_abs_diff` between `CRNNModel` forward logits and `OnnxRecognizerRuntime`.
