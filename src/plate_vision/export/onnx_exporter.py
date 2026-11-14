import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.detection.trainer import load_detector_checkpoint
from plate_vision.recognition.trainer import load_recognizer_checkpoint
from plate_vision.utils.exceptions import DataIOError, ModelLoadError

ONNX_MAGIC_HEADER: Final[bytes] = b"ONNXPV01"


class OnnxModelMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_type: str = Field(min_length=1)
    opset_version: int = Field(ge=11, le=21)
    quantized: bool
    input_shape: tuple[int, ...]
    output_names: tuple[str, ...]
    sha256: str = Field(min_length=64, max_length=64)


class ParityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_type: str = Field(min_length=1)
    max_abs_diff: float = Field(ge=0.0)
    mean_abs_diff: float = Field(ge=0.0)
    tolerance: float = Field(gt=0.0)
    passed: bool


def _quantize_float_matrix(matrix: list[list[float]]) -> dict[str, Any]:
    flat = [abs(val) for row in matrix for val in row]
    max_abs = max(flat) if flat else 0.0
    scale = (max_abs / 127.0) if max_abs > 1e-12 else 1.0
    q_rows = [[max(-127, min(127, round(val / scale))) for val in row] for row in matrix]
    dequant_rows = [[round(q * scale, 8) for q in row] for row in q_rows]
    return {
        "scale": scale,
        "int8_data": q_rows,
        "float_data": dequant_rows,
    }


def _pack_onnx_payload(payload: dict[str, Any], output_path: Path) -> str:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        compressed = zlib.compress(raw_json, level=6)
        blob = ONNX_MAGIC_HEADER + struct.pack(">I", len(compressed)) + compressed
        output_path.write_bytes(blob)
        return hashlib.sha256(blob).hexdigest()
    except OSError as exc:
        raise DataIOError(
            f"Failed to write ONNX artifact to {output_path}",
            details={"path": str(output_path), "reason": str(exc)},
        ) from exc


def unpack_onnx_payload(onnx_path: Path) -> dict[str, Any]:
    if not onnx_path.is_file():
        raise ModelLoadError(
            f"ONNX model file does not exist: {onnx_path}",
            details={"path": str(onnx_path)},
        )
    try:
        blob = onnx_path.read_bytes()
    except OSError as exc:
        raise ModelLoadError(
            f"Failed to read ONNX model file: {onnx_path}",
            details={"path": str(onnx_path), "reason": str(exc)},
        ) from exc

    header_len = len(ONNX_MAGIC_HEADER)
    if len(blob) < header_len + 4 or blob[:header_len] != ONNX_MAGIC_HEADER:
        raise ModelLoadError(
            f"Invalid ONNX binary header in {onnx_path}",
            details={"path": str(onnx_path)},
        )
    payload_len = struct.unpack(">I", blob[header_len : header_len + 4])[0]
    compressed = blob[header_len + 4 : header_len + 4 + payload_len]
    if len(compressed) != payload_len:
        raise ModelLoadError(
            f"Truncated ONNX payload in {onnx_path}",
            details={"path": str(onnx_path)},
        )
    try:
        decompressed = zlib.decompress(compressed)
        parsed = json.loads(decompressed.decode("utf-8"))
    except (zlib.error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelLoadError(
            f"Corrupted ONNX graph payload in {onnx_path}",
            details={"path": str(onnx_path), "reason": str(exc)},
        ) from exc
    if not isinstance(parsed, dict):
        raise ModelLoadError(
            f"Invalid ONNX root structure in {onnx_path}",
            details={"path": str(onnx_path)},
        )
    return parsed


def export_detector_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset_version: int = 17,
    dynamic_quantize: bool = False,
) -> OnnxModelMetadata:
    ckpt = load_detector_checkpoint(checkpoint_path)
    input_shape = (1, 3, ckpt.input_size, ckpt.input_size)
    output_names = ("boxes", "scores", "class_ids")
    state_copy = dict(ckpt.state)
    if dynamic_quantize:
        raw_score_bias = float(state_copy.get("score_bias", 0.85))
        q_scale = max(abs(raw_score_bias), 1e-6) / 127.0
        q_val = max(-127, min(127, round(raw_score_bias / q_scale)))
        state_copy["score_bias"] = round(q_val * q_scale, 8)
        state_copy["quantized_int8"] = True

    payload: dict[str, Any] = {
        "model_type": "detector",
        "opset_version": opset_version,
        "quantized": dynamic_quantize,
        "input_shape": list(input_shape),
        "output_names": list(output_names),
        "model_architecture": ckpt.model_architecture,
        "input_size": ckpt.input_size,
        "state": state_copy,
    }
    digest = _pack_onnx_payload(payload, output_path)
    return OnnxModelMetadata(
        model_type="detector",
        opset_version=opset_version,
        quantized=dynamic_quantize,
        input_shape=input_shape,
        output_names=output_names,
        sha256=digest,
    )


def export_recognizer_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    opset_version: int = 17,
    dynamic_quantize: bool = False,
    image_height: int = 32,
    image_width: int = 128,
) -> OnnxModelMetadata:
    ckpt = load_recognizer_checkpoint(checkpoint_path)
    input_shape = (1, 1, image_height, image_width)
    output_names = ("ctc_logits",)
    model_state = dict(ckpt.model_state)

    if dynamic_quantize:
        for key in ("conv_weights", "recurrent_weights", "classifier_weights"):
            raw_matrix = model_state.get(key)
            if isinstance(raw_matrix, list):
                q_info = _quantize_float_matrix(raw_matrix)
                model_state[key] = q_info["float_data"]
                model_state[f"{key}_int8"] = q_info["int8_data"]
                model_state[f"{key}_scale"] = q_info["scale"]

    payload: dict[str, Any] = {
        "model_type": "recognizer",
        "opset_version": opset_version,
        "quantized": dynamic_quantize,
        "input_shape": list(input_shape),
        "output_names": list(output_names),
        "charset": ckpt.charset,
        "model_state": model_state,
    }
    digest = _pack_onnx_payload(payload, output_path)
    return OnnxModelMetadata(
        model_type="recognizer",
        opset_version=opset_version,
        quantized=dynamic_quantize,
        input_shape=input_shape,
        output_names=output_names,
        sha256=digest,
    )


def quantize_onnx_model(
    input_onnx_path: Path,
    output_onnx_path: Path,
) -> OnnxModelMetadata:
    payload = unpack_onnx_payload(input_onnx_path)
    model_type = str(payload.get("model_type", ""))
    opset_version = int(payload.get("opset_version", 17))
    input_shape = tuple(int(x) for x in payload.get("input_shape", [1, 1, 32, 128]))
    output_names = tuple(str(x) for x in payload.get("output_names", ["output"]))

    payload["quantized"] = True
    if model_type == "recognizer":
        model_state = dict(payload.get("model_state", {}))
        for key in ("conv_weights", "recurrent_weights", "classifier_weights"):
            raw_matrix = model_state.get(key)
            if isinstance(raw_matrix, list):
                q_info = _quantize_float_matrix(raw_matrix)
                model_state[key] = q_info["float_data"]
                model_state[f"{key}_int8"] = q_info["int8_data"]
                model_state[f"{key}_scale"] = q_info["scale"]
        payload["model_state"] = model_state
    elif model_type == "detector":
        state_copy = dict(payload.get("state", {}))
        raw_score_bias = float(state_copy.get("score_bias", 0.85))
        q_scale = max(abs(raw_score_bias), 1e-6) / 127.0
        q_val = max(-127, min(127, round(raw_score_bias / q_scale)))
        state_copy["score_bias"] = round(q_val * q_scale, 8)
        state_copy["quantized_int8"] = True
        payload["state"] = state_copy
    else:
        raise ModelLoadError(
            f"Unsupported ONNX model_type for quantization: {model_type}",
            details={"model_type": model_type},
        )

    digest = _pack_onnx_payload(payload, output_onnx_path)
    return OnnxModelMetadata(
        model_type=model_type,
        opset_version=opset_version,
        quantized=True,
        input_shape=input_shape,
        output_names=output_names,
        sha256=digest,
    )


__all__ = [
    "ONNX_MAGIC_HEADER",
    "OnnxModelMetadata",
    "ParityReport",
    "export_detector_to_onnx",
    "export_recognizer_to_onnx",
    "quantize_onnx_model",
    "unpack_onnx_payload",
]
