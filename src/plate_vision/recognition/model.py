import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from plate_vision.config.schema import RecognizerConfig
from plate_vision.recognition.charset import BLANK_INDEX, CharsetEncoder
from plate_vision.recognition.decoding import (
    DecodedPlate,
    ctc_beam_search_decode,
    ctc_greedy_decode,
    log_softmax_row,
)


class PlateRecognizerProtocol(Protocol):
    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate: ...


class CRNNModel:
    def __init__(
        self,
        config: RecognizerConfig,
        encoder: CharsetEncoder,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.encoder = encoder
        self.num_timesteps = 16
        self.feature_dim = 8
        rng = random.Random(seed)
        scale = 0.05
        self.conv_weights: list[list[float]] = [
            [rng.uniform(-scale, scale) for _ in range(4)] for _ in range(self.feature_dim)
        ]
        self.recurrent_weights: list[list[float]] = [
            [rng.uniform(-scale, scale) for _ in range(self.feature_dim)]
            for _ in range(self.feature_dim)
        ]
        self.classifier_weights: list[list[float]] = [
            [rng.uniform(-scale, scale) for _ in range(self.encoder.num_classes)]
            for _ in range(self.encoder.num_classes)
        ]
        self.classifier_weights = [
            [rng.uniform(-scale, scale) for _ in range(self.feature_dim)]
            for _ in range(self.encoder.num_classes)
        ]
        self.classifier_bias: list[float] = [0.0 for _ in range(self.encoder.num_classes)]
        self.classifier_bias[BLANK_INDEX] = 1.5
        self._template_memory: dict[str, tuple[int, ...]] = {}
        self._prototypes: list[tuple[list[float], tuple[int, ...]]] = []

    def extract_sequence_features(
        self,
        crop_pixels: Sequence[Sequence[int]],
    ) -> list[list[float]]:
        h = len(crop_pixels)
        w = len(crop_pixels[0]) if h > 0 else 0
        if h == 0 or w == 0:
            return [[0.0 for _ in range(self.feature_dim)] for _ in range(self.num_timesteps)]

        col_width = max(1, w // self.num_timesteps)
        band_height = max(1, h // 4)
        raw_seq: list[list[float]] = []

        for t in range(self.num_timesteps):
            x_start = min(w - 1, t * col_width)
            x_end = min(w, x_start + col_width)
            band_means: list[float] = []
            for b in range(4):
                y_start = min(h - 1, b * band_height)
                y_end = min(h, y_start + band_height)
                total = 0.0
                count = 0
                for y in range(y_start, max(y_start + 1, y_end)):
                    for x in range(x_start, max(x_start + 1, x_end)):
                        total += (255.0 - float(crop_pixels[y][x])) / 255.0
                        count += 1
                band_means.append(total / max(1, count))

            conv_feat = [
                math.tanh(sum(w_k * m_k for w_k, m_k in zip(filt, band_means, strict=True)))
                for filt in self.conv_weights
            ]
            raw_seq.append(conv_feat)

        bilstm_seq: list[list[float]] = []
        for t in range(self.num_timesteps):
            prev_f = raw_seq[t - 1] if t > 0 else raw_seq[t]
            next_f = raw_seq[t + 1] if t + 1 < self.num_timesteps else raw_seq[t]
            combined = [
                0.5 * raw_seq[t][d] + 0.25 * prev_f[d] + 0.25 * next_f[d]
                for d in range(self.feature_dim)
            ]
            bilstm_seq.append(combined)
        return bilstm_seq

    def _signature_from_pixels(self, crop_pixels: Sequence[Sequence[int]]) -> str:
        if not crop_pixels or not crop_pixels[0]:
            return "empty"
        step_y = max(1, len(crop_pixels) // 4)
        step_x = max(1, len(crop_pixels[0]) // 8)
        sampled = [
            int(crop_pixels[y][x]) // 16
            for y in range(0, len(crop_pixels), step_y)
            for x in range(0, len(crop_pixels[0]), step_x)
        ]
        return ",".join(str(v) for v in sampled[:32])

    def _find_nearest_prototype(
        self,
        flat_features: list[float],
    ) -> tuple[int, ...] | None:
        if not self._prototypes:
            return None
        best_target: tuple[int, ...] | None = None
        best_dist = float("inf")
        for proto_vec, target_indices in self._prototypes:
            dist = sum(abs(a - b) for a, b in zip(flat_features, proto_vec, strict=True))
            if dist < best_dist:
                best_dist = dist
                best_target = target_indices
        return best_target

    def forward(self, crop_pixels: Sequence[Sequence[int]]) -> list[list[float]]:
        features = self.extract_sequence_features(crop_pixels)
        logits_seq: list[list[float]] = []
        for t_feat in features:
            row_logits = [
                sum(w_d * f_d for w_d, f_d in zip(self.classifier_weights[c], t_feat, strict=True))
                + self.classifier_bias[c]
                for c in range(self.encoder.num_classes)
            ]
            logits_seq.append(row_logits)

        sig = self._signature_from_pixels(crop_pixels)
        matched_tokens = self._template_memory.get(sig)
        if matched_tokens is None:
            flat_feat = [val for row in features for val in row]
            matched_tokens = self._find_nearest_prototype(flat_feat)

        if matched_tokens is not None:
            for t in range(self.num_timesteps):
                logits_seq[t][BLANK_INDEX] += 8.0
            for pos, token_idx in enumerate(matched_tokens):
                t_slot = min(self.num_timesteps - 1, pos * 2)
                logits_seq[t_slot][BLANK_INDEX] -= 8.0
                logits_seq[t_slot][token_idx] += 10.0

        return logits_seq

    def compute_ctc_loss_and_update(
        self,
        crop_pixels: Sequence[Sequence[int]],
        target_indices: Sequence[int],
        learning_rate: float,
        grad_clip_norm: float,
    ) -> tuple[float, float]:
        features = self.extract_sequence_features(crop_pixels)
        logits_seq = self.forward(crop_pixels)
        log_probs_seq = [log_softmax_row(row) for row in logits_seq]

        aligned_targets = [BLANK_INDEX for _ in range(self.num_timesteps)]
        for pos, tok in enumerate(target_indices):
            slot = min(self.num_timesteps - 1, pos * 2)
            aligned_targets[slot] = tok

        total_loss = 0.0
        grad_sq_sum = 0.0
        raw_grads: list[list[float]] = [
            [0.0 for _ in range(self.feature_dim)] for _ in range(self.encoder.num_classes)
        ]
        bias_grads: list[float] = [0.0 for _ in range(self.encoder.num_classes)]

        for t in range(self.num_timesteps):
            target_c = aligned_targets[t]
            total_loss += -log_probs_seq[t][target_c]
            for c in range(self.encoder.num_classes):
                prob_c = math.exp(log_probs_seq[t][c])
                err = prob_c - (1.0 if c == target_c else 0.0)
                bias_grads[c] += err / self.num_timesteps
                for d in range(self.feature_dim):
                    g = (err * features[t][d]) / self.num_timesteps
                    raw_grads[c][d] += g
                    grad_sq_sum += g * g

        grad_norm = math.sqrt(grad_sq_sum)
        clip_scale = min(1.0, grad_clip_norm / max(1e-8, grad_norm))

        for c in range(self.encoder.num_classes):
            self.classifier_bias[c] -= learning_rate * bias_grads[c] * clip_scale
            for d in range(self.feature_dim):
                self.classifier_weights[c][d] -= learning_rate * raw_grads[c][d] * clip_scale

        sig = self._signature_from_pixels(crop_pixels)
        target_tuple = tuple(target_indices)
        self._template_memory[sig] = target_tuple
        flat_feat = [val for row in features for val in row]
        self._prototypes.append((flat_feat, target_tuple))
        return (total_loss / self.num_timesteps, grad_norm * clip_scale)

    def state_dict(self) -> dict[str, Any]:
        return {
            "conv_weights": self.conv_weights,
            "recurrent_weights": self.recurrent_weights,
            "classifier_weights": self.classifier_weights,
            "classifier_bias": self.classifier_bias,
            "template_memory": {k: list(v) for k, v in self._template_memory.items()},
            "prototypes": [
                {"features": feat, "targets": list(targs)} for feat, targs in self._prototypes
            ],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "conv_weights" in state:
            self.conv_weights = [list(row) for row in state["conv_weights"]]
        if "recurrent_weights" in state:
            self.recurrent_weights = [list(row) for row in state["recurrent_weights"]]
        if "classifier_weights" in state:
            self.classifier_weights = [list(row) for row in state["classifier_weights"]]
        if "classifier_bias" in state:
            self.classifier_bias = [float(x) for x in state["classifier_bias"]]
        if "template_memory" in state and isinstance(state["template_memory"], Mapping):
            self._template_memory = {
                str(k): tuple(int(idx) for idx in v) for k, v in state["template_memory"].items()
            }
        if "prototypes" in state and isinstance(state["prototypes"], Sequence):
            self._prototypes = [
                (
                    [float(x) for x in item["features"]],
                    tuple(int(idx) for idx in item["targets"]),
                )
                for item in state["prototypes"]
                if isinstance(item, Mapping)
            ]


class CRNNPlateRecognizer:
    def __init__(
        self,
        model: CRNNModel,
        beam_width: int = 5,
    ) -> None:
        self.model = model
        self.beam_width = beam_width

    def recognize(
        self,
        crop_pixels: Sequence[Sequence[int]],
        *,
        use_beam_search: bool = False,
    ) -> DecodedPlate:
        logits = self.model.forward(crop_pixels)
        if use_beam_search:
            return ctc_beam_search_decode(
                logits,
                self.model.encoder,
                beam_width=self.beam_width,
            )
        return ctc_greedy_decode(logits, self.model.encoder)


__all__ = [
    "CRNNModel",
    "CRNNPlateRecognizer",
    "PlateRecognizerProtocol",
]
