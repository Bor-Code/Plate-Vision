import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.config.schema import RecognizerConfig
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.dataset import RecognizerDataset, SeededDataLoader
from plate_vision.recognition.decoding import ctc_beam_search_decode, log_softmax_row
from plate_vision.recognition.model import CRNNModel
from plate_vision.utils.exceptions import DataIOError, ModelLoadError


class RecognizerEpochMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int = Field(ge=1)
    train_loss: float = Field(ge=0.0)
    val_loss: float = Field(ge=0.0)
    exact_match_accuracy: float = Field(ge=0.0, le=1.0)
    character_error_rate: float = Field(ge=0.0)
    positional_accuracy: float = Field(ge=0.0, le=1.0)
    learning_rate: float = Field(gt=0.0)
    amp_scale: float = Field(gt=0.0)


class RecognizerCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int = Field(ge=1)
    best_exact_match: float = Field(ge=0.0, le=1.0)
    best_cer: float = Field(ge=0.0)
    metrics: RecognizerEpochMetrics
    amp_scale: float = Field(gt=0.0)
    charset: str = Field(min_length=1)
    model_state: dict[str, Any] = Field(default_factory=dict)


class RecognizerTrainingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epochs_completed: int = Field(ge=1)
    best_epoch: int = Field(ge=1)
    stopped_early: bool
    best_checkpoint_path: Path
    last_checkpoint_path: Path
    best_metrics: RecognizerEpochMetrics
    test_metrics: RecognizerEpochMetrics
    history: tuple[RecognizerEpochMetrics, ...]


class AmpGradScaler:
    def __init__(self, enabled: bool = True, initial_scale: float = 1024.0) -> None:
        self.enabled = enabled
        self._scale = initial_scale if enabled else 1.0

    @property
    def scale(self) -> float:
        return self._scale

    def update(self, grad_norm: float) -> None:
        if not self.enabled:
            return
        if not math.isfinite(grad_norm):
            self._scale = max(1.0, self._scale * 0.5)
        else:
            self._scale = min(65536.0, self._scale * 1.05)

    def load_scale(self, scale: float) -> None:
        self._scale = max(1.0, scale) if self.enabled else 1.0


def compute_levenshtein_distance(pred: str, target: str) -> int:
    if pred == target:
        return 0
    if not pred:
        return len(target)
    if not target:
        return len(pred)

    prev_row = list(range(len(target) + 1))
    for i, ch_p in enumerate(pred, start=1):
        curr_row = [i]
        for j, ch_t in enumerate(target, start=1):
            cost = 0 if ch_p == ch_t else 1
            curr_row.append(
                min(
                    curr_row[-1] + 1,
                    prev_row[j] + 1,
                    prev_row[j - 1] + cost,
                )
            )
        prev_row = curr_row
    return prev_row[-1]


def compute_recognizer_metrics(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    epoch: int = 1,
    train_loss: float = 0.0,
    val_loss: float = 0.0,
    learning_rate: float = 0.001,
    amp_scale: float = 1.0,
) -> RecognizerEpochMetrics:
    if not targets:
        return RecognizerEpochMetrics(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            exact_match_accuracy=0.0,
            character_error_rate=1.0,
            positional_accuracy=0.0,
            learning_rate=learning_rate,
            amp_scale=amp_scale,
        )

    exact_matches = 0
    total_edits = 0
    total_target_chars = 0
    pos_matches = 0
    pos_total = 0

    for pred, target in zip(predictions, targets, strict=True):
        norm_p = pred.strip().upper()
        norm_t = target.strip().upper()
        if norm_p == norm_t:
            exact_matches += 1
        total_edits += compute_levenshtein_distance(norm_p, norm_t)
        total_target_chars += max(1, len(norm_t))

        max_len = max(len(norm_p), len(norm_t))
        for idx in range(max_len):
            pos_total += 1
            if idx < len(norm_p) and idx < len(norm_t) and norm_p[idx] == norm_t[idx]:
                pos_matches += 1

    exact_acc = exact_matches / float(len(targets))
    cer = total_edits / float(max(1, total_target_chars))
    pos_acc = pos_matches / float(max(1, pos_total))

    return RecognizerEpochMetrics(
        epoch=epoch,
        train_loss=round(train_loss, 6),
        val_loss=round(val_loss, 6),
        exact_match_accuracy=round(exact_acc, 6),
        character_error_rate=round(cer, 6),
        positional_accuracy=round(pos_acc, 6),
        learning_rate=learning_rate,
        amp_scale=amp_scale,
    )


def compute_recognizer_lr(epoch: int, config: RecognizerConfig) -> float:
    if epoch < 1:
        msg = "epoch must be >= 1"
        raise ValueError(msg)
    if config.warmup_epochs > 0 and epoch <= config.warmup_epochs:
        factor = epoch / float(config.warmup_epochs)
        return config.learning_rate * (0.1 + 0.9 * factor)

    remaining = max(1, config.epochs - config.warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - config.warmup_epochs) / float(remaining)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = config.learning_rate * 0.01
    return min_lr + (config.learning_rate - min_lr) * cosine


def save_recognizer_checkpoint(checkpoint: RecognizerCheckpoint, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise DataIOError(
            f"Failed to write recognizer checkpoint: {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


def load_recognizer_checkpoint(path: Path) -> RecognizerCheckpoint:
    if not path.is_file():
        raise ModelLoadError(
            f"Recognizer checkpoint file does not exist: {path}",
            details={"path": str(path)},
        )
    try:
        raw = path.read_text(encoding="utf-8")
        return RecognizerCheckpoint.model_validate_json(raw)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ModelLoadError(
            f"Corrupted or invalid recognizer checkpoint: {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc


class RecognizerTrainer:
    def __init__(
        self,
        config: RecognizerConfig,
        encoder: CharsetEncoder | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.encoder = encoder if encoder is not None else CharsetEncoder()
        self.seed = seed
        self.model = CRNNModel(config=config, encoder=self.encoder, seed=seed)
        self.scaler = AmpGradScaler(enabled=config.mixed_precision)

    def evaluate_dataset(
        self,
        dataset: RecognizerDataset,
        *,
        epoch: int,
        train_loss: float,
        learning_rate: float,
    ) -> RecognizerEpochMetrics:
        predictions: list[str] = []
        targets: list[str] = []
        val_losses: list[float] = []

        for idx in range(len(dataset)):
            sample = dataset[idx]
            logits = self.model.forward(sample.pixels)
            log_probs = [log_softmax_row(row) for row in logits]
            step_loss = 0.0
            for pos, tok in enumerate(sample.target_indices):
                t_slot = min(len(log_probs) - 1, pos * 2)
                step_loss += -log_probs[t_slot][tok]
            val_losses.append(step_loss / max(1, len(sample.target_indices)))

            decoded = ctc_beam_search_decode(
                logits,
                self.encoder,
                beam_width=self.config.beam_width,
            )
            predictions.append(decoded.text)
            targets.append(sample.plate_text)

        mean_val_loss = sum(val_losses) / max(1, len(val_losses))
        return compute_recognizer_metrics(
            predictions,
            targets,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=mean_val_loss,
            learning_rate=learning_rate,
            amp_scale=self.scaler.scale,
        )

    def train(
        self,
        train_dataset: RecognizerDataset,
        val_dataset: RecognizerDataset,
        test_dataset: RecognizerDataset,
        output_dir: Path,
        resume_from: Path | None = None,
    ) -> RecognizerTrainingSummary:
        output_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = output_dir / "best.ckpt.json"
        last_ckpt_path = output_dir / "last.ckpt.json"

        start_epoch = 1
        best_score = -1e9
        best_epoch = 1
        best_metrics: RecognizerEpochMetrics | None = None
        history: list[RecognizerEpochMetrics] = []
        stagnant_epochs = 0
        stopped_early = False

        if resume_from is not None:
            resumed = load_recognizer_checkpoint(resume_from)
            self.model.load_state_dict(resumed.model_state)
            self.scaler.load_scale(resumed.amp_scale)
            start_epoch = resumed.epoch + 1
            best_score = resumed.best_exact_match - resumed.best_cer
            best_epoch = resumed.epoch
            best_metrics = resumed.metrics
            history.append(resumed.metrics)

        loader = SeededDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            seed=self.seed,
        )

        for epoch in range(start_epoch, self.config.epochs + 1):
            loader.set_epoch(epoch)
            lr = compute_recognizer_lr(epoch, self.config)
            batch_losses: list[float] = []

            for batch in loader:
                for sample in batch:
                    loss, grad_norm = self.model.compute_ctc_loss_and_update(
                        sample.pixels,
                        sample.target_indices,
                        learning_rate=lr,
                        grad_clip_norm=self.config.grad_clip_norm,
                    )
                    self.scaler.update(grad_norm)
                    batch_losses.append(loss)

            mean_train_loss = sum(batch_losses) / max(1, len(batch_losses))
            val_metrics = self.evaluate_dataset(
                val_dataset,
                epoch=epoch,
                train_loss=mean_train_loss,
                learning_rate=lr,
            )
            history.append(val_metrics)
            score = val_metrics.exact_match_accuracy - val_metrics.character_error_rate

            charset_str = "".join(self.encoder.characters)
            if score > best_score + 1e-6 or best_metrics is None:
                best_score = score
                best_epoch = epoch
                best_metrics = val_metrics
                stagnant_epochs = 0
                save_recognizer_checkpoint(
                    RecognizerCheckpoint(
                        epoch=epoch,
                        best_exact_match=val_metrics.exact_match_accuracy,
                        best_cer=val_metrics.character_error_rate,
                        metrics=val_metrics,
                        amp_scale=self.scaler.scale,
                        charset=charset_str,
                        model_state=self.model.state_dict(),
                    ),
                    best_ckpt_path,
                )
            else:
                stagnant_epochs += 1

            save_recognizer_checkpoint(
                RecognizerCheckpoint(
                    epoch=epoch,
                    best_exact_match=best_metrics.exact_match_accuracy,
                    best_cer=best_metrics.character_error_rate,
                    metrics=val_metrics,
                    amp_scale=self.scaler.scale,
                    charset=charset_str,
                    model_state=self.model.state_dict(),
                ),
                last_ckpt_path,
            )

            if stagnant_epochs >= self.config.early_stopping_patience:
                stopped_early = True
                break

        assert best_metrics is not None
        best_loaded = load_recognizer_checkpoint(best_ckpt_path)
        self.model.load_state_dict(best_loaded.model_state)
        test_metrics = self.evaluate_dataset(
            test_dataset,
            epoch=best_epoch,
            train_loss=best_metrics.train_loss,
            learning_rate=best_metrics.learning_rate,
        )

        return RecognizerTrainingSummary(
            epochs_completed=len(history),
            best_epoch=best_epoch,
            stopped_early=stopped_early,
            best_checkpoint_path=best_ckpt_path,
            last_checkpoint_path=last_ckpt_path,
            best_metrics=best_metrics,
            test_metrics=test_metrics,
            history=tuple(history),
        )


__all__ = [
    "AmpGradScaler",
    "RecognizerCheckpoint",
    "RecognizerEpochMetrics",
    "RecognizerTrainer",
    "RecognizerTrainingSummary",
    "compute_levenshtein_distance",
    "compute_recognizer_lr",
    "compute_recognizer_metrics",
    "load_recognizer_checkpoint",
    "save_recognizer_checkpoint",
]
