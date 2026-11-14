from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from plate_vision.config.schema import RecognizerConfig
from plate_vision.recognition import (
    AmpGradScaler,
    RecognizerCheckpoint,
    RecognizerEpochMetrics,
    compute_levenshtein_distance,
    compute_recognizer_lr,
    compute_recognizer_metrics,
    load_recognizer_checkpoint,
    save_recognizer_checkpoint,
)
from plate_vision.utils.exceptions import DataIOError, ModelLoadError


def test_levenshtein_and_recognizer_metrics() -> None:
    assert compute_levenshtein_distance("34ABC123", "34ABC123") == 0
    assert compute_levenshtein_distance("", "34") == 2
    assert compute_levenshtein_distance("34", "") == 2
    assert compute_levenshtein_distance("34ABC123", "34ABD123") == 1

    empty_m = compute_recognizer_metrics([], [])
    assert empty_m.exact_match_accuracy == 0.0
    assert empty_m.character_error_rate == 1.0

    m = compute_recognizer_metrics(["34ABC123", "06DEF456"], ["34ABC123", "06DEF459"])
    assert m.exact_match_accuracy == 0.5
    assert m.character_error_rate == round(1 / 16, 6)
    assert m.positional_accuracy == round(15 / 16, 6)


def test_amp_scaler_and_lr_schedule() -> None:
    scaler = AmpGradScaler(enabled=True, initial_scale=512.0)
    assert scaler.scale == 512.0
    scaler.update(1.5)
    assert scaler.scale > 512.0
    scaler.update(math.nan)
    assert scaler.scale < 512.0
    scaler.load_scale(256.0)
    assert scaler.scale == 256.0

    disabled = AmpGradScaler(enabled=False)
    disabled.update(math.nan)
    assert disabled.scale == 1.0

    cfg = RecognizerConfig(epochs=10, warmup_epochs=2, learning_rate=0.002)
    with pytest.raises(ValueError, match="epoch must be >= 1"):
        compute_recognizer_lr(0, cfg)
    assert compute_recognizer_lr(1, cfg) < compute_recognizer_lr(2, cfg) == 0.002
    assert compute_recognizer_lr(10, cfg) < compute_recognizer_lr(2, cfg)


def test_recognizer_checkpoint_io_and_errors(tmp_path: Path) -> None:
    metrics = RecognizerEpochMetrics(
        epoch=1,
        train_loss=0.5,
        val_loss=0.4,
        exact_match_accuracy=0.9,
        character_error_rate=0.05,
        positional_accuracy=0.95,
        learning_rate=0.001,
        amp_scale=1024.0,
    )
    ckpt = RecognizerCheckpoint(
        epoch=1,
        best_exact_match=0.9,
        best_cer=0.05,
        metrics=metrics,
        amp_scale=1024.0,
        charset="0123456789AB",
        model_state={"classifier_bias": [0.0, 1.0]},
    )
    path = tmp_path / "rec.ckpt.json"
    save_recognizer_checkpoint(ckpt, path)
    assert load_recognizer_checkpoint(path) == ckpt

    with pytest.raises(ModelLoadError, match="does not exist"):
        load_recognizer_checkpoint(tmp_path / "missing.json")

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{bad_json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="Corrupted or invalid"):
        load_recognizer_checkpoint(corrupt)

    with (
        patch.object(Path, "write_text", side_effect=OSError("disk error")),
        pytest.raises(DataIOError, match="Failed to write recognizer checkpoint"),
    ):
        save_recognizer_checkpoint(ckpt, tmp_path / "fail.json")
