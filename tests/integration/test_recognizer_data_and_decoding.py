from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from plate_vision.config.schema import DataConfig, RecognizerConfig
from plate_vision.data import (
    AugmentationConfig,
    AugmentationPipeline,
    DatasetSplit,
    SyntheticPlateConfig,
    TurkeyCountryProfile,
    export_recognizer_manifests,
    generate_synthetic_dataset,
    stratified_split,
)
from plate_vision.recognition import (
    CharsetEncoder,
    CRNNModel,
    CRNNPlateRecognizer,
    RecognizerDataset,
    SeededDataLoader,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_synthetic_to_recognizer_dataset_dataloader_and_crnn(tmp_path: Path) -> None:
    records = generate_synthetic_dataset(
        count=10,
        output_dir=tmp_path / "synth",
        config=SyntheticPlateConfig(seed=42),
        profile=TurkeyCountryProfile(),
    )
    splits = stratified_split(records, DataConfig().split_ratios, seed=42)
    manifests = export_recognizer_manifests(splits, tmp_path / "manifests")

    encoder = CharsetEncoder()
    aug = AugmentationPipeline(AugmentationConfig(probability=0.5))
    train_dataset = RecognizerDataset(
        manifests[DatasetSplit.TRAIN],
        encoder=encoder,
        augmentation=aug,
        seed=42,
    )
    assert len(train_dataset) == len(splits.train)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        SeededDataLoader(train_dataset, batch_size=0)

    loader = SeededDataLoader(train_dataset, batch_size=2, shuffle=True, seed=42)
    loader.set_epoch(1)
    assert len(loader) == (len(train_dataset) + 1) // 2

    model = CRNNModel(config=RecognizerConfig(), encoder=encoder, seed=42)
    for batch in loader:
        for sample in batch:
            model.compute_ctc_loss_and_update(
                sample.pixels,
                sample.target_indices,
                learning_rate=0.02,
                grad_clip_norm=5.0,
            )

    eval_dataset = RecognizerDataset(manifests[DatasetSplit.TRAIN], encoder=encoder, seed=42)
    for idx in range(len(eval_dataset)):
        sample = eval_dataset[idx]
        model.compute_ctc_loss_and_update(
            sample.pixels,
            sample.target_indices,
            learning_rate=0.02,
            grad_clip_norm=5.0,
        )

    recognizer = CRNNPlateRecognizer(model=model, beam_width=3)
    first_sample = eval_dataset[0]
    decoded = recognizer.recognize(first_sample.pixels, use_beam_search=True)
    assert decoded.text == first_sample.plate_text
