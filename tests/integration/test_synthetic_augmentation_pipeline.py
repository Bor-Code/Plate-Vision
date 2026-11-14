import random
from pathlib import Path

from plate_vision.config.schema import DataConfig
from plate_vision.data import (
    AugmentationConfig,
    AugmentationPipeline,
    SyntheticPlateConfig,
    TurkeyCountryProfile,
    deduplicate_records,
    export_recognizer_manifests,
    generate_synthetic_dataset,
    inspect_image_bytes,
    stratified_split,
)


def test_synthetic_generation_to_split_and_manifest_export(tmp_path: Path) -> None:
    synth_dir = tmp_path / "synthetic"
    profile = TurkeyCountryProfile()
    synth_config = SyntheticPlateConfig(
        image_width=128,
        image_height=32,
        seed=101,
    )
    records = generate_synthetic_dataset(
        count=15,
        output_dir=synth_dir,
        config=synth_config,
        profile=profile,
    )
    assert len(records) == 15

    for record in records:
        assert profile.is_valid(record.annotations[0].plate_text)
        raw_bytes = record.image.path.read_bytes()
        w, h, mime = inspect_image_bytes(
            raw_bytes,
            max_bytes=1_048_576,
            min_width=32,
            min_height=16,
            source_path=str(record.image.path),
        )
        assert (w, h, mime) == (128, 32, "image/png")

    dedup = deduplicate_records(records)
    assert dedup.removed_count == 0

    data_cfg = DataConfig(seed=101)
    splits = stratified_split(dedup.unique_records, data_cfg.split_ratios, seed=data_cfg.seed)
    manifests = export_recognizer_manifests(splits, tmp_path / "manifests")
    assert len(manifests) == 3

    aug_pipeline = AugmentationPipeline(AugmentationConfig(probability=0.8))
    rng = random.Random(101)
    sample_grid = [[200 for _ in range(128)] for _ in range(32)]
    augmented = aug_pipeline.apply(sample_grid, rng)
    assert len(augmented) == 32
    assert len(augmented[0]) == 128
