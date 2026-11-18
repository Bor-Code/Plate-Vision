import json
from pathlib import Path

from plate_vision.config.schema import DataConfig, DatasetSplitRatios
from plate_vision.data.adapters import RealDatasetFormat, load_real_dataset
from plate_vision.data.deduplication import deduplicate_records
from plate_vision.data.exporters import export_recognizer_manifests, export_yolo_dataset
from plate_vision.data.splitter import stratified_split
from plate_vision.data.synthetic import encode_grayscale_png


def test_real_dataset_adapters_end_to_end_export(tmp_path: Path) -> None:
    ds_dir = tmp_path / "real_openalpr"
    ds_dir.mkdir(parents=True)
    for idx in range(6):
        img_path = ds_dir / f"cam_{idx:02d}.png"
        grid = [[100 + idx * 10 for _ in range(128)] for _ in range(64)]
        img_path.write_bytes(encode_grayscale_png(grid))
        sidecar = {
            "plates": [
                {
                    "plate_text": f"34ABC12{idx}",
                    "country_code": "TR",
                    "bbox": {"x_min": 12.0, "y_min": 12.0, "x_max": 110.0, "y_max": 44.0},
                }
            ]
        }
        (ds_dir / f"cam_{idx:02d}.json").write_text(json.dumps(sidecar), encoding="utf-8")

    records = load_real_dataset(
        ds_dir,
        RealDatasetFormat.OPENALPR_SIDECAR,
        config=DataConfig(),
    )
    dedup = deduplicate_records(records)
    assert dedup.removed_count == 0
    split_ds = stratified_split(dedup.unique_records, DatasetSplitRatios(), seed=42)

    out_dir = tmp_path / "exported"
    yolo_yaml = export_yolo_dataset(split_ds, out_dir / "yolo")
    manifests = export_recognizer_manifests(split_ds, out_dir / "recognizer")

    assert yolo_yaml.is_file()
    assert any(path.is_file() for path in manifests.values())
