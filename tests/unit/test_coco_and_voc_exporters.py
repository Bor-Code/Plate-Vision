from pathlib import Path

from plate_vision.config.schema import DataConfig
from plate_vision.data import (
    BoundingBox,
    CcpdFilenameAdapter,
    CocoJsonAdapter,
    DatasetSplit,
    PascalVocXmlAdapter,
    SplitDataset,
    encode_grayscale_png,
    export_coco_dataset,
    export_pascal_voc_dataset,
    ingest_manifest_records,
)


def test_export_coco_and_pascal_voc_round_trip(tmp_path: Path) -> None:
    ds_dir = tmp_path / "source"
    ds_dir.mkdir(parents=True, exist_ok=True)
    img_path = ds_dir / "sample_car.png"
    pixels = [[128 for _ in range(320)] for _ in range(240)]
    img_path.write_bytes(encode_grayscale_png(pixels))

    records = ingest_manifest_records(
        [
            {
                "record_id": "sample_car",
                "image_path": img_path,
                "source_dataset": "roundtrip",
                "annotations": [
                    {
                        "plate_text": "34ABC123",
                        "country_code": "TR",
                        "bbox": BoundingBox(x_min=40.0, y_min=80.0, x_max=200.0, y_max=130.0),
                        "corners": (
                            (40.0, 80.0),
                            (200.0, 80.0),
                            (200.0, 130.0),
                            (40.0, 130.0),
                        ),
                    }
                ],
            }
        ],
        config=DataConfig(min_image_width=16, min_image_height=16),
    )
    assert isinstance(CcpdFilenameAdapter(), object)
    split_ds = SplitDataset(train=tuple(records), val=(), test=())

    coco_out = tmp_path / "coco_export"
    coco_paths = export_coco_dataset(split_ds, coco_out)
    assert set(coco_paths.keys()) == {DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST}
    assert coco_paths[DatasetSplit.TRAIN].is_file()

    coco_adapter = CocoJsonAdapter(annotations_filename="coco_train.json")
    coco_reloaded = coco_adapter.load_records(
        coco_out,
        config=DataConfig(min_image_width=16, min_image_height=16),
    )
    assert len(coco_reloaded) == 1
    assert coco_reloaded[0].annotations[0].plate_text == "34ABC123"
    assert coco_reloaded[0].annotations[0].country_code == "TR"
    assert coco_reloaded[0].annotations[0].corners is not None

    voc_out = tmp_path / "voc_export"
    voc_paths = export_pascal_voc_dataset(split_ds, voc_out)
    assert len(voc_paths[DatasetSplit.TRAIN]) == 1
    assert voc_paths[DatasetSplit.TRAIN][0].is_file()

    (voc_out / "sample_car.png").write_bytes(img_path.read_bytes())
    voc_adapter = PascalVocXmlAdapter(annotations_dirname="train")
    voc_reloaded = voc_adapter.load_records(
        voc_out,
        config=DataConfig(min_image_width=16, min_image_height=16),
    )
    assert len(voc_reloaded) == 1
    assert voc_reloaded[0].annotations[0].plate_text == "34ABC123"
    assert voc_reloaded[0].annotations[0].bbox.x_max == 200.0
