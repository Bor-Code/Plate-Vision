from pathlib import Path

import pytest

from plate_vision.config.schema import DataConfig
from plate_vision.data import (
    CcpdFilenameAdapter,
    PascalVocXmlAdapter,
    RealDatasetFormat,
    encode_grayscale_png,
    load_real_dataset,
    parse_ccpd_filename,
)
from plate_vision.utils.exceptions import DatasetValidationError


def _write_png(path: Path, width: int = 800, height: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = [[128 for _ in range(width)] for _ in range(height)]
    path.write_bytes(encode_grayscale_png(pixels))


def test_parse_ccpd_filename_extracts_plate_bbox_and_corners() -> None:
    parsed = parse_ccpd_filename(
        "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )
    assert parsed["plate_text"] == "AHAY339S"
    assert parsed["bbox"] == {
        "x_min": 154.0,
        "y_min": 383.0,
        "x_max": 386.0,
        "y_max": 473.0,
        "class_id": 0,
    }
    assert parsed["corners"] == (
        (154.0, 383.0),
        (363.0, 402.0),
        (386.0, 473.0),
        (177.0, 454.0),
    )
    assert parsed["area"] == pytest.approx(25.0)
    assert parsed["tilt"] == (95.0, 113.0)
    assert parsed["brightness"] == 37
    assert parsed["blurriness"] == 15


def test_parse_ccpd_filename_validation_errors() -> None:
    with pytest.raises(DatasetValidationError):
        parse_ccpd_filename("invalid_name.jpg")

    with pytest.raises(DatasetValidationError):
        parse_ccpd_filename(
            "025-bad_tilt-10&20_30&40-30&40_10&40_10&20_30&20-0_0_1_2_3_4_5-10-5.jpg"
        )

    with pytest.raises(DatasetValidationError):
        parse_ccpd_filename(
            "025-95_113-100&200_50&300-30&40_10&40_10&20_30&20-0_0_1_2_3_4_5-10-5.jpg"
        )

    with pytest.raises(DatasetValidationError):
        parse_ccpd_filename(
            "025-95_113-10&20_30&40-30&40_10&40_10&20_30&20-99_0_1_2_3_4_5-10-5.jpg"
        )

    with pytest.raises(DatasetValidationError):
        parse_ccpd_filename("025-95_113-10&20_30&40-30&40_10&40_10&20_30&20-0_0_1_2_3-10-5.jpg")


def test_ccpd_filename_adapter_loads_records(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ccpd"
    filename = (
        "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.png"
    )
    _write_png(ds_dir / filename, 800, 600)

    cfg = DataConfig(min_image_width=16, min_image_height=16)
    records = load_real_dataset(
        ds_dir,
        RealDatasetFormat.CCPD_FILENAME,
        config=cfg,
        default_country_code="CN",
    )
    assert len(records) == 1
    assert records[0].source_dataset == "real_ccpd"
    assert records[0].annotations[0].plate_text == "AHAY339S"
    assert records[0].annotations[0].country_code == "CN"
    assert records[0].annotations[0].corners is not None

    direct = CcpdFilenameAdapter().load_records(ds_dir, cfg)
    assert len(direct) == 1


def test_pascal_voc_xml_adapter_loads_and_validates(tmp_path: Path) -> None:
    ds_dir = tmp_path / "voc"
    _write_png(ds_dir / "JPEGImages" / "car_01.png", 640, 480)
    (ds_dir / "Annotations").mkdir(parents=True, exist_ok=True)
    (ds_dir / "Annotations" / "car_01.xml").write_text(
        "<annotation>"
        "<object>"
        "<name>plate</name>"
        "<plate_text>34ABC123</plate_text>"
        "<country_code>TR</country_code>"
        "<bndbox><xmin>100</xmin><ymin>200</ymin><xmax>260</xmax><ymax>245</ymax></bndbox>"
        "</object>"
        "<object>"
        "<name>plate</name>"
        "<bndbox><xmin>300</xmin><ymin>210</ymin><xmax>410</xmax><ymax>250</ymax></bndbox>"
        "</object>"
        "</annotation>",
        encoding="utf-8",
    )

    cfg = DataConfig(min_image_width=16, min_image_height=16)
    records = load_real_dataset(
        ds_dir,
        RealDatasetFormat.PASCAL_VOC_XML,
        config=cfg,
        default_country_code="DE",
    )
    assert len(records) == 1
    assert records[0].source_dataset == "real_pascal_voc"
    assert len(records[0].annotations) == 2
    assert records[0].annotations[0].plate_text == "34ABC123"
    assert records[0].annotations[0].country_code == "TR"
    assert records[0].annotations[1].plate_text == "UNKNOWN"
    assert records[0].annotations[1].country_code == "DE"

    (ds_dir / "Annotations" / "car_01.xml").write_text("<annotation><broken>", encoding="utf-8")
    with pytest.raises(DatasetValidationError):
        PascalVocXmlAdapter().load_records(ds_dir, cfg)

    (ds_dir / "Annotations" / "car_01.xml").write_text(
        "<annotation><object><name>plate</name></object></annotation>",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError):
        PascalVocXmlAdapter().load_records(ds_dir, cfg)
