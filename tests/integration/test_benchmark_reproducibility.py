import random
import time
from pathlib import Path

from plate_vision.config.schema import DetectorConfig, RecognizerConfig
from plate_vision.data.exporters import export_recognizer_manifests, export_yolo_dataset
from plate_vision.data.models import (
    BoundingBox,
    DatasetRecord,
    DatasetSplit,
    ImageMetadata,
    PlateAnnotation,
    SplitDataset,
)
from plate_vision.data.profiles import get_country_profile
from plate_vision.data.synthetic import SyntheticPlateConfig, generate_synthetic_sample
from plate_vision.detection.trainer import DetectorTrainer
from plate_vision.evaluation.analyzer import build_evaluation_report
from plate_vision.export.onnx_exporter import export_detector_to_onnx, export_recognizer_to_onnx
from plate_vision.export.runtime import (
    OnnxDetectorRuntime,
    OnnxRecognizerRuntime,
    verify_recognizer_parity,
)
from plate_vision.recognition.charset import CharsetEncoder
from plate_vision.recognition.dataset import RecognizerDataset
from plate_vision.recognition.trainer import RecognizerTrainer


def test_benchmark_reproducibility_matches_documented_metrics(tmp_path: Path) -> None:
    profile = get_country_profile("TR")
    rng = random.Random(42)
    fleet_plates = [profile.sample_plate(rng) for _ in range(3)]

    def build_records(split_tag: str, noise: float) -> list[DatasetRecord]:
        out_dir = tmp_path / f"imgs_{split_tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = SyntheticPlateConfig(
            image_width=128,
            image_height=32,
            perspective_shift=0.0,
            noise_std=noise,
            blur_strength=0.0,
            lighting_gradient=0.0,
            dirt_spots=0,
            seed=42,
        )
        recs: list[DatasetRecord] = []
        for idx, ptext in enumerate(fleet_plates):
            sample_rng = random.Random(100 + idx)
            _, png_bytes, bbox = generate_synthetic_sample(ptext, cfg, sample_rng)
            fpath = out_dir / f"{split_tag}_{idx}_{ptext}.png"
            fpath.write_bytes(png_bytes)
            recs.append(
                DatasetRecord(
                    record_id=f"{split_tag}_{idx:03d}",
                    image=ImageMetadata(
                        path=fpath,
                        width=128,
                        height=32,
                        size_bytes=len(png_bytes),
                        sha256=f"{idx:064x}",
                        mime_type="image/png",
                    ),
                    annotations=(PlateAnnotation(plate_text=ptext, bbox=bbox, country_code="TR"),),
                    source_dataset="synthetic_fleet",
                    stratum_key=f"TR:{ptext[:2]}",
                )
            )
        return recs

    split_ds = SplitDataset(
        train=tuple(build_records("train", 0.0)),
        val=tuple(build_records("val", 1.0)),
        test=tuple(build_records("test", 1.5)),
    )
    det_yaml = export_yolo_dataset(split_ds, tmp_path / "yolo_ds")
    manifests = export_recognizer_manifests(split_ds, tmp_path / "manifests")

    det_summary = DetectorTrainer(
        config=DetectorConfig(epochs=3, warmup_epochs=1),
        seed=42,
    ).train(det_yaml, tmp_path / "det_ckpt")

    encoder = CharsetEncoder()
    train_ds = RecognizerDataset(manifests[DatasetSplit.TRAIN], encoder, seed=42)
    val_ds = RecognizerDataset(manifests[DatasetSplit.VAL], encoder, seed=42)
    test_ds = RecognizerDataset(manifests[DatasetSplit.TEST], encoder, seed=42)

    rec_summary = RecognizerTrainer(
        config=RecognizerConfig(epochs=3, batch_size=3, warmup_epochs=1),
        encoder=encoder,
        seed=42,
    ).train(train_ds, val_ds, test_ds, tmp_path / "rec_ckpt")

    det_onnx = tmp_path / "detector.onnx"
    rec_onnx = tmp_path / "recognizer.onnx"
    export_detector_to_onnx(det_summary.best_checkpoint_path, det_onnx)
    export_recognizer_to_onnx(rec_summary.best_checkpoint_path, rec_onnx)

    parity = verify_recognizer_parity(
        rec_summary.best_checkpoint_path,
        rec_onnx,
        [list(test_ds[i].pixels) for i in range(len(test_ds))],
    )
    assert parity.passed is True

    det_rt = OnnxDetectorRuntime(det_onnx)
    rec_rt = OnnxRecognizerRuntime(rec_onnx)
    latencies: list[float] = []
    preds: list[str] = []
    targets: list[str] = []
    det_preds = {}
    det_gts = {}
    sids: list[str] = []

    for idx, rec in enumerate(split_ds.test):
        sid = rec.record_id
        sids.append(sid)
        t0 = time.perf_counter()
        decoded = rec_rt.recognize(test_ds[idx].pixels, use_beam_search=True)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds.append(decoded.text)
        targets.append(rec.annotations[0].plate_text)
        det_preds[sid] = det_rt.detect(rec.image.path.read_bytes(), 128, 32)
        det_gts[sid] = [BoundingBox(x_min=25.6, y_min=11.2, x_max=102.4, y_max=20.8)]

    report = build_evaluation_report(
        detector_predictions=det_preds,
        detector_ground_truths=det_gts,
        sample_ids=sids,
        recognizer_predictions=preds,
        recognizer_targets=targets,
        end_to_end_predictions=preds,
        end_to_end_targets=targets,
        latencies_ms=latencies,
    )
    assert report.detector.map50 == 1.0
    assert report.end_to_end.plate_level_accuracy == 1.0
    assert report.end_to_end.throughput_fps > 0.0
