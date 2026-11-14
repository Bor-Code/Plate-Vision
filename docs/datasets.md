# Datasets and Data Versioning

## Approved Datasets

Only datasets whose licenses permit the intended training and evaluation usage are ingested into the `plate-vision` pipeline:

| Dataset Identifier | Source | License | Version | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `synthetic_tr_v1` | Internal deterministic synthetic plate generator (`plate_vision.data`) | MIT | `1.0.0` | Recognizer pre-training, edge-case enrichment, and Turkey (`TR`) format coverage |
| `open_images_v7_plates` | Google Open Images Dataset V7 (`Vehicle registration plate` subset) | CC BY 4.0 (annotations) / CC BY 2.0 (images) | `v7` | License plate detection training and evaluation |
| `ufpr_alpr_academic` | UFPR-ALPR Dataset (restricted to academic benchmark evaluation only when explicitly licensed) | Academic Non-Commercial (excluded from commercial builds) | `1.0.0` | Optional academic comparative reference |

## Data Versioning with DVC

Raw images, processed splits, and trained model weights are never committed to Git. All dataset artifacts are tracked via DVC:

- `.dvc/config`: Configures the DVC storage remote.
- `dvc.yaml`: Defines reproducible dataset pipeline stages and dependencies.

## Reproducible Pipeline Stages

1. **Ingestion and Validation (`plate_vision.data.ingestion`)**:
   - Validates binary image headers (`PNG`, `JPEG`, `WebP`), dimensions (`min_image_width`, `min_image_height`), and file sizes (`max_image_bytes`).
   - Validates bounding box coordinates against image bounds and normalizes plate annotations.
2. **Deduplication (`plate_vision.data.deduplication`)**:
   - Computes SHA-256 content digests and eliminates duplicate images in deterministic `record_id` order prior to splitting.
3. **Stratified Splitting (`plate_vision.data.splitter`)**:
   - Partitions records into `train`, `val`, and `test` splits stratified by `stratum_key` (dataset source, country code, and province prefix, plus negative and multi-plate strata) using a fixed random seed.
   - Verifies zero SHA-256 overlap across splits.
4. **Exporters (`plate_vision.data.exporters`)**:
   - Exports normalized YOLO detection labels (`labels/{train,val,test}/*.txt`, `train.txt`, `val.txt`, `test.txt`, `dataset.yaml`).
   - Exports recognizer manifests in both JSONL (`recognizer_{train,val,test}.jsonl`) and CSV (`recognizer_{train,val,test}.csv`) formats.
