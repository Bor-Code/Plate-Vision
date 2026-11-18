# Datasets and Data Versioning

## Approved Datasets

Only datasets whose licenses permit the intended training and evaluation usage are ingested into the `plate-vision` pipeline:

| Dataset Identifier | Source | License | Version | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `synthetic_tr_v1` | Internal deterministic synthetic plate generator (`plate_vision.data.synthetic`) | MIT | `1.0.0` | Recognizer pre-training, edge-case enrichment, and country format coverage |
| `open_images_v7_plates` | Google Open Images Dataset V7 (`Vehicle registration plate` subset) | CC BY 4.0 (annotations) / CC BY 2.0 (images) | `v7` | License plate detection training and evaluation |
| `ufpr_alpr_academic` | UFPR-ALPR Dataset (restricted to academic benchmark evaluation only when explicitly licensed) | Academic Non-Commercial (excluded from commercial builds) | `1.0.0` | Optional academic comparative reference |

## Real-World Dataset Adapters (`plate_vision.data.adapters`)

Real captured license plate datasets can be ingested and normalized into `DatasetRecord` collections via `load_real_dataset`:

- **`JsonlManifestAdapter` (`jsonl_manifest`)**: Reads `<dataset_dir>/manifest.jsonl` containing `record_id`, relative or absolute `image_path`, and `annotations`.
- **`YoloDirectoryAdapter` (`yolo_directory`)**: Reads images from `<dataset_dir>` (or `<dataset_dir>/images`) with adjacent `.txt` or `<dataset_dir>/labels/<stem>.txt` YOLO annotation lines (`<class_id> <x_center> <y_center> <norm_w> <norm_h> [plate_text] [country_code]`).
- **`OpenAlprSidecarAdapter` (`openalpr_sidecar`)**: Reads images paired with `<stem>.json` sidecar files containing `"plates"` (`plate_text`, optional `country_code`, `bbox`, and optional 4-point `corners`).

All adapters pass image payloads through `inspect_image_bytes` and `ingest_manifest_records` for binary header verification, bounding-box validation, and SHA-256 content hashing.

### Preparing Real and Hybrid Datasets via CLI

```bash
uv run prepare_dataset --output-dir data/prepared_real --real-dataset-dir data/raw/real_openalpr --real-format openalpr_sidecar --synthetic-count 0 --country-code TR
uv run prepare_dataset --output-dir data/prepared_hybrid --real-dataset-dir data/raw/real_yolo --real-format yolo_directory --synthetic-count 50 --country-code TR --seed 42
```

## Data Versioning with DVC

Raw images, processed splits, and trained model weights are never committed to Git. All dataset artifacts are tracked via DVC:

- `.dvc/config`: Configures the DVC storage remote.
- `dvc.yaml`: Defines reproducible dataset pipeline stages and dependencies.

## Country Profiles and Turkey (`TR`) Specification

Country-specific rules are encapsulated behind `CountryProfileProtocol` and managed by `CountryProfileRegistry` across 8 registered jurisdictions (`TR`, `DE`, `GB`, `AZ`, `FR`, `IT`, `ES`, `NL`):

- **Turkey (`TR`) Profile (`TurkeyCountryProfile`)**:
  - Province code: `01` through `81`.
  - Letters: 1 to 3 uppercase ASCII characters from `A-Z` excluding `Q`, `W`, and `X`.
  - Trailing digits: 2 to 4 digits depending on letter group length (`1 letter -> 4 digits`, `2 letters -> 3-4 digits`, `3 letters -> 2-3 digits`).
  - Positional character confusion correction for `0/O`, `1/I`, `8/B`, `5/S`, and `2/Z`.
- New country profiles can be registered at runtime via `register_country_profile(profile)` without modifying core pipeline logic.

## Synthetic Plate Generation and Augmentation

- `plate_vision.data.synthetic`: Renders configurable license plate crops with custom glyph scaling, background/foreground luminance ranges, perspective shear, illumination gradients, dirt spots, blur, and Gaussian noise, outputting valid PNG files and `DatasetRecord` metadata.
- `plate_vision.data.augmentation`: Provides experiment-configurable training augmentations (brightness/contrast, rotation, motion blur, coarse occlusion dropout, and Gaussian noise) along with Albumentations transform specifications.
