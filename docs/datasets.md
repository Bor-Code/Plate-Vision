# Datasets and Data Versioning

## Approved Datasets

Only datasets whose licenses permit the intended training and evaluation usage are ingested into the `plate-vision` pipeline:

| Dataset Identifier | Source | License | Version | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `synthetic_tr_v1` | Internal deterministic synthetic plate generator (`plate_vision.data.synthetic`) | MIT | `1.0.0` | Recognizer pre-training, edge-case enrichment, and Turkey (`TR`) format coverage |
| `open_images_v7_plates` | Google Open Images Dataset V7 (`Vehicle registration plate` subset) | CC BY 4.0 (annotations) / CC BY 2.0 (images) | `v7` | License plate detection training and evaluation |
| `ufpr_alpr_academic` | UFPR-ALPR Dataset (restricted to academic benchmark evaluation only when explicitly licensed) | Academic Non-Commercial (excluded from commercial builds) | `1.0.0` | Optional academic comparative reference |

## Data Versioning with DVC

Raw images, processed splits, and trained model weights are never committed to Git. All dataset artifacts are tracked via DVC:

- `.dvc/config`: Configures the DVC storage remote.
- `dvc.yaml`: Defines reproducible dataset pipeline stages and dependencies.

## Country Profiles and Turkey (`TR`) Specification

Country-specific rules are encapsulated behind `CountryProfileProtocol` and managed by `CountryProfileRegistry`:

- **Turkey (`TR`) Profile (`TurkeyCountryProfile`)**:
  - Province code: `01` through `81`.
  - Letters: 1 to 3 uppercase ASCII characters from `A-Z` excluding `Q`, `W`, and `X`.
  - Trailing digits: 2 to 5 digits depending on letter group length (`1 letter + 4-5 digits`, `2 letters + 3-4 digits`, `3 letters + 2-3 digits`).
  - Positional character confusion correction for `0/O`, `1/I`, `8/B`, `5/S`, and `2/Z`.
- New country profiles can be registered at runtime via `register_country_profile(profile)` without modifying core pipeline logic.

## Synthetic Plate Generation and Augmentation

- `plate_vision.data.synthetic`: Renders configurable license plate crops with custom glyph scaling, background/foreground luminance ranges, perspective shear, illumination gradients, dirt spots, blur, and Gaussian noise, outputting valid PNG files and `DatasetRecord` metadata.
- `plate_vision.data.augmentation`: Provides experiment-configurable training augmentations (brightness/contrast, rotation, motion blur, coarse occlusion dropout, and Gaussian noise) along with Albumentations transform specifications.
