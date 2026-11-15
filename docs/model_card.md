# Model Card: Plate-Vision Detector and CRNN Recognizer

## Model Details

- **System Name**: Plate-Vision Two-Stage ALPR
- **Version**: `0.1.0`
- **Stage 1 (Detector)**:
  - **Architecture**: YOLO bounding-box detector (`yolo11n.pt` configuration profile, input resolution `640x640`, stride `32`).
  - **Outputs**:Normalized or pixel-space bounding boxes `(x_min, y_min, x_max, y_max)`, objectness confidence `[0.0, 1.0]`, and class ID `0` (`license_plate`).
- **Stage 2 (Recognizer)**:
  - **Architecture**: Convolutional Recurrent Neural Network (`CRNNModel`) with spatial-to-sequence feature extraction (`16` time steps), recurrent sequence modeling, linear projection over `34` classes (`33` Turkish plate characters + `1` CTC blank token at index `0`), and Connectionist Temporal Classification (CTC) training objective.
  - **Input Resolution**: `32x128` single-channel grayscale normalized plate crop.
  - **Decoding**: CTC prefix beam search (`beam_width=5`) and greedy decoding.
- **Serving Format**: ONNX (`opset_version=17`) in both `fp32` and dynamically quantized `int8` variants.

## Intended Use

- **Primary Use Cases**: Automated parking access control, toll collection, fleet gate management, and traffic monitoring for Turkish (`TR`) and extensible international license plates.
- **Out-of-Scope Uses**: Surveillance unrelated to vehicle license plate recognition or deployment in jurisdictions without an explicitly registered `CountryProfileProtocol`.

## Country Profile Specification (`TR`)

- **Province Code**: `01` through `81` (2 digits).
- **Letter Group**: `1` to `3` uppercase letters from `ABCDEFGHIJKLMNOPRSTUVYZ` (excluding `Q`, `W`, `X`).
- **Numeric Suffix**: `2` to `4` digits depending on letter group length (`1` letter -> `4` digits; `2` letters -> `3-4` digits; `3` letters -> `2-3` digits).
- **Confusion Correction Map**:
  - Digit-to-Letter: `0 -> O`, `1 -> I`, `8 -> B`, `5 -> S`, `2 -> Z`
  - Letter-to-Digit: `O -> 0`, `D -> 0`, `I -> 1`, `B -> 8`, `S -> 5`, `Z -> 2`

## Limitations and Edge Cases

- Extreme occlusion covering more than two characters cannot be recovered purely from optical confusion rules and will be flagged with `is_valid_format=False`.
- Crops smaller than `12x6` pixels are filtered prior to recognition to avoid false-positive noise reads.
- Video streams with rapid camera pan should use `VideoStreamProcessor` with `PlateTracker` and `TemporalVoter` to aggregate multi-frame observations.
