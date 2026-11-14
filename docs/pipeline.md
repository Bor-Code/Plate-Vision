# Two-Stage Inference Pipeline

The `plate_vision.pipeline` package implements the production two-stage Automatic License Plate Recognition (ALPR) inference pipeline.

## Pipeline Stages

1. **Input Validation and Decoding (`decode_image_to_grayscale`)**:
   - Rejects empty payloads and payloads exceeding `max_payload_bytes` with `InvalidInputError`.
   - Rejects corrupted or non-image binary uploads with `CorruptedImageError`.
2. **Stage 1 - Plate Detection (`PlateDetectorProtocol` + `apply_nms`)**:
   - Detects candidate plate bounding boxes and applies Non-Maximum Suppression (`confidence_threshold`, `iou_threshold`, `max_detections`).
   - Filters out tiny detections smaller than `min_plate_width` x `min_plate_height`.
   - Handles images with zero plates (returns empty `plates` tuple or raises `EmptyDetectionError` when `require_detection=True`) and images with multiple plates.
3. **Intermediate Rectification and Normalization**:
   - `extract_plate_crop`: Extracts bounded rectangular crops.
   - `rectify_perspective_crop`: Warps 4-point quadrilateral coordinates of rotated or skewed plates into an axis-aligned rectangular grid using bilinear interpolation.
   - `normalize_plate_crop`: Resizes crops to `(target_height, target_width)`, applies contrast stretching for low-light images, and applies horizontal unsharp masking for motion-blurred crops.
4. **Stage 2 - Plate Recognition (`PlateRecognizerProtocol`)**:
   - Runs the CRNN recognizer with CTC decoding (greedy or beam search).
5. **Post-Processing (`postprocess_plate_text`)**:
   - Normalizes plate strings, corrects positional optical character confusions (`0/O`, `1/I`, `8/B`, `5/S`, `2/Z`), and validates country-specific regex rules via `CountryProfileProtocol`.
6. **Timeout Enforcement**:
   - Raises `InferenceTimeoutError` if execution time exceeds `timeout_seconds`.
