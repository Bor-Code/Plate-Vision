# License Plate Recognition (CRNN + CTC)

## Architecture

Stage 2 of `plate-vision` transcribes character sequences from rectified license plate crops (`plate_vision.recognition`):

1. **`CharsetEncoder` (`plate_vision.recognition.charset`)**:
   - Maps alphanumeric plate characters to integer class indices `1..N` while reserving index `0` for the CTC `<blank>` token.
2. **`CRNNModel` (`plate_vision.recognition.model`)**:
   - Extracts horizontal sequence features from `32x128` grayscale crops, applies bidirectional recurrent contextualization across `T=16` time steps, projects to `num_classes` logits per time step, and supports analytical CTC loss updates.
3. **Decoders (`plate_vision.recognition.decoding`)**:
   - `ctc_greedy_decode`: Selects maximum-probability tokens per time step, collapses consecutive repeats, removes blanks, and computes geometric-mean confidence.
   - `ctc_beam_search_decode`: Performs log-domain prefix beam search tracking blank and non-blank path probabilities (`(p_b, p_nb)`) with configurable `beam_width`.
4. **Dataset and Seeded DataLoader (`plate_vision.recognition.dataset`)**:
   - `RecognizerDataset` loads exported JSONL manifests, decodes PNG crops, encodes target labels, and applies optional training augmentations.
   - `SeededDataLoader` provides deterministic epoch-seeded shuffling and batch iteration.

## Training Loop and Evaluation Metrics

`RecognizerTrainer` (`plate_vision.recognition.trainer`) manages CRNN optimization:

- **Mixed Precision (`AmpGradScaler`)**: Dynamically scales loss gradients and adjusts scale factor on overflow/underflow.
- **Gradient Clipping and Scheduling**: Clips gradient norms (`grad_clip_norm`) and schedules learning rate with linear warmup and cosine decay (`compute_recognizer_lr`).
- **Evaluation Metrics (`compute_recognizer_metrics`)**:
  - `exact_match_accuracy`: Plate-level exact string match ratio.
  - `character_error_rate`: Levenshtein edit distance divided by total ground-truth characters.
  - `positional_accuracy`: Character match accuracy at corresponding indices.
- **Checkpointing and Resume**: Persists `best.ckpt.json` and `last.ckpt.json` and supports resuming from any saved checkpoint via `--resume-from`.

## CLI Usage

```bash
uv run train_recognizer \
  --config configs/recognition/default.yaml \
  --manifest-dir data/processed/recognizer \
  --output-dir checkpoints/recognizer \
  --seed 42
```
