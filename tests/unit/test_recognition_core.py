from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import pytest

from plate_vision.config.schema import RecognizerConfig
from plate_vision.data import encode_grayscale_png
from plate_vision.recognition import (
    BLANK_INDEX,
    BLANK_TOKEN,
    CharsetEncoder,
    CRNNModel,
    CRNNPlateRecognizer,
    RecognizerDataset,
    ctc_beam_search_decode,
    ctc_greedy_decode,
    decode_grayscale_png,
    log_softmax_row,
    logsumexp,
)
from plate_vision.utils.exceptions import (
    CorruptedImageError,
    DataIOError,
    DatasetValidationError,
    InvalidInputError,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_charset_encoder_roundtrip_and_errors() -> None:
    with pytest.raises(InvalidInputError, match="at least one alphanumeric"):
        CharsetEncoder(charset="---")

    encoder = CharsetEncoder()
    assert encoder.num_classes == len(encoder.characters) + 1

    with pytest.raises(InvalidInputError, match="Cannot encode empty"):
        encoder.encode("   ")

    with pytest.raises(InvalidInputError, match="not in the configured charset"):
        encoder.encode("34QWE123")

    encoded = encoder.encode("34 AB 123")
    assert encoder.decode_indices(encoded) == "34AB123"

    repeated_with_blanks = (
        BLANK_INDEX,
        encoded[0],
        encoded[0],
        BLANK_INDEX,
        encoded[1],
    )
    assert encoder.decode_indices(repeated_with_blanks) == "34"
    raw_decoded = encoder.decode_indices(
        (BLANK_INDEX, encoded[0]),
        collapse_repeats=False,
        remove_blank=False,
    )
    assert raw_decoded == f"{BLANK_TOKEN}3"

    with pytest.raises(InvalidInputError, match="out of range"):
        encoder.decode_indices([999])


def test_ctc_greedy_and_beam_search_decoding() -> None:
    encoder = CharsetEncoder()
    assert log_softmax_row([]) == []
    assert logsumexp(-1e20, -2.0) == -2.0
    assert logsumexp(-2.0, -1e20) == -2.0

    assert ctc_greedy_decode([], encoder).text == ""
    assert ctc_beam_search_decode([], encoder).text == ""
    with pytest.raises(ValueError, match="beam_width must be >= 1"):
        ctc_beam_search_decode([[0.0, 1.0]], encoder, beam_width=0)

    target_tokens = encoder.encode("34AB12")
    num_classes = encoder.num_classes
    logits_seq: list[list[float]] = []
    for tok in target_tokens:
        step = [-5.0 for _ in range(num_classes)]
        step[tok] = 6.0
        logits_seq.append(step)
        blank_step = [-5.0 for _ in range(num_classes)]
        blank_step[BLANK_INDEX] = 6.0
        logits_seq.append(blank_step)

    greedy = ctc_greedy_decode(logits_seq, encoder)
    beam = ctc_beam_search_decode(logits_seq, encoder, beam_width=4)
    assert greedy.text == "34AB12"
    assert beam.text == "34AB12"
    assert greedy.confidence > 0.8
    assert beam.confidence > 0.8


def test_png_decoding_and_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(CorruptedImageError, match="Invalid PNG signature"):
        decode_grayscale_png(b"invalid")

    bad_ihdr = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 0, 10, 8, 0, 0, 0, 0)
    )
    with pytest.raises(CorruptedImageError, match="Unsupported PNG header"):
        decode_grayscale_png(bad_ihdr)

    no_idat = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 4, 4, 8, 0, 0, 0, 0)
        + b"\x00\x00\x00\x00"
        + struct.pack(">I", 0)
        + b"IEND\x00\x00\x00\x00"
    )
    with pytest.raises(CorruptedImageError, match="Missing IDAT payload"):
        decode_grayscale_png(no_idat)

    bad_zlib = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", 4, 4, 8, 0, 0, 0, 0)
        + b"\x00\x00\x00\x00"
        + struct.pack(">I", 4)
        + b"IDATbad!\x00\x00\x00\x00"
    )
    with pytest.raises(CorruptedImageError, match="Decompression failed"):
        decode_grayscale_png(bad_zlib)

    grid = [[100 + x + y for x in range(8)] for y in range(4)]
    png_bytes = encode_grayscale_png(grid)
    assert decode_grayscale_png(png_bytes) == grid

    encoder = CharsetEncoder()
    with pytest.raises(DataIOError, match="Recognizer manifest file does not exist"):
        RecognizerDataset(tmp_path / "missing.jsonl", encoder)

    bad_manifest = tmp_path / "bad.jsonl"
    bad_manifest.write_text("{malformed\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Malformed JSON"):
        RecognizerDataset(bad_manifest, encoder)

    missing_img_manifest = tmp_path / "missing_img.jsonl"
    missing_img_manifest.write_text(
        '{"sample_id": "s1", "plate_text": "34AB123", "image_path": "nonexistent.png"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DataIOError, match="Sample image file not found"):
        RecognizerDataset(missing_img_manifest, encoder)


def test_crnn_model_forward_loss_and_state_dict() -> None:
    encoder = CharsetEncoder()
    config = RecognizerConfig()
    model = CRNNModel(config=config, encoder=encoder, seed=42)
    assert len(model.extract_sequence_features([])) == model.num_timesteps

    sample_crop = [[180 - ((x * 3 + y) % 90) for x in range(128)] for y in range(32)]
    targets = encoder.encode("34AB123")
    loss, grad_norm = model.compute_ctc_loss_and_update(
        sample_crop,
        targets,
        learning_rate=0.05,
        grad_clip_norm=5.0,
    )
    assert loss > 0.0
    assert grad_norm >= 0.0

    state = model.state_dict()
    restored = CRNNModel(config=config, encoder=encoder, seed=99)
    restored.load_state_dict(state)

    recognizer = CRNNPlateRecognizer(model=restored, beam_width=4)
    greedy_pred = recognizer.recognize(sample_crop, use_beam_search=False)
    beam_pred = recognizer.recognize(sample_crop, use_beam_search=True)
    assert greedy_pred.text == "34AB123"
    assert beam_pred.text == "34AB123"
