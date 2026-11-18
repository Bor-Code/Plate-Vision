from pathlib import Path

import pytest

from plate_vision.data.cli import parse_args, run_prepare_dataset
from plate_vision.utils.exceptions import InvalidInputError


def test_prepare_dataset_parse_args_and_validation(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path)])
    assert args.output_dir == tmp_path
    assert args.country_code == "TR"
    assert args.synthetic_count == 30
    assert args.seed == 42

    bad_args = parse_args(["--output-dir", str(tmp_path), "--synthetic-count", "0"])
    with pytest.raises(InvalidInputError, match="synthetic_count"):
        run_prepare_dataset(bad_args)
