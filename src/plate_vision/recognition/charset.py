from collections.abc import Sequence
from typing import Final

from plate_vision.data.profiles import TURKEY_CHARSET
from plate_vision.utils.exceptions import InvalidInputError

BLANK_INDEX: Final[int] = 0
BLANK_TOKEN: Final[str] = "<blank>"


class CharsetEncoder:
    def __init__(self, charset: str = TURKEY_CHARSET) -> None:
        unique_chars = sorted({ch.upper() for ch in charset if ch.isalnum()})
        if not unique_chars:
            raise InvalidInputError("Charset must contain at least one alphanumeric character")
        self._characters: tuple[str, ...] = tuple(unique_chars)
        self._char_to_index: dict[str, int] = {
            ch: idx + 1 for idx, ch in enumerate(self._characters)
        }
        self._index_to_char: dict[int, str] = {
            idx + 1: ch for idx, ch in enumerate(self._characters)
        }

    @property
    def characters(self) -> tuple[str, ...]:
        return self._characters

    @property
    def num_classes(self) -> int:
        return len(self._characters) + 1

    def encode(self, text: str) -> tuple[int, ...]:
        cleaned = "".join(ch for ch in text.strip().upper() if not ch.isspace() and ch != "-")
        if not cleaned:
            raise InvalidInputError("Cannot encode empty plate text")
        indices: list[int] = []
        for ch in cleaned:
            token_idx = self._char_to_index.get(ch)
            if token_idx is None:
                raise InvalidInputError(
                    f"Character '{ch}' is not in the configured charset",
                    details={"character": ch, "text": text},
                )
            indices.append(token_idx)
        return tuple(indices)

    def decode_indices(
        self,
        indices: Sequence[int],
        *,
        collapse_repeats: bool = True,
        remove_blank: bool = True,
    ) -> str:
        chars: list[str] = []
        prev_idx: int | None = None
        for idx in indices:
            if collapse_repeats and idx == prev_idx:
                continue
            prev_idx = idx
            if idx == BLANK_INDEX:
                if not remove_blank:
                    chars.append(BLANK_TOKEN)
                continue
            ch = self._index_to_char.get(idx)
            if ch is None:
                raise InvalidInputError(
                    f"Token index {idx} is out of range for charset",
                    details={"index": idx, "num_classes": self.num_classes},
                )
            chars.append(ch)
        return "".join(chars)


__all__ = [
    "BLANK_INDEX",
    "BLANK_TOKEN",
    "CharsetEncoder",
]
