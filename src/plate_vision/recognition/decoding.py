import math
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from plate_vision.recognition.charset import BLANK_INDEX, CharsetEncoder

NEG_INF: Final[float] = -1e18


class DecodedPlate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    token_indices: tuple[int, ...] = Field(default=())
    timestep_confidences: tuple[float, ...] = Field(default=())


def logsumexp(a: float, b: float) -> float:
    if a <= NEG_INF:
        return b
    if b <= NEG_INF:
        return a
    high = max(a, b)
    low = min(a, b)
    return high + math.log1p(math.exp(low - high))


def log_softmax_row(logits: Sequence[float]) -> list[float]:
    if not logits:
        return []
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    log_denom = max_val + math.log(sum(exps))
    return [val - log_denom for val in logits]


def ctc_greedy_decode(
    logits_sequence: Sequence[Sequence[float]],
    encoder: CharsetEncoder,
) -> DecodedPlate:
    if not logits_sequence:
        return DecodedPlate(text="", confidence=0.0)

    raw_indices: list[int] = []
    step_probs: list[float] = []

    for row in logits_sequence:
        log_probs = log_softmax_row(row)
        best_idx = max(range(len(log_probs)), key=lambda idx: log_probs[idx])
        best_prob = min(1.0, max(0.0, math.exp(log_probs[best_idx])))
        raw_indices.append(best_idx)
        step_probs.append(round(best_prob, 6))

    text = encoder.decode_indices(raw_indices, collapse_repeats=True, remove_blank=True)
    mean_log_prob = sum(math.log(max(1e-12, p)) for p in step_probs) / len(step_probs)
    confidence = min(1.0, max(0.0, math.exp(mean_log_prob)))

    return DecodedPlate(
        text=text,
        confidence=round(confidence, 6),
        token_indices=tuple(raw_indices),
        timestep_confidences=tuple(step_probs),
    )


def ctc_beam_search_decode(
    logits_sequence: Sequence[Sequence[float]],
    encoder: CharsetEncoder,
    beam_width: int = 5,
) -> DecodedPlate:
    if not logits_sequence:
        return DecodedPlate(text="", confidence=0.0)
    if beam_width < 1:
        msg = "beam_width must be >= 1"
        raise ValueError(msg)

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}

    for row in logits_sequence:
        log_probs = log_softmax_row(row)
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = logsumexp(p_b, p_nb)
            blank_lp = log_probs[BLANK_INDEX]
            cur_b, cur_nb = next_beams.get(prefix, (NEG_INF, NEG_INF))
            next_beams[prefix] = (logsumexp(cur_b, p_total + blank_lp), cur_nb)

            last_token = prefix[-1] if prefix else None
            for token_idx in range(1, len(log_probs)):
                token_lp = log_probs[token_idx]
                new_prefix = (*prefix, token_idx)
                if token_idx == last_token:
                    old_b, old_nb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                    next_beams[prefix] = (old_b, logsumexp(old_nb, p_nb + token_lp))

                    nb_b, nb_nb = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (nb_b, logsumexp(nb_nb, p_b + token_lp))
                else:
                    nb_b, nb_nb = next_beams.get(new_prefix, (NEG_INF, NEG_INF))
                    next_beams[new_prefix] = (nb_b, logsumexp(nb_nb, p_total + token_lp))

        ranked = sorted(
            next_beams.items(),
            key=lambda item: logsumexp(item[1][0], item[1][1]),
            reverse=True,
        )
        beams = dict(ranked[:beam_width])

    best_prefix, (best_pb, best_pnb) = max(
        beams.items(),
        key=lambda item: logsumexp(item[1][0], item[1][1]),
    )
    total_log_prob = logsumexp(best_pb, best_pnb)
    seq_len = max(1, len(logits_sequence))
    confidence = min(1.0, max(0.0, math.exp(total_log_prob / seq_len)))
    text = encoder.decode_indices(best_prefix, collapse_repeats=False, remove_blank=True)

    return DecodedPlate(
        text=text,
        confidence=round(confidence, 6),
        token_indices=best_prefix,
        timestep_confidences=(),
    )


__all__ = [
    "DecodedPlate",
    "ctc_beam_search_decode",
    "ctc_greedy_decode",
    "log_softmax_row",
    "logsumexp",
]
