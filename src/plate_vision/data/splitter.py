import math
import random
from collections import defaultdict
from collections.abc import Sequence

from plate_vision.config.schema import DatasetSplitRatios
from plate_vision.data.models import DatasetRecord, SplitDataset
from plate_vision.utils.exceptions import DatasetValidationError


def allocate_stratum_counts(
    total: int,
    ratios: DatasetSplitRatios,
) -> tuple[int, int, int]:
    if total == 0:
        return (0, 0, 0)
    if total == 1:
        return (1, 0, 0)
    if total == 2:
        return (1, 1, 0)

    raw_train = total * ratios.train
    raw_val = total * ratios.val
    raw_test = total * ratios.test

    train_count = max(1, math.floor(raw_train))
    val_count = max(1, math.floor(raw_val))
    test_count = max(1, math.floor(raw_test))

    assigned = train_count + val_count + test_count
    remainders = [
        (raw_train - math.floor(raw_train), 0),
        (raw_val - math.floor(raw_val), 1),
        (raw_test - math.floor(raw_test), 2),
    ]
    remainders.sort(key=lambda item: (-item[0], item[1]))

    counts = [train_count, val_count, test_count]
    idx = 0
    while assigned < total:
        target_bucket = remainders[idx % 3][1]
        counts[target_bucket] += 1
        assigned += 1
        idx += 1

    while assigned > total:
        if counts[0] > 1:
            counts[0] -= 1
        elif counts[1] > 1:
            counts[1] -= 1
        else:
            counts[2] -= 1
        assigned -= 1

    return (counts[0], counts[1], counts[2])


def verify_no_split_leakage(split_dataset: SplitDataset) -> None:
    train_hashes = {rec.image.sha256 for rec in split_dataset.train}
    val_hashes = {rec.image.sha256 for rec in split_dataset.val}
    test_hashes = {rec.image.sha256 for rec in split_dataset.test}

    overlap = (
        (train_hashes & val_hashes) | (train_hashes & test_hashes) | (val_hashes & test_hashes)
    )
    if overlap:
        raise DatasetValidationError(
            "Duplicate image hashes detected across dataset splits",
            details={"overlapping_hashes": sorted(overlap)},
        )


def stratified_split(
    records: Sequence[DatasetRecord],
    ratios: DatasetSplitRatios,
    seed: int,
) -> SplitDataset:
    if not records:
        raise DatasetValidationError("Cannot split an empty dataset")

    strata: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda item: item.record_id):
        strata[record.stratum_key].append(record)

    rng = random.Random(seed)
    train_records: list[DatasetRecord] = []
    val_records: list[DatasetRecord] = []
    test_records: list[DatasetRecord] = []

    for stratum_key in sorted(strata.keys()):
        group = list(strata[stratum_key])
        rng.shuffle(group)
        n_train, n_val, _ = allocate_stratum_counts(len(group), ratios)
        train_records.extend(group[:n_train])
        val_records.extend(group[n_train : n_train + n_val])
        test_records.extend(group[n_train + n_val :])

    result = SplitDataset(
        train=tuple(sorted(train_records, key=lambda rec: rec.record_id)),
        val=tuple(sorted(val_records, key=lambda rec: rec.record_id)),
        test=tuple(sorted(test_records, key=lambda rec: rec.record_id)),
    )
    verify_no_split_leakage(result)
    return result


__all__ = [
    "allocate_stratum_counts",
    "stratified_split",
    "verify_no_split_leakage",
]
