from collections.abc import Sequence

from plate_vision.data.models import DatasetRecord, DeduplicationResult


def deduplicate_records(records: Sequence[DatasetRecord]) -> DeduplicationResult:
    ordered_records = sorted(records, key=lambda rec: rec.record_id)
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    unique: list[DatasetRecord] = []
    duplicates: list[str] = []

    for record in ordered_records:
        if record.record_id in seen_ids or record.image.sha256 in seen_hashes:
            duplicates.append(record.record_id)
            continue
        seen_ids.add(record.record_id)
        seen_hashes.add(record.image.sha256)
        unique.append(record)

    return DeduplicationResult(
        unique_records=tuple(unique),
        duplicate_record_ids=tuple(duplicates),
        removed_count=len(duplicates),
    )


__all__ = ["deduplicate_records"]
