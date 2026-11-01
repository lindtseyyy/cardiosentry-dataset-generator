#!/usr/bin/env python3
"""Create a copy-only merge of prototype_1152 and a combination dataset.

The prototype contributes every row except AF+LVH. The combination dataset
contributes exactly render_k=0 for every record, which keeps one of the four
training augmentations and the sole validation/test image. When a retained
prototype patient would span splits after the merge, the prototype rows for
that patient are excluded and recorded in ``excluded_prototype_rows.csv``.

Neither input is ever modified. The output directory and archive must not
already exist.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


CORE_FIELDS = [
    "rel_path",
    "record",
    "record_id",
    "original_id",
    "cls",
    "STEMI",
    "AF",
    "LVH",
    "NORMAL",
    "split",
    "source",
    "render_k",
    "grid_palette",
    "rotation_deg",
    "has_annotation",
    "patient_id",
    "age",
    "sex",
    "note",
]
LABELS = ["STEMI", "AF", "LVH", "NORMAL"]
SPLITS = ["train", "val", "test"]
SPLIT_ORDER = {split: index for index, split in enumerate(SPLITS)}
MANIFEST_FIELDS = CORE_FIELDS + [
    "origin_dataset",
    "origin_run_id",
    "origin_rel_path",
    "origin_record",
    "origin_sha256",
    "origin_bytes",
    "merge_action",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def require_fields(rows: list[dict[str, str]], fields: list[str], label: str) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")


def group_splits(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        if row["patient_id"]:
            result[row["patient_id"]].add(row["split"])
    return result


def counts_by_split(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = collections.Counter(row["split"] for row in rows)
    return {split: counts.get(split, 0) for split in SPLITS}


def counts_by_split_and_class(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts = collections.Counter(row["cls"] for row in rows if row["split"] == split)
        result[split] = dict(sorted(counts.items()))
    return result


def positives_by_split(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            label: sum(int(row[label]) for row in rows if row["split"] == split)
            for label in LABELS
        }
        for split in SPLITS
    }


def validate_source_variants(rows: list[dict[str, str]]) -> None:
    by_record: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        by_record[row["record_id"]].append(row)

    errors: list[str] = []
    for record_id, record_rows in by_record.items():
        splits = {row["split"] for row in record_rows}
        if len(splits) != 1:
            errors.append(f"{record_id}: spans source splits {sorted(splits)}")
            continue
        split = next(iter(splits))
        render_ks = sorted(int(row["render_k"]) for row in record_rows)
        expected = [0, 1, 2, 3] if split == "train" else [0]
        if render_ks != expected:
            errors.append(f"{record_id}: {split} render_k={render_ks}, expected {expected}")
    if errors:
        raise ValueError("source augmentation validation failed: " + "; ".join(errors[:10]))


def output_row_from_source(row: dict[str, str]) -> dict[str, str]:
    record_id = row["record_id"]
    if not record_id or any(char in record_id for char in "/\\"):
        raise ValueError(f"unsafe or empty source record_id: {record_id!r}")
    result = {field: row.get(field, "") for field in CORE_FIELDS}
    result["record"] = record_id
    result["rel_path"] = f"images/{row['split']}/{record_id}_a0.jpg"
    result["render_k"] = "0"
    return result


def copy_rows(
    rows: list[dict[str, str]],
    destination_root: Path,
    prototype_root: Path,
    combinations_root: Path,
) -> None:
    total = len(rows)
    for index, row in enumerate(rows, 1):
        destination = destination_root / row["rel_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if row["origin_dataset"] == "prototype_1152":
            source = prototype_root / row["origin_rel_path"]
        else:
            source = combinations_root / row["origin_rel_path"]
        if not source.is_file():
            raise FileNotFoundError(f"source image is missing: {source}")
        shutil.copy2(source, destination)
        if index % 500 == 0 or index == total:
            print(f"  copied {index:,}/{total:,} images", flush=True)


def create_archive(dataset_root: Path, archive_path: Path) -> None:
    temporary_archive = archive_path.with_name(archive_path.name + ".building")
    if temporary_archive.exists():
        raise FileExistsError(f"temporary archive already exists: {temporary_archive}")
    with tarfile.open(temporary_archive, "w", format=tarfile.PAX_FORMAT) as archive:
        archive.add(dataset_root, arcname=dataset_root.name, recursive=True)
    os.replace(temporary_archive, archive_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prototype", type=Path, required=True)
    parser.add_argument("--combinations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()

    prototype_root = args.prototype.resolve()
    combinations_root = args.combinations.resolve()
    output_root = args.out.resolve()
    archive_path = args.archive.resolve()
    temporary_output = output_root.with_name("." + output_root.name + ".building")

    for path in (output_root, archive_path, archive_path.with_suffix(archive_path.suffix + ".sha256")):
        if path.exists():
            sys.exit(f"refusing to overwrite existing output: {path}")
    if temporary_output.exists():
        sys.exit(f"refusing to reuse existing temporary directory: {temporary_output}")
    if archive_path.suffix != ".tar":
        sys.exit("--archive must end in .tar")

    prototype_index = prototype_root / "index.csv"
    combinations_index = combinations_root / "labels.csv"
    prototype_rows = read_csv(prototype_index)
    combination_rows_all = read_csv(combinations_index)
    require_fields(prototype_rows, CORE_FIELDS, "prototype index")
    require_fields(
        combination_rows_all,
        [field for field in CORE_FIELDS if field != "rel_path"]
        + ["image_path", "sha256", "bytes", "dataset_run_id"],
        "combination labels",
    )
    validate_source_variants(combination_rows_all)

    combination_rows = [row for row in combination_rows_all if row["render_k"] == "0"]
    if len({row["record_id"] for row in combination_rows}) != len(combination_rows):
        raise ValueError("render_k=0 selection does not contain one unique row per record_id")

    prototype_af_lvh = [row for row in prototype_rows if row["cls"] == "AF+LVH"]
    combination_af_lvh_ids = {
        row["record_id"] for row in combination_rows if row["cls"] == "AF+LVH"
    }
    unmatched_af_lvh = [
        row["record_id"]
        for row in prototype_af_lvh
        if row["record_id"] not in combination_af_lvh_ids
    ]
    if unmatched_af_lvh:
        raise ValueError(
            "prototype AF+LVH records missing from newer combination data: "
            + ", ".join(unmatched_af_lvh[:10])
        )

    af_lvh_same_split = sum(
        1
        for row in prototype_af_lvh
        if any(
            newer["record_id"] == row["record_id"] and newer["split"] == row["split"]
            for newer in combination_rows
        )
    )

    prototype_without_af_lvh = [
        row for row in prototype_rows if row["cls"] != "AF+LVH"
    ]
    prototype_patient_splits = group_splits(prototype_without_af_lvh)
    combination_patient_splits = group_splits(combination_rows)
    conflicting_patients = sorted(
        patient_id
        for patient_id in prototype_patient_splits.keys() & combination_patient_splits.keys()
        if len(prototype_patient_splits[patient_id] | combination_patient_splits[patient_id]) > 1
    )
    retained_prototype = [
        row
        for row in prototype_without_af_lvh
        if row["patient_id"] not in conflicting_patients
    ]
    patient_conflict_rows = [
        row
        for row in prototype_without_af_lvh
        if row["patient_id"] in conflicting_patients
    ]

    excluded_rows: list[dict[str, str]] = []
    for row in prototype_af_lvh:
        excluded_rows.append(
            {
                "exclusion_reason": "replaced_by_newer_AF+LVH_record_id_match",
                **row,
            }
        )
    for row in patient_conflict_rows:
        excluded_rows.append(
            {
                "exclusion_reason": "excluded_to_prevent_cross_split_patient_leakage",
                **row,
            }
        )

    manifest_rows: list[dict[str, str]] = []
    for row in retained_prototype:
        core = {field: row.get(field, "") for field in CORE_FIELDS}
        manifest_rows.append(
            {
                **core,
                "origin_dataset": "prototype_1152",
                "origin_run_id": "",
                "origin_rel_path": row["rel_path"],
                "origin_record": row["record"],
                "origin_sha256": "",
                "origin_bytes": "",
                "merge_action": "retained_from_prototype",
            }
        )
    for row in combination_rows:
        core = output_row_from_source(row)
        manifest_rows.append(
            {
                **core,
                "origin_dataset": "af_lvh_stemi_combinations_v1",
                "origin_run_id": row["dataset_run_id"],
                "origin_rel_path": row["image_path"],
                "origin_record": row["record"],
                "origin_sha256": row["sha256"],
                "origin_bytes": row["bytes"],
                "merge_action": "selected_render_k_0",
            }
        )

    manifest_rows.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            row["origin_dataset"],
            row["record_id"],
        )
    )
    final_rows = [{field: row[field] for field in CORE_FIELDS} for row in manifest_rows]

    record_ids = [row["record_id"] for row in final_rows]
    records = [row["record"] for row in final_rows]
    rel_paths = [row["rel_path"] for row in final_rows]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("final record_id values are not unique")
    if len(set(records)) != len(records):
        raise ValueError("final record values are not unique")
    if len(set(rel_paths)) != len(rel_paths):
        raise ValueError("final rel_path values are not unique")
    invalid_splits = sorted({row["split"] for row in final_rows} - set(SPLITS))
    if invalid_splits:
        raise ValueError(f"invalid final splits: {invalid_splits}")
    leaking_patients = {
        patient_id: splits
        for patient_id, splits in group_splits(final_rows).items()
        if len(splits) > 1
    }
    if leaking_patients:
        raise ValueError(f"patients still span splits: {list(leaking_patients.items())[:5]}")
    bad_normal = [
        row["record_id"]
        for row in final_rows
        if row["NORMAL"] == "1" and any(row[label] == "1" for label in LABELS[:3])
    ]
    if bad_normal:
        raise ValueError(f"NORMAL co-occurs with pathology: {bad_normal[:5]}")

    bytes_to_copy = 0
    for row in manifest_rows:
        if row["origin_dataset"] == "prototype_1152":
            source_path = prototype_root / row["origin_rel_path"]
        else:
            source_path = combinations_root / row["origin_rel_path"]
        bytes_to_copy += source_path.stat().st_size
    free_bytes = shutil.disk_usage(output_root.parent).free
    required_bytes = bytes_to_copy * 2 + 256 * 1024 * 1024
    if free_bytes < required_bytes:
        raise OSError(
            f"insufficient free space: need about {required_bytes:,} bytes for "
            f"dataset plus tar, have {free_bytes:,}"
        )

    temporary_output.mkdir(parents=True)
    print(f"building {len(final_rows):,}-image dataset in {temporary_output}")
    copy_rows(
        manifest_rows,
        temporary_output,
        prototype_root,
        combinations_root,
    )

    print("validating copied combination-image checksums")
    checksum_mismatches: list[str] = []
    for row in manifest_rows:
        if row["origin_dataset"] != "af_lvh_stemi_combinations_v1":
            continue
        actual = sha256_file(temporary_output / row["rel_path"])
        if actual != row["origin_sha256"]:
            checksum_mismatches.append(row["record_id"])
    if checksum_mismatches:
        raise ValueError(
            "copied combination-image checksum mismatches: "
            + ", ".join(checksum_mismatches[:10])
        )

    write_csv(temporary_output / "index.csv", final_rows, CORE_FIELDS)
    write_csv(temporary_output / "merge_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    exclusion_fields = ["exclusion_reason"] + CORE_FIELDS
    write_csv(
        temporary_output / "excluded_prototype_rows.csv",
        excluded_rows,
        exclusion_fields,
    )

    source_af_lvh_hashes = {
        row["sha256"] for row in combination_rows if row["cls"] == "AF+LVH"
    }
    prototype_af_lvh_hashes = {
        sha256_file(prototype_root / row["rel_path"]) for row in prototype_af_lvh
    }
    exact_af_lvh_image_hash_overlaps = len(
        source_af_lvh_hashes & prototype_af_lvh_hashes
    )

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "image_pipeline/merge_prototype_combinations.py 1.0",
        "purpose": "copy-only merged training dataset",
        "sources": {
            "prototype_1152": {
                "root": str(prototype_root),
                "index": str(prototype_index),
                "index_sha256": sha256_file(prototype_index),
                "rows": len(prototype_rows),
            },
            "af_lvh_stemi_combinations_v1": {
                "root": str(combinations_root),
                "index": str(combinations_index),
                "index_sha256": sha256_file(combinations_index),
                "rows": len(combination_rows_all),
                "records": len(combination_rows),
            },
        },
        "selection": {
            "prototype": (
                "Retain all prototype rows except AF+LVH rows and rows excluded "
                "to prevent patient-level cross-split leakage."
            ),
            "combination_train": (
                "Select render_k=0 (augmentation_number=1), exactly one of four "
                "available images for every training record."
            ),
            "combination_val_test": (
                "Select render_k=0, the only available image for every validation "
                "and test record."
            ),
            "record_namespace": (
                "Combination records and copied filenames use globally unique "
                "record_id values; origin_record preserves the source ECG name "
                "in merge_manifest.csv."
            ),
        },
        "deduplication": {
            "key": "record_id",
            "prototype_af_lvh_removed": len(prototype_af_lvh),
            "prototype_af_lvh_matched_in_newer_source": (
                len(prototype_af_lvh) - len(unmatched_af_lvh)
            ),
            "af_lvh_matches_with_same_split": af_lvh_same_split,
            "af_lvh_matches_with_changed_split": len(prototype_af_lvh)
            - af_lvh_same_split,
            "exact_af_lvh_sha256_image_overlaps": exact_af_lvh_image_hash_overlaps,
            "note": (
                "Record-level duplicates were removed even when render styling "
                "made the JPEG bytes different."
            ),
        },
        "split_integrity": {
            "conflicting_patient_count_before_exclusion": len(conflicting_patients),
            "prototype_rows_excluded_for_patient_split_integrity": len(
                patient_conflict_rows
            ),
            "conflicting_patient_ids": conflicting_patients,
            "patients_spanning_splits_after_merge": 0,
            "records_spanning_splits_after_merge": 0,
        },
        "counts": {
            "final_images": len(final_rows),
            "final_by_split": counts_by_split(final_rows),
            "final_by_split_and_class": counts_by_split_and_class(final_rows),
            "positive_labels_by_split": positives_by_split(final_rows),
            "retained_prototype_images": len(retained_prototype),
            "selected_combination_images": len(combination_rows),
            "excluded_prototype_images": len(excluded_rows),
        },
        "validation": {
            "source_train_variants_per_record": 4,
            "source_val_test_variants_per_record": 1,
            "selected_variants_per_record": 1,
            "record_id_unique": True,
            "record_unique": True,
            "rel_path_unique": True,
            "normal_pathology_cooccurrences": 0,
            "combination_copy_sha256_mismatches": 0,
        },
        "source_datasets_modified": False,
    }
    (temporary_output / "MERGE_PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    readme = f"""# CardioSentry merged prototype and combination dataset

This copy-only dataset combines `prototype_1152` with the exact AF/LVH/STEMI
combination dataset from run `{combination_rows[0]['dataset_run_id']}`.

## Selection

- All {len(prototype_af_lvh):,} AF+LVH rows from `prototype_1152` were replaced
  by their newer matching records, using stable `record_id` values.
- The combination training split keeps `render_k=0`, one of the four available
  augmentations for each record.
- Combination validation and test records are complete; each has only
  `render_k=0`.
- {len(patient_conflict_rows):,} additional prototype rows from
  {len(conflicting_patients):,} patients were excluded to prevent a patient
  from spanning train, validation, and test after the merge.
- Neither source dataset was modified.

## Contents

```
{output_root.name}/
├── index.csv
├── merge_manifest.csv
├── excluded_prototype_rows.csv
├── MERGE_PROVENANCE.json
├── checksums.sha256
└── images/{{train,val,test}}/
```

`index.csv` is training-compatible with `prototype_1152`. The additional
`merge_manifest.csv` preserves each row's source dataset, source path, original
ECG record name, and available source checksum.

## Counts

| split | images / records |
| --- | ---: |
| train | {counts_by_split(final_rows)['train']:,} |
| val | {counts_by_split(final_rows)['val']:,} |
| test | {counts_by_split(final_rows)['test']:,} |
| total | {len(final_rows):,} |

Every final record has exactly one image. Stable `record_id`, output `record`,
and `rel_path` values are unique. No patient spans splits.

See `MERGE_PROVENANCE.json` for label counts, duplicate analysis, source hashes,
and the full selection policy. See `excluded_prototype_rows.csv` for every row
left out of the new copy and its reason.
"""
    (temporary_output / "README.md").write_text(readme, encoding="utf-8")

    print("writing checksums.sha256")
    checksum_targets = sorted(
        path
        for path in temporary_output.rglob("*")
        if path.is_file() and path.name != "checksums.sha256"
    )
    with (temporary_output / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for index, path in enumerate(checksum_targets, 1):
            relative = path.relative_to(temporary_output).as_posix()
            handle.write(f"{sha256_file(path)}  {relative}\n")
            if index % 1000 == 0 or index == len(checksum_targets):
                print(f"  hashed {index:,}/{len(checksum_targets):,} files", flush=True)

    os.replace(temporary_output, output_root)
    print(f"dataset complete: {output_root}")
    print(f"creating archive: {archive_path}")
    create_archive(output_root, archive_path)
    archive_digest = sha256_file(archive_path)
    archive_checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    archive_checksum_path.write_text(
        f"{archive_digest}  {archive_path.name}\n", encoding="utf-8"
    )
    print(f"archive complete: {archive_path}")
    print(f"archive SHA-256: {archive_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
