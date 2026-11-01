#!/usr/bin/env python3
"""Download only the HEEDB WFDB records selected in the CardioSentry CSV.

The script streams the CSV, so it does not load the multi-gigabyte manifest into
memory.  For each record it downloads the WFDB header first, parses the signal
filename(s) from that header, and downloads only those referenced objects.

Safety defaults:
  * Running without --execute is a five-record dry run.
  * Existing non-empty files are skipped.
  * Downloads are written to temporary files and atomically renamed.
  * A checkpoint advances only after an entire batch succeeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


DEFAULT_ACCESS_POINT_ARN = (
    "arn:aws:s3:us-east-1:184438910517:accesspoint/"
    "bdsp-credentialed-access-point-1"
)
DEFAULT_REGION = "us-east-1"
DEFAULT_SELECTED_CSV = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "heedb_cardiosentry_ecg_selection_2026-09-27"
    / "heedb_cardiosentry_selected_ecgs.csv"
)
REQUIRED_COLUMNS = {
    "site_subset",
    "FileName",
    "FileID",
    "metadata_match_verified",
    "fileid_matches_filename_basename",
}
EXPECTED_SELECTED_RECORDS = 6_489_196
ESTIMATED_BYTES_PER_RECORD = 121_070
READ_CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class Record:
    row_number: int
    site: str
    file_id: str
    labels: str
    record_key: str


@dataclass(frozen=True)
class RecordResult:
    row_number: int
    site: str
    file_id: str
    header_key: str
    signal_keys: tuple[str, ...]
    downloaded_objects: int
    skipped_objects: int
    downloaded_bytes: int


@dataclass(frozen=True)
class RecordFailure:
    row_number: int
    site: str
    file_id: str
    record_key: str
    error_type: str
    error: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the deduplicated HEEDB ECGs listed in the CardioSentry "
            "selection CSV. The default mode is a dry run."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_SELECTED_CSV,
        help=f"Master selection CSV (default: {DEFAULT_SELECTED_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination root. Required with --execute.",
    )
    parser.add_argument(
        "--access-point",
        default=DEFAULT_ACCESS_POINT_ARN,
        help="S3 access-point ARN or its s3:// URI.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION}).",
    )
    parser.add_argument(
        "--profile",
        help="Optional AWS CLI/shared-credentials profile name.",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=8,
        help="Concurrent ECG records to download (default: 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=256,
        help="Records per checkpoint batch (default: 256).",
    )
    parser.add_argument(
        "--attempts",
        type=positive_int,
        default=5,
        help="Whole-object attempts, in addition to SDK request retries (default: 5).",
    )
    parser.add_argument(
        "--start-row",
        type=positive_int,
        default=1,
        help="One-based CSV data-row number at which to start (default: 1).",
    )
    parser.add_argument(
        "--limit",
        type=nonnegative_int,
        help="Maximum records to process. Omit for all records in execute mode.",
    )
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=1_000,
        help="Report progress after approximately this many records (default: 1000).",
    )
    parser.add_argument(
        "--on-error",
        choices=("stop", "continue"),
        default="stop",
        help=(
            "Stop before advancing the failed batch's checkpoint, or log failures "
            "and continue (default: stop)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download objects again even if a non-empty local file exists.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing checkpoint and use --start-row.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually download. Without this flag the script only previews paths.",
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def normalize_access_point(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith("s3://"):
        normalized = normalized[5:]
    normalized = normalized.rstrip("/")
    if not normalized.startswith("arn:aws:s3:") or ":accesspoint/" not in normalized:
        raise ValueError("--access-point must be an S3 access-point ARN or s3:// ARN URI")
    return normalized


def safe_posix_relative(value: str, field_name: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts:
        raise ValueError(f"{field_name} must be a non-empty relative path: {value!r}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"Unsafe {field_name}: {value!r}")
    return candidate


def record_from_row(row: dict[str, str], row_number: int) -> Record:
    site = (row.get("site_subset") or "").strip()
    if site not in {"I0001", "I0006"}:
        raise ValueError(f"Unsupported site_subset at CSV row {row_number}: {site!r}")

    if (row.get("metadata_match_verified") or "").strip() != "1":
        raise ValueError(f"Unverified metadata path at CSV row {row_number}")
    if (row.get("fileid_matches_filename_basename") or "").strip() != "1":
        raise ValueError(f"Unverified FileID/path match at CSV row {row_number}")

    raw_name = (row.get("FileName") or "").strip().replace("\\", "/")
    if not raw_name:
        raise ValueError(f"Missing FileName at CSV row {row_number}")
    relative_name = raw_name.lstrip("/")
    if relative_name.startswith("WFDB/"):
        relative_name = relative_name[len("WFDB/") :]
    if relative_name.endswith(".hea"):
        relative_name = relative_name[:-4]
    relative_path = safe_posix_relative(relative_name, "FileName")

    file_id = (row.get("FileID") or "").strip()
    if not file_id:
        raise ValueError(f"Missing FileID at CSV row {row_number}")
    if file_id != relative_path.name:
        raise ValueError(
            f"FileID does not match FileName basename at CSV row {row_number}: "
            f"{file_id!r} != {relative_path.name!r}"
        )

    record_key = str(PurePosixPath("ECG", site, "WFDB") / relative_path)
    return Record(
        row_number=row_number,
        site=site,
        file_id=file_id,
        labels=(row.get("cardiosentry_labels") or "").strip(),
        record_key=record_key,
    )


def iter_records(csv_path: Path, start_row: int, limit: int | None) -> Iterator[Record]:
    csv.field_size_limit(max(csv.field_size_limit(), 10 * 1024 * 1024))
    yielded = 0
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Selection CSV is missing columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=1):
            if row_number < start_row:
                continue
            if limit is not None and yielded >= limit:
                return
            yield record_from_row(row, row_number)
            yielded += 1


def batches(records: Iterable[Record], size: int) -> Iterator[list[Record]]:
    batch: list[Record] = []
    for record in records:
        batch.append(record)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def local_path_for_key(output_dir: Path, key: str) -> Path:
    relative = safe_posix_relative(key, "S3 key")
    return output_dir.joinpath(*relative.parts)


def parse_wfdb_signal_references(header_path: Path) -> tuple[PurePosixPath, ...]:
    with header_path.open("r", encoding="utf-8", errors="replace") as handle:
        content_lines = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not content_lines:
        raise ValueError(f"Empty WFDB header: {header_path}")

    record_fields = content_lines[0].split()
    if len(record_fields) < 2:
        raise ValueError(f"Malformed WFDB record line: {content_lines[0]!r}")
    if "/" in record_fields[0]:
        raise ValueError(
            f"Multi-segment WFDB records are not supported: {record_fields[0]!r}"
        )
    try:
        signal_count = int(record_fields[1])
    except ValueError as exc:
        raise ValueError(f"Invalid WFDB signal count: {record_fields[1]!r}") from exc
    if signal_count <= 0:
        raise ValueError(f"Invalid WFDB signal count: {signal_count}")

    signal_lines = content_lines[1 : 1 + signal_count]
    if len(signal_lines) != signal_count:
        raise ValueError(
            f"WFDB header declares {signal_count} signals but has "
            f"{len(signal_lines)} signal lines: {header_path}"
        )

    references: list[PurePosixPath] = []
    seen: set[str] = set()
    for line in signal_lines:
        filename = line.split(maxsplit=1)[0]
        if filename == "~":
            continue
        reference = safe_posix_relative(filename, "WFDB signal reference")
        reference_text = str(reference)
        if reference_text not in seen:
            seen.add(reference_text)
            references.append(reference)
    if not references:
        raise ValueError(f"No signal files referenced by WFDB header: {header_path}")
    return tuple(references)


def import_aws_modules() -> tuple[Any, Any]:
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.config import Config  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is required for execute mode. Use the project environment "
            "(.venv/bin/python) or install boto3 in the Python environment."
        ) from exc
    return boto3, Config


def create_s3_client(args: argparse.Namespace) -> Any:
    boto3, Config = import_aws_modules()
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    config = Config(
        region_name=args.region,
        connect_timeout=15,
        read_timeout=120,
        max_pool_connections=max(16, args.workers * 2),
        retries={"mode": "adaptive", "max_attempts": 10},
        s3={"use_arn_region": True},
    )
    return session.client("s3", config=config)


def download_object_once(
    s3_client: Any,
    access_point_arn: str,
    key: str,
    destination: Path,
    overwrite: bool,
) -> tuple[bool, int]:
    if not overwrite and destination.is_file() and destination.stat().st_size > 0:
        return False, 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.part-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
    )
    body = None
    try:
        response = s3_client.get_object(
            Bucket=access_point_arn,
            Key=key,
        )
        body = response["Body"]
        expected_size = int(response.get("ContentLength", -1))
        written = 0
        with temporary.open("xb") as handle:
            while True:
                chunk = body.read(READ_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
        if expected_size >= 0 and written != expected_size:
            raise OSError(
                f"Downloaded size mismatch for {key}: expected {expected_size}, got {written}"
            )
        os.replace(temporary, destination)
        return True, written
    finally:
        if body is not None:
            body.close()
        temporary.unlink(missing_ok=True)


def download_object(
    s3_client: Any,
    access_point_arn: str,
    key: str,
    destination: Path,
    overwrite: bool,
    attempts: int,
) -> tuple[bool, int]:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return download_object_once(
                s3_client,
                access_point_arn,
                key,
                destination,
                overwrite,
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = min(30.0, 2.0 ** (attempt - 1)) + random.random()
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def download_record(
    record: Record,
    output_dir: Path,
    s3_client: Any,
    access_point_arn: str,
    overwrite: bool,
    attempts: int,
) -> RecordResult:
    header_key = f"{record.record_key}.hea"
    header_path = local_path_for_key(output_dir, header_key)
    downloaded_objects = 0
    skipped_objects = 0
    downloaded_bytes = 0

    header_downloaded, header_bytes = download_object(
        s3_client,
        access_point_arn,
        header_key,
        header_path,
        overwrite=overwrite,
        attempts=attempts,
    )
    downloaded_objects += int(header_downloaded)
    skipped_objects += int(not header_downloaded)
    downloaded_bytes += header_bytes

    try:
        references = parse_wfdb_signal_references(header_path)
    except (OSError, ValueError):
        # A pre-existing non-empty header could still be stale or truncated.
        # Refresh it once before treating the record as a failure.
        if header_downloaded or overwrite:
            raise
        refreshed, refreshed_bytes = download_object(
            s3_client,
            access_point_arn,
            header_key,
            header_path,
            overwrite=True,
            attempts=attempts,
        )
        downloaded_objects += int(refreshed)
        skipped_objects -= 1
        downloaded_bytes += refreshed_bytes
        references = parse_wfdb_signal_references(header_path)

    header_parent_key = PurePosixPath(header_key).parent
    signal_keys: list[str] = []
    for reference in references:
        signal_key = str(header_parent_key / reference)
        signal_path = header_path.parent.joinpath(*reference.parts)
        did_download, byte_count = download_object(
            s3_client,
            access_point_arn,
            signal_key,
            signal_path,
            overwrite=overwrite,
            attempts=attempts,
        )
        downloaded_objects += int(did_download)
        skipped_objects += int(not did_download)
        downloaded_bytes += byte_count
        signal_keys.append(signal_key)

    return RecordResult(
        row_number=record.row_number,
        site=record.site,
        file_id=record.file_id,
        header_key=header_key,
        signal_keys=tuple(signal_keys),
        downloaded_objects=downloaded_objects,
        skipped_objects=skipped_objects,
        downloaded_bytes=downloaded_bytes,
    )


def csv_signature(csv_path: Path) -> dict[str, Any]:
    stat = csv_path.stat()
    return {
        "path": str(csv_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def load_resume_row(
    checkpoint_path: Path,
    signature: dict[str, Any],
    access_point_arn: str,
    region: str,
) -> int | None:
    if not checkpoint_path.is_file():
        return None
    with checkpoint_path.open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    if checkpoint.get("csv") != signature:
        raise ValueError(
            f"Checkpoint belongs to a different or modified CSV: {checkpoint_path}. "
            "Use --no-resume or a different output directory if intentional."
        )
    if checkpoint.get("access_point_arn") != access_point_arn:
        raise ValueError("Checkpoint access-point ARN does not match --access-point")
    if checkpoint.get("region") != region:
        raise ValueError("Checkpoint region does not match --region")
    return int(checkpoint["last_completed_row"]) + 1


def write_checkpoint(
    checkpoint_path: Path,
    signature: dict[str, Any],
    access_point_arn: str,
    region: str,
    last_completed_row: int,
    run_totals: dict[str, int],
) -> None:
    payload = {
        "version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv": signature,
        "access_point_arn": access_point_arn,
        "region": region,
        "last_completed_row": last_completed_row,
        "current_run_totals": run_totals,
    }
    temporary = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, checkpoint_path)


def append_failures(path: Path, failures: list[RecordFailure]) -> None:
    if not failures:
        return
    fieldnames = [
        "timestamp_utc",
        "row_number",
        "site",
        "file_id",
        "record_key",
        "error_type",
        "error",
    ]
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        timestamp = datetime.now(timezone.utc).isoformat()
        for failure in failures:
            writer.writerow({"timestamp_utc": timestamp, **asdict(failure)})


def print_capacity_notice(output_dir: Path, full_run: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    free_gib = usage.free / (1024**3)
    print(f"Destination free space: {free_gib:,.1f} GiB", file=sys.stderr)
    if full_run:
        estimated_gib = EXPECTED_SELECTED_RECORDS * ESTIMATED_BYTES_PER_RECORD / (1024**3)
        print(
            f"Estimated logical waveform size: {estimated_gib:,.1f} GiB for "
            f"{EXPECTED_SELECTED_RECORDS:,} records; filesystem overhead is additional.",
            file=sys.stderr,
        )
        if free_gib < 800:
            print(
                "WARNING: Less than 800 GiB is free. The complete selection may not fit.",
                file=sys.stderr,
            )
        statvfs = os.statvfs(output_dir)
        if statvfs.f_favail:
            print(f"Destination free inodes: {statvfs.f_favail:,}", file=sys.stderr)
            if statvfs.f_favail < EXPECTED_SELECTED_RECORDS * 2:
                print(
                    "WARNING: Fewer than two free inodes per selected ECG are available.",
                    file=sys.stderr,
                )


def preview(args: argparse.Namespace, access_point_arn: str) -> int:
    preview_limit = 5 if args.limit is None else args.limit
    if preview_limit == 0:
        print("Dry run requested with --limit 0; no records to preview.")
        return 0
    output_dir = args.output_dir or Path("heedb-selected-ecgs")
    count = 0
    for record in iter_records(args.csv, args.start_row, preview_limit):
        header_key = f"{record.record_key}.hea"
        header_path = local_path_for_key(output_dir, header_key)
        print(f"CSV row:       {record.row_number}")
        print(f"Site/FileID:   {record.site} / {record.file_id}")
        if record.labels:
            print(f"Labels:        {record.labels}")
        print(f"Header source: s3://{access_point_arn}/{header_key}")
        print(f"Header output: {header_path}")
        print("Signal source: discovered from the downloaded .hea file")
        print()
        count += 1
    print(
        f"Dry run only: previewed {count} record(s); no directories or files were created."
    )
    print("Add --execute and --output-dir PATH to download.")
    return 0


def execute(args: argparse.Namespace, access_point_arn: str) -> int:
    assert args.output_dir is not None
    output_dir = args.output_dir.resolve()
    print_capacity_notice(output_dir, full_run=args.limit is None)

    signature = csv_signature(args.csv)
    checkpoint_path = output_dir / ".heedb-download-state.json"
    failure_path = output_dir / "heedb-download-failures.csv"
    start_row = args.start_row
    if not args.no_resume:
        resume_row = load_resume_row(
            checkpoint_path,
            signature,
            access_point_arn,
            args.region,
        )
        if resume_row is not None:
            start_row = max(start_row, resume_row)
            print(f"Resuming at CSV data row {start_row:,}.", file=sys.stderr)

    s3_client = create_s3_client(args)
    records = iter_records(args.csv, start_row, args.limit)
    totals = {
        "records_completed": 0,
        "records_failed": 0,
        "objects_downloaded": 0,
        "objects_skipped": 0,
        "bytes_downloaded": 0,
    }
    began = time.monotonic()
    next_progress = args.progress_every
    last_checkpoint_row = start_row - 1

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for batch in batches(records, args.batch_size):
                future_to_record = {
                    executor.submit(
                        download_record,
                        record,
                        output_dir,
                        s3_client,
                        access_point_arn,
                        args.overwrite,
                        args.attempts,
                    ): record
                    for record in batch
                }
                successes: list[RecordResult] = []
                failures: list[RecordFailure] = []
                for future in as_completed(future_to_record):
                    record = future_to_record[future]
                    try:
                        successes.append(future.result())
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except Exception as exc:
                        failures.append(
                            RecordFailure(
                                row_number=record.row_number,
                                site=record.site,
                                file_id=record.file_id,
                                record_key=record.record_key,
                                error_type=type(exc).__name__,
                                error=str(exc),
                            )
                        )

                totals["records_completed"] += len(successes)
                totals["records_failed"] += len(failures)
                totals["objects_downloaded"] += sum(
                    result.downloaded_objects for result in successes
                )
                totals["objects_skipped"] += sum(
                    result.skipped_objects for result in successes
                )
                totals["bytes_downloaded"] += sum(
                    result.downloaded_bytes for result in successes
                )
                append_failures(failure_path, failures)

                if failures and args.on_error == "stop":
                    first = failures[0]
                    print(
                        f"ERROR: {len(failures)} record(s) failed in the current batch. "
                        f"First failure: CSV row {first.row_number}, {first.error_type}: "
                        f"{first.error}",
                        file=sys.stderr,
                    )
                    print(
                        "The checkpoint was not advanced for this batch. Rerunning will "
                        "skip its completed files and retry the failure(s).",
                        file=sys.stderr,
                    )
                    return 2

                last_checkpoint_row = batch[-1].row_number
                write_checkpoint(
                    checkpoint_path,
                    signature,
                    access_point_arn,
                    args.region,
                    last_checkpoint_row,
                    totals,
                )

                processed = totals["records_completed"] + totals["records_failed"]
                if processed >= next_progress:
                    elapsed = max(time.monotonic() - began, 0.001)
                    mib = totals["bytes_downloaded"] / (1024**2)
                    print(
                        f"Processed {processed:,} records through CSV row "
                        f"{last_checkpoint_row:,}; downloaded "
                        f"{totals['objects_downloaded']:,} objects ({mib:,.1f} MiB), "
                        f"skipped {totals['objects_skipped']:,}, failed "
                        f"{totals['records_failed']:,}; {processed / elapsed:,.2f} records/s.",
                        file=sys.stderr,
                    )
                    while next_progress <= processed:
                        next_progress += args.progress_every
    except KeyboardInterrupt:
        print(
            "Interrupted. The last fully completed batch remains checkpointed; rerun "
            "the same command to resume.",
            file=sys.stderr,
        )
        return 130

    processed = totals["records_completed"] + totals["records_failed"]
    print(
        f"Finished: {processed:,} records processed, "
        f"{totals['records_completed']:,} completed, "
        f"{totals['records_failed']:,} failed, "
        f"{totals['objects_downloaded']:,} objects downloaded, "
        f"{totals['objects_skipped']:,} existing objects skipped.",
        file=sys.stderr,
    )
    if totals["records_failed"]:
        print(f"Failures were logged to {failure_path}.", file=sys.stderr)
        return 3
    if processed == 0:
        print("No records remained to process.", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.csv = args.csv.expanduser().resolve()
    if not args.csv.is_file():
        print(f"ERROR: Selection CSV not found: {args.csv}", file=sys.stderr)
        return 2
    if args.execute and args.output_dir is None:
        print("ERROR: --output-dir is required with --execute.", file=sys.stderr)
        return 2
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser()
    try:
        access_point_arn = normalize_access_point(args.access_point)
        if args.execute:
            return execute(args, access_point_arn)
        return preview(args, access_point_arn)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
