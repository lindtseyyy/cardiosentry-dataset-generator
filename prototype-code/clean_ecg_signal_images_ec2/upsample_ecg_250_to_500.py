#!/usr/bin/env python3
"""Create separate 500 Hz WFDB copies of 250 Hz ECG records.

The conversion is deliberately conservative: original signed 16-bit ADC
samples occupy every even output index unchanged; each odd index is the
rounded midpoint of its two neighbors. The final odd sample repeats the last
source sample because no later measurement exists. There is no filtering,
normalization, smoothing, or extrapolation. No new extrema can be created.
Interpolation adds timestamps but cannot recover detail above the source's
125 Hz Nyquist frequency.

Outputs contain only converted 250 Hz records and never overwrite the source
ECG tree. Run again after companion-file downloads finish to process records
that were previously missing their signal file.

Dry run:
    python upsample_ecg_250_to_500.py
Convert all available 250 Hz records:
    python upsample_ecg_250_to_500.py --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "dataset-files/HEEDB/extracted/ECG"
DEFAULT_OUTPUT = REPO_ROOT / "dataset-files/HEEDB/extracted/ECG_500Hz"
DEFAULT_REPORT = REPO_ROOT / "dataset-files/HEEDB/extracted/ecg_250_to_500_report.csv"
LEADS = {"I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"}


class InvalidRecord(ValueError):
    """A source header or signal cannot be converted safely."""


@dataclass(frozen=True)
class Header:
    path: Path
    first_fields: tuple[str, ...]
    lead_fields: tuple[tuple[str, ...], ...]
    comments: tuple[str, ...]
    signal_path: Path
    signal_format: str
    samples: int


def sampling_rate_from_first_line(path: Path) -> float:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        parts = handle.readline().split()
    if len(parts) < 4:
        raise InvalidRecord("Incomplete WFDB record line")
    try:
        return float(parts[2].split("/", 1)[0].split("(", 1)[0])
    except ValueError as exc:
        raise InvalidRecord("Invalid sampling frequency") from exc


def parse_header(path: Path, input_root: Path) -> Header:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise InvalidRecord("Empty WFDB header")
    first = lines[0].split()
    if len(first) < 4 or "/" in first[0] or first[0] != path.stem:
        raise InvalidRecord("Unsupported or malformed WFDB record line")
    try:
        nleads = int(first[1])
        fs = float(first[2].split("/", 1)[0])
        samples = int(first[3])
    except ValueError as exc:
        raise InvalidRecord("Invalid lead count, frequency, or sample count") from exc
    if nleads != 12 or fs != 250 or samples < 2:
        raise InvalidRecord(f"Need 12 leads at 250 Hz; found {nleads} leads, {fs} Hz, {samples} samples")

    signal_lines: list[tuple[str, ...]] = []
    comments: list[str] = []
    for line in lines[1:]:
        if line.startswith("#"):
            comments.append(line)
        elif line.strip():
            signal_lines.append(tuple(line.split()))
    if len(signal_lines) != 12 or any(len(row) != 9 for row in signal_lines):
        raise InvalidRecord("Expected exactly twelve nine-field WFDB signal lines")
    names = [row[8] for row in signal_lines]
    if len(set(names)) != 12 or set(names) != LEADS:
        raise InvalidRecord("Missing, duplicate, or unknown ECG leads")
    filenames = {row[0] for row in signal_lines}
    formats = {row[1] for row in signal_lines}
    if len(filenames) != 1 or len(formats) != 1:
        raise InvalidRecord("Multiple signal files or mixed formats are unsupported")
    filename = next(iter(filenames))
    signal_format = next(iter(formats))
    if filename != Path(filename).name or filename in (".", ".."):
        raise InvalidRecord("Unsafe signal filename")
    if signal_format == "16":
        if not filename.lower().endswith(".dat"):
            raise InvalidRecord("WFDB format 16 must reference .dat in this dataset")
    elif signal_format in {"16+24", "16x1+24"}:
        if not filename.lower().endswith(".mat"):
            raise InvalidRecord("WFDB MAT format must reference .mat in this dataset")
    else:
        raise InvalidRecord(f"Unsupported WFDB signal format: {signal_format}")
    for row in signal_lines:
        gain_field = row[2]
        if not gain_field.endswith("/mV"):
            raise InvalidRecord(f"Expected mV units, found {gain_field}")
        try:
            gain = float(gain_field.split("(", 1)[0].split("/", 1)[0])
            adc_resolution = int(row[3])
            adc_zero = int(row[4])
        except ValueError as exc:
            raise InvalidRecord("Invalid gain, ADC resolution, or ADC zero") from exc
        if not np.isfinite(gain) or gain <= 0 or adc_resolution != 16:
            raise InvalidRecord("Invalid gain or unsupported ADC resolution")
        if "(" in gain_field:
            baseline = gain_field.split("(", 1)[1].split(")", 1)[0]
            try:
                int(baseline)
            except ValueError as exc:
                raise InvalidRecord("Invalid ADC baseline") from exc

    signal_path = path.parent / filename
    if not signal_path.resolve(strict=False).is_relative_to(input_root.resolve()):
        raise InvalidRecord("Signal path escapes input directory")
    return Header(path, tuple(first), tuple(signal_lines), tuple(comments),
                  signal_path, signal_format, samples)


def read_source(header: Header) -> np.ndarray:
    path = header.signal_path
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty companion: {path}")
    if header.signal_format == "16":
        expected_bytes = 12 * header.samples * 2
        if path.stat().st_size != expected_bytes:
            raise InvalidRecord(f".dat length {path.stat().st_size} != {expected_bytes} bytes")
        raw = np.fromfile(path, dtype="<i2").reshape(header.samples, 12).T
    else:
        try:
            raw = np.asarray(loadmat(path, variable_names=["val"])["val"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise InvalidRecord(f"Cannot read MAT val: {exc}") from exc
        if raw.shape == (header.samples, 12):
            raw = raw.T
        if raw.shape != (12, header.samples) or raw.dtype.kind != "i":
            raise InvalidRecord(f"Unexpected MAT val shape/type: {raw.shape}/{raw.dtype}")
        if int(raw.min()) < -32768 or int(raw.max()) > 32767:
            raise InvalidRecord("MAT digital samples exceed signed 16-bit range")
    return np.ascontiguousarray(raw, dtype="<i2")


def upsample_exact(raw: np.ndarray) -> np.ndarray:
    """Double the rate while preserving all measured digital samples exactly."""
    if raw.ndim != 2 or raw.shape[0] != 12 or raw.shape[1] < 2 or raw.dtype != np.dtype("<i2"):
        raise InvalidRecord("Expected a 12 x N signed int16 source array")
    output = np.empty((12, 2 * raw.shape[1]), dtype="<i2")
    output[:, ::2] = raw
    neighbor_sums = raw[:, :-1].astype(np.int32) + raw[:, 1:].astype(np.int32)
    output[:, 1:-1:2] = np.rint(neighbor_sums / 2).astype("<i2")
    output[:, -1] = raw[:, -1]  # 2 ms after the last measured point: hold, do not extrapolate.

    if not np.array_equal(output[:, ::2], raw):
        raise AssertionError("Original sample preservation failed")
    lower = np.minimum(raw[:, :-1], raw[:, 1:])
    upper = np.maximum(raw[:, :-1], raw[:, 1:])
    inserted = output[:, 1:-1:2]
    if np.any(inserted < lower) or np.any(inserted > upper):
        raise AssertionError("Interpolation created a new local extremum")
    if not (np.array_equal(output.min(axis=1), raw.min(axis=1)) and
            np.array_equal(output.max(axis=1), raw.max(axis=1))):
        raise AssertionError("Per-lead amplitude range changed")
    return output


def signed_checksum(samples: np.ndarray) -> int:
    """WFDB signed 16-bit sum of all samples in one lead."""
    return (int(np.sum(samples, dtype=np.int64)) + 32768) % 65536 - 32768


def output_header_text(header: Header, output: np.ndarray) -> str:
    record = list(header.first_fields)
    record[2] = "500"
    record[3] = str(output.shape[1])
    lines = [" ".join(record)]
    for index, source in enumerate(header.lead_fields):
        lines.append(" ".join((
            f"{header.path.stem}.dat", "16", source[2], source[3], source[4],
            str(int(output[index, 0])), str(signed_checksum(output[index])), "0", source[8],
        )))
    lines.extend(header.comments)
    lines.append("# Upsampling: 250 to 500 Hz by midpoint interpolation; original samples unchanged")
    lines.append("# Final interpolated sample: held equal to last source sample")
    return "\n".join(lines) + "\n"


def verify_output(path: Path, expected: np.ndarray, source: Header) -> bool:
    header_path = path
    data_path = path.with_suffix(".dat")
    if not header_path.is_file() or not data_path.is_file():
        return False
    if data_path.stat().st_size != expected.size * 2:
        return False
    try:
        if header_path.read_text(encoding="utf-8") != output_header_text(source, expected):
            return False
        actual = np.fromfile(data_path, dtype="<i2").reshape(expected.shape[1], 12).T
        return np.array_equal(actual, expected)
    except (OSError, IndexError, ValueError):
        return False


def convert_record(source_header: Path, input_root: Path, output_root: Path) -> tuple[str, int, str]:
    header = parse_header(source_header, input_root)
    raw = read_source(header)
    output = upsample_exact(raw)
    relative = source_header.relative_to(input_root)
    destination_header = output_root / relative
    destination_data = destination_header.with_suffix(".dat")
    if verify_output(destination_header, output, header):
        return "verified_existing", 0, hashlib.sha256(raw.tobytes()).hexdigest()

    destination_header.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_data = destination_data.with_name(f".{destination_data.name}.part-{token}")
    temporary_header = destination_header.with_name(f".{destination_header.name}.part-{token}")
    try:
        interleaved = np.ascontiguousarray(output.T, dtype="<i2")
        with temporary_data.open("xb") as handle:
            interleaved.tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary_data.stat().st_size != output.size * 2:
            raise OSError("Staged output data length mismatch")
        staged = np.fromfile(temporary_data, dtype="<i2").reshape(output.shape[1], 12).T
        if not np.array_equal(staged, output) or not np.array_equal(staged[:, ::2], raw):
            raise OSError("Staged output differs from original or interpolated samples")
        with temporary_header.open("x", encoding="utf-8") as handle:
            handle.write(output_header_text(header, output))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_data, destination_data)
        os.replace(temporary_header, destination_header)  # Header is published last.
        if not verify_output(destination_header, output, header):
            raise OSError("Published output verification failed")
    finally:
        temporary_data.unlink(missing_ok=True)
        temporary_header.unlink(missing_ok=True)
    return "converted", output.size * 2, hashlib.sha256(raw.tobytes()).hexdigest()


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=positive_int, help="Process only the first N 250 Hz records")
    parser.add_argument("--progress-every", type=positive_int, default=100)
    parser.add_argument("--execute", action="store_true", help="Write converted files; default is a dry run")
    args = parser.parse_args(argv)
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    if not input_root.is_dir():
        parser.error(f"Input ECG tree does not exist: {input_root}")
    if (input_root == output_root or input_root.is_relative_to(output_root)
            or output_root.is_relative_to(input_root)):
        parser.error("Output must be a separate tree outside the source ECG directory")

    selected: list[Path] = []
    scan_errors: list[tuple[str, str]] = []
    for index, path in enumerate(input_root.rglob("*.hea"), start=1):
        try:
            if sampling_rate_from_first_line(path) == 250:
                selected.append(path)
        except (OSError, InvalidRecord) as exc:
            scan_errors.append((str(path), str(exc)))
        if index % 25_000 == 0:
            print(f"Scanned {index:,} headers...", file=sys.stderr, flush=True)
    selected.sort()
    total_250_hz = len(selected)
    if args.limit:
        selected = selected[:args.limit]
    print(f"Found {total_250_hz:,} 250 Hz records; header scan errors: {len(scan_errors):,}")
    if args.limit:
        print(f"Selected first {len(selected):,} records (--limit {args.limit:,}).")
    if not args.execute:
        print(f"Input: {input_root}\nOutput: {output_root}")
        print("Dry run only. Add --execute to convert available records.")
        return 1 if scan_errors else 0

    output_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    counts = {"converted": 0, "verified_existing": 0, "missing_companion": 0, "failed": 0}
    fieldnames = ["source_header", "output_header", "status", "output_bytes", "source_sha256", "error"]
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, path in enumerate(selected, start=1):
            relative = path.relative_to(input_root)
            try:
                status, byte_count, digest = convert_record(path, input_root, output_root)
                error = ""
            except FileNotFoundError as exc:
                status, byte_count, digest, error = "missing_companion", 0, "", str(exc)
            except Exception as exc:
                status, byte_count, digest = "failed", 0, ""
                error = f"{type(exc).__name__}: {exc}"
            counts[status] += 1
            writer.writerow({"source_header": str(path),
                             "output_header": str(output_root / relative),
                             "status": status, "output_bytes": byte_count,
                             "source_sha256": digest, "error": error})
            if index % args.progress_every == 0 or index == len(selected):
                handle.flush()
                print(f"{index:,}/{len(selected):,}: converted={counts['converted']:,}, "
                      f"verified_existing={counts['verified_existing']:,}, "
                      f"missing={counts['missing_companion']:,}, failed={counts['failed']:,}",
                      flush=True)
    print(f"Output: {output_root}\nReport: {args.report}")
    return 1 if counts["missing_companion"] or counts["failed"] or scan_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
