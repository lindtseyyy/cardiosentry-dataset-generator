#!/usr/bin/env python3
"""Build an independent, all-500-Hz HEEDB ECG tree without changing sources.

For each record in extracted/ECG, use the original pair if its header says
500 Hz, or the corresponding converted pair from extracted/ECG_500Hz if its
header says 250 Hz. Copy into a temporary sibling directory, compare every
copied file byte for byte, then publish extracted/ECG_combined_500Hz.
"""

from __future__ import annotations

import filecmp
import os
import shutil
import sys
from collections import Counter
from pathlib import Path


EXTRACTED = Path(__file__).resolve().parents[2] / "dataset-files/HEEDB/extracted"
ORIGINAL = EXTRACTED / "ECG"
UPSAMPLED = EXTRACTED / "ECG_500Hz"
DESTINATION = EXTRACTED / "ECG_combined_500Hz"
STAGING = EXTRACTED / ".ECG_combined_500Hz.building"


def inspect_pair(header: Path, expected_rate: str) -> Path:
    lines = header.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Empty header: {header}")
    record = lines[0].split()
    if len(record) < 4 or record[0] != header.stem or record[1] != "12" or record[2] != expected_rate:
        raise ValueError(f"Unexpected record line in {header}: {lines[0]}")
    signals = [line.split() for line in lines[1:] if line.strip() and not line.startswith("#")]
    if len(signals) != 12 or any(not signal for signal in signals):
        raise ValueError(f"Expected 12 signal lines in {header}")
    filenames = {signal[0] for signal in signals}
    if len(filenames) != 1:
        raise ValueError(f"Expected one signal file in {header}")
    filename = filenames.pop()
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ValueError(f"Unsafe signal filename in {header}: {filename}")
    signal = header.parent / filename
    if not signal.is_file() or signal.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty signal file: {signal}")
    return signal


def main() -> int:
    if not ORIGINAL.is_dir() or not UPSAMPLED.is_dir():
        raise FileNotFoundError("The original and upsampled ECG folders must both exist")
    if DESTINATION.exists() or STAGING.exists():
        raise FileExistsError(f"Destination or staging folder already exists: {DESTINATION}, {STAGING}")

    # Each entry is (selected header, selected signal, relative header, relative signal).
    selected: list[tuple[Path, Path, Path, Path]] = []
    counts: Counter[str] = Counter()
    required_bytes = 0
    for index, original_header in enumerate(ORIGINAL.rglob("*.hea"), 1):
        with original_header.open(encoding="utf-8") as handle:
            first = handle.readline().split()
        if len(first) < 4:
            raise ValueError(f"Incomplete header: {original_header}")
        rate = first[2]
        if rate not in {"250", "500"}:
            raise ValueError(f"Unsupported sampling rate {rate} in {original_header}")
        original_signal = inspect_pair(original_header, rate)
        relative_header = original_header.relative_to(ORIGINAL)
        selected_header = original_header if rate == "500" else UPSAMPLED / relative_header
        selected_signal = original_signal if rate == "500" else inspect_pair(selected_header, "500")
        relative_signal = selected_signal.relative_to(ORIGINAL if rate == "500" else UPSAMPLED)
        selected.append((selected_header, selected_signal, relative_header, relative_signal))
        counts[rate] += 1
        required_bytes += selected_header.stat().st_size + selected_signal.stat().st_size
        if index % 50_000 == 0:
            print(f"Preflight: {index:,} original records inspected", flush=True)

    if counts["250"] != sum(1 for _ in UPSAMPLED.rglob("*.hea")):
        raise ValueError("Converted record count does not match original 250 Hz record count")
    free_bytes = shutil.disk_usage(EXTRACTED).free
    if free_bytes < required_bytes + 2 * 1024**3:
        raise OSError(f"Insufficient free disk space: need {required_bytes:,} bytes plus 2 GiB margin; have {free_bytes:,}")
    print(f"Preflight passed: {counts['500']:,} original 500 Hz + {counts['250']:,} converted 250 Hz records")
    print(f"Copying {required_bytes / 1024**3:.2f} GiB into {STAGING}", flush=True)

    STAGING.mkdir()
    for index, (header, signal, relative_header, relative_signal) in enumerate(selected, 1):
        destination_header = STAGING / relative_header
        destination_signal = STAGING / relative_signal
        destination_header.parent.mkdir(parents=True, exist_ok=True)
        if destination_header.exists() or destination_signal.exists():
            raise FileExistsError(f"Duplicate destination at {destination_header}")
        shutil.copy2(signal, destination_signal)
        shutil.copy2(header, destination_header)
        if index % 10_000 == 0 or index == len(selected):
            print(f"Copied {index:,}/{len(selected):,} records", flush=True)

    print("Verifying every copied header and signal byte for byte...", flush=True)
    for index, (header, signal, relative_header, relative_signal) in enumerate(selected, 1):
        destination_header = STAGING / relative_header
        destination_signal = STAGING / relative_signal
        if not filecmp.cmp(header, destination_header, shallow=False):
            raise OSError(f"Header copy mismatch: {destination_header}")
        if not filecmp.cmp(signal, destination_signal, shallow=False):
            raise OSError(f"Signal copy mismatch: {destination_signal}")
        if index % 10_000 == 0 or index == len(selected):
            print(f"Verified {index:,}/{len(selected):,} records", flush=True)

    output_headers = sum(1 for _ in STAGING.rglob("*.hea"))
    output_signals = sum(1 for path in STAGING.rglob("*") if path.is_file() and path.suffix in {".mat", ".dat"})
    if output_headers != len(selected) or output_signals != len(selected):
        raise OSError(f"Unexpected final file counts: {output_headers} headers, {output_signals} signals")
    os.rename(STAGING, DESTINATION)
    print(f"Published {DESTINATION}")
    print(f"Total: {len(selected):,} records, all 500 Hz; original 500 Hz: {counts['500']:,}; converted 250 Hz: {counts['250']:,}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Build stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
