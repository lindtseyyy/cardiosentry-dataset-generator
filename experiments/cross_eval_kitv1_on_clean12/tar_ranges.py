"""Random access to members of a plain (uncompressed) TAR stored in S3.

A plain TAR has no central directory: every member is a 512-byte header
followed by its data, padded to 512 bytes. To find one member without
downloading the archive, the index is built from the headers alone, read
with small S3 byte-range requests.

Walking 100k+ headers one after another would take an hour, so the archive is
cut into segments that are walked in parallel. Each segment finds its first
header by scanning 512-byte-aligned blocks for the "ustar" magic plus a valid
header checksum. Each walker must then land exactly on the next segment's
first header. That proves the whole chain of headers was followed without
gaps, so a false match inside PNG data cannot go unnoticed.

Supported formats: POSIX ustar (with prefix), PAX (long names, float mtimes,
large sizes) and GNU (././@LongLink names, base-256 sizes). Only `read(offset,
length)` and `size` are needed from the reader, so tests can use bytes in
memory instead of S3.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

TAR_BLOCK = 512
HEADER_PROBE_BYTES = 4096          # header + a small PAX/GNU payload + the next header
SYNC_WINDOW_BYTES = 1 << 20        # scanned per request while looking for a segment's first header
EXTENSION_TYPES = (b"x", b"g", b"L", b"K")


def tar_padded(size: int) -> int:
    return (size + TAR_BLOCK - 1) // TAR_BLOCK * TAR_BLOCK


def tar_number(field: bytes) -> int:
    """Octal or GNU base-256 numeric header field (as tarfile.nti)."""
    if field[:1] in (b"\x80", b"\xff"):
        value = 0
        for byte in field[1:]:
            value = (value << 8) + byte
        if field[0] == 0xFF:
            value = -(256 ** (len(field) - 1) - value)
        return value
    text = field.split(b"\0", 1)[0].strip()
    return int(text or b"0", 8)


def tar_header_is_valid(block: bytes) -> bool:
    if len(block) != TAR_BLOCK or block[257:262] != b"ustar":
        return False
    try:
        stored = tar_number(block[148:156])
    except ValueError:
        return False
    unsigned = sum(block[:148]) + 8 * 32 + sum(block[156:])
    if stored == unsigned:
        return True
    signed = sum(b - 256 if b > 127 else b for b in block[:148]) + 8 * 32 + sum(
        b - 256 if b > 127 else b for b in block[156:])
    return stored == signed


def tar_header_fields(block: bytes) -> dict:
    name = block[0:100].split(b"\0", 1)[0]
    if block[257:265] == b"ustar\x0000":           # POSIX: name may continue in the prefix field
        prefix = block[345:500].split(b"\0", 1)[0]
        if prefix:
            name = prefix + b"/" + name
    return {
        "name": name.decode("utf-8", "surrogateescape"),
        "size": tar_number(block[124:136]),
        "type": block[156:157] if block[156:157] not in (b"", b"\0") else b"0",
    }


def parse_pax_records(payload: bytes) -> dict:
    records, pos = {}, 0
    while pos < len(payload) and payload[pos:pos + 1] not in (b"\0", b""):
        space = payload.index(b" ", pos)
        length = int(payload[pos:space])
        key, _, value = payload[space + 1:pos + length - 1].partition(b"=")
        records[key.decode("utf-8")] = value.decode("utf-8", "surrogateescape")
        pos += length
    return records


def _read_exact(reader, offset: int, length: int) -> bytes:
    data = reader.read(offset, length)
    if len(data) != length:
        raise IOError(f"short read at byte {offset}: {len(data)} of {length}")
    return data


def find_first_header(reader, start: int, limit: int, window_bytes: int = SYNC_WINDOW_BYTES) -> int | None:
    """First valid header at a 512-aligned offset in [start, limit)."""
    offset = start - start % TAR_BLOCK
    while offset < limit:
        window = reader.read(offset, min(window_bytes, limit - offset))
        if len(window) < TAR_BLOCK:
            return None
        for position in range(0, len(window) - TAR_BLOCK + 1, TAR_BLOCK):
            if window[position + 257:position + 262] == b"ustar" and tar_header_is_valid(
                    window[position:position + TAR_BLOCK]):
                return offset + position
        offset += len(window) - len(window) % TAR_BLOCK
    return None


def walk_segment(reader, start: int, stop: int | None) -> list[dict]:
    """Follow the header chain from `start` until `stop` (None: end of archive).

    Returns one entry per real member. An extended header (PAX / GNU long
    name) that sits before `stop` is resolved against the member header that
    follows it, even when that header is exactly `stop`. The next walker then
    lists the same member with a truncated name, and merge_segments keeps
    this entry.
    """
    entries, offset = [], start
    attributes, extension_start = {}, None
    cached_offset, cached = -1, b""          # last probe: a PAX header's member header is usually inside it
    while True:
        if stop is not None and offset >= stop and extension_start is None:
            if offset != stop:
                raise RuntimeError(f"header chain overshot the next segment: {offset} != {stop}")
            return entries
        if offset + TAR_BLOCK > reader.size:
            if stop is None and extension_start is None:
                return entries                      # archive without end-of-archive blocks
            raise RuntimeError(f"header chain ran past the end of the archive at {offset}")
        if cached_offset <= offset and offset + TAR_BLOCK <= cached_offset + len(cached):
            probe = cached[offset - cached_offset:]
        else:
            probe = reader.read(offset, min(HEADER_PROBE_BYTES, reader.size - offset))
            cached_offset, cached = offset, probe
        block = probe[:TAR_BLOCK]
        if block == b"\0" * TAR_BLOCK:
            if stop is not None or extension_start is not None:
                raise RuntimeError(f"end-of-archive block inside the header chain at {offset}")
            return entries
        if not tar_header_is_valid(block):
            raise RuntimeError(f"invalid TAR header at byte {offset}")
        header = tar_header_fields(block)
        size, kind = header["size"], header["type"]
        if kind in EXTENSION_TYPES:
            payload = (probe[TAR_BLOCK:TAR_BLOCK + size] if TAR_BLOCK + size <= len(probe)
                       else _read_exact(reader, offset + TAR_BLOCK, size))
            if kind == b"x":
                attributes.update(parse_pax_records(payload))
            elif kind == b"L":
                attributes["path"] = payload.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")
            if extension_start is None:
                extension_start = offset
            offset += TAR_BLOCK + tar_padded(size)
            continue
        if "size" in attributes:
            size = int(attributes["size"])
        entries.append({
            "name": attributes.get("path", header["name"]).rstrip("/") or "/",   # as tarfile: "dir/" -> "dir"
            "type": kind.decode("ascii", "replace"),
            "header_offset": extension_start if extension_start is not None else offset,
            "data_offset": offset + TAR_BLOCK,
            "size": size,
            "extended_name": "path" in attributes,
        })
        attributes, extension_start = {}, None
        offset += TAR_BLOCK + tar_padded(size)
        if stop is not None and entries[-1]["header_offset"] < stop <= entries[-1]["data_offset"] - TAR_BLOCK:
            # This member's extended header started before `stop`; the next
            # walker owns everything from its own first header onwards.
            return entries


def merge_segments(segment_entries: list[list[dict]], archive_size: int) -> list[dict]:
    """Deduplicate by data offset and prove the headers form one gap-free chain.

    A member listed twice was seen by two walkers; the one that started at its
    extended header (smaller header offset) also saw the full name.
    """
    by_data = {}
    for entries in segment_entries:
        for entry in entries:
            kept = by_data.get(entry["data_offset"])
            if kept is None or entry["header_offset"] < kept["header_offset"]:
                by_data[entry["data_offset"]] = entry
    ordered = [by_data[key] for key in sorted(by_data)]
    if not ordered:
        raise RuntimeError("the archive index is empty")
    expected_header = 0
    for entry in ordered:
        if entry["header_offset"] != expected_header:
            raise RuntimeError(
                f"index gap: member {entry['name']!r} header at {entry['header_offset']}, "
                f"expected {expected_header}")
        expected_header = entry["data_offset"] + tar_padded(entry["size"])
        if expected_header > archive_size:
            raise RuntimeError(f"member {entry['name']!r} runs past the end of the archive")
    return ordered


def build_tar_index(reader, segments: int = 128, workers: int = 64, log=print,
                    min_segment_bytes: int = 4 * SYNC_WINDOW_BYTES,
                    sync_window_bytes: int = SYNC_WINDOW_BYTES) -> dict[str, dict]:
    """name -> {data_offset, size, type, header_offset} for every member of the archive."""
    started = time.time()
    size = reader.size
    segments = max(1, min(segments, size // max(min_segment_bytes, TAR_BLOCK)))
    bounds = [(index * size // segments) // TAR_BLOCK * TAR_BLOCK for index in range(segments)] + [size]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        starts = list(pool.map(
            lambda k: 0 if k == 0 else find_first_header(reader, bounds[k], bounds[k + 1], sync_window_bytes),
            range(segments)))
    starts = sorted({start for start in starts if start is not None})
    if not starts or starts[0] != 0:
        raise RuntimeError("no TAR header at byte 0: not a plain TAR archive")
    stops = starts[1:] + [None]
    done = [0]
    lock = threading.Lock()

    def run(pair):
        entries = walk_segment(reader, *pair)
        with lock:
            done[0] += 1
            if log and (done[0] % max(1, len(starts) // 8) == 0 or done[0] == len(starts)):
                log(f"  index: {done[0]}/{len(starts)} segments walked ({time.time() - started:.0f}s)")
        return entries

    with ThreadPoolExecutor(max_workers=workers) as pool:
        segment_entries = list(pool.map(run, zip(starts, stops)))
    ordered = merge_segments(segment_entries, size)
    index = {}
    for entry in ordered:
        if entry["name"] in index and entry["type"] in ("0", "7"):
            raise RuntimeError(f"duplicate archive member {entry['name']!r}")
        index[entry["name"]] = {key: entry[key] for key in ("data_offset", "size", "type", "header_offset")}
    if log:
        log(f"  index: {len(index):,} members in {time.time() - started:.0f}s")
    return index


def read_member(reader, index: dict, name: str) -> bytes:
    entry = index[name]
    return _read_exact(reader, entry["data_offset"], entry["size"]) if entry["size"] else b""
