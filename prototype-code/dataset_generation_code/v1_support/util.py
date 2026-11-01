"""Small shared helpers: hashing, deterministic RNG, atomic writes, logging."""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def rng_from_parts(seed: int, *parts) -> np.random.Generator:
    """Deterministic per-identity RNG.

    blake2b-keyed (never Python hash(), which is salted per process), so the
    same (seed, parts) yields the same stream on any machine and any run.
    """
    key = "|".join(str(p) for p in parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big") ^ int(seed))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write via tmp+rename so a crash never leaves a half-written manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class V1Error(Exception):
    """Base error: always loud, always carries the offending identities."""


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise V1Error(msg)
