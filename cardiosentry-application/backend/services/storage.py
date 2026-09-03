"""Capture directory storage: naming, atomic JSON, listing.

The filesystem + JSON is the source of truth (plan §2.2). Every JSON write goes
through a temp file + `os.replace` so a crash mid-write can never leave a
corrupt record; capture directories are named timestamp + 6 random hex so they
are collision-proof, sortable, and browseable through date shards.
"""
from __future__ import annotations

import json
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

from backend.schemas.capture import CaptureRecord
from backend.settings import settings

CAPTURE_ID_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{6})_([0-9a-f]{6})$")

# The only file names /api/captures/{id}/files/{name} will ever serve (§19.2).
FILE_WHITELIST = ("original.jpg", "preview.jpg", "overlay.jpg",
                  "corrected.png", "model_input.png")

# Explain runs write one PNG pair PER LABEL, so their names cannot be a fixed
# whitelist the way the stage files are. They are constrained by shape instead:
# <method>_<LABEL>_<heatmap|overlay>.png, plus the optional raw array bundle.
# Every serve still re-resolves the path and asserts containment in the run dir
# (§19.2), so the pattern is a first gate and not the only one.
EXPLAIN_ID_RE = re.compile(r"^xai_\d{8}_\d{6}_\d{3}_[A-Za-z0-9_.-]+$")
EXPLAIN_FILE_RE = re.compile(
    r"^(?:[a-z_]+_[A-Za-z0-9_.-]+_(?:heatmap|overlay)\.png|record\.json|raw\.npz)$")


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_capture_dir() -> tuple[Path, str]:
    now = datetime.now(timezone.utc)
    date_shard = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    hex6 = secrets.token_hex(3)
    capture_id = f"{stamp}_{hex6}"
    d = settings.data_dir / date_shard / capture_id
    d.mkdir(parents=True, exist_ok=False)
    return d, capture_id


def capture_dir(capture_id: str) -> Path:
    """Resolve a capture id to its directory, refusing anything malformed."""
    m = CAPTURE_ID_RE.fullmatch(capture_id)
    if not m:
        raise KeyError(f"malformed capture_id: {capture_id!r}")
    date_shard = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    d = settings.data_dir / date_shard / capture_id
    if not d.is_dir():
        raise KeyError(capture_id)
    return d


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_capture_json(d: Path) -> dict:
    return json.loads((d / "capture.json").read_text(encoding="utf-8"))


def save_capture_record(d: Path, record: CaptureRecord) -> None:
    atomic_write_json(d / "capture.json", record.model_dump(mode="json"))


def iter_captures(descending: bool = True) -> list[Path]:
    """All capture dirs, newest first. Cheap at the intended scale (≤ few k)."""
    if not settings.data_dir.is_dir():
        return []
    dirs = [p for p in settings.data_dir.glob("*/[0-9]*") if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name, reverse=descending)


def free_space_gb() -> float:
    try:
        return shutil.disk_usage(settings.data_dir).free / 1024**3
    except OSError:
        return float("inf")


def trash_capture(capture_id: str) -> Path:
    """Move a capture to captures/_trash/ rather than unlinking (§17.8)."""
    d = capture_dir(capture_id)
    trash = settings.data_dir / "_trash"
    trash.mkdir(parents=True, exist_ok=True)
    dst = trash / d.parent.name / d.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    d.rename(dst)
    return dst


def explain_dir(capture_id: str, explain_id: str, create: bool = False) -> Path:
    """Resolve one explain run's directory, refusing a malformed id.

    The id is pattern-checked before it ever reaches the filesystem, so a
    traversal attempt fails on the regex rather than on a path comparison.
    """
    if not EXPLAIN_ID_RE.fullmatch(explain_id):
        raise KeyError(f"malformed explain_id: {explain_id!r}")
    d = capture_dir(capture_id) / "explain" / explain_id
    if create:
        d.mkdir(parents=True, exist_ok=False)
        return d
    if not d.is_dir():
        raise KeyError(explain_id)
    return d


def new_explain_dir(capture_id: str, method: str, model_id: str) -> tuple[Path, str]:
    """Create captures/<date>/<id>/explain/<explain_id>/ for one run.

    The id carries the method and the model in the NAME because these runs are
    compared by eye in a directory listing: `xai_20260828_141233_907_gradcam`
    beside `..._integrated_gradients` is the comparison the feature exists for.
    """
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    explain_id = f"xai_{stamp}_{method}_{model_id}"
    return explain_dir(capture_id, explain_id, create=True), explain_id


def iter_explain_runs(capture_id: str) -> list[Path]:
    """This capture's explain run dirs, newest first."""
    try:
        d = capture_dir(capture_id) / "explain"
    except KeyError:
        return []
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir()
                   if p.is_dir() and EXPLAIN_ID_RE.fullmatch(p.name)),
                  key=lambda p: p.name, reverse=True)


def app_env_record() -> dict:
    """Versions stamped into every record (§17.9: versions must travel with data)."""
    import sys
    import cv2
    import numpy
    import PIL
    import timm
    import torch
    import torchvision
    try:
        from importlib.metadata import version as _v
        fastapi_v = _v("fastapi")
    except Exception:
        fastapi_v = "?"
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": getattr(timm, "__version__", "?"),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "pillow": PIL.__version__,
        "fastapi": fastapi_v,
    }
