"""Ingest: validate, EXIF-orient, hash, and store the original upload.

Rules (plan §7.3, §16):
- Magic bytes + PIL decode, never trust Content-Type.
- Byte cap + `MAX_IMAGE_PIXELS` decompression-bomb guard.
- The stored `original.jpg` is the uploaded bytes, unmodified, with one
  permitted exception: GPS EXIF is surgically stripped when
  settings.strip_gps is true (§13.4) — recorded as `gps_stripped`.
- EXIF orientation is a real transformation of the pixel grid and is therefore
  *recorded* (`exif_orientation`, `orientation_applied`, `oriented_size`). It
  is applied in memory at decode time; the stored bytes are never rotated.
  Every downstream geometry (detection quads, quality metrics) is expressed in
  the ORIENTED pixel grid, which is why `oriented_size` travels in the record.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from backend.errors import ApiError
from backend.settings import settings

_ACCEPTED_MAGIC = {
    b"\xff\xd8\xff": ("JPEG", "jpg"),
    b"\x89PNG\r\n\x1a\n": ("PNG", "png"),
}
_ACCEPTED_FORMATS = {"jpeg", "jpg", "png"}


@dataclass
class IngestResult:
    path: Path                       # stored original.jpg
    width: int                       # decoded size (pre-orientation)
    height: int
    oriented_width: int              # size of the pixel grid everything downstream uses
    oriented_height: int
    sha256: str                      # of the stored bytes
    uploaded_sha256: str             # of the upload bytes exactly as received
    format: str
    bytes: int
    exif: dict = field(default_factory=dict)
    exif_orientation: int = 1
    exif_gps_present: bool = False
    gps_stripped: bool = False
    orientation_applied: bool = False

    @property
    def oriented_size(self) -> tuple[int, int]:
        return (self.oriented_width, self.oriented_height)


def _load_heif_support() -> None:
    """If pillow-heif is installed, its import registers HEIF with PIL."""
    try:
        import pillow_heif  # noqa: F401
        pillow_heif.register_heif_opener()
        _ACCEPTED_FORMATS.update({"heic", "heif"})
    except ImportError:
        pass


def _magic_check(data: bytes) -> tuple[str, str]:
    for magic, (fmt, ext) in _ACCEPTED_MAGIC.items():
        if data[: len(magic)] == magic:
            return fmt, ext
    # HEIC container: 'ftyp' box at offset 4
    if data[4:8] == b"ftyp":
        for brand in (b"heic", b"heix", b"hevc", b"mif1"):
            if brand in data[8:24]:
                return "HEIC", "heic"
    raise ApiError(415, "UNSUPPORTED_FORMAT",
                   "Unsupported file. Accepted formats: "
                   + ", ".join(sorted(_ACCEPTED_FORMATS)))


def _strip_gps_surgical(data: bytes) -> tuple[bytes, bool]:
    """Remove the GPS IFD from JPEG EXIF without re-encoding pixels (piexif)."""
    try:
        import piexif
    except ImportError:
        raise ApiError(500, "GPS_STRIP_UNAVAILABLE",
                       "strip_gps is enabled but piexif is not installed")
    try:
        exif_dict = piexif.load(data)
    except Exception:
        return data, False            # no parseable EXIF -> nothing to strip
    if "GPS" not in exif_dict:
        return data, False
    del exif_dict["GPS"]
    exif_bytes = piexif.dump(exif_dict)
    try:
        return piexif.insert(exif_bytes, data), True
    except ValueError:
        # Never substitute a blank EXIF block: that would drop the orientation
        # tag and silently unrotate the stored image. Keep the bytes as-is and
        # record that stripping failed (the record is honest either way).
        return data, False


def ingest_image(raw: bytes, declared_filename: str, dst: Path) -> IngestResult:
    """Validate + store `raw` at `dst`; return metadata. Raises ApiError."""
    _load_heif_support()

    if not raw:
        raise ApiError(422, "EMPTY_UPLOAD", "The upload was empty.")
    if len(raw) > settings.max_upload_bytes:
        raise ApiError(413, "IMAGE_TOO_LARGE",
                       f"Image exceeds the {settings.max_upload_bytes // 2**20} MB limit.")

    fmt, ext = _magic_check(raw)
    uploaded_sha = hashlib.sha256(raw).hexdigest()

    # Decompression-bomb guard, then decode.
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            width, height = im.size
            exif_orientation = int(im.getexif().get(0x0112, 1) or 1)
            exif_dict = im.getexif()
            exif = {str(k): str(v) for k, v in exif_dict.items()
                    if k in (0x010F, 0x0110, 0x8827, 0x829A, 0x829D, 0x920A)}
            gps_present = bool(im.getexif().get_ifd(0x8825))
            # useful exif fields, humanised
            exif["make"] = exif_dict.get(0x010F, "")
            exif["model"] = exif_dict.get(0x0110, "")
            exif["iso"] = exif_dict.get(0x8827, "")
            exif["exposure_time"] = exif_dict.get(0x829A, "")
            exif["f_number"] = exif_dict.get(0x829D, "")
            exif["focal_length_mm"] = exif_dict.get(0x920A, "")
            exif["orientation"] = exif_orientation
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(422, "CORRUPT_IMAGE",
                       "The photo didn't upload completely — try again.") from exc

    # PII hygiene: strip GPS from the stored bytes when configured (§13.4).
    stored = raw
    gps_stripped = False
    if settings.strip_gps and gps_present:
        stored, gps_stripped = _strip_gps_surgical(raw)

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(stored)
    tmp.replace(dst)                 # atomic

    orientation_applied = exif_orientation not in (1, 0, None)
    if orientation_applied:
        ow, oh = (height, width) if exif_orientation in (5, 6, 7, 8) else (width, height)
    else:
        ow, oh = width, height

    return IngestResult(
        path=dst, width=width, height=height,
        oriented_width=ow, oriented_height=oh,
        sha256=hashlib.sha256(stored).hexdigest(),
        uploaded_sha256=uploaded_sha,
        format=fmt, bytes=len(stored),
        exif=exif, exif_orientation=exif_orientation,
        exif_gps_present=gps_present, gps_stripped=gps_stripped,
        orientation_applied=orientation_applied,
    )


def decode_oriented_rgb(res: IngestResult) -> np.ndarray:
    """Decode the stored original into an ORIENTED RGB numpy array.

    This is the single named place where EXIF orientation is applied. All
    downstream geometry operates on this grid.
    """
    with Image.open(res.path) as im:
        if res.orientation_applied:
            im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    if arr.shape[1] != res.oriented_width or arr.shape[0] != res.oriented_height:
        raise ApiError(422, "CORRUPT_IMAGE",
                       "Decoded image size did not match the ingest record.")
    return arr


def decode_record_original(capture_dir: Path, record: dict) -> np.ndarray:
    """Re-decode a stored capture's original.jpg into the oriented grid,
    rebuilding the IngestResult from capture.json so API and offline scripts
    share exactly one decode path."""
    orig = record["original"]
    res = IngestResult(
        path=capture_dir / "original.jpg",
        width=orig["width"], height=orig["height"],
        oriented_width=orig["oriented_width"], oriented_height=orig["oriented_height"],
        sha256=orig["sha256"], uploaded_sha256=orig.get("uploaded_sha256", ""),
        format=orig["format"], bytes=orig["bytes"],
        exif=orig.get("exif", {}),
        exif_orientation=orig.get("exif_orientation", 1),
        exif_gps_present=orig.get("exif_gps_present", False),
        gps_stripped=orig.get("gps_stripped", False),
        orientation_applied=orig.get("orientation_applied", False),
    )
    return decode_oriented_rgb(res)
