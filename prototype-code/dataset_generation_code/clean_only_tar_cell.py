# Scratch: clean-only TAR for fast prototyping (train/val/test clean images).
# Run AFTER interrupting §17. Same S3 folder as the full archives.
import csv as _csv
import importlib as _il
import inspect as _insp
import shutil as _sh
from pathlib import Path as _P

from v1_support import package as _PKG
from v1_support import s3util as _S3U
_il.reload(_S3U)
assert "v1.17" in _insp.getsource(_S3U), \
    "copy the v1.17 s3util.py to EC2 first (no-checksum fallback required)"

_CLEAN_SUBSETS = ["train", "val_clean", "test_clean"]
_tar_name = "cardiosentry_v1_10s_768x1024_clean_only.tar"

# --- stage a clean-only mirror (hardlinks: instant, no extra disk) ---
_stg = _P(cfg.work_dir) / "clean_pkg"
if _stg.exists():
    _sh.rmtree(_stg)
_n_img = 0
for _s in _CLEAN_SUBSETS:
    (_stg / _s).mkdir(parents=True, exist_ok=True)
    with open(_P(cfg.dataset_dir) / _s / "metadata.csv", newline="") as _fh:
        _rd = _csv.DictReader(_fh)
        _fields = list(_rd.fieldnames or [])
        assert "variant_id" in _fields, f"no variant_id column in {_s}"
        _clean = [_r for _r in _rd if _r["variant_id"] == "clean"]
    assert _clean, f"no clean rows in {_s}/metadata.csv"
    with open(_stg / _s / "metadata.csv", "w", newline="") as _fh:
        _w = _csv.DictWriter(_fh, fieldnames=_fields)
        _w.writeheader()
        _w.writerows(_clean)
    for _r in _clean:
        _dst = _stg / _r["image_path"]
        _dst.parent.mkdir(parents=True, exist_ok=True)
        _dst.hardlink_to(_P(cfg.dataset_dir) / _r["image_path"])
        _n_img += 1
    print(f"{_s}: {len(_clean)} clean images staged")
with open(_P(cfg.dataset_dir) / "full_metadata.csv", newline="") as _fh:
    _rd = _csv.DictReader(_fh)
    _fields = list(_rd.fieldnames or [])
    _clean_full = [_r for _r in _rd if _r["variant_id"] == "clean"]
assert len(_clean_full) == _n_img, (len(_clean_full), _n_img)
with open(_stg / "full_metadata.csv", "w", newline="") as _fh:
    _w = _csv.DictWriter(_fh, fieldnames=_fields)
    _w.writeheader()
    _w.writerows(_clean_full)
for _f in ("generation_config.json", "grid_palette_by_label.csv",
           "grid_palette_audit.json", "font_by_label.csv",
           "distortion_report.json", "failures.csv"):
    _sh.copy(_P(cfg.dataset_dir) / _f, _stg / _f)
_sh.copytree(_P(cfg.dataset_dir) / "splits", _stg / "splits")
print(f"staged {_n_img} clean images + metadata")

# --- tar + verify listing ---
_top = _CLEAN_SUBSETS + ["splits", "full_metadata.csv",
        "generation_config.json", "grid_palette_by_label.csv",
        "grid_palette_audit.json", "font_by_label.csv",
        "distortion_report.json", "failures.csv"]
_tar = _PKG.build_tar(_stg, _P(cfg.work_dir) / _tar_name, _top)
_PKG.verify_tar_listing(
    _tar["path"], _stg / "full_metadata.csv",
    set(_top) | {"splits/train_records.csv", "splits/val_records.csv",
                 "splits/test_records.csv", "splits/duplicate_groups.csv",
                 "splits/conflicting_duplicate_groups.csv",
                 "splits/patient_recovery_report.txt",
                 "splits/split_report.txt"})
print(f"clean TAR ready: {_tar['path']} ({_tar['bytes'] / 2**30:.1f} GB)")

# --- upload to the same S3 folder ---
import boto3 as _b3
_s3 = _b3.client("s3")
_key = f"{cfg.s3_prefix}/{_tar_name}"
_rec = _S3U.upload_and_verify(_s3, _tar["path"], cfg.s3_bucket, _key,
                              force_redownload=cfg.verify_s3_bytes)
_side = _P(str(_tar["path"]) + ".sha256")
_s3.upload_file(str(_side), cfg.s3_bucket, _key + ".sha256")
print(f"\nclean TAR uploaded: {_rec['s3_uri']}  bytes={_rec['bytes']}  "
      f"crc64nvme={_rec['crc64nvme']}  verify={_rec['verify_method']}")
print("sidecar:", _key + ".sha256")
