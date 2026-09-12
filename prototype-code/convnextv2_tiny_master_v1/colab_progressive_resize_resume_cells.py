# %% [markdown]
# Cell 1: progressively replace extracted JPEGs with verified, lossless,
# pre-resized WebP files (method 4). Run only after interrupting training
# and backing up the completed epoch-4 checkpoint in Google Drive.

import gc
import io
import os
import json
import hashlib
import math
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, features
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

EXPECTED_RESUME_EPOCH = 4
backup_path = RUN_DIR / "last_epoch04_before_resize.pt"
if not backup_path.is_file():
    raise FileNotFoundError(f"Missing protected checkpoint: {backup_path}")
protected_checkpoint = torch.load(backup_path, map_location="cpu", weights_only=False)
if int(protected_checkpoint.get("epoch", -1)) != EXPECTED_RESUME_EPOCH:
    raise RuntimeError(
        f"Expected epoch {EXPECTED_RESUME_EPOCH} backup, "
        f"found {protected_checkpoint.get('epoch')}"
    )
del protected_checkpoint

if not features.check("webp"):
    raise RuntimeError("This Pillow build does not support WebP")

# Persistent DataLoader workers must be stopped before any source JPEG is
# deleted. This does not restart the runtime or delete /content.
for loader_name in ("train_loader", "val_loader", "test_loader"):
    loader = globals().get(loader_name)
    if loader is None:
        continue
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        try:
            iterator._shutdown_workers()
        except Exception as exc:
            print(f"Worker shutdown warning for {loader_name}: {exc}")
        loader._iterator = None
    globals().pop(loader_name, None)
gc.collect()
print("Existing DataLoader workers stopped.")

LOCAL_CONTENT_ROOT = Path("/content").resolve()
DRIVE_MOUNT_ROOT = (LOCAL_CONTENT_ROOT / "drive").resolve()


def is_within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_local_content_path(path, label):
    resolved = Path(path).resolve()
    if not is_within(resolved, LOCAL_CONTENT_ROOT):
        raise RuntimeError(
            f"{label} must be under {LOCAL_CONTENT_ROOT}, found: {resolved}"
        )
    if is_within(resolved, DRIVE_MOUNT_ROOT):
        raise RuntimeError(
            f"{label} must not be inside the Google Drive mount: {resolved}"
        )
    return resolved


LOCAL_EXTRACT_ROOT = require_local_content_path(EXTRACT_ROOT, "EXTRACT_ROOT")
SOURCE_IMAGES_ROOT = require_local_content_path(images_root, "images_root")
if not is_within(SOURCE_IMAGES_ROOT, LOCAL_EXTRACT_ROOT):
    raise RuntimeError("images_root must be contained by EXTRACT_ROOT")

# Keep the prepared cache beside EXTRACT_ROOT, not inside it. A later cleanup
# that removes EXTRACT_ROOT therefore cannot also remove the WebP cache.
CACHE_ROOT = require_local_content_path(
    LOCAL_EXTRACT_ROOT.parent / "resized_768x1024_webp_lossless_m4",
    "CACHE_ROOT",
)
if is_within(CACHE_ROOT, LOCAL_EXTRACT_ROOT):
    raise RuntimeError("CACHE_ROOT must not be inside EXTRACT_ROOT")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_IDENTITY_PATH = CACHE_ROOT / "cache_identity.json"
LOCAL_PROGRESS_PATH = CACHE_ROOT / "conversion_progress.json"
LOCAL_MANIFEST_PATH = CACHE_ROOT / "cache_manifest.csv"

DRIVE_EVIDENCE_ROOT = (
    PROJECT_ROOT / "experiments" / "convnextv2_tiny_master_v1" / "data"
)
DRIVE_EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_COMPLETE_MARKER = (
    DRIVE_EVIDENCE_ROOT / "progressive_resize_768x1024_webp_lossless_m4_complete.json"
)
DRIVE_MANIFEST_PATH = (
    DRIVE_EVIDENCE_ROOT / "progressive_resize_768x1024_webp_lossless_m4_manifest.csv"
)

RESIZE_WORKERS = min(8, max(1, os.cpu_count() or 1))
CONVERSION_CHUNK_SIZE = 512
MIN_FREE_BYTES = 10 * 2**30
CHUNK_WRITE_BUFFER_BYTES = 1 * 2**30
resize_only = transforms.Resize(
    INPUT_SIZE,
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)

# Preserve the original audited source paths across safe reruns of this cell.
if "source_local_path" not in index_df.columns:
    index_df["source_local_path"] = index_df["local_path"].astype(str)

def make_cached_path(source_path):
    raw_source = Path(source_path)
    if raw_source.is_symlink():
        raise RuntimeError(f"Source image must not be a symbolic link: {raw_source}")
    source = require_local_content_path(raw_source, "source image")
    try:
        relative = source.relative_to(SOURCE_IMAGES_ROOT)
    except ValueError as exc:
        raise RuntimeError(
            f"Source is outside the audited image root: {source}"
        ) from exc
    if source.suffix.lower() not in {".jpg", ".jpeg"}:
        raise RuntimeError(f"Unexpected source extension: {source}")
    destination = require_local_content_path(
        (CACHE_ROOT / relative).with_suffix(".webp"),
        "cache destination",
    )
    if not is_within(destination, CACHE_ROOT):
        raise RuntimeError(f"Cache destination escaped CACHE_ROOT: {destination}")
    return str(destination)


index_df["cached_local_path"] = [
    make_cached_path(path) for path in index_df["source_local_path"]
]

# These checks happen before any deletion. Each source must map to exactly one
# destination and each destination must belong to exactly one dataset row.
for column in ("source_local_path", "cached_local_path"):
    duplicated = index_df[column].duplicated(keep=False)
    if duplicated.any():
        examples = index_df.loc[
            duplicated, ["record", "split", column]
        ].head(10).to_dict("records")
        raise RuntimeError(
            f"{column} is not unique. No images were deleted. Examples: {examples}"
        )

source_paths = [Path(path).resolve() for path in index_df["source_local_path"]]
cached_paths = [Path(path).resolve() for path in index_df["cached_local_path"]]
if set(source_paths) & set(cached_paths):
    raise RuntimeError(
        "At least one cache destination overlaps a source image. "
        "No images were deleted."
    )
if len(set(source_paths)) != len(index_df):
    raise RuntimeError("Source-path uniqueness validation failed")
if len(set(cached_paths)) != len(index_df):
    raise RuntimeError("Cache-path uniqueness validation failed")

mapping_text = "".join(
    f"{source}\t{cached}\n"
    for source, cached in zip(source_paths, cached_paths)
)
cache_identity = {
    "version": 3,
    "index_sha256": INDEX_SHA256,
    "mapping_sha256": hashlib.sha256(mapping_text.encode()).hexdigest(),
    "image_count": int(len(index_df)),
    "input_size_hw": list(INPUT_SIZE),
    "interpolation": "torchvision PIL bicubic",
    "antialias": True,
    "format": "lossless RGB WebP",
    "webp_lossless": True,
    "webp_quality": 100,
    "webp_method": 4,
    "webp_exact": True,
}

if CACHE_IDENTITY_PATH.is_file():
    observed_identity = json.loads(CACHE_IDENTITY_PATH.read_text())
    if observed_identity != cache_identity:
        raise RuntimeError(
            "The existing cache belongs to a different index or mapping. "
            "No images were deleted."
        )
else:
    identity_temporary = CACHE_IDENTITY_PATH.with_suffix(".json.tmp")
    unexpected_entries = [
        path for path in CACHE_ROOT.iterdir() if path != identity_temporary
    ]
    if unexpected_entries:
        raise RuntimeError(
            "The cache directory is non-empty but has no trusted identity file. "
            "No images were deleted."
        )
    identity_temporary.unlink(missing_ok=True)
    identity_temporary.write_text(json.dumps(cache_identity, indent=2, sort_keys=True))
    os.replace(identity_temporary, CACHE_IDENTITY_PATH)

print("Unique source/cache mappings verified:", f"{len(index_df):,}")
print("Cache identity:", cache_identity["mapping_sha256"][:16] + "...")


def load_webp_array(path):
    try:
        with Image.open(path) as image:
            image.load()
            if (
                image.format != "WEBP"
                or image.size != (INPUT_SIZE[1], INPUT_SIZE[0])
            ):
                return None
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, ValueError):
        return None


def prepare_verified_cache(job):
    row_index, source, destination = job
    destination.parent.mkdir(parents=True, exist_ok=True)

    # A missing source is accepted only inside this fingerprinted cache. The
    # conversion sequence below finalizes and verifies the WebP before unlinking
    # its source, so this state is safe after an interrupted prior attempt.
    if not source.is_file():
        if load_webp_array(destination) is None:
            raise RuntimeError(
                f"Neither a usable source nor verified cache exists for {source}"
            )
        return row_index, source, destination, False

    with Image.open(source) as image:
        resized = resize_only(image.convert("RGB"))
        expected_array = np.array(resized, dtype=np.uint8, copy=True)

    actual_array = load_webp_array(destination)
    created = False
    if actual_array is None or not np.array_equal(expected_array, actual_array):
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.unlink(missing_ok=True)
        resized.save(
            temporary,
            format="WEBP",
            lossless=True,
            quality=100,
            method=4,
            exact=True,
        )
        actual_array = load_webp_array(temporary)
        if actual_array is None or not np.array_equal(expected_array, actual_array):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Lossless cache verification failed for {source}")
        os.replace(temporary, destination)
        created = True

    # This explicit equality check is the semantic bridge between the old
    # on-demand resize and the new stored cache. It runs before source.unlink().
    if actual_array is None or not np.array_equal(expected_array, actual_array):
        raise RuntimeError(
            f"Cached WebP differs from the expected resized RGB pixels: {source}"
        )

    return row_index, source, destination, created


jobs = [
    (row_index, source, destination)
    for row_index, (source, destination) in enumerate(zip(source_paths, cached_paths))
]

# Estimate the final local footprint before the first source deletion. Because
# each verified WebP replaces its JPEG progressively, only one bounded chunk
# overlaps during conversion.
remaining_jobs = [job for job in jobs if job[1].is_file()]
upper_95_mean_webp_bytes = 0.0
if remaining_jobs:
    estimate_sample_count = min(128, len(remaining_jobs))
    estimate_rng = np.random.default_rng(SEED)
    sample_positions = np.sort(
        estimate_rng.choice(
            len(remaining_jobs),
            size=estimate_sample_count,
            replace=False,
        )
    )
    estimated_sizes = []
    for position in sample_positions:
        _, source, _ = remaining_jobs[int(position)]
        with Image.open(source) as image:
            resized = resize_only(image.convert("RGB"))
            buffer = io.BytesIO()
            resized.save(
                buffer,
                format="WEBP",
                lossless=True,
                quality=100,
                method=4,
                exact=True,
            )
            estimated_sizes.append(buffer.tell())

    estimated_sizes = np.asarray(estimated_sizes, dtype=np.float64)
    mean_webp_bytes = float(estimated_sizes.mean())
    size_standard_error = (
        float(estimated_sizes.std(ddof=1) / math.sqrt(len(estimated_sizes)))
        if len(estimated_sizes) > 1 else 0.0
    )
    upper_95_mean_webp_bytes = mean_webp_bytes + 1.96 * size_standard_error
    current_free_bytes = shutil.disk_usage(CACHE_ROOT).free
    remaining_source_bytes = sum(job[1].stat().st_size for job in remaining_jobs)
    missing_cache_count = sum(not job[2].is_file() for job in remaining_jobs)
    conservative_new_cache_bytes = upper_95_mean_webp_bytes * missing_cache_count
    conservative_projected_final_free = (
        current_free_bytes + remaining_source_bytes - conservative_new_cache_bytes
    )
    print(f"Current free space: {current_free_bytes / 2**30:.1f} GiB")
    print(f"JPEG bytes still replaceable: {remaining_source_bytes / 2**30:.1f} GiB")
    print(
        "Conservative additional WebP bytes (95% upper estimate): "
        f"{conservative_new_cache_bytes / 2**30:.1f} GiB"
    )
    print(
        "Conservative projected free space after replacement: "
        f"{conservative_projected_final_free / 2**30:.1f} GiB"
    )
    if conservative_projected_final_free < 10 * 2**30:
        raise RuntimeError(
            "Conservative projected final free space is below the 10-GiB safety margin. "
            "No source images were deleted."
        )

created_total = 0
deleted_total = 0
conversion_started = time.time()
progress = tqdm(total=len(jobs), desc="progressive lossless WebP resize")

with ThreadPoolExecutor(max_workers=RESIZE_WORKERS) as executor:
    for offset in range(0, len(jobs), CONVERSION_CHUNK_SIZE):
        chunk = jobs[offset:offset + CONVERSION_CHUNK_SIZE]

        # Leave enough room for a conservative full chunk of new WebPs plus a
        # 1-GiB write buffer before beginning work on the chunk.
        chunk_sources_present = sum(source.is_file() for _, source, _ in chunk)
        conservative_chunk_write_bytes = int(
            math.ceil(upper_95_mean_webp_bytes * chunk_sources_present)
        )
        free_before_chunk = shutil.disk_usage(CACHE_ROOT).free
        required_before_chunk = (
            MIN_FREE_BYTES
            + (CHUNK_WRITE_BUFFER_BYTES if chunk_sources_present else 0)
            + conservative_chunk_write_bytes
        )
        if free_before_chunk < required_before_chunk:
            raise RuntimeError(
                "Conversion safely paused before the next chunk because free "
                f"space is {free_before_chunk / 2**30:.1f} GiB but "
                f"{required_before_chunk / 2**30:.1f} GiB is required. "
                "No source image in the unstarted chunk was deleted."
            )

        # No source in this chunk is deleted unless every output in the chunk
        # has been prepared and pixel-verified successfully.
        results = list(executor.map(prepare_verified_cache, chunk))

        for _, source, destination, created in results:
            if load_webp_array(destination) is None:
                raise RuntimeError(
                    f"Cache became invalid before source deletion: {destination}"
                )
            if source.is_file():
                deletion_target = require_local_content_path(
                    source, "source deletion target"
                )
                if not is_within(deletion_target, SOURCE_IMAGES_ROOT):
                    raise RuntimeError(
                        f"Refusing to delete a file outside images_root: {deletion_target}"
                    )
                if is_within(deletion_target, DRIVE_MOUNT_ROOT):
                    raise RuntimeError(
                        f"Refusing to delete a Google Drive file: {deletion_target}"
                    )
                source.unlink()
                deleted_total += 1
            created_total += int(created)

        progress.update(len(chunk))
        progress_payload = {
            **cache_identity,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_jobs": min(offset + len(chunk), len(jobs)),
            "created_or_repaired_webps_this_run": created_total,
            "deleted_jpegs_this_run": deleted_total,
            "free_bytes_after_chunk": shutil.disk_usage(CACHE_ROOT).free,
        }
        progress_temporary = LOCAL_PROGRESS_PATH.with_suffix(".json.tmp")
        progress_temporary.write_text(json.dumps(progress_payload, indent=2, sort_keys=True))
        os.replace(progress_temporary, LOCAL_PROGRESS_PATH)

        free_after_chunk = shutil.disk_usage(CACHE_ROOT).free
        if free_after_chunk < MIN_FREE_BYTES:
            progress.close()
            raise RuntimeError(
                "Conversion safely paused after a completed, verified chunk "
                f"because free space fell to {free_after_chunk / 2**30:.1f} GiB. "
                "Every deleted source in completed chunks has an exact WebP "
                "replacement; increase free space before rerunning Cell 1."
            )

progress.close()

# Full postcondition: one valid WebP per index row, no original indexed JPEGs,
# and a unique path mapping. Sidecar JSON annotations are intentionally retained.
invalid_cached = [str(path) for path in cached_paths if load_webp_array(path) is None]
if invalid_cached:
    raise RuntimeError(
        f"{len(invalid_cached):,} cached WebPs failed final validation; "
        f"first: {invalid_cached[:3]}"
    )
remaining_sources = [str(path) for path in source_paths if path.is_file()]
if remaining_sources:
    raise RuntimeError(
        f"{len(remaining_sources):,} indexed JPEGs remain; first: {remaining_sources[:3]}"
    )

manifest = pd.DataFrame({
    "record": index_df["record"].astype(str),
    "split": index_df["split"].astype(str),
    "render_k": index_df["render_k"].astype(int),
    "relative_cached_path": [
        path.relative_to(CACHE_ROOT).as_posix() for path in cached_paths
    ],
    "cached_bytes": [path.stat().st_size for path in cached_paths],
})
if manifest["relative_cached_path"].duplicated().any():
    raise RuntimeError("Final cache manifest contains duplicate paths")

manifest_temporary = LOCAL_MANIFEST_PATH.with_suffix(".csv.tmp")
manifest.to_csv(manifest_temporary, index=False)
os.replace(manifest_temporary, LOCAL_MANIFEST_PATH)

drive_manifest_temporary = DRIVE_MANIFEST_PATH.with_suffix(".csv.tmp")
manifest.to_csv(drive_manifest_temporary, index=False)
os.replace(drive_manifest_temporary, DRIVE_MANIFEST_PATH)

complete_payload = {
    **cache_identity,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "cache_root": str(CACHE_ROOT),
    "cache_bytes": int(manifest["cached_bytes"].sum()),
    "conversion_minutes_this_run": (time.time() - conversion_started) / 60,
    "original_indexed_jpegs_remaining": 0,
}
complete_temporary = DRIVE_COMPLETE_MARKER.with_suffix(".json.tmp")
complete_temporary.write_text(json.dumps(complete_payload, indent=2, sort_keys=True))
os.replace(complete_temporary, DRIVE_COMPLETE_MARKER)

# Switch the in-memory frames to the cached WebPs. This does not create loaders
# or resume training yet.
index_df["local_path"] = index_df["cached_local_path"]
train_df = index_df[index_df.split == "train"].reset_index(drop=True)
val_df = index_df[index_df.split == "val"].reset_index(drop=True)
test_df = index_df[index_df.split == "test"].reset_index(drop=True)

print("Progressive resize complete.")
print("Cached images:", f"{len(manifest):,}")
print("Cache size:", f"{complete_payload['cache_bytes'] / 2**30:.1f} GiB")
print("Indexed source JPEGs remaining: 0")
print("Cache root:", CACHE_ROOT)
print("Drive evidence:", DRIVE_COMPLETE_MARKER)
print("Do not run the long-training cell until Cell 3 restores epoch 4.")


# %% [markdown]
# Cell 2 (recommended): persist the resized cache to Google Drive as eight
# uncompressed tar shards. WebP is already compressed, and tar files are written
# directly to Drive, so this does not create a second cache-sized file in
# /content. Keep the local WebP cache until training is finished.

DRIVE_CACHE_ARCHIVE_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "convnextv2_tiny_master_v1"
    / "data"
    / "resized_768x1024_webp_lossless_m4_archives"
)
DRIVE_CACHE_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
LOCAL_TAR_LIST_ROOT = CACHE_ROOT / ".tar_lists"
LOCAL_TAR_LIST_ROOT.mkdir(parents=True, exist_ok=True)

train_files = sorted((CACHE_ROOT / "train").rglob("*.webp"))
val_files = sorted((CACHE_ROOT / "val").rglob("*.webp"))
test_files = sorted((CACHE_ROOT / "test").rglob("*.webp"))
if len(train_files) != EXPECTED_IMAGES_BY_SPLIT["train"]:
    raise RuntimeError(f"Unexpected cached train count: {len(train_files):,}")
if len(val_files) != EXPECTED_IMAGES_BY_SPLIT["val"]:
    raise RuntimeError(f"Unexpected cached validation count: {len(val_files):,}")
if len(test_files) != EXPECTED_IMAGES_BY_SPLIT["test"]:
    raise RuntimeError(f"Unexpected cached test count: {len(test_files):,}")

archive_groups = []
for shard_index, group in enumerate(np.array_split(np.array(train_files, dtype=object), 6)):
    archive_groups.append((f"images_train_{shard_index:02d}_768x1024_webp.tar", list(group)))
archive_groups.extend([
    ("images_val_768x1024_webp.tar", val_files),
    ("images_test_768x1024_webp.tar", test_files),
])

archive_records = []
for archive_name, files in archive_groups:
    archive_path = DRIVE_CACHE_ARCHIVE_ROOT / archive_name
    checksum_path = DRIVE_CACHE_ARCHIVE_ROOT / f"{archive_name}.sha256"

    if archive_path.is_file() and checksum_path.is_file():
        expected_sha = checksum_path.read_text().split()[0].lower()
        observed_sha = sha256_file(archive_path)
        if observed_sha == expected_sha:
            print("reuse verified Drive archive:", archive_name)
            archive_records.append({
                "archive": archive_name,
                "bytes": int(archive_path.stat().st_size),
                "sha256": observed_sha,
                "files": len(files),
            })
            continue
        raise RuntimeError(
            f"Existing Drive archive checksum mismatch: {archive_path}"
        )

    relative_paths = [path.relative_to(CACHE_ROOT).as_posix() for path in files]
    list_path = LOCAL_TAR_LIST_ROOT / f"{archive_name}.txt"
    list_path.write_text("".join(f"{path}\n" for path in relative_paths))

    temporary_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")
    temporary_archive.unlink(missing_ok=True)
    print("writing directly to Drive:", archive_name)
    subprocess.run([
        "tar", "-cf", str(temporary_archive),
        "-C", str(CACHE_ROOT),
        "-T", str(list_path),
    ], check=True)
    os.replace(temporary_archive, archive_path)

    archive_sha = sha256_file(archive_path)
    checksum_temporary = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    checksum_temporary.write_text(f"{archive_sha}  {archive_name}\n")
    os.replace(checksum_temporary, checksum_path)
    archive_records.append({
        "archive": archive_name,
        "bytes": int(archive_path.stat().st_size),
        "sha256": archive_sha,
        "files": len(files),
    })
    print("verified Drive archive:", archive_name, archive_sha[:16] + "...")

DRIVE_ARCHIVE_MANIFEST = DRIVE_CACHE_ARCHIVE_ROOT / "archive_manifest.json"
archive_manifest_payload = {
    **cache_identity,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "archives": archive_records,
    "total_files": int(sum(record["files"] for record in archive_records)),
    "total_bytes": int(sum(record["bytes"] for record in archive_records)),
}
if archive_manifest_payload["total_files"] != len(index_df):
    raise RuntimeError("Drive archive file total does not match the image index")
archive_manifest_temporary = DRIVE_ARCHIVE_MANIFEST.with_suffix(".json.tmp")
archive_manifest_temporary.write_text(
    json.dumps(archive_manifest_payload, indent=2, sort_keys=True)
)
os.replace(archive_manifest_temporary, DRIVE_ARCHIVE_MANIFEST)
print("Drive cache persistence complete:", DRIVE_ARCHIVE_MANIFEST)
print("Keep the local WebP cache in /content while training continues.")


# %% [markdown]
# Cell 3: build cache-only loaders and restore the protected epoch-4 state.
# After this succeeds, rerun only the original long-running training cell.

if not DRIVE_COMPLETE_MARKER.is_file():
    raise FileNotFoundError("The verified cache completion marker is missing")
completion_evidence = json.loads(DRIVE_COMPLETE_MARKER.read_text())
for key, value in cache_identity.items():
    if completion_evidence.get(key) != value:
        raise RuntimeError(f"Cache completion identity mismatch for {key}")

# No resize remains in the per-sample training path.
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

index_df["local_path"] = index_df["cached_local_path"]
train_df = index_df[index_df.split == "train"].reset_index(drop=True)
val_df = index_df[index_df.split == "val"].reset_index(drop=True)
test_df = index_df[index_df.split == "test"].reset_index(drop=True)

train_dataset = ECGImageDataset(train_df)
val_dataset = ECGImageDataset(val_df)
test_dataset = ECGImageDataset(test_df)

for dataset_name, dataset in (
    ("train", train_dataset),
    ("val", val_dataset),
    ("test", test_dataset),
):
    image, target, metadata = dataset[0]
    if tuple(image.shape) != (3, INPUT_SIZE[0], INPUT_SIZE[1]):
        raise RuntimeError(f"Unexpected cached tensor shape for {dataset_name}: {image.shape}")
    print(dataset_name, tuple(image.shape), metadata["record"])

loader_options = dict(
    batch_size=MICRO_BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
    worker_init_fn=seed_data_worker,
)
TRAIN_GENERATOR = torch.Generator()
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    drop_last=True,
    generator=TRAIN_GENERATOR,
    **loader_options,
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    drop_last=False,
    **loader_options,
)
test_loader = DataLoader(
    test_dataset,
    shuffle=False,
    drop_last=False,
    **loader_options,
)

usable_train_batches = len(train_loader) - (len(train_loader) % ACCUMULATION_STEPS)
if usable_train_batches <= 0:
    raise RuntimeError("Cached training loader is too short")
OPTIMIZER_STEPS_PER_EPOCH = usable_train_batches // ACCUMULATION_STEPS

# Roll back all partial epoch-5 state in memory to the protected epoch-4 state.
checkpoint = torch.load(backup_path, map_location=DEVICE, weights_only=False)
if int(checkpoint.get("epoch", -1)) != EXPECTED_RESUME_EPOCH:
    raise RuntimeError(f"Protected checkpoint epoch changed: {checkpoint.get('epoch')}")
if checkpoint.get("config_fingerprint") != CONFIG_FINGERPRINT:
    raise RuntimeError(
        "The protected checkpoint configuration does not match this active run"
    )

model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
scaler.load_state_dict(checkpoint["scaler"])
optimizer.zero_grad(set_to_none=True)

if checkpoint.get("train_generator_state") is not None:
    TRAIN_GENERATOR.set_state(checkpoint["train_generator_state"].cpu())
else:
    TRAIN_GENERATOR.manual_seed(SEED)
if checkpoint.get("python_random_state") is not None:
    random.setstate(checkpoint["python_random_state"])
if checkpoint.get("numpy_random_state") is not None:
    np.random.set_state(checkpoint["numpy_random_state"])
if checkpoint.get("torch_random_state") is not None:
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
if checkpoint.get("cuda_random_state_all") is not None:
    torch.cuda.set_rng_state_all([
        state.cpu() for state in checkpoint["cuda_random_state_all"]
    ])

start_epoch = int(checkpoint["epoch"]) + 1
best_macro_average_precision = float(checkpoint["best_macro_average_precision"])
bad_epochs = int(checkpoint["bad_epochs"])
history = list(checkpoint["history"])
if start_epoch != 5:
    raise RuntimeError(f"Expected to resume at epoch 5, got epoch {start_epoch}")

transition_path = RUN_DIR / "data_pipeline_transition_epoch05.json"
transition_payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "completed_checkpoint_epoch": EXPECTED_RESUME_EPOCH,
    "resume_epoch": start_epoch,
    "checkpoint": str(backup_path),
    "checkpoint_config_fingerprint": checkpoint["config_fingerprint"],
    "model": MODEL_NAME,
    "input_size_hw": list(INPUT_SIZE),
    "semantic_preprocessing_change": False,
    "before": "source JPEG decoded and bicubic-resized per sample",
    "after": "pixel-identical lossless pre-resized WebP decoded per sample",
    "cache_identity": cache_identity,
    "cache_completion_marker": str(DRIVE_COMPLETE_MARKER),
    "cache_manifest": str(DRIVE_MANIFEST_PATH),
    "drive_archive_manifest": (
        str(DRIVE_ARCHIVE_MANIFEST)
        if "DRIVE_ARCHIVE_MANIFEST" in globals() and DRIVE_ARCHIVE_MANIFEST.is_file()
        else None
    ),
}
transition_temporary = transition_path.with_suffix(".json.tmp")
transition_temporary.write_text(json.dumps(transition_payload, indent=2, sort_keys=True))
os.replace(transition_temporary, transition_path)

print("Cached loader batches:", len(train_loader), len(val_loader), len(test_loader))
print("Micro-batch / accumulation:", MICRO_BATCH_SIZE, ACCUMULATION_STEPS)
print("Protected checkpoint restored:", backup_path)
print("Resume epoch:", start_epoch)
print("Pipeline transition evidence:", transition_path)
print("READY: rerun only the original long-running training cell.")
