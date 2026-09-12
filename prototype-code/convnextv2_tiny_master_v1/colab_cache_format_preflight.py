"""Non-destructive Colab cell for choosing a pre-resized cache format."""

import io
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, features
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

PREFLIGHT_SAMPLE_SIZE = 128
PREFLIGHT_SAFETY_MARGIN_GIB = 10.0

if not features.check("webp"):
    raise RuntimeError(
        "This Pillow build has no WebP support. Install a Pillow build with "
        "WebP enabled before testing a lossless WebP cache."
    )

source_column = (
    "source_local_path" if "source_local_path" in index_df.columns else "local_path"
)
source_paths = [Path(path) for path in index_df[source_column].astype(str)]
missing_sources = [str(path) for path in source_paths if not path.is_file()]
if missing_sources:
    raise RuntimeError(
        f"{len(missing_sources):,} source images are missing; first: "
        f"{missing_sources[:3]}. Do not start a new conversion."
    )

# A deterministic random sample is representative of the actual indexed rows.
sample_count = min(PREFLIGHT_SAMPLE_SIZE, len(index_df))
sample_rng = np.random.default_rng(SEED)
sample_indices = np.sort(sample_rng.choice(len(index_df), size=sample_count, replace=False))
sample_paths = [source_paths[int(index)] for index in sample_indices]

resize_only = transforms.Resize(
    INPUT_SIZE,
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)

candidates = [
    {
        "name": "webp_lossless_m4",
        "format": "WEBP",
        "save_kwargs": {"lossless": True, "quality": 100, "method": 4, "exact": True},
    },
    {
        "name": "webp_lossless_m6",
        "format": "WEBP",
        "save_kwargs": {"lossless": True, "quality": 100, "method": 6, "exact": True},
    },
    {
        "name": "jpeg_q95_444",
        "format": "JPEG",
        "save_kwargs": {"quality": 95, "subsampling": 0, "optimize": True},
    },
    {
        "name": "jpeg_q98_444",
        "format": "JPEG",
        "save_kwargs": {"quality": 98, "subsampling": 0, "optimize": True},
    },
    {
        "name": "jpeg_q100_444",
        "format": "JPEG",
        "save_kwargs": {"quality": 100, "subsampling": 0, "optimize": True},
    },
]

statistics = {
    candidate["name"]: {
        "encoded_bytes": [],
        "encode_seconds": 0.0,
        "decode_seconds": 0.0,
        "absolute_error_sum": 0,
        "squared_error_sum": 0.0,
        "changed_values": 0,
        "value_count": 0,
        "max_absolute_error": 0,
        "exact_images": 0,
    }
    for candidate in candidates
}
original_decode_resize_seconds = 0.0

for source_path in tqdm(sample_paths, desc="cache format preflight"):
    started = time.perf_counter()
    with Image.open(source_path) as source_image:
        expected = np.array(
            resize_only(source_image.convert("RGB")),
            dtype=np.uint8,
            copy=True,
        )
    original_decode_resize_seconds += time.perf_counter() - started

    for candidate in candidates:
        candidate_stats = statistics[candidate["name"]]
        buffer = io.BytesIO()

        started = time.perf_counter()
        Image.fromarray(expected, mode="RGB").save(
            buffer,
            format=candidate["format"],
            **candidate["save_kwargs"],
        )
        candidate_stats["encode_seconds"] += time.perf_counter() - started
        encoded = buffer.getvalue()
        candidate_stats["encoded_bytes"].append(len(encoded))

        started = time.perf_counter()
        with Image.open(io.BytesIO(encoded)) as decoded_image:
            decoded_image.load()
            decoded = np.array(decoded_image.convert("RGB"), dtype=np.uint8, copy=True)
        candidate_stats["decode_seconds"] += time.perf_counter() - started

        if decoded.shape != expected.shape:
            raise RuntimeError(
                f"{candidate['name']} decoded to {decoded.shape}, expected {expected.shape}"
            )

        difference = decoded.astype(np.int16) - expected.astype(np.int16)
        absolute_difference = np.abs(difference)
        candidate_stats["absolute_error_sum"] += int(absolute_difference.sum())
        candidate_stats["squared_error_sum"] += float(
            np.square(difference.astype(np.float64)).sum()
        )
        candidate_stats["changed_values"] += int(np.count_nonzero(difference))
        candidate_stats["value_count"] += int(difference.size)
        candidate_stats["max_absolute_error"] = max(
            candidate_stats["max_absolute_error"], int(absolute_difference.max())
        )
        candidate_stats["exact_images"] += int(np.array_equal(decoded, expected))

total_source_bytes = sum(path.stat().st_size for path in source_paths)
current_free_bytes = shutil.disk_usage(source_paths[0].parent).free
safety_margin_bytes = int(PREFLIGHT_SAFETY_MARGIN_GIB * 2**30)
maximum_safe_cache_bytes = current_free_bytes + total_source_bytes - safety_margin_bytes

rows = []
for candidate in candidates:
    candidate_stats = statistics[candidate["name"]]
    sizes = np.asarray(candidate_stats["encoded_bytes"], dtype=np.float64)
    mean_bytes = float(sizes.mean())
    standard_error = float(sizes.std(ddof=1) / math.sqrt(len(sizes))) if len(sizes) > 1 else 0.0
    upper_95_mean_bytes = mean_bytes + 1.96 * standard_error
    estimated_cache_bytes = mean_bytes * len(index_df)
    conservative_cache_bytes = upper_95_mean_bytes * len(index_df)
    value_count = candidate_stats["value_count"]
    mae = candidate_stats["absolute_error_sum"] / value_count
    rmse = math.sqrt(candidate_stats["squared_error_sum"] / value_count)
    psnr = float("inf") if rmse == 0 else 20 * math.log10(255.0 / rmse)

    rows.append({
        "candidate": candidate["name"],
        "sample_images": sample_count,
        "sample_exact_images": candidate_stats["exact_images"],
        "pixel_exact": candidate_stats["exact_images"] == sample_count,
        "mean_encoded_mib": mean_bytes / 2**20,
        "estimated_cache_gib": estimated_cache_bytes / 2**30,
        "upper_95_estimated_cache_gib": conservative_cache_bytes / 2**30,
        "estimated_final_free_gib": (
            current_free_bytes + total_source_bytes - estimated_cache_bytes
        ) / 2**30,
        "conservative_final_free_gib": (
            current_free_bytes + total_source_bytes - conservative_cache_bytes
        ) / 2**30,
        "fits_with_10_gib_margin": conservative_cache_bytes <= maximum_safe_cache_bytes,
        "mae_0_255": mae,
        "rmse_0_255": rmse,
        "psnr_db": psnr,
        "max_absolute_error": candidate_stats["max_absolute_error"],
        "changed_channel_values_percent": 100 * candidate_stats["changed_values"] / value_count,
        "mean_encode_ms_per_image": 1000 * candidate_stats["encode_seconds"] / sample_count,
        "mean_decode_ms_per_image_from_memory": (
            1000 * candidate_stats["decode_seconds"] / sample_count
        ),
    })

results = pd.DataFrame(rows)
results.insert(
    len(results.columns),
    "original_decode_plus_resize_ms_per_image",
    1000 * original_decode_resize_seconds / sample_count,
)

webp_rows = results[results["candidate"].str.startswith("webp_lossless")]
if not bool(webp_rows["pixel_exact"].all()):
    raise RuntimeError("Lossless WebP failed exact RGB pixel verification on the sample")

REPORT_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "convnextv2_tiny_master_v1"
    / "data"
    / "cache_format_preflight"
)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
csv_path = REPORT_ROOT / "cache_format_preflight.csv"
json_path = REPORT_ROOT / "cache_format_preflight.json"
results.to_csv(csv_path, index=False)

json_payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "index_sha256": INDEX_SHA256,
    "image_count": int(len(index_df)),
    "sample_size": sample_count,
    "input_size_hw": list(INPUT_SIZE),
    "current_free_gib": current_free_bytes / 2**30,
    "original_source_gib": total_source_bytes / 2**30,
    "safety_margin_gib": PREFLIGHT_SAFETY_MARGIN_GIB,
    "maximum_safe_cache_gib": maximum_safe_cache_bytes / 2**30,
    "results": results.replace({np.inf: None}).to_dict("records"),
}
temporary_json_path = json_path.with_suffix(".json.tmp")
temporary_json_path.write_text(json.dumps(json_payload, indent=2, sort_keys=True))
os.replace(temporary_json_path, json_path)

pd.set_option("display.max_columns", None)
display(results)
print("\nNo files were created, replaced, or deleted in /content.")
print("Maximum safe cache size:", f"{maximum_safe_cache_bytes / 2**30:.1f} GiB")
print("CSV report:", csv_path)
print("JSON report:", json_path)
print(
    "Choose lossless WebP only if pixel_exact=True and "
    "fits_with_10_gib_margin=True. Otherwise compare the JPEG error columns."
)
