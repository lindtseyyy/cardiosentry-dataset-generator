"""Kit rendering driver (PLAN 6.x, 13.13).

Runs the vendored batch driver per staged chunk (Arial pass for all records,
seeded-font pass for train records), collects exactly the PNGs, and removes
kit side products (.hea/.dat/.json the kit drops next to renders).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

from .config import FIXED_FONT, KIT_ENV, KIT_RENDER_FLAGS, SEED
from .util import V1Error, ensure_dir, log, rng_from_parts

KIT_SCRIPT = "gen_ecg_images_from_data_batch.py"
KIT_SUBPATH = Path("kit") / "ecg_image_v1_code" / "ecg-image-generator"


def resolve_kit_dir(work_dir: str | Path, kit_dir: str | Path = "") -> str:
    """Self-healing kit path (PLAN v1.12).

    §1 rebuilds cfg from scratch, wiping the kit path §2 found — and the §12
    approval instruction tells the user to do exactly that (edit §1, re-run
    §1). So no consumer may trust cfg.kit_dir blindly: a valid entry passes
    through, anything else is re-derived from work_dir (the data persists on
    disk; only the cfg field was wiped). Raises only if §2 never extracted.
    """
    if kit_dir and (Path(kit_dir) / KIT_SCRIPT).is_file():
        return str(kit_dir)
    derived = Path(work_dir) / KIT_SUBPATH
    if (derived / KIT_SCRIPT).is_file():
        log(f"kit_dir recovered -> {derived} "
            f"(cfg was rebuilt after §2; re-derived, no re-run needed)")
        return str(derived)
    raise V1Error(f"kit not extracted at {derived} — run §2 "
                  f"(bundle extract) before rendering")


def kit_fonts(kit_dir: str | Path) -> list[str]:
    fonts = sorted((Path(kit_dir) / "Fonts").glob("*.ttf"))
    if not fonts:
        raise V1Error(f"no fonts in {kit_dir}/Fonts")
    return [f.name for f in fonts]


def check_patches(kit_dir: str | Path) -> None:
    """Refuse to render unless all patch_kit markers are present."""
    kit = Path(kit_dir)
    need = {
        "ecg_plot.py": ["PATCHED by image_pipeline/patch_kit.py",
                        "PATCHED-LEADNAME", "PATCHED-RHYTHMNAME",
                        "PATCHED-INKWIDTH", "PATCHED-LAYOUT", "PATCHED-PAPER"],
        "extract_leads.py": ["PATCHED-COLUMNS"],
        "gen_ecg_images_from_data_batch.py": ["PATCHED-NUMPY2"],
        "gen_ecg_image_from_data.py": ["PATCHED-LAZYDEPS", "PATCHED-FIXEDFONT"],
        "helper_functions.py": ["PATCHED-UNUSED-IMGAUG"],
    }
    missing = []
    for name, markers in need.items():
        text = (kit / name).read_text()
        for mk in markers:
            if mk not in text:
                missing.append(f"{name}:{mk}")
    if missing:
        raise V1Error(f"kit at {kit} is not patched ({len(missing)} markers "
                      f"missing, e.g. {missing[:3]}); refusing to render")
    log(f"kit patch gate OK ({kit})")


def draw_train_font(record_id: str, fonts: list[str], seed: int = SEED) -> str:
    """Seeded font for a train record's augmented-master pass (PLAN 6.7).

    Keyed on record_id only — never label/source/split — so fonts cannot
    correlate with anything clinical.
    """
    rng = rng_from_parts(seed, record_id, "font")
    return fonts[int(rng.integers(len(fonts)))]


def run_batch(kit_dir: str | Path, kit_python: str, staged_dir: str | Path,
              out_dir: str | Path, font: str, seed: int = SEED,
              timeout_s: int = 3600,
              extra_flags: list[str] | None = None) -> None:
    """Invoke gen_ecg_images_from_data_batch.py once for a staged directory.

    extra_flags: appended verbatim (used ONLY by the preview annotation
    probe: ["--store_config", "1", "--lead_bbox", "--lead_name_bbox"]).
    """
    if not str(kit_dir):
        raise V1Error("kit_dir is empty (cfg was rebuilt by a §1 re-run "
                      "after §2?) — call render.resolve_kit_dir(work_dir) "
                      "or re-run §2 first")
    kit_dir = Path(kit_dir).resolve()
    if not (kit_dir / KIT_SCRIPT).is_file():
        raise V1Error(f"kit script missing: {kit_dir / KIT_SCRIPT} — "
                      f"re-run §2 (bundle extract) first")
    out_dir = ensure_dir(Path(out_dir).resolve())
    staged_dir = Path(staged_dir).resolve()
    # NOTE: absolute paths are mandatory here — the kit subprocess runs with
    # cwd=kit_dir, so relative -i/-o would resolve inside the kit tree.
    cmd = [kit_python, KIT_SCRIPT,
           "-i", str(staged_dir), "-o", str(out_dir),
           "-se", str(seed), *KIT_RENDER_FLAGS, *(extra_flags or [])]
    env = dict(os.environ)
    env.update(KIT_ENV)
    env["ECG_FIXED_FONT"] = font
    # one numerical thread per renderer (FCMAE pattern); never the GPU
    env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    proc = subprocess.run(cmd, cwd=str(kit_dir), capture_output=True,
                          text=True, timeout=timeout_s, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise V1Error(f"kit batch failed ({staged_dir} -> {out_dir}, "
                      f"font={font}): {'; '.join(tail)}")


def collect_pngs(render_out: str | Path, record_ids: set[str]) -> dict[str, Path]:
    """Map record_id -> rendered PNG. The kit names outputs <record>-0.png.

    Raises if any record is missing its render (never a silent skip).
    """
    found: dict[str, Path] = {}
    for png in Path(render_out).rglob("*.png"):
        stem = png.stem  # <record_id>-0
        rid = stem[:-2] if stem.endswith("-0") else stem
        if rid in record_ids:
            found[rid] = png
    missing = sorted(record_ids - set(found))
    if missing:
        raise V1Error(f"kit produced no PNG for {len(missing)} records, e.g. "
                      f"{missing[:5]}")
    return found


def clean_render_dir(render_out: str | Path) -> None:
    """Remove kit side products; only *.png may remain."""
    for pat in ("*.hea", "*.dat", "*.json", "*.csv"):
        for p in Path(render_out).rglob(pat):
            p.unlink()


def plan_font_jobs(train_records: list[dict], fonts: list[str],
                   seed: int = SEED) -> dict[str, list[dict]]:
    """Group train records by their seeded font for the font-master pass."""
    jobs: dict[str, list[dict]] = defaultdict(list)
    for r in train_records:
        jobs[draw_train_font(r["record_id"], fonts, seed)].append(r)
    return jobs


def clear_dir(path: str | Path) -> None:
    p = Path(path)
    if p.is_dir():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
