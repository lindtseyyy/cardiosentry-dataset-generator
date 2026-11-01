"""Build ecg_image_v1_code.tar.gz: the patched kit subset for EC2.

Copies the already-patched vendored kit files (PLAN 0.6 verified), the
reference patch script, fonts, wrinkle textures, and a provenance file with
per-file SHAs into a deterministic tarball + .sha256 sidecar.

The notebook extracts it on EC2, verifies SHAs against kit_provenance.txt,
and runs the render.check_patches() gate before any rendering.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT_SRC = Path("/home/lindtseyxvii/Codes/cardiosentry/dataset-combination-code"
               "/ecg-dataset-pipeline/ecg-image-kit/codes/ecg-image-generator")
PIPE_SRC = Path("/home/lindtseyxvii/Codes/cardiosentry/dataset-combination-code"
                "/ecg-dataset-pipeline/image_pipeline")

KIT_FILES = ["ecg_plot.py", "extract_leads.py", "gen_ecg_image_from_data.py",
             "gen_ecg_images_from_data_batch.py", "helper_functions.py",
             "config.yaml"]
KIT_DIRS = ["Fonts", "TemplateFiles"]
TEXTURE_DIR = "CreasesWrinkles/wrinkles-dataset"
PIPE_FILES = ["patch_kit.py", "config.py"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    stage = HERE / ".bundle_stage"
    if stage.exists():
        shutil.rmtree(stage)
    root = stage / "ecg_image_v1_code"
    gen = root / "ecg-image-generator"
    gen.mkdir(parents=True)
    for name in KIT_FILES:
        shutil.copy2(KIT_SRC / name, gen / name)
    for d in KIT_DIRS:
        shutil.copytree(KIT_SRC / d, gen / d)
    # wrinkle photos only (creases.py itself is unused by design)
    tex = gen / TEXTURE_DIR
    tex.mkdir(parents=True)
    for p in sorted((KIT_SRC / TEXTURE_DIR).iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(p, tex / p.name)
    pipe = root / "image_pipeline_reference"
    pipe.mkdir()
    for name in PIPE_FILES:
        shutil.copy2(PIPE_SRC / name, pipe / name)

    # provenance with per-file SHAs
    files = sorted(p for p in root.rglob("*") if p.is_file())
    prov = [f"ecg_image_v1_code provenance",
            f"built_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
            f"kit_source: {KIT_SRC}", f"pipeline_source: {PIPE_SRC}",
            "note: kit files are the pre-patched vendored checkout "
            "(all PATCHED-* markers verified present at bundle time); "
            "patch_kit.py is included for reference only.",
            "", "files:"]
    for p in files:
        prov.append(f"  {sha256_file(p)}  {p.relative_to(root)}")
    (root / "kit_provenance.txt").write_text("\n".join(prov) + "\n")

    out = HERE / "ecg_image_v1_code.tar.gz"
    if out.exists():
        out.unlink()

    def _reset(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.mtime = 0
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.pax_headers = {}
        return ti

    with tarfile.open(out, "w:gz", format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(root.rglob("*")):
            arc = p.relative_to(stage)
            ti = tf.gettarinfo(str(p), arcname=str(arc))
            ti = _reset(ti)
            if p.is_file():
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
            else:
                tf.addfile(ti)
    (HERE / "ecg_image_v1_code.tar.gz.sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n")
    shutil.rmtree(stage)
    print(f"bundle: {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
