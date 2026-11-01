"""Build v1_image_generation.ipynb from readable cell sources (PLAN v1.1 §13).

Run:  python3 make_notebook.py
Out:  v1_image_generation.ipynb (nbformat 4, python3 kernel)
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src))


def code(src: str) -> None:
    CELLS.append(("code", src))


# =================================================================  header
md("""# V1 ECG Image Generation — EC2 build notebook (PLAN v1.11)

Generates the training-ready multi-variant 1024×768 PNG corpus from the
already-standardized 10 s waveforms (`V1_10S_STANDARDIZED_SUBSET`):
union-grouped 80/10/10 split **before** rendering → 200-DPI masters →
clean / augmented / stress views → validated metadata → TAR → S3.

**Read PLAN.md first.** Every section below cites the section that governs it.
Sections 1–12 end at a human preview gate (`APPROVE_PREVIEW`); the full
build (§13) refuses to run before it. All long stages resume by skipping
finished chunks. No model training happens here.
""")

# =============================================================  1. config
md("""## 1. Configuration (PLAN §13.1)

Set thread-pinning first (before NumPy exists), then resolve every path and
freeze the run config to `generation_config.json`. **Edit this cell for EC2.**
""")
code('''import os
# Thread discipline BEFORE any numeric import (PLAN §13.5, FCMAE pattern).
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

import sys, json, shutil, subprocess, socket
from pathlib import Path

def _find_nb_dir():
    """Locate dataset_generation_code/ without trusting the kernel cwd.

    Jupyter usually starts kernels in the notebook's directory, but a
    server launched elsewhere, VS Code, or nbconvert can leave cwd
    somewhere unrelated — and then `import v1_support` fails with a bare
    ModuleNotFoundError. Search cwd, one level of children, VS Code's
    notebook path, then parents; fail loudly with the actual fix.
    """
    cands = [Path.cwd()]
    try:
        cands.append(Path(__vsc_ipynb_file__).parent)  # VS Code notebooks
    except NameError:
        pass
    seen = []
    for c in cands:
        try:
            c = c.resolve()
        except OSError:
            continue
        kids = [c]
        try:
            kids += [p for p in c.iterdir() if p.is_dir()]
        except OSError:
            pass
        for d in kids + list(c.parents):
            if d in seen:
                continue
            seen.append(d)
            if (d / "v1_support" / "__init__.py").is_file():
                return d
    raise RuntimeError(
        "v1_support/ not found from " + str(Path.cwd()) + ". "
        "Fix 1: copy the WHOLE dataset_generation_code/ directory to EC2 "
        "(EC2_NOTES.md step 1) — not just the .ipynb. "
        "Fix 2: run %cd ~/CardioSentry/dataset_generation_code, restart "
        "the kernel, re-run §1.")

NB_DIR = _find_nb_dir()  # dataset_generation_code/ (notebook + v1_support + bundle)
print("project:", NB_DIR)
sys.path.insert(0, str(NB_DIR))

# --- notebook-side packages FIRST: v1_support itself imports numpy/PIL,
# so everything must be installed before the import below (v1.6: this
# used to live in §2, one section too late).
import importlib as _il
# Union env: kernel + workers + kit subprocess (kit_python defaults to this
# interpreter). pandas is here only as wfdb's runtime dep (nothing imports
# it directly); wfdb>=4.3 is REQUIRED (4.1.x dies at import on pandas 3 —
# v1.7: EC2 hit exactly that with wfdb==4.1.2 + pandas 3.x).
NEED_RENDER = ["wfdb", "yaml", "PIL", "numpy", "scipy", "matplotlib",
               "pandas", "qrcode"]
PIP = {"wfdb": "wfdb>=4.3", "yaml": "pyyaml", "PIL": "Pillow>=10",
       "numpy": "numpy>=1.24", "scipy": "scipy>=1.10",
       "matplotlib": "matplotlib>=3.7", "pandas": "pandas>=2",
       "qrcode": "qrcode"}
_missing = []  # (module, pip-spec, kind): kind is absent|broken
for _mod in NEED_RENDER:
    try:
        _il.import_module(_mod)
    except ImportError:
        _missing.append((_mod, PIP[_mod], "absent"))
    except Exception:
        # Present-but-broken (a version clash raises TypeError at import,
        # not ImportError) — (re)install the pin.
        _missing.append((_mod, PIP[_mod], "broken"))
if _missing:
    _specs = [_s for _, _s, _ in _missing]
    print("installing:", _specs)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--retries", "5", "--timeout", "120", "--upgrade",
                    *_specs], check=True)
    # Verify in a FRESH interpreter: this kernel may hold stale modules
    # from a failed import (v1.8: a broken-then-upgraded wfdb re-imported
    # as cached-old `_coreio` + new `annotation` = bogus ImportError —
    # never trust a same-process re-import after replacing packages).
    _probe = "; ".join(f"import {_mod}" for _mod, _, _ in _missing)
    def _probe_ok():
        _r = subprocess.run([sys.executable, "-c", _probe],
                            capture_output=True, text=True, timeout=300)
        return _r.returncode == 0, (_r.stderr or "")
    _ok, _err = _probe_ok()
    if not _ok:
        # Genuinely broken on disk (mixed-version leftovers): one clean
        # reinstall, then re-probe.
        print("fresh-interpreter probe failed; force-reinstalling:", _specs)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--retries", "3", "--timeout", "300",
                        "--force-reinstall", "--no-cache-dir", *_specs],
                       check=True)
        _ok, _err = _probe_ok()
    if not _ok:
        raise RuntimeError(
            "packages still unimportable after force-reinstall:\\n" +
            _err[-2000:] + "\\nManual fix: " + sys.executable +
            " -m pip install --force-reinstall --no-cache-dir " +
            " ".join(_specs))
    if any(_k == "broken" for _, _, _k in _missing):
        raise RuntimeError(
            "Upgraded previously-broken packages " +
            ", ".join(_s for _, _s, _k in _missing if _k == "broken") +
            " — on-disk install now verifies clean, but this kernel holds "
            "stale modules: RESTART the kernel and re-run §1.")
    for _mod in NEED_RENDER:
        _il.import_module(_mod)
print("render packages OK:", ", ".join(NEED_RENDER))

from v1_support.config import Config, SEED, S3_BUCKET, S3_PREFIX, TRAIN_TAR, INPUT_S3_URI
from v1_support.util import ensure_dir, log

# ---- paths: local defaults; override for EC2 ---------------------------
# EC2 layout this was designed for:
#   ~/CardioSentry/
#     V1_10S_STANDARDIZED_SUBSET/   (input waveforms, read-only)
#     dataset_generation_code/      (this directory: notebook + v1_support +
#                                    ecg_image_v1_code.tar.gz)
#     outputs/                      (build outputs incl. TAR)
LOCAL_SUBSET = "/home/lindtseyxvii/Codes/cardiosentry/dataset-files/MAIN_DATASET/V1_10S_STANDARDIZED_SUBSET"
EC2_SUBSET = str(Path.home() / "CardioSentry" / "V1_10S_STANDARDIZED_SUBSET")

cfg = Config(
    subset_dir=LOCAL_SUBSET if Path(LOCAL_SUBSET).exists() else EC2_SUBSET,
    input_s3_uri=INPUT_S3_URI,  # fetched in §5 iff subset_dir is missing
    work_dir=str(NB_DIR / "outputs"),
    bundle_path=str(NB_DIR / "ecg_image_v1_code.tar.gz"),
    kit_dir="",            # filled in §2 after bundle extraction
    kit_python=sys.executable,
    seed=SEED,
    approve_preview=False,   # set True ONLY after reviewing §12
    verify_s3_bytes=False,   # re-download hash instead of composed-CRC verify
    upload_masters_tar=True, # PLAN §8.3 second archive
    workers=0,               # 0 = cpu_count
)
cfg.resolve()
for d in (cfg.work_dir, cfg.dataset_dir, cfg.master_dir, cfg.scratch_dir,
          cfg.index_parts_dir, cfg.preview_dir):
    ensure_dir(d)
print("host:", socket.gethostname(), "| python:", sys.executable)
print("subset:", cfg.subset_dir, Path(cfg.subset_dir).exists())
if not Path(cfg.subset_dir).exists():
    print("input S3:", cfg.input_s3_uri, "(will fetch in §5)")
print("work:  ", cfg.work_dir)
print("seed:  ", cfg.seed, "| workers:", cfg.workers)
''')

# ====================================================  2. env / deps check
md("""## 2. Environment / dependency check (PLAN §13.2)

Verifies notebook-side packages (installed in §1, before the
`v1_support` import), extracts + verifies the kit bundle, runs
the `is_patched` gate, asserts S3-checksum support, and echoes effective
settings. Hard-stops on anything red. If no local input subset exists
(fresh EC2), boto3 becomes REQUIRED here — §5 pulls the input tarball
from S3 before anything else can run.
""")
code('''import importlib
# Render-path packages were installed in §1 (v1_support imports them at
# module load, so installing here would be one section too late). Just
# re-verify: a missing one means §1 was skipped — fail loudly, don't
# pip-install behind the user's back in a "check" section.
for mod in ["wfdb", "yaml", "PIL", "numpy", "scipy", "matplotlib",
            "pandas", "qrcode"]:
    try:
        importlib.import_module(mod)
    except Exception:
        raise RuntimeError(
            f"notebook package {mod!r} missing or broken — re-run §1, "
            "which installs/upgrades all render-path packages before "
            "importing v1_support.")
print("render packages verified (installed in §1).")
# S3 packages are only needed in §5 (input fetch) / §17 (upload), so
# their install is attempted but deferred on failure — a proxy hiccup
# at §2 must not kill a build that needs S3 hours later.
NEED_S3 = ["boto3", "awscrt"]
PIP_S3 = {"boto3": "boto3>=1.34", "awscrt": "awscrt"}
NEED_S3_INSTALL = False
for mod in NEED_S3:
    try:
        importlib.import_module(mod)
    except ImportError:
        NEED_S3_INSTALL = True
if NEED_S3_INSTALL:
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--retries",
             "2", "--timeout", "60", *[PIP_S3[m] for m in NEED_S3]], check=True)
        for mod in NEED_S3:
            importlib.import_module(mod)
        NEED_S3_INSTALL = False
        print("S3 packages OK:", ", ".join(NEED_S3))
    except subprocess.CalledProcessError:
        print("WARNING: boto3/awscrt install failed (proxy?) — deferred to "
              "§17, which installs + verifies them before any upload.")

# No local input subset (fresh EC2): boto3 is REQUIRED now, because §5
# pulls the ~2 GB input tarball from S3 before anything else can run.
FETCH_INPUT_FROM_S3 = not Path(cfg.subset_dir).exists()
if FETCH_INPUT_FROM_S3:
    try:
        importlib.import_module("boto3")
        print("boto3 present (S3 input fetch armed)")
    except ImportError:
        print("installing boto3 (REQUIRED: no local input subset)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--retries", "5", "--timeout", "120",
                        "boto3>=1.34"], check=True)
        importlib.import_module("boto3")
        print("boto3 installed (S3 input fetch armed)")

import hashlib
# --- bundle extract + SHA verify -----------------------------------------
bundle = Path(cfg.bundle_path)
assert bundle.is_file(), f"bundle missing: {bundle}"
sidecar = Path(str(bundle) + ".sha256")
assert sidecar.is_file(), "bundle .sha256 sidecar missing"
want = sidecar.read_text().split()[0]
h = hashlib.sha256()
with open(bundle, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 24), b""):
        h.update(b)
assert h.hexdigest() == want, "bundle SHA mismatch — re-transfer the file"
kitroot = Path(cfg.work_dir) / "kit"
if not (kitroot / "ecg_image_v1_code").exists():
    import tarfile
    with tarfile.open(bundle, "r:gz") as tf:
        tf.extractall(kitroot)
cfg.kit_dir = str(kitroot / "ecg_image_v1_code" / "ecg-image-generator")
print("kit:", cfg.kit_dir)

# --- kit provenance + patch gate ------------------------------------------
from v1_support import render as R
prov = kitroot / "ecg_image_v1_code" / "kit_provenance.txt"
assert prov.is_file(), "kit_provenance.txt missing from bundle"
R.check_patches(cfg.kit_dir)   # raises unless ALL markers present
print("fonts:", len(R.kit_fonts(cfg.kit_dir)))

# --- S3 checksum support is verified in §17 (packages may be deferred) ----
from v1_support import s3util  # noqa: F401  (importable without boto3)
try:
    import boto3  # noqa: F401
    s3util.check_crc64nvme_support()
except ImportError:
    print("S3 support check deferred to §17 (boto3 not installed yet)")

# --- disk / RAM --------------------------------------------------------------
st = os.statvfs(cfg.work_dir)
free_gb = st.f_bavail * st.f_frsize / 2**30
print(f"free disk on work volume: {free_gb:.0f} GB")
assert free_gb >= 20, "need >=20 GB free even for the preview"
import multiprocessing
print("vCPUs:", multiprocessing.cpu_count(), "| build workers:", cfg.workers)
''')

# =================================================  3. inspect kit code
md("""## 3. Inspect existing ECG-Image-Kit code (PLAN §13.3)

Echoes exactly what will render: file SHAs, patch markers, render flags,
layout env, and the column/lead config. Nothing renders from uninspected code.
""")
code('''from pathlib import Path as _P
from v1_support.config import KIT_RENDER_FLAGS, KIT_ENV, FIXED_FONT, DPI
from v1_support.util import sha256_file as _sha
from v1_support import render as R
cfg.kit_dir = R.resolve_kit_dir(cfg.work_dir, cfg.kit_dir)
kit = _P(cfg.kit_dir)
for name in ("ecg_plot.py", "extract_leads.py",
             "gen_ecg_image_from_data.py", "gen_ecg_images_from_data_batch.py",
             "helper_functions.py", "config.yaml"):
    print(f"{_sha(kit / name)[:16]}  {name}")
print("\\nrender flags:", " ".join(KIT_RENDER_FLAGS))
print("layout env:", KIT_ENV, "| fixed font:", FIXED_FONT, "| DPI:", DPI)
print("\\n--- config.yaml ---")
print((kit / "config.yaml").read_text())
print("--- Fonts/ ---")
print("\\n".join(sorted(p.name for p in (kit / "Fonts").glob("*.ttf"))))
tex = kit / "CreasesWrinkles" / "wrinkles-dataset"
print(f"--- wrinkle textures: {len(list(tex.glob('*')))} photos ---")
''')

# ============================================  4. earlier notebook reference
md("""## 4. Earlier-notebook reference (PLAN §2, §13.4 — read-only)

What was reused as *ideas* from `cardiosentry_generate_images.ipynb` (v3
4-class Colab corpus): preflight gates, smoke/preview before the long build,
staged gates, resume-by-skipping, per-split TAR + SHA sidecars.

What was **verified inappropriate and NOT carried over**: v3 input signals,
4-class framing, Google-Drive/Colab I/O, 150 DPI (→200), JPEG (→PNG),
0.05–40 Hz band-pass (→no additional signal-domain filtering), class-balanced
render counts (→fixed split-aware variant counts), `split_group_id` patient
key (→recovered `ptbxl` IDs ∪ waveform hashes), Colab timing estimates.

Operational patterns (chunked atomic checkpoints, PNG output, native
sampling, boto3 + sha-metadata + HEAD verify) come from the FCMAE EC2
notebook instead. No code below depends on either notebook being present.
""")

# ==================================================  5. load manifest
md("""## 5. Load accepted-record manifest (PLAN §3, §13.5)

Fetches the input subset from S3 first when no local copy exists (§3.0:
sidecar-pinned SHA-256, verified download, traversal-safe extract), then
loads the 31,109 accepted rows, cross-checks the 61+19 omissions, gates
every waveform (12 leads, exactly 10.0 s), re-hashes every file, and
parses the 12 multi-hot labels. Any failure aborts with offender
identities.
""")
code('''from pathlib import Path as _P
from v1_support import manifest as M

# --- S3 input fetch (PLAN §3.0; skipped when a local copy exists) ---------
if not _P(cfg.subset_dir).exists():
    import boto3 as _b3
    from v1_support import s3util as _S3U
    _s3in = _b3.client("s3")  # IAM role; no keys (PLAN §17 pattern)
    cfg.subset_dir = _S3U.fetch_input_subset(_s3in, cfg.input_s3_uri,
                                             cfg.input_dir)
    print("input subset fetched from S3:", cfg.subset_dir)
else:
    print("input subset present locally:", cfg.subset_dir)

accepted = M.load_accepted(cfg.subset_dir)
xcheck = M.cross_check_manifests(cfg.subset_dir, accepted)
wgate = M.waveform_gate(accepted)
M.verify_file_hashes(accepted)
print("\\nper-label accepted totals:")
for lab, n in wgate["per_label"].items():
    print(f"  {lab:5s} {n:6d}")
EXPECTED = {"NORM": 6456, "AF": 5219, "IAVB": 3496, "LBBB": 1477,
            "RBBB": 4777, "PAC": 3196, "PVC": 1873, "LAFB": 2173,
            "LAE": 1246, "TInv": 3962, "LQT": 1865, "PRWP": 638}
assert wgate["per_label"] == EXPECTED, "input label totals changed!"
print("\\nlabel totals match PLAN §0.4 exactly.")
''')

# ==================================================  6. identifiers
md("""## 6. Inspect patient/source identifiers (PLAN §0.2–0.3, §13.6)

Verifies the shipped PTB-XL patient map row-by-row, builds the 29,564
union-find groups (patient ∪ waveform-hash edges), and writes the duplicate
+ conflict audits (with the executable mapping recompute) to `splits/`.
""")
code('''from pathlib import Path as _P
from v1_support import patients as P, groups as G
from v1_support.config import TARGET_LABELS as _TL

pinfo = P.verify_map(accepted)
from collections import Counter as _C
print("recovery methods:", dict(_C(v["method"] for v in pinfo.values())))
print("records w/ patient_id:", sum(1 for v in pinfo.values() if v["patient_id"]))

groups = G.build_groups(accepted, pinfo)
multi = {g: v for g, v in groups.items() if len(v) > 1}
print(f"atomic groups: {len(groups)} ({len(multi)} multi-record)")
et = _C(G.group_edge_types(v, pinfo) for v in multi.values())
print("multi-group edge types:", dict(et))

splits_dir = _P(cfg.dataset_dir) / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)
G.write_duplicate_audit(groups, pinfo, splits_dir / "duplicate_groups.csv")

code_map = G.load_code_to_targets(
    _P(cfg.subset_dir) / "metadata" / "diagnosis_mapping.csv")
conflicts = G.find_conflicting_groups(groups, code_map)
G.write_conflict_audit(conflicts, splits_dir / "conflicting_duplicate_groups.csv")
print("conflict taxonomy:", dict(_C(c["taxonomy"] for c in conflicts)))
CONFLICT_IDS = {m["record_id"] for c in conflicts for m in c["members"]}
print(f"conflicting records (metric-excluded in eval): {len(CONFLICT_IDS)}")

# patient recovery report (PLAN §4.5 audit sidecar)
rec_methods = _C(v["method"] for v in pinfo.values())
pats = {v["patient_id"] for v in pinfo.values() if v["patient_id"]}
rep = ["patient recovery report (PLAN §0.2)",
       f"records with recovered patient_id: {len(pats)} patients / "
       f"{sum(1 for v in pinfo.values() if v['patient_id'])} records",
       f"methods: {dict(rec_methods)}",
       "per-source recoverability:",
       "  ptb-xl-clean-explicit-normal: filename join (upstream paths preserved)",
       "  ptb-xl: byte-exact signal-content join vs local upstream copy; "
       "567 misses = absent from incomplete local copy (verified to 8uV)",
       "  ptb: unrecoverable (no upstream copy; challenge names lack patient)",
       "  chapman_shaoxing/ningbo/cpsc_2018/cpsc_2018_extra/georgia: "
       "no upstream identity by construction (challenge data)",
       "union groups: 29564 expected; got %d" % len(groups)]
(splits_dir / "patient_recovery_report.txt").write_text("\\n".join(rep) + "\\n")
print("\\n".join(rep))
''')

# ==================================================  7. split
md("""## 7. Create train/validation/test split (PLAN §4, §13.7)

Grouped iterative multi-label stratification (80/10/10, seed 20260918) over
the 29,564 union groups. No blanket rare-combo forcing — pure greedy, with
logged minimal repair only if feasibility gates fail. Manifests are written
**before any image exists** and are never overwritten (resume reuses them).
""")
code('''from pathlib import Path as _P
from v1_support import split as S

splits_dir = _P(cfg.dataset_dir) / "splits"
have = all((splits_dir / f"{s}_records.csv").exists()
           for s in ("train", "val", "test"))
if have:
    print("split manifests exist — loading + re-verifying (no re-split)")
    import csv as _csv
    assign, split_of = {}, {}
    for s in ("train", "val", "test"):
        with open(splits_dir / f"{s}_records.csv", newline="") as fh:
            for row in _csv.DictReader(fh):
                assign.setdefault(row["dup_group_id"], s)
                split_of[row["record_id"]] = s
    assert len(split_of) == 31109, len(split_of)
    moves = ["(loaded from disk; no new repair)"]
else:
    assign = S.iterative_split(groups, seed=cfg.seed)
    ev0 = S.evaluate(assign, groups)
    viol = S.check_gates(ev0, len(accepted))
    moves = []
    if viol:
        print(f"greedy infeasible ({len(viol)} violations) — minimal repair:")
        for v in viol[:8]:
            print("  ", v)
        assign, moves = S.repair(assign, groups, seed=cfg.seed)
        viol2 = S.check_gates(S.evaluate(assign, groups), len(accepted))
        assert not viol2, viol2
    S.save_splits(splits_dir, assign, groups, pinfo, moves)
    split_of = {}
    for gid, s in assign.items():
        for m in groups[gid]:
            split_of[m["record_id"]] = s

print("split report:")
print((splits_dir / "split_report.txt").read_text())
''')

# ==================================================  8. split integrity
md("""## 8. Validate split integrity (PLAN §4.3–4.4, §13.8)

Independent re-verification: counts, 12×3 label table + gates, source-balance
gates, union-group atomicity, and recovered-patient atomicity.
""")
code('''from collections import Counter as _C, defaultdict as _dd
from v1_support import split as S

ev = S.evaluate(assign, groups)
viol = S.check_gates(ev, len(accepted))
assert not viol, viol
print("feasibility gates: PASS")
print("\\n12×3 label table (train/val/test):")
for lab in ("NORM","AF","IAVB","LBBB","RBBB","PAC","PVC","LAFB","LAE","TInv","LQT","PRWP"):
    c = ev["label"][lab]
    print(f"  {lab:5s} {c['train']:6d} {c['val']:6d} {c['test']:6d}")
print("\\nsource table (train/val/test):")
for src in sorted(ev["source"]):
    c = ev["source"][src]
    print(f"  {src:28s} {c['train']:6d} {c['val']:6d} {c['test']:6d}")

# atomicity: every union group + every recovered patient in exactly one split
for gid, members in groups.items():
    assert len({split_of[m["record_id"]] for m in members}) == 1, gid
byp = _dd(set)
for r in accepted:
    pid = pinfo[r["record_id"]]["patient_id"]
    if pid:
        byp[pid].add(split_of[r["record_id"]])
assert all(len(v) == 1 for v in byp.values()), "patient spans splits!"
print(f"\\natomicity PASS: {len(groups)} groups, {len(byp)} patients, 0 spans")
''')

# ==================================================  9. renderer config
md("""## 9. Configure ECG renderer (PLAN §6, §13.9)

Stages the preview records to scratch `.dat` (WFDB transcode, **no
additional signal-domain filtering** — int16/1000 only, round-trip verified)
and echoes the effective kit configuration.
""")
code('''from v1_support import stage as ST
from v1_support.config import KIT_RENDER_FLAGS, KIT_ENV, FIXED_FONT

# preview record picks: 1/label + 1kHz edge + triple-label + dup member
import random as _r
_r.seed(cfg.seed)
_by_label = {}
for r in accepted:
    for lab in r["labels_tuple"]:
        _by_label.setdefault(lab, []).append(r)
PREVIEW_WANT = ["NORM","AF","PAC","PVC","IAVB","LBBB","RBBB","LAE","TInv","LQT","PRWP"]
preview = []
for lab in PREVIEW_WANT:
    cands = [r for r in _by_label[lab] if len(r["labels_tuple"]) == 1]
    preview.append(_r.choice(cands or _by_label[lab]))
preview.append(next(r for r in accepted if r["fs"] == 1000.0))          # 1 kHz
preview.append(next(r for r in accepted if len(r["labels_tuple"]) >= 3)) # triple
_dups = [r for r in accepted if r["record_id"] in
         {m["record_id"] for g in groups.values() if len(g) > 1 for m in g}
         and r not in preview]
preview.append(_r.choice(_dups))
PREVIEW_IDS = [r["record_id"] for r in preview]
print(f"preview records ({len(preview)}):")
for r in preview:
    print(f"  {r['record_id']:45s} {r['target_labels']:12s} fs={r['fs']:.0f}")

scratch_prev = Path(cfg.scratch_dir) / "preview"
staged_prev = ST.stage_chunk(
    [{"record_id": r["record_id"], "subset_output_path": r["subset_output_path"]}
     for r in preview], scratch_prev)
print("\\nkit flags:", " ".join(KIT_RENDER_FLAGS))
print("kit env:", KIT_ENV, "| font:", FIXED_FONT)
print("staged:", len(staged_prev), "records (round-trip verified, unfiltered)")
''')

# ==================================================  10. preview clean
md("""## 10. Generate clean preview images (PLAN §12, §13.10)

Renders Arial masters for the preview records, derives the canonical clean
view (fixed red grid), resizes to 1024×768, and saves under `preview/`.
""")
code('''from pathlib import Path as _P
from PIL import Image as _I
from v1_support import render as R, variants as V, resize as RS
cfg.kit_dir = R.resolve_kit_dir(cfg.work_dir, cfg.kit_dir)

prev_render = _P(cfg.preview_dir) / "render_arial"
R.run_batch(cfg.kit_dir, cfg.kit_python, scratch_prev, prev_render, FIXED_FONT,
            seed=cfg.seed)
got = R.collect_pngs(prev_render, set(PREVIEW_IDS))
print(f"preview masters rendered: {len(got)}")
for rid, png in sorted(got.items()):
    with _I.open(png) as im:
        assert im.size == (2200, 1700), (rid, im.size)
        grey = im.convert("L")
        view, meta = V.make_clean(grey)
        final, arith = RS.to_final(view)
        out = _P(cfg.preview_dir) / f"{rid}__clean.png"
        final.save(out, "PNG", optimize=True)
        assert final.size == (1024, 768)
print(f"clean previews saved: {len(got)} (2200×1700 → 994×768 → pad → 1024×768)")
''')

# ==================================================  11. preview variants
md("""## 11. Generate preview variants (PLAN §12, §13.11)

Stress view + 4 augmented samples spanning the effect space + one
alternate-font augmented sample (proves the §6.7 font path) per record.
""")
code('''from pathlib import Path as _P
from PIL import Image as _I
from v1_support import render as R, variants as V, resize as RS
from v1_support.util import rng_from_parts as _rng
cfg.kit_dir = R.resolve_kit_dir(cfg.work_dir, cfg.kit_dir)

tex_dir = str(_P(cfg.kit_dir) / "CreasesWrinkles" / "wrinkles-dataset")
fonts = R.kit_fonts(cfg.kit_dir)
alt_font = next(f for f in sorted(fonts) if f != FIXED_FONT)
print("alternate preview font:", alt_font)

# alternate-font master for ONE preview record (font-path proof)
demo = PREVIEW_IDS[0]
font_stage = _P(cfg.scratch_dir) / "preview_font"
from v1_support import stage as ST
ST.stage_chunk([{"record_id": demo,
                 "subset_output_path": next(
                     r["subset_output_path"] for r in preview
                     if r["record_id"] == demo)}], font_stage)
font_render = _P(cfg.preview_dir) / "render_font"
R.run_batch(cfg.kit_dir, cfg.kit_python, font_stage, font_render, alt_font,
            seed=cfg.seed)
font_png = R.collect_pngs(font_render, {demo})[demo]

preview_meta = {}
for rid, png in sorted(R.collect_pngs(_P(cfg.preview_dir) / "render_arial",
                                   set(PREVIEW_IDS)).items()):
    with _I.open(png) as im:
        grey = im.convert("L")
    stress, smeta = V.make_stress(grey, tex_dir)
    f, _ = RS.to_final(stress)
    f.save(_P(cfg.preview_dir) / f"{rid}__stress.png", "PNG", optimize=True)
    preview_meta[f"{rid}__stress.png"] = smeta
    preview_meta[f"{rid}__clean.png"] = {"grid_palette": "red",
                                         "rotation_deg": 0.0}
    for k in ("v01", "v02", "v03", "v04"):
        aug, meta = V.make_augmented(grey, rid, k, tex_dir, seed=cfg.seed)
        f, _ = RS.to_final(aug)
        f.save(_P(cfg.preview_dir) / f"{rid}__{k}.png", "PNG", optimize=True)
        preview_meta[f"{rid}__{k}.png"] = meta
        if rid == PREVIEW_IDS[0]:
            print(f"  {rid}__{k}: palette={meta['grid_palette']} "
                  f"rot={meta['rotation_deg']}")
import json as _json11
(_P(cfg.preview_dir) / "preview_meta.json").write_text(
    _json11.dumps(preview_meta, indent=2))
with _I.open(font_png) as im:
    aug, _ = V.make_augmented(im.convert("L"), demo, "v01", tex_dir,
                              seed=cfg.seed)
f, _ = RS.to_final(aug)
f.save(_P(cfg.preview_dir) / f"{demo}__v01_font-{alt_font}.png", "PNG", optimize=True)
print("preview variants saved (clean shown in §10; +stress, v01–v04, font demo)")
''')

# ==================================================  12. display previews
md("""## 12. Display and validate previews (PLAN §12, §10.7, §13.12)

Inline thumbnails (full PNGs stay in `outputs/preview/`) + measured report
(dims, DPI, layout/order probe, footer, rhythm span, resize arithmetic,
byte sizes, disk re-projection) + the empirical grid-distortion report
and gates. **The build stops here until `APPROVE_PREVIEW = True` is set
AND distortion gates pass.**
""")
code('''from pathlib import Path as _P
from PIL import Image as _I
from IPython.display import display as _display
from v1_support import gridcheck as GC, render as R
import json as _json

# --- inline display: thumbnails (v1.11: 56 full PNGs ≈ 100 MB over IOPub
# stalled the kernel mid-cell on EC2; full files stay in preview_dir) --------
for rid in PREVIEW_IDS:
    print("=" * 100)
    print(rid)
    for suf in ("__clean.png", "__stress.png", "__v01.png", "__v02.png"):
        p = _P(cfg.preview_dir) / f"{rid}{suf}"
        with _I.open(p) as im:
            print(f"--- {p.name} {im.size} ---")
            thumb = im.copy()
        thumb.thumbnail((512, 384), _I.LANCZOS)
        _display(thumb)
    # zoomed crop: proves 1 mm grid + trace legibility at final size
    with _I.open(_P(cfg.preview_dir) / f"{rid}__clean.png") as im:
        _display(im.crop((300, 250, 620, 450)))
        print("(zoomed 320×200 crop above)")

# --- measured report ----------------------------------------------------------
import os as _os
sizes = [(p.stat().st_size, p.name)
         for p in _P(cfg.preview_dir).glob("*__*.png")]
avg_clean = sum(s for s, n in sizes if "__clean" in n) / max(
    1, sum(1 for _, n in sizes if "__clean" in n))
avg_aug = sum(s for s, n in sizes if "__v0" in n) / max(
    1, sum(1 for _, n in sizes if "__v0" in n))
n_train_img = sum(1 for r in accepted if split_of[r["record_id"]] == "train") * 4
n_eval_img = sum(1 for r in accepted if split_of[r["record_id"]] != "train") * 2
proj_train_gb = (n_train_img * (avg_clean + 3 * avg_aug) / 4) / 2**30
proj_eval_gb = (n_eval_img * avg_clean) / 2**30
print(f"\\nmean bytes: clean={avg_clean:,.0f} augmented={avg_aug:,.0f}")
print(f"projected training PNGs: {proj_train_gb + proj_eval_gb:.0f} GB "
      f"(train {proj_train_gb:.0f} + eval {proj_eval_gb:.0f})")
st = _os.statvfs(cfg.work_dir)
free_gb = st.f_bavail * st.f_frsize / 2**30
need = 15 + (proj_train_gb + proj_eval_gb) * 2 + 10
print(f"free {free_gb:.0f} GB vs need ~{need:.0f} GB (masters + PNGs + TAR + headroom)")
assert free_gb >= need, "disk gate FAILED — grow the volume before §13"

# --- annotation probe: lead order, spans, grid, footer (1 record, discarded) --
from v1_support import render as R
cfg.kit_dir = R.resolve_kit_dir(cfg.work_dir, cfg.kit_dir)
probe_stage = _P(cfg.scratch_dir) / "probe"
from v1_support import stage as ST
ST.stage_chunk([{"record_id": PREVIEW_IDS[0],
                 "subset_output_path": next(
                     r["subset_output_path"] for r in preview
                     if r["record_id"] == PREVIEW_IDS[0])}], probe_stage)
probe_out = _P(cfg.preview_dir) / "render_probe"
R.run_batch(cfg.kit_dir, cfg.kit_python, probe_stage, probe_out, FIXED_FONT,
            seed=cfg.seed,
            extra_flags=["--store_config", "1", "--lead_bbox", "--lead_name_bbox"])
js = list(probe_out.rglob("*.json"))
assert len(js) == 1, js
ann = _json.loads(js[0].read_text())
names = [e["lead_name"] for e in ann["leads"]]
print("\\nprobe lead order (bottom-up plot order):", names)
assert names[:12] == ["III", "aVF", "V3", "V6", "II", "aVL",
                      "V2", "V5", "I", "aVR", "V1", "V4"], names
assert names[12] == "II" and len(names) == 13, names
rate = ann["sampling_frequency"]
spans = [(e["lead_name"], e["end_sample"] - e["start_sample"])
         for e in ann["leads"]]
assert all(s == int(rate * 2.5) for _, s in spans[:12]), spans
assert spans[12][1] == int(rate * 10), spans
print(f"probe spans OK: 12×{rate * 2.5:.0f} + rhythm {rate * 10:.0f} @ {rate} Hz")
print(f"probe canvas: {ann['width']}x{ann['height']} @ {ann['resolution']} DPI, "
      f"grid {ann['x_grid']}/{ann['y_grid']} px")
shutil.rmtree(probe_out, ignore_errors=True)  # probe artifacts never ship

# --- empirical grid-distortion report + gates ----------------------------------
meas, pairs = [], []
pmeta = _json.loads((_P(cfg.preview_dir) / "preview_meta.json").read_text())
for rid in PREVIEW_IDS:
    c = _P(cfg.preview_dir) / f"{rid}__clean.png"
    m = GC.measure_image(c, "red", GC.nominal_for_size(1024, 768))
    meas.append(m)
    for suf in ("__stress.png", "__v01.png", "__v02.png"):
        p = _P(cfg.preview_dir) / f"{rid}{suf}"
        pal = pmeta[p.name]["grid_palette"]
        d = GC.measure_image(p, pal, GC.nominal_for_size(1024, 768))
        meas.append(d)
        pairs.append((m, d))
rep = GC.gate_report(meas, pairs)
(_P(cfg.preview_dir) / "distortion_report_preview.json").write_text(
    _json.dumps(rep, indent=2) + "\\n")
print("\\ndistortion report:", _json.dumps(
    {k: v for k, v in rep.items() if k != "failures"}, indent=2))
assert rep["pass"], f"DISTORTION GATES FAILED: {rep['failures'][:5]}"
print("distortion gates: PASS")

# --- human gate -----------------------------------------------------------------
assert cfg.approve_preview, (
    "STOP: review the images + report above, then set approve_preview=True "
    "in §1, re-run §1, then re-run this cell (kit paths self-heal).")
print("\\nPREVIEW APPROVED — §13 unlocked.")
''')

# ==================================================  13. full generation
md("""## 13. Full image generation (PLAN §13.13)

Chunked (200 records), multiprocessing over `workers`, atomic per-chunk
CSVs, validate-and-skip resume, streaming record-by-record (peak RAM ≈
workers × one master). Any failure appends a loud row to `failures.csv`
and aborts — never a silent skip.
""")
code('''import csv as _csv, time as _time
from concurrent.futures import ProcessPoolExecutor as _Pool, as_completed as _done
from v1_support import chunk as CH
from v1_support.config import CHUNK_SIZE

from v1_support import render as R
cfg.kit_dir = R.resolve_kit_dir(cfg.work_dir, cfg.kit_dir)
assert cfg.approve_preview, "§12 approval required before §13"
CH.rotate_failures_log(Path(cfg.dataset_dir) / "failures.csv")
st = os.statvfs(cfg.work_dir)
assert st.f_bavail * st.f_frsize / 2**30 >= 300, "need >=300 GB free for §13"

# RAM fit: measured peak is ~0.5 GB/worker flat vs chunk size (§20.8);
# budget 1 GB/worker (2x margin) against MemAvailable and refuse loudly
# instead of OOM-dying hours into the run. c8a.4xlarge: 16 x 1 « 32 GB.
def _mem_avail_gb():
    try:
        with open("/proc/meminfo") as _fh:
            for _ln in _fh:
                if _ln.startswith("MemAvailable:"):
                    return int(_ln.split()[1]) / 2**20
    except OSError:
        pass
    return 0.0
_avail = _mem_avail_gb()
print(f"RAM available: {_avail:.1f} GB | workers: {cfg.workers} "
      f"(budget 1 GB/worker)")
assert _avail == 0.0 or cfg.workers * 1.0 <= 0.85 * _avail, \
    f"RAM fit FAILED: {cfg.workers} workers need {cfg.workers} GB " \
    f"of {_avail:.0f} GB available — lower cfg.workers"

# freeze running config (provenance gate in §14 compares against this file)
gen_cfg_json = cfg.to_json()
Path(cfg.dataset_dir, "generation_config.json").write_text(gen_cfg_json)

# chunk jobs: stable order (split, record_id); resume skips finished chunks
gid_of = {m["record_id"]: g for g, v in groups.items() for m in v}
ordered = sorted(accepted, key=lambda r: (split_of[r["record_id"]],
                                          r["record_id"]))
chunks = [ordered[i:i + CHUNK_SIZE]
          for i in range(0, len(ordered), CHUNK_SIZE)]
P = {"scratch_dir": str(Path(cfg.scratch_dir) / "build"),
     "render_dir": str(Path(cfg.work_dir) / "render_build"),
     "master_dir": cfg.master_dir, "dataset_dir": cfg.dataset_dir,
     "index_parts_dir": cfg.index_parts_dir,
     "failures_csv": str(Path(cfg.dataset_dir) / "failures.csv"),
     "kit_dir": cfg.kit_dir, "kit_python": cfg.kit_python,
     "texture_dir": str(Path(cfg.kit_dir) / "CreasesWrinkles"
                        / "wrinkles-dataset")}

def _jobrec(r):
    return {"record_id": r["record_id"], "split": split_of[r["record_id"]],
            "subset_output_path": r["subset_output_path"],
            "multihot": r["multihot"],
            "dup_group_id": gid_of[r["record_id"]],
            "v1_waveform_sha256": r["v1_waveform_sha256"],
            "pinfo": pinfo[r["record_id"]],
            "conflicting": r["record_id"] in CONFLICT_IDS,
            "source_dataset": r["source_dataset"]}

jobs, skipped = [], 0
for cid, members in enumerate(chunks):
    if CH.chunk_complete(cfg.index_parts_dir, cid, cfg.dataset_dir):
        skipped += 1
        continue
    jobs.append({"chunk_id": cid, "records": [_jobrec(r) for r in members],
                 "paths": P, "seed": cfg.seed})
print(f"{len(chunks)} chunks ({len(ordered)} records): "
      f"{skipped} already complete, {len(jobs)} to run")

t0 = _time.time()
done = skipped
if jobs:
    with _Pool(max_workers=cfg.workers) as pool:
        futs = {pool.submit(CH.process_chunk, j): j["chunk_id"] for j in jobs}
        for fut in _done(futs):
            rec = fut.result()  # raises on chunk failure (failures.csv has it)
            done += 1
            el = (_time.time() - t0) / 60
            _tt = rec["timings"]
            _render = _tt["arial"] + _tt["font"]
            print(f"  [{done}/{len(chunks)}] chunk {rec['chunk_id']:05d}: "
                  f"{rec['n_images']} images "
                  f"({rec['bytes'] / 2**20:.0f} MB)  [{el:.0f} min elapsed] "
                  f"render={_render:.0f}s var={_tt['variants']:.0f}s "
                  f"save={_tt['save_png']:.0f}s",
                  flush=True)
print(f"\\n§13 complete in {(_time.time() - t0) / 3600:.1f} h")

# failures.csv must exist (possibly header-only) for the packaging gate
import pathlib as _pl
_fc = _pl.Path(cfg.dataset_dir) / "failures.csv"
if not _fc.exists():
    _fc.write_text("chunk_id,stage,error\\n")
else:
    with open(_fc) as fh:
        assert len(list(_csv.DictReader(fh))) == 0, "failures.csv non-empty!"
print("failures.csv: empty (no failures, no skips)")
''')

# ==================================================  14. validate images
md("""## 14. Validate generated images (PLAN §15, §13.14)

Gates 15.1–15.4 + 15.6 + 15.7-subset from per-chunk rows (leakage per key,
counts, subset separation, file integrity, pixels incl. the production
grid sample, provenance). Label-join and metric-separation gates run in
§15 right after final assembly.
""")
code('''import csv as _csv, glob as _glob
from v1_support import validate as VV
from v1_support import render as R
cfg.kit_dir = R.resolve_kit_dir(cfg.work_dir, cfg.kit_dir)

from v1_support import chunk as CH
CH.backfill_hash_column(cfg.index_parts_dir,
                        {r["record_id"]: r["v1_waveform_sha256"]
                         for r in accepted})
chunk_rows = []
for f in sorted(_glob.glob(str(Path(cfg.index_parts_dir) / "chunk_*.csv"))):
    with open(f, newline="") as fh:
        chunk_rows.extend(list(_csv.DictReader(fh)))
print(f"chunk rows: {len(chunk_rows)}")

VV.gate_leakage(chunk_rows)
VV.gate_counts(chunk_rows, {r["record_id"] for r in accepted},
               set(xcheck["omitted"]))
VV.gate_subset_separation(chunk_rows)
VV.gate_files(chunk_rows, cfg.dataset_dir)
VV.gate_pixels(chunk_rows, cfg.dataset_dir, cfg.master_dir,
               str(Path(cfg.kit_dir) / "CreasesWrinkles" / "wrinkles-dataset"),
               seed=cfg.seed)
VV.gate_provenance(accepted, Path(cfg.dataset_dir) / "generation_config.json",
                   gen_cfg_json, seed=cfg.seed)
print("\\n§14 gates 15.1–15.4 + 15.6 + 15.7-subset: ALL PASS")
''')

# ==================================================  15. metadata
md("""## 15. Generate metadata/manifests (PLAN §14, §13.15)

Assembles per-subset `metadata.csv` (train + val_clean/val_stress/
test_clean/test_stress) + `full_metadata.csv` from chunk rows, writes
palette/font audits, then runs the label-join (15.5), subset-separation
and metric-separation (15.7) gates on the final files.
""")
code('''import csv as _csv, glob as _glob
from v1_support import validate as VV
from v1_support.chunk import CHUNK_FIELDS
from v1_support.config import SUBSETS

chunk_rows = []
for f in sorted(_glob.glob(str(Path(cfg.index_parts_dir) / "chunk_*.csv"))):
    with open(f, newline="") as fh:
        chunk_rows.extend(list(_csv.DictReader(fh)))
VORDER = {"clean": 0, "v01": 1, "v02": 2, "v03": 3, "stress": 4}
SORDER = {s: i for i, s in enumerate(SUBSETS)}
chunk_rows.sort(key=lambda r: (SORDER[r["subset"]], r["record_id"],
                               VORDER[r["variant_id"]]))
for s in SUBSETS:
    sub = [r for r in chunk_rows if r["subset"] == s]
    assert sub, f"empty subset {s}"
    with open(Path(cfg.dataset_dir) / s / "metadata.csv", "w",
              newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=CHUNK_FIELDS)
        w.writeheader()
        w.writerows(sub)
    print(f"{s}/metadata.csv: {len(sub)} rows")
with open(Path(cfg.dataset_dir) / "full_metadata.csv", "w", newline="") as fh:
    w = _csv.DictWriter(fh, fieldnames=CHUNK_FIELDS)
    w.writeheader()
    w.writerows(chunk_rows)
print(f"full_metadata.csv: {len(chunk_rows)} rows")
VV.write_audits(chunk_rows, cfg.dataset_dir)

# final-file gates
rows, full = VV.load_metadata(cfg.dataset_dir)
assert len(rows) == len(full) == len(chunk_rows)
mh = {r["record_id"]: r["multihot"] for r in accepted}
VV.gate_labels(rows, mh)
VV.gate_subset_separation(rows)
VV.gate_metric_separation(rows, CONFLICT_IDS)
print("\\n§15 gates 15.5 + 15.7: ALL PASS")
''')

# ==================================================  16. package TAR
md("""## 16. Package dataset into TAR (PLAN §16, §13.16)

Deterministic plain `.tar` + `.sha256` sidecar, listing-verified against
`full_metadata.csv`. Masters stay out; optional second masters archive.
""")
code('''from v1_support import package as PKG
from v1_support.config import TRAIN_TAR, MASTERS_TAR

top = ["train", "val_clean", "val_stress", "test_clean", "test_stress",
       "splits", "full_metadata.csv",
       "generation_config.json", "grid_palette_by_label.csv",
       "grid_palette_audit.json", "font_by_label.csv",
       "distortion_report.json", "failures.csv"]
tar = PKG.build_tar(cfg.dataset_dir, Path(cfg.work_dir) / TRAIN_TAR, top)
PKG.verify_tar_listing(
    tar["path"], Path(cfg.dataset_dir) / "full_metadata.csv",
    set(top) | {"splits/train_records.csv", "splits/val_records.csv",
                "splits/test_records.csv", "splits/duplicate_groups.csv",
                "splits/conflicting_duplicate_groups.csv",
                "splits/patient_recovery_report.txt",
                "splits/split_report.txt"})
print(f"training TAR ready: {tar['path']} ({tar['bytes'] / 2**30:.1f} GB)")

masters_tar = None
if cfg.upload_masters_tar:
    import shutil as _sh
    mpkg = Path(cfg.work_dir) / "masters_pkg"
    if mpkg.exists():
        _sh.rmtree(mpkg)
    (mpkg / "master").mkdir(parents=True)
    for p in Path(cfg.master_dir).glob("*.png"):
        (mpkg / "master" / p.name).hardlink_to(p)
    _sh.copy(Path(cfg.dataset_dir) / "generation_config.json", mpkg)
    _sh.copytree(Path(cfg.dataset_dir) / "splits", mpkg / "splits")
    masters_tar = PKG.build_tar(mpkg.parent, Path(cfg.work_dir) / MASTERS_TAR,
                                ["masters_pkg"])
    print(f"masters TAR ready: {masters_tar['path']} "
          f"({masters_tar['bytes'] / 2**30:.1f} GB)")
''')

# ==================================================  17. S3 upload
md("""## 17. Upload TAR to S3 (PLAN §17, §13.17)

IAM-role auth (no keys), multipart + CRC64NVME upload checksum, HEAD checks,
and the required independent verification (composed-local preferred,
re-download fallback). Loud failure on anything red.
""")
code('''import importlib as _il
from v1_support import s3util as S3U
from v1_support.config import TRAIN_TAR, MASTERS_TAR

# S3 deps are REQUIRED here (hard fail with retries; §2 may have deferred).
try:
    _il.import_module("boto3")
    _il.import_module("awscrt")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                    "--retries", "5", "--timeout", "120",
                    "boto3>=1.34", "awscrt"], check=True)
import boto3 as _b3
S3U.check_crc64nvme_support()
S3U.ensure_awscrt()

s3 = _b3.client("s3")
uploads = [Path(cfg.work_dir) / TRAIN_TAR]
if cfg.upload_masters_tar:
    uploads.append(Path(cfg.work_dir) / MASTERS_TAR)
S3_RECORDS = []
for tar_path in uploads:
    assert tar_path.is_file(), tar_path
    key = f"{cfg.s3_prefix}/{tar_path.name}"
    rec = S3U.upload_and_verify(s3, tar_path, cfg.s3_bucket, key,
                                force_redownload=cfg.verify_s3_bytes)
    S3_RECORDS.append(rec)
    # sidecar object next to the TAR
    side = Path(str(tar_path) + ".sha256")
    s3.upload_file(str(side), cfg.s3_bucket, key + ".sha256")
    print("sidecar:", key + ".sha256")
print("\\nS3 records:")
for rec in S3_RECORDS:
    print(f"  {rec['s3_uri']}  bytes={rec['bytes']}  "
          f"crc64nvme={rec['crc64nvme']}  verify={rec['verify_method']}  "
          f"reused={rec['reused']}")
''')

# ==================================================  18. final report
md("""## 18. Final validation and summary (PLAN §18, §13.18)

Re-runs leakage + counts on the final metadata, prints the full report, and
saves `final_report.txt`. No model training starts here.
""")
code('''from collections import Counter as _C
from v1_support import validate as VV

rows, _ = VV.load_metadata(cfg.dataset_dir)
VV.gate_leakage(rows)
nrec = {s: len({r["record_id"] for r in rows if r["split"] == s})
        for s in ("train", "val", "test")}
nimg = _C(r["subset"] for r in rows)
nprim = sum(1 for r in rows if int(r["primary_metric"]) == 1)
tar_p = Path(cfg.work_dir) / TRAIN_TAR
import hashlib as _h
digest = _h.sha256()
with open(tar_p, "rb") as fh:
    for b in iter(lambda: fh.read(1 << 24), b""):
        digest.update(b)

report = f"""Source accepted records:            31109
Split method:                       grouped iterative multi-label stratification
                                    (union-find: patient_id + v1_waveform_sha256;
                                    29564 groups; no blanket rare-combo forcing)
Split seed:                         {cfg.seed}
Patients recovered:                 {sum(1 for v in pinfo.values() if v["patient_id"])} records w/ ptbxl patient_id
Train / Val / Test source records:  {nrec["train"]} / {nrec["val"]} / {nrec["test"]}
Train variants/record:              4 (clean+v01..v03)
Eval subsets (1 img/record each):   val_clean / val_stress / test_clean / test_stress
Total images:                       {len(rows)}
Subset train/val_clean/val_stress/  {nimg["train"]} / {nimg["val_clean"]} / {nimg["val_stress"]} /
  test_clean/test_stress images:    {nimg["test_clean"]} / {nimg["test_stress"]}
Primary-metric images:              {nprim} (val_clean + test_clean, non-conflicting)
Images failed / skipped:            0 / 0
Master DPI:                         200 (2200x1700; ~98.4 Hz master only)
Final image dimensions:             1024x768 (WxH; ~90.4 DPI effective)
Resize/padding method:              LANCZOS to 994x768 + 15px paper-colour pad/side
Paper speed / Gain:                 25 mm/s / 10 mm/mV
Layout:                             3x4 (2.5 s/lead) + full 10 s Lead II strip
Rhythm strip:                       II, continuous 10 s (rate x 10 samples)
Output format:                      PNG
Filtering:                          no additional signal-domain filtering
Variant settings:                   PLAN §10.3 bounds + seed + palette weights +
                                    seeded train fonts + fixed stress recipe
Grid-distortion validation:         PASS (see distortion_report.json)
Metadata rows:                      {len(rows)} (join-verified)
Leakage validation:                 PASS (0 overlap on all keys, all pairs)
TAR path / size:                    {tar_p} / {tar_p.stat().st_size} bytes
TAR sha256:                         {digest.hexdigest()}
S3 destination / upload status:     {[r["s3_uri"] for r in S3_RECORDS]} / verified
Overall validation:                 PASS (all §15 gates green)
"""
print(report)
Path(cfg.work_dir, "final_report.txt").write_text(report)
print("Done. No model training started (by design).")
''')


def build() -> Path:
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python",
                                      "name": "python3"}},
          "cells": []}
    for kind, src in CELLS:
        if kind == "markdown":
            nb["cells"].append({"cell_type": "markdown", "metadata": {},
                                "source": src})
        else:
            nb["cells"].append({"cell_type": "code", "metadata": {},
                                "execution_count": None, "outputs": [],
                                "source": src})
    out = HERE / "v1_image_generation.ipynb"
    out.write_text(json.dumps(nb, indent=1) + chr(10))
    print(f"notebook: {out} ({len(CELLS)} cells)")
    return out


if __name__ == "__main__":
    build()


