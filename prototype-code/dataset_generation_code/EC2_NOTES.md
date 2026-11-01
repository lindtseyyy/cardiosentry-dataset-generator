# EC2 handoff notes — V1 ECG image generation (PLAN v1.11)

## What this directory is

`dataset_generation_code/` contains everything the EC2 build needs:

```text
dataset_generation_code/
├── PLAN.md                        # the approved plan (v1.11 + §20 calibrations)
├── EC2_NOTES.md                   # this file
├── v1_image_generation.ipynb      # the 18-section build notebook (36 cells)
├── make_notebook.py               # regenerates the notebook (reviewable source)
├── build_code_bundle.py           # rebuilds ecg_image_v1_code.tar.gz
├── build_patient_map.py           # rebuilds v1_support/data (needs upstream copy)
├── ecg_image_v1_code.tar.gz       # patched kit + fonts + textures (+ .sha256)
├── v1_support/                    # notebook-imported package (all stages)
│   ├── data/                      # shipped PTB-XL patient map + db extract
│   └── tests/                     # 58 tests green (gridcheck 9, subsets 15, package 5, s3input 16, bootstrap 6, paths 6, thumbnails 1)
└── outputs/                       # build outputs (see below)
```

## Transfer to EC2

1. Copy this whole directory to the instance, e.g.
   `~/CardioSentry/dataset_generation_code/`.
2. The input dataset pulls itself from S3 on first run — no manual copy
   needed. If `~/CardioSentry/V1_10S_STANDARDIZED_SUBSET/` is absent, §5
   downloads `s3://cardiosentry-fcmae-758821198776-us-east-1-an/datasets/raw_signals_data/V1/V1_10S_STANDARDIZED_SUBSET.tar.gz`
   (~2 GB, sidecar-pinned SHA-256, resume-by-hash), verifies it, and
   extracts to `outputs/input/`. A manual copy to that path still works
   as a fallback (it is used as-is when present; override
   `cfg.subset_dir` in §1 if yours differs). Either way §5 re-hashes
   all 62,218 files and aborts on any mismatch.
3. `outputs/` already contains `dataset/splits/` from the local preview
   rehearsal (same code + seed 20260918 → identical on EC2). The notebook
   reuses them (never re-splits); to force a fresh split, delete
   `outputs/dataset/splits/*_records.csv` first (audits + report regenerate).

## Instance sizing (measured, not guessed)

Target: **c8a.4xlarge (16 vCPU / 32 GB)**, CPU-only, 16 workers
(`workers=0` = `cpu_count`; every worker is single-threaded, so 16
workers subscribe all 16 threads — more cannot help, fewer idle cores).

- **≥ 400 GB free gp3** on the build volume (measured peak ≈ 310 GB:
  ~11 GB masters + ~143 GB PNGs + ~143 GB TAR + headroom). §12
  re-projects from measured bytes and refuses §13 on short disk; §13
  floors at 300 GB.
- RAM: measured worker peak is ~0.5 GB flat vs chunk size → 16 workers
  ≈ 7 GB of 32 GB. §13 enforces a RAM-fit guard (workers × 1 GB ≤ 85%
  MemAvailable) and refuses to start rather than OOM mid-run.
- Expected wall time: **≈ 7–10 h total**, likely ~8 h (§13 ≈ 5–7 h at
  the measured 9.0 s/record pace on a much weaker chip — Genoa will
  beat it; validate/pack ≈ 1 h; S3 upload 1–2 h for the ~150 GB TAR).
  Each §13 chunk line prints live render/variant/save timings, so the
  real rate is visible within the first half hour. Every stage resumes.

## Launch

```bash
cd ~/CardioSentry/dataset_generation_code
python3 -m venv .venv && . .venv/bin/activate
pip install jupyter
jupyter lab v1_image_generation.ipynb   # or: jupyter nbconvert --to notebook --execute (not recommended: §12 needs eyes)
```

§1 pip-installs the remaining Python deps (numpy/scipy/matplotlib/…)
into the kernel env itself before importing anything — expect a few
minutes on first run, then it is a no-op on re-runs.

Run §§1–12, **look at the preview images**, then set
`cfg.approve_preview = True` in §1 and continue. The full build (§13)
cannot start before that flag AND the distortion gates pass.

## What was already verified locally (rehearsal §§1–12, exit 0)

- 31,109 accepted rows; 61+19 omissions absent; 62,218 file SHAs match.
- 11,132 patient IDs verified; 29,564 union groups; 31 conflict groups
  audited (mapping recompute clean); split 24,881/3,115/3,113, 0 repair.
- Kit probe: lead order, 12×2.5 s spans + 10 s rhythm, 200 DPI, 39.37 px
  grid — all asserted, all pass. Native-1 kHz record renders correctly.
- 85 preview images inspected (clean/stress/4×augmented/font-demo):
  layout, labels, pulses, footer, pad seams, ±2.88° extremes — all good.
- Distortion gates PASS (worst nominal 3.0%, worst mismatch 2.8%).
- Chunk worker (§13 unit) tested end-to-end on 3 records (train + val +
  test) incl. font pass: 8 images land in `train/` + the four eval
  subset dirs with matching `subset` column; subset-separation,
  metric-separation, file-integrity, and label-join gates pass; 5
  per-subset `metadata.csv` + `full_metadata.csv` assemble and reload;
  deterministic mini-TAR verifies with no legacy `val/`/`test/` dirs.
- `verify_tar_listing` bug fixed (v1.2): it rejected per-subset
  `metadata.csv` members, which would have failed §16 on EC2.
- v1.3: EC2 pulls the input tarball from S3 when no local copy exists
  (sidecar-pinned SHA, verified download, safe extract). Live-tested
  against the real bucket (sidecar object round-trips byte-identical);
  full 2 GB pull runs on EC2. Regression suite: 42 tests green
  (`gridcheck` 9 + `subsets` 15 + `package` 5 + `s3input` 13).
- Rehearsal re-run green after the v1.2 regeneration (exit 0 with the
  §12 disk/human gates acknowledged: 87 GB free locally vs ~310 GB need,
  so the disk gate correctly refuses §13 on this box; distortion PASS at
  worst nominal 2.96% / mismatch 2.75%).
- v1.4: chunk worker profiled (4- and 12-record runs): linear scaling,
  9.0 s/record, ~0.43 GB peak RSS flat, stage split render 44% /
  variants 46% / PNG save 8% / resize 2%. Per-chunk timings now logged
  live in §13; RAM-fit guard added (16 × 1 GB « 32 GB on c8a.4xlarge).

## Not yet run (EC2-only by nature)

- §13 full build (112k images), §14–15 production validation sample,
  §16 TAR (~150 GB), §17 S3 upload to
  `s3://cardiosentry-fcmae-758821198776-us-east-1-an/datasets/V1/`.
- IAM role (exact policy below); no keys in the notebook. S3 uses
  CRC64NVME upload checksums + independent verification
  (composed-local via awscrt, else re-download hash).

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "InputRead", "Effect": "Allow",
     "Action": ["s3:GetObject"],
     "Resource": "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an/datasets/raw_signals_data/V1/*"},
    {"Sid": "OutputReadWrite", "Effect": "Allow",
     "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
     "Resource": "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an/datasets/V1/*"},
    {"Sid": "HeadAndList", "Effect": "Allow",
     "Action": ["s3:ListBucket"],
     "Resource": "arn:aws:s3:::cardiosentry-fcmae-758821198776-us-east-1-an",
     "Condition": {"StringLike": {"s3:prefix": ["datasets/*"]}}}
  ]
}
```

(`s3:GetObject` on the output prefix is required, not just `PutObject`:
the §17 resume check HEADs destination keys. Attach via EC2 console →
instance → Actions → Security → Modify IAM role; takes effect
immediately, no reboot. Verify on the instance:
`aws sts get-caller-identity` should show an AssumedRole.)

## If something fails

- `ModuleNotFoundError: No module named 'v1_support'` in §1 means the
  kernel didn't start in the project dir (v1.5 finds it automatically —
  re-copy the notebook if yours is older): either start Jupyter from
  inside `dataset_generation_code/`, or run
  `%cd ~/CardioSentry/dataset_generation_code`, restart the kernel, and
  re-run §1. If it persists, `v1_support/` itself is missing — re-copy
  the whole directory, not just the `.ipynb`.
- `TypeError ... StringArray` from `wfdb` at §1 import means a stale
  `wfdb==4.1.2` met pandas 3 (v1.7 pins `wfdb>=4.3` and upgrades broken
  installs in place — re-copy the notebook and re-run §1; no venv
  rebuild needed).
- `ImportError: cannot import name 'CLOUD_PROTOCOLS'` right after a pip
  upgrade means the running kernel holds stale modules from the failed
  import (the on-disk install is already fine): just restart the kernel
  and re-run §1. v1.8 detects this case and tells you to restart
  instead of re-importing in-process.
- `No AWS credentials found` / S3 `refused access` at §5 means the
  instance has no IAM role (or the role lacks a grant): attach the
  policy from §Transfer/IAM above, then re-run §5 — no kernel restart
  needed (v1.9 raises this as a short actionable error instead of a
  botocore traceback).
- `waveform gate failed for 31109: ... missing file` means absolute CSV
  paths weren't rebased (v1.10 re-anchors them in `load_accepted` —
  re-copy `v1_support/manifest.py`, restart the kernel, re-run from §5;
  the notebook file itself is unchanged apart from its version label).
- §12 stuck mid-display (`--- ...__v01.png ---` with no progress) means
  the 56 full-size inline PNGs (~100 MB) stalled IOPub: Kernel →
  Interrupt, view the full PNGs in `outputs/preview/` via the file
  browser, then run `import IPython.display; IPython.display.display =
  lambda *a, **k: None` in a scratch cell and re-run §12 (all gates
  still run; v1.11 thumbnails this permanently).
- Any stage aborts LOUDLY with offender identities (`failures.csv` for
  §13). Fix the cause, re-run the notebook: splits are reused, finished
  chunks are validate-and-skipped, S3 objects resume by HEAD match.
- Do not delete `outputs/index_parts/` mid-build (those are the resume
  checkpoints). `outputs/staged_scratch/` and `outputs/render_build/`
  are transient and safe to clear while the notebook is stopped.
