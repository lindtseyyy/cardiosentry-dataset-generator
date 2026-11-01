# V1 ECG Image Generation — Plan (v1.17, APPROVED with amendments)

## Changelog v1.16 → v1.17 (no-checksum S3 fallback)

1. **Missing S3 checksum routes to SHA-256 re-download proof.**
   S3 accepted the 146 GB multipart upload (size-correct) yet stored
   no `ChecksumCRC64NVME` — an SDK/S3-side storage gap, not an
   integrity signal — which the old code treated as fatal.
   CRC64NVME stays preferred (requested on every upload, composed
   path used whenever S3 stores it), but a missing checksum now
   falls back to the pre-approved path (b) re-download SHA-256
   byte-proof instead of raising; size-matching checksum-less
   objects are reused (no 146 GB re-upload on §17 re-run) and
   corruption is still caught by the re-hash. Covered by
   `test_s3_fallback.py` (5 tests, fake S3 client, no network).
   Notebook untouched; EC2 = copy `s3util.py`, re-run §17.

## Changelog v1.15 → v1.16 (determinism re-derive save params)

1. **`validate.rederive_final()` mirrors chunk.py byte-exactly.** The
   determinism spot-check saved its re-derives with default PNG
   params while chunk.py uses `optimize=True`: identical pixels,
   different bytes, phantom "nondeterministic re-derive" on EC2
   after all real gates passed. The re-derive is now a helper
   replicating chunk.py's mode pipeline + save params exactly
   (mode pipeline proven harmless: both recolours converge to L
   internally), called by `gate_pixels`. Covered by
   `test_rederive.py` (3 tests, incl. proof the params matter).
   Notebook untouched; EC2 = copy `validate.py`, run the two
   remaining gates.

## Changelog v1.14 → v1.15 (missing hash column false-LEAKAGE)

1. **Chunk rows carry `v1_waveform_sha256`; §14 backfills pre-v1.15
   csvs.** The column was never written (absent from CHUNK_FIELDS,
   never passed through `_jobrec`), so `gate_leakage` read `''` for
   all 111,980 rows and raised a phantom LEAKAGE spanning all
   splits. No data was wrong: the manifest holds the hash for all
   31,109 records (0 empties). `_jobrec` passes it through, new
   chunks write it, `chunk.backfill_hash_column` adds it to old
   csvs (atomic tmp+replace, idempotent, conflicts raise, no
   re-render), called at the top of §14; it flows into shipped
   `metadata.csv`/`full_metadata.csv` via CHUNK_FIELDS. The gate
   gained an all-empty-column pre-check naming the column instead
   of crying LEAKAGE. Covered by `test_chunk_schema.py` (7 tests).

## Changelog v1.13 → v1.14 (stale failures.csv on §13 resume)

1. **`chunk.rotate_failures_log()` rotates stale failure rows at §13
   start.** A chunk failure appends to `failures.csv` and kills the
   cell; the resumed run that overcomes it tripped the end-of-§13
   empty assert on the stale row (observed: run-1 chunk_00009
   staging row vs a fully complete run-2 build). Data-bearing logs
   move to `failures_<UTC-stamp>.csv` (audit preserved, kept out of
   the TAR); header-only/missing logs are untouched. Covered by
   `test_failures_rotation.py` (4 tests).

## Changelog v1.12 → v1.13 (INT16_MIN staging guard)

1. **`write_dat` accepts the true int16 range [-32768, 32767].**
   The old `abs().max() > 32767` guard rejected the legal value
   -32768 (INT16_MIN, ADC negative rail; 25 accepted records contain
   it, e.g. `ningbo__JS23786` with 26 railed samples), killing §13.
   Full-subset scan proves the fix is complete: all 21,727 `.mat`
   `val` arrays are int16 dtype, global range exactly
   [-32768, 32767], zero records outside int16 — so post-fix
   overflow is impossible and all 31,109 records round-trip
   bit-exactly with no omission, clipping, rescale, or filtering.
   Covered by `test_wfdbio_int16.py` (3 tests). Notebook untouched;
   EC2 resume = copy `v1_support/wfdbio.py`, re-run §13.

## Changelog v1.11 → v1.12 (§1 re-run self-healing kit path)

1. **Kit path self-heals after any §1 re-run.**
   `render.resolve_kit_dir(work_dir, kit_dir)` passes a valid entry
   through, else re-derives `<work_dir>/kit/ecg_image_v1_code/
   ecg-image-generator` (the data persists on disk; only the cfg field
   was wiped), and raises naming §2 only if nothing was extracted.
   Every kit consumer (§§3, 10, 11, 12 probe, 13, 14-gates) calls it
   first, so the approve flow (§1 edit → §1 re-run → §12) can no
   longer fail with the bare subprocess `can't open file` this
   replaced: `run_batch` now refuses empty/missing kits with an
   actionable `V1Error`. Covered by `test_kitresolve.py` (6 tests).

## Changelog v1.10 → v1.11 (§12 inline thumbnails)

1. **§12 displays thumbnails (≤512×384), not full PNGs.** 56 full-size
   inline images (~100 MB over IOPub) stalled the kernel mid-cell on
   EC2; full files stay in `outputs/preview/` for the approval eyeball.
   All §12 gates/report/probe logic unchanged. Covered by
   `test_thumbnails.py` (executes the real loop slice, asserts display
   sizes). Stopgap for a stuck §12 without re-copying: stub
   `IPython.display.display` to a no-op in a scratch cell, then run §12.

## Changelog v1.9 → v1.10 (rebase absolute CSV paths onto runtime dir)

1. **`load_accepted` re-anchors `subset_output_path` under the runtime
   `subset_dir`.** The column stores ABSOLUTE cutting-machine paths, used
   verbatim by every consumer (gates, staging, content-join, splits,
   chunk jobs, previews) — valid locally, "missing file" for all 31,109
   records on EC2. The portable tail after `V1_10S_STANDARDIZED_SUBSET/`
   is now re-anchored at the single choke point; already-relative
   values anchor directly, marker-less absolute values fail loudly.
   Verified: 31,109/31,109 exist post-rebase, zero strings changed on
   the cutting machine. Covered by `test_paths.py`.

## Changelog v1.8 → v1.9 (actionable S3 credential errors + IAM fix)

1. **Known S3 failures raise short actionable errors**: `download_file`
   translates `NoCredentialsError`/`PartialCredentialsError` and 403s
   into a `V1Error` naming the missing IAM grant + the re-run step
   (duck-typed — botocore stays optional); anything unrecognized still
   propagates raw. Covered by `test_s3input.py`.
2. **IAM docs corrected**: the role needs `s3:GetObject` on the output
   prefix too (the §17 resume check HEADs destination keys), not just
   `PutObject`. EC2_NOTES now carries the exact policy JSON.

## Changelog v1.7 → v1.8 (stale-kernel re-import fix)

1. **§1 verifies installs in a fresh interpreter + demands restart after
   replacing broken packages.** Root cause of the EC2 `ImportError:
   CLOUD_PROTOCOLS`: a failed import leaves stale old submodules cached
   in `sys.modules` (proven: old `wfdb.io._coreio` without the symbol
   survives the TypeError), so the same-process re-import after pip
   upgrade mixes cached-old with new files. §1 now probes
   `sys.executable -c "import …"` (clean interpreter): on failure it
   force-reinstalls once and re-probes (loud manual-fix error if still
   red); on success after replacing broken packages it raises a restart
   demand instead of re-importing in-process. Fresh installs (merely
   absent packages) still flow straight through. Covered by
   `test_bootstrap.py` (fake pip + scripted probe outcomes).

## Changelog v1.6 → v1.7 (wfdb/pandas-3 version-clash fix)

1. **Render pin `wfdb==4.1.2` → `wfdb>=4.3`**: 4.1.x dies at import on
   pandas 3 (`set_index` over a `StringArray`, TypeError) — reproduced
   exactly here; 4.3.1 (the venv-proven version, exercised by the full
   rehearsal incl. kit renders) imports cleanly on pandas 3.0.5.
2. **Present-but-broken counts as missing**: the §1 detector and the §2
   re-verifier catch `Exception`, not just `ImportError` (a version
   clash raises TypeError, not ImportError), and §1 installs with
   `--upgrade` so a broken env heals in place on re-run — no venv
   rebuild needed. Covered by `test_bootstrap.py` (fake pip + a
   simulated broken install).

## Changelog v1.5 → v1.6 (install-before-import ordering fix)

1. **Render-package install moved from §2 to the top of §1**, before the
   `v1_support` import: `v1_support` modules import numpy/PIL at load, so
   installing in §2 was one section too late and a fresh kernel died with
   `ModuleNotFoundError: No module named 'numpy'` in §1. §2 now
   re-verifies (loud "re-run §1" on a skip) and keeps only the S3
   attempt/defer + fetch-case boto3 rule, with S3-only version pins.

## Changelog v1.4 → v1.5 (§1 finds the project without trusting cwd)

1. **Robust §1 bootstrap**: the notebook locates `dataset_generation_code/`
   by searching cwd, one level of children, VS Code's notebook path, then
   parents — instead of assuming the kernel started in the project dir.
   A bare `ModuleNotFoundError: No module named 'v1_support'` (server
   launched elsewhere, or only the `.ipynb` copied) is replaced by either
   finding the project or a loud error naming the two fixes. Covered by
   `v1_support/tests/test_bootstrap.py` (executes the real §1 cell).

## Changelog v1.3 → v1.4 (c8a.4xlarge sizing: instrumentation + worker fit)

1. **Per-chunk stage timings**: the chunk worker reports
   `arial/font/variants/resize/save_png/csv` seconds; §13 prints them live
   per chunk (§20.8). Additive only — output bytes unchanged.
2. **RAM-fit guard in §13**: refuses to start when
   `workers × 1 GB > 85% MemAvailable` (2× margin over the measured
   ~0.5 GB/worker peak) instead of OOM-dying mid-run.
3. **Measured sizing replaces scaled guesses** (§13.5, §20.8): 9.0 s/record
   on a weak mobile i3 (conservative upper bound), perfectly linear in
   chunk size, ~0.5 GB/worker flat. c8a.4xlarge (16 vCPU / 32 GB) runs
   16 workers at ~7 GB with an estimated ≈ 7–10 h total.

## Changelog v1.2 → v1.3 (S3 input fetch)

1. **EC2 pulls its input from S3.** The input subset ships as
   `V1_10S_STANDARDIZED_SUBSET.tar.gz` (+ `.sha256` sidecar object) at
   `s3://cardiosentry-fcmae-758821198776-us-east-1-an/datasets/raw_signals_data/V1/`.
   When no local copy exists, §5 downloads (sidecar-pinned SHA-256,
   verified, resume-by-hash) and extracts (traversal-safe) it before
   loading any manifest; a local copy is still used as-is when present
   (§3.0, §13.2, §13.5). boto3 becomes REQUIRED in §2 only in the
   fetch case.
2. **No behavior change otherwise**: same 31,109 accepted rows, same
   split/seed, same renders, same subsets, same TAR/S3 output path.

## Changelog v1.1 → v1.2 (eval-subset separation; train logic unchanged, pending later decision)

1. **Four eval subsets replace undifferentiated val/test image sets.**
   Every val/test ECG yields a clean/stress pair that must stay in the same
   split; the pair members are separated into first-class subsets
   `val_clean`, `val_stress`, `test_clean`, `test_stress` — separate image
   directories, separate per-subset `metadata.csv` files, and a `subset`
   column on every row of `full_metadata.csv` (§10.1, §11, §14). No
   undifferentiated val/test image set is ever assembled.
2. **Gate 15.7 extended**: every row's `subset` must equal the value derived
   from (`split`, `variant_id`); every eval record owns exactly one
   `<split>_clean` + one `<split>_stress` row (same record, same split);
   per-subset image counts must equal record counts exactly (§15).
3. TAR top-level entries and the §18 report enumerate the five image
   subsets (`train` + four eval subsets) (§16, §18).
4. Train rendering (`clean` + `v01`–`v03`, subset label `train`) is
   **unchanged** by this revision; training-set logic will be decided
   separately.

## Changelog v1.0 → v1.1 (user amendments, all incorporated before implementation)

1. **Subject grouping**: waveform-hash-only splitting replaced by transitive
   union of recovered patient identity + shared `v1_waveform_sha256`
   (§0.2, §4.1). PTB-XL patient IDs recovered for 11,132/11,699 records
   (6,456 by filename, 4,676 by signal-content join); method per source
   documented; union-find yields 29,564 atomic groups.
2. **Conflicting duplicates**: 31 identical-waveform groups with disagreeing
   target labels are audited (taxonomy + per-group mapping investigation, no
   mapping errors found — all annotation disagreements), flagged, and
   **excluded from primary val/test metrics**; labels never altered (§0.3,
   §4.5, §14.2, §15).
3. **No blanket rare-combo forcing**: pure grouped iterative stratification;
   forced placement only via logged minimal repair if feasibility gates fail
   (§4.3).
4. **Artifact-stress view**: val/test additionally get one deterministic
   `__stress.png` each, evaluated separately, never pooled into the primary
   clean metric (§10.1, §10.6, §11, §14–§15, §18).
5. **Resolution docs corrected**: ~98.4 Hz applies to the 200-DPI master only;
   final 1024×768 carries ~90.4 DPI effective paper resolution (~44.5 Hz
   spatial Nyquist) via LANCZOS antialiased resampling (§6.5, §7).
6. **Filtering wording**: "no **additional signal-domain** filtering";
   rasterization/resampling acknowledged as spatial resampling (§9, §18).
7. **No 0.14% claim**: replaced by empirical multi-location grid-spacing
   validation with recorded distortion and tighten-if-needed rule (§10.4,
   §10.7, §12, §15).
8. **Fonts**: clean images fixed Arial everywhere; augmented train variants
   use seeded per-record font diversity with correlation audit (§6.7, §10.2).
9. **Packaging**: confirmed unchanged — masters outside training TAR, plain
   TAR, SHA-256 sidecars, optional masters archive (§8, §16).
10. **S3 integrity upgraded**: S3-native upload checksum (CRC64NVME)
    enabled + verified; HEAD+user-metadata honestly described as NOT
    independent verification; local SHA-256 retained for provenance (§17).

Target notebook runs on AWS EC2 and produces a training-ready multi-variant
768×1024 PNG corpus from
`dataset-files/MAIN_DATASET/V1_10S_STANDARDIZED_SUBSET`.
This plan is written **before** any notebook code, per instructions.
Nothing below has been implemented yet.

Plan location (this file) and all future notebook outputs:

```text
/home/lindtseyxvii/Codes/cardiosentry/prototype-code/dataset_generation_code/
```

Code inspected for this plan:

```text
dataset-combination-code/ecg-dataset-pipeline/ecg-image-kit/codes/ecg-image-generator/
  ecg_plot.py, extract_leads.py, gen_ecg_image_from_data.py,
  gen_ecg_images_from_data_batch.py, helper_functions.py, config.yaml,
  ImageAugmentation/augment.py, Fonts/, CreasesWrinkles/, HandwrittenText/
dataset-combination-code/ecg-dataset-pipeline/image_pipeline/
  config.py, patch_kit.py, stage1_manifest_from_csv.py, stage2_transcode.py,
  stage3_render.py, stage4_augment.py, stage5_verify.py, install.sh
prototype-code/cardiosentry_generate_images/
  cardiosentry_generate_images.ipynb (51 cells),
  fcmae_generate_synthetic_images_aws_ec2.ipynb (18 cells),
  fcmae_synthetic_image_generator.py
dataset-files/MAIN_DATASET/V1_10S_STANDARDIZED_SUBSET/
  metadata/subset_records_extended.csv, manifests/standardization_manifest.csv,
  metadata/subset_source_summary.csv, records/ (spot headers + .mat key check)
```

---

## 0. Inspection findings that drive this plan (verified, not assumed)

### 0.1 Accepted set = exactly 31,109, files match manifest exactly

- `subset_records_extended.csv`: **31,109 rows**, every row has empty
  `exclusion_reason`. Action mix: `copied_as_is` 28,183,
  `cropped_centered` 1,758, `cropped_af_verified` 630,
  `cropped_pac_localized` 463, `cropped_pvc_localized` 64,
  `cropped_af_flagged` 11.
- `standardization_manifest.csv`: **31,189 rows** = 31,109 accepted +
  **61 `unresolved`** + **19 `excluded`** — exactly the records the
  instructions say to omit. Filter rule is therefore a direct
  `action_taken` membership test, cross-checked against this manifest.
- `records/` contains exactly **31,109 `.hea` + 21,727 `.mat` + 9,382 `.dat`**
  — no stray waveform files outside the accepted set.
- All 31,109 `subset_output_path` entries exist (checked programmatically,
  0 missing). Paths have **two depths**: most are
  `records/<source>/<group>/*.hea`, the 6,456 `ptb-xl-clean-explicit-normal`
  records sit one level deeper. The notebook must resolve paths **from the
  CSV**, never by glob.
- All records are exactly **10.0 s**: 31,094 @ 500 Hz / 5,000 samples,
  15 (all `ptb`, all AF) @ 1,000 Hz / 10,000 samples.

### 0.2 Patient identity: recovered where feasible, documented everywhere

V1.1 amendment 1 required the strongest recoverable subject grouping instead
of hash-only splitting. Investigation result (verified by joining, not assumed):

**Recoverable — PTB-XL family (11,699 records):**

- `V1/metadata/ptbxl_reference/ptbxl_database.csv` (21,799 rows: `patient_id`,
  `ecg_id`, `strat_fold`, `filename_hr`, age/sex) exists locally, and a full
  upstream signal copy exists at `dataset-files/ptb-xl-dataset/records500/`.
- `ptb-xl-clean-explicit-normal` (6,456 records) preserves upstream paths
  (`records500/XXXXX/XXXXX_hr`) → **6,456/6,456 joined by filename**
  (6,116 distinct patients; 283 multi-record patients covering 623 records).
- `ptb-xl` challenge-renamed HR records (5,243) → joined by **byte-exact
  signal-content hash** against the upstream copy: **4,676 matched (89.1%)**,
  age/sex cross-checked (4,626/4,676 age-consistent ±1 yr; byte-identical
  matches accepted regardless since identical bytes = identical recording).
  The 567 misses have no counterpart even at 8 µV coarse quantization: the
  local upstream copy is incomplete (19,526/21,799 `records500` files present),
  so these are **unrecoverable from local data**, not transform-mismatched.
- Method per record is stored (`filename` / `content`); the precomputed map
  ships in the code bundle with provenance, and the notebook re-verifies
  every entry (filename joins recomputed from the bundled database extract;
  content entries verified by re-hashing the subset `.mat`). Re-running the
  join against a complete upstream copy (e.g. on EC2 with network + PhysioNet
  credentialed access) is documented as an option but NOT required.
- `strat_fold` is recovered into metadata for provenance/analysis but the
  unified 80/10/10 grouped split is NOT replaced by PTB-XL folds (single
  split design across all 8 sources, per the approved plan).

**Unrecoverable — documented with reason:**

- `ptb` (15 records): no upstream copy locally; challenge `Sxxxx` names do
  not encode the upstream patient folders → hash grouping + documented limit.
- `chapman_shaoxing`, `ningbo`, `cpsc_2018`, `cpsc_2018_extra`, `georgia`:
  upstream CinC-Challenge data carries no patient identifiers by construction
  → hash grouping (this is a data property, not a gap in the recovery).

**Union grouping (transitive closure over patient edges + hash edges):**

- **29,564 atomic groups**: 28,303 singletons + 1,261 multi-record groups
  (2,806 records). Of the multi-record groups: 799 patient-only (same
  patient, different ECGs — the key gain over hash-only), 458 hash-only,
  4 both.
- **69 patients span `ptb-xl` ↔ `ptb-xl-clean-explicit-normal`** (same
  patient's NORM and abnormal ECGs now correctly grouped — this leakage
  vector is closed by amendment 1).
- `patient_id` metadata column is now **filled** (`ptbxl:<id>`) where
  recovered, empty elsewhere with `patient_recovery_method` documenting why.

### 0.3 LEAKAGE FINDING: 462 duplicate-waveform groups, 173 cross-source

Grouping accepted records by `v1_waveform_sha256`:

- **462 groups (939 records)**: 449 pairs, 11 triples, 2 quads.
- **173 groups span 2+ sources** — e.g. `chapman_shaoxing__JS01332` and
  `ningbo__JS23830` are byte-identical (Shaoxing-hospital overlap, plus
  CPSC-2018 / CPSC-2018-extra overlap). A naive record-level shuffle would
  put identical ECGs in train *and* test.
- **31 groups carry DISAGREEING target labels for identical bytes.**
  V1.1 amendment 2 investigation (per-group original-label/dx-code audit —
  see `conflicting_duplicate_groups.csv` spec in §4.5):
  taxonomy = 24 SUBSET_SUPERSET (one side's labels ⊂ the other's) +
  6 DISJOINT (e.g. `PAC` vs `PVC`, `LBBB` vs `IAVB`, `AF` vs `RBBB`) +
  1 PARTIAL. 30/31 are `cpsc_2018` (sparse 1–2-code annotation) vs
  `cpsc_2018_extra` (rich 3–6-code annotation incl. OldMI/MI/NSSTTA) — the
  same physical ECG published in both sets with different annotation depth.
  1/31 is an internal PTB-XL pair (`HR07779` LAFB|LAE vs `HR07780` LAFB,
  both confirmed same upstream ecg_id 7780 / patient 8978).
  **Mapping-error check: none found.** Every member's target set is the
  deterministic normalization of its own dx codes (spot cases AFL→non-target
  vs AF→target, ILBBB/IRBBB→non-target vs LBBB/RBBB→target all follow
  `diagnosis_mapping.csv`); all 31 are genuine source-annotation
  disagreements → verdict `annotation_disagreement_unresolved` for all.
  Mitigation: keep all 31,109 records with labels byte-unchanged; force each
  group into **one** split; flag all members; and **exclude unresolved
  contradictory groups' val/test images from the primary metric**
  (`primary_metric=0`, `metric_note` set, §14.2) while reporting them
  separately. Train images from these groups are kept (flagged) since
  training tolerates label noise that evaluation must not contain.
- All 457 `subset_waveform_sha256` duplicate groups are contained within the
  `v1_waveform_sha256` groups, so the hash edge of the union grouping covers
  both. Hash-only split units would have been 30,632 groups
  (30,170 singletons + 462 multi-record); with v1.1 patient-union edges the
  effective split units are **29,564 groups** (§0.2, §4.1).

### 0.4 Source × label structure (dataset property, flagged not fixed)

Per-label totals: NORM 6,456 / AF 5,219 / IAVB 3,496 / LBBB 1,477 /
RBBB 4,777 / PAC 3,196 / PVC 1,873 / LAFB 2,173 / LAE 1,246 / TInv 3,962 /
LQT 1,865 / PRWP 638. 209 distinct combinations, 4,559 multi-label rows,
133 rare combinations with <10 records.

Structural confounds (verified in the accepted CSV):

- **NORM ⟺ `ptb-xl-clean-explicit-normal`**: all 6,456 NORM records come from
  that source and it contains nothing else.
- PRWP only in `ningbo`; AF absent from `ningbo`; PVC absent from `cpsc_2018`
  and `ptb-xl`; LAE ≈ `georgia`; `ptb` = 15 AF-only records @ 1 kHz.
- Mitigations in this plan: identical rendering for every record (§6),
  **no printed header** (native filenames are source-identifying, §6.7),
  label stratification with post-hoc source-balance verification (§4).
  The confound itself is inherent to V1 and is reported, not hidden.

### 0.5 Kit `.mat` gain bug CONFIRMED — WFDB transcode to scratch is required

- Standardized `.mat` files use key `'val'`, `int16` raw ADC counts
  (verified by loading `JS00001.mat`: shape `(12, 5000)`).
- `helper_functions.load_recording()` returns `loadmat()['val']` **without**
  applying the header ADC gain for `.mat`, while `.dat` goes through
  `wfdb.rdrecord().p_signal` in mV. `extract_leads.py` reads the gain into a
  variable and never applies it. Rendering `.mat` directly therefore plots
  1,000× over-scale.
- **Consequence:** reuse the `stage2_transcode.py` *format* path (read via
  `wfdb` → write scratch `.dat` in mV, canonical 12-lead order) but **without**
  its filter (§9). Scratch is deleted per chunk; source files are never touched.

### 0.6 All `patch_kit.py` patches are already applied to the vendored kit

Every `PATCHED-*` marker is present in the vendored checkout
(`ecg_plot.py` ×7, `extract_leads.py` ×3, single-entry ×2, batch-entry ×1,
helpers ×1). The notebook will still run the equivalent of
`patch_kit.is_patched()` as a preflight gate and refuse to render otherwise.

---

## 1. Which existing ECG-Image-Kit code will be reused

### 1.1 Reused (render path — the verified working pipeline)

| File | Role in our build |
|---|---|
| `ecg_plot.py` | Sheet renderer (grid, traces, labels, calibration pulse, footer). Patched copy. |
| `extract_leads.py` | 10 s segmentation: 4 column windows + full-Lead-II rhythm. Patched copy. |
| `gen_ecg_images_from_data_batch.py` | Batch driver invoked per chunk (numpy-2 shim included). |
| `gen_ecg_image_from_data.py` | Single-file path with lazy heavy deps + fixed-font support. |
| `helper_functions.py` | WFDB/`.mat` loading, header parsing (dead `imgaug` import removed). |
| `config.yaml` | `paper_len: 10.0`, `format_4_by_3` column groups, `leadNames_12` order. |
| `Fonts/` | Glyph source; one face pinned via `ECG_FIXED_FONT` (deterministic clean). |
| `CreasesWrinkles/wrinkles-dataset/` (19 photos) | **Texture source only** for the stage-4-style shading field (§10). |
| `image_pipeline/patch_kit.py` | All 10 patches (header inset, lead-name-above-trace, ink widths, any-N columns, full-sheet layout, paper-size/centering, numpy-2 shim, lazy deps, dead-import removal, fixed font). |
| `image_pipeline/stage4_augment.py` approach | Custom augmentation math (LUT recolour, tilt, perspective, texture, creases, lighting, exposure/contrast, noise, defocus, paper renormalisation) — adapted to PNG + our variant counts (§10). |
| `image_pipeline/stage1/stage5` ideas | Grouped-split + integrity/leakage/balance check structure (§4, §15). |

### 1.2 Deliberately NOT reused, with reasons

| Code | Reason |
|---|---|
| `ImageAugmentation/augment.py` (`--augment`) | Documented defects in `stage4_augment.py` header, verified in source: `-t` temperature is dead code (always a heavy orange/blue cast, no neutral); `-noise 0` crashes; `iaa.Affine` fills rotation corners black and clips the sheet; `--augment` silently requires `--store_config`; `--lead_bbox` silently disables cropping. |
| `CreasesWrinkles/creases.py` (`--wrinkles`) | Even at minimum settings lays a heavy grey texture over the whole sheet and buries the trace. We reuse its *photos* with our own bounded blend instead. |
| `HandwrittenText/` (`--hw_text`) | Needs TensorFlow/spacy/seaborn/cv2 (~1 GB, ~7 s import per chunk); random text can obscure P/QRS/T; not requested. |
| `stage2_transcode.py` **filter** | 2nd-order Butterworth 0.05–40 Hz zero-phase + median subtraction. Rejected per §9 (no new filtering without cause). Only its *format* path is reused. |
| `stage4_photo.py` / `photo_config.py` | Photo-realistic camera-model path from a different experiment; our bounded scan/print model (§10) is the approved scope. |
| `--print_header` | Prints `ID: <record name>` + Age/Sex. Native names are source-identifying (leak), and the block is model-irrelevant chrome. Matches `image_pipeline` default OFF. |
| JPEG output (`q85/4:2:0`) | Instructions mandate PNG. (§8 reports the size tradeoff.) |
| Annotation sidecars (`--store_config/--lead_bbox/--lead_name_bbox`) | Lead-bbox JSONs are for segmentation/digitisation training; expensive (per-pixel `plotted_pixels` tracing) and unrequested for this classification corpus. OFF. |
| Digitizer codes (`ecg-image-digitizer/`) | Reverse direction (image→signal); irrelevant. |
| `qrcode`, `--add_qr_code` | Not requested. (`qrcode` stays installed only because the kit imports it unconditionally.) |

### 1.3 How the kit reaches EC2

The notebook cannot assume this repo exists on EC2. Companion packaging
(built here, beside the notebook):

```text
dataset_generation_code/
├── PLAN.md                          # this file
├── v1_image_generation.ipynb        # the notebook (to be written after approval)
├── v1_support/                      # notebook-imported package (split, variants,
│                                    # render, resize, validate, s3util)
├── ecg_image_v1_code.tar.gz         # versioned bundle: patched kit subset +
│                                    # patch_kit.py + wrinkle photos + SHA file
└── outputs/                         # build outputs (also the TAR staging area)
```

The notebook's environment section downloads (or side-loads) the bundle,
verifies its SHA-256, extracts it to EBS, runs the `is_patched()` gate, and
only then proceeds. All custom logic (split, variants, resize, validation)
lives in `v1_support/` + notebook cells so it is reviewable without digging
through the kit.

Dependencies for the render venv: `numpy, scipy, matplotlib, wfdb, pyyaml,
Pillow, qrcode, pandas, boto3` (+ `jupyter`). No TensorFlow, spacy, imgaug,
scikit-image, or opencv — none of the enabled code paths import them after
the lazy-dependency patches.

---

## 2. What parts of the earlier notebook are useful as reference

`prototype-code/cardiosentry_generate_images/cardiosentry_generate_images.ipynb`
is a 51-cell Colab notebook that builds the older **v3 4-class
(STEMI/AF/LVH/NORMAL) JPEG corpus** via `image_pipeline` stages 1–5 to Google
Drive. Useful as **structure and hard-won practice**, not as code to copy:

**Reuse the ideas:**

- Preflight gates that hard-stop in the first minutes (inputs present, disk/RAM,
  dependencies, kit patches, effective-settings echo).
- Smoke test on a handful of records (incl. edge cases) before the long build.
- Staged build with a human-readable gate after manifest creation
  (record counts + class sanity) before burning hours on rendering.
- Stage-3 checkpoint (`checkpoint_rendered.tar`) since rendering is the
  long pole; resume-by-skipping-finished-work in every stage.
- Per-split TAR packaging + SHA-256 sidecars; never thousands of loose files
  to remote storage.

**Do NOT carry over (verified inappropriate for V1):**

- Input (`harmonized.tar` v3 signals), 4-class single-label framing, Drive I/O,
  Colab-only constructs — none apply to V1/EC2/S3.
- `image_pipeline/config.py` *values*: 150 DPI → we use 200 (§6.5);
  JPEG → PNG; 0.05–40 Hz band-pass → none (§9); class-balanced
  `RENDERS_PER_RECORD` → fixed split-aware variant counts (§10);
  `STEMI/AF/LVH/NORMAL` label columns → our 12 multi-hot columns.
- Its patient grouping (`split_group_id`) — V1 has no such column (§0.2);
  we union recovered patient IDs with waveform hashes (§0.2, §4).
- Its 8–12 h Colab timing assumptions — re-estimated for EC2 in §13.5.

The **FCMAE EC2 notebook** (`fcmae_generate_synthetic_images_aws_ec2.ipynb`)
is the closer operational reference and its patterns are adopted directly:
EC2-native paths, thread-pinning env (`OMP/MKL/NUMEXPR_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=`), 200-record atomic chunk checkpoints
(`index_parts/chunk_XXXXX.csv`) with validate-and-skip resume, PNG output,
native sampling preserved, no printed header, `generation_config.json` +
`failures.csv` + palette-audit sidecars, and boto3 S3 upload with
`sha256` object metadata + HEAD verification + optional full re-download
hash check (see §17).

---

## 3. Input dataset and accepted-record filtering

### 3.0 S3 source + fetch-if-missing (v1.3)

The input subset ships as a deterministic tarball + sidecar:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/datasets/raw_signals_data/V1/
  V1_10S_STANDARDIZED_SUBSET.tar.gz          # 1,981,727,518 bytes
  V1_10S_STANDARDIZED_SUBSET.tar.gz.sha256   # SHA-256 authority for the fetch
```

Fetch rule (§5 preamble): if `cfg.subset_dir` exists locally it is used
as-is (manual copy still supported); otherwise the notebook downloads the
sidecar object, downloads the tarball (skipped when a local copy already
hashes correctly — resume by hash), SHA-256-verifies it (a mismatch after
a fresh download deletes the file and aborts), and extracts it with a
traversal-safe extractor (absolute/`..` members rejected) into
`<work_dir>/input/`. boto3 is REQUIRED in §2 only when the fetch will run
(IAM role auth, same §17 pattern; no keys). The filtering gates below
(steps 1–5: 31,109 rows, 62,218 file hashes) then run identically on
either source, so a corrupt fetch or partial extraction still aborts
loudly.

Source (read-only, never modified) once present:

```text
V1_10S_STANDARDIZED_SUBSET/
```

Accepted-record manifest (the filter input):

```text
metadata/subset_records_extended.csv   # 31,109 rows — every row is accepted
```

Filtering procedure in the notebook:

1. Load `subset_records_extended.csv`; assert 31,109 rows.
2. Cross-check against `manifests/standardization_manifest.csv` (31,189 rows):
   assert the accepted IDs equal exactly the manifest rows whose
   `action_taken` ∈ {`copied_as_is`, `cropped_centered`,
   `cropped_af_verified`, `cropped_pac_localized`, `cropped_pvc_localized`,
   `cropped_af_flagged`} — i.e. assert the 61 `unresolved` and 19 `excluded`
   record IDs are absent, and assert no `exclusion_reason` is non-empty.
3. Per-record waveform gate: `subset_output_path` exists; header parses to
   12 leads; `samples / fs == 10.0 s` exactly (5,000 @ 500 Hz or
   10,000 @ 1,000 Hz); lead names cover the canonical 12 (case-insensitive).
   Any failure → row in `failures.csv`, hard stop before splitting
   (a missing/corrupt waveform must be investigated, never skipped silently).
4. Integrity spot-check: recompute SHA-256 of every `.hea` + waveform file and
   compare to `subset_header_sha256` / `subset_waveform_sha256`.
   (Full re-hash of ~31 k small files takes minutes and proves the EC2 copy
   is intact; failures abort the build.)
5. Labels: parse `target_labels` (`|`-separated) into 12 fixed multi-hot
   columns in this order:
   `NORM AF IAVB LBBB RBBB PAC PVC LAFB LAE TInv LQT PRWP`.
   No combination classes (e.g. `AF+RBBB`) are ever created; multi-label rows
   stay multi-hot. Assert every row has ≥1 positive label and that per-label
   totals match §0.4 exactly (guards against a silently changed input file).

---

## 4. Train / validation / test split strategy

### 4.1 Identifier used for splitting (v1.1: union grouping)

- **Atomic split groups = connected components (union-find) over two edge
  types: shared recovered patient identity (§0.2) and shared
  `v1_waveform_sha256` (§0.3).** Result: **29,564 groups**
  (28,303 singletons + 1,261 multi-record). This is the strongest
  recoverable subject grouping: same-patient/different-ECG pairs (799
  patient-only groups, incl. 69 spanning PTB-XL ↔ clean-explicit-normal)
  and identical-waveform copies (458 hash-only groups, 173 cross-source)
  are all atomic.
- Within a group, each `record_id` keeps its own row, labels, and images —
  grouping affects split assignment only.
- `patient_id` is **filled where recovered** (`ptbxl:<id>`, 11,132 records)
  with `patient_recovery_method` ∈ {`filename`, `content`,
  `unrecoverable_incomplete_upstream`, `no_upstream_identity`} documenting
  every record's provenance. `dup_group_id` (= canonical union-group id,
  `G000000`…; member hashes/patients listed in `splits/duplicate_groups.csv`)
  is the grouping key used by the splitter and leakage checks.

### 4.2 Ratio, seed, reproducibility

- Ratio **80 / 10 / 10** on groups (≈24,887 / 3,111 / 3,111 records;
  exact counts reported after the split).
- Fixed seed **`20260918`**, recorded in `generation_config.json` and every
  manifest header comment.
- Reproducibility is fourfold: (1) fixed seed; (2) deterministic input order
  (groups sorted by hash); (3) deterministic algorithm (below — no
  `PYTHONHASHSEED`-dependent hashing, `blake2b`-keyed RNG as in stage 4);
  (4) frozen outputs — `splits/*.csv` are written once **before any rendering**
  and the build refuses to re-split over them (resume reuses them).

### 4.3 Multi-label stratification (iterative, Sechidis-style)

`sklearn`'s stratifiers are single-label; with 209 combinations and 133 rare
ones (<10 records), exact combination stratification is impossible. The
notebook implements iterative multi-label stratification (~60 lines,
numpy/pandas only, in `v1_support/split.py`):

1. Operate on the 29,564 union groups. Each group's label vector = OR over
   members (for the 31 conflicting groups the OR is used for stratification
   arithmetic only — member labels on disk are never modified).
2. Repeatedly place the currently-rarest-label group into the split that most
   needs that label (desired counts = ratio × label total), ties broken by
   seeded RNG. **Individual-label stratification is primary; higher-order
   combinations are preserved best-effort by the greedy pass — there is NO
   blanket rule forcing rare combinations to train (v1.1 amendment 3).**
3. Post-split gates (hard stops): every one of the 12 labels must appear in
   every split with ≥ max(30, 5% of its total) records; per-label proportions
   must be within ±25% relative of 80/10/10 — the notebook prints the full
   12×3 table. Rarest label PRWP (638) yields ≈64 test records, so the
   absolute floor is satisfiable.
4. **Feasibility-only repair (the sole forcing mechanism):** if and only if a
   gate in (3) fails, apply minimal logged repair moves (relocate the
   smallest feasible groups into the deficient split, patient/hash-atomic,
   seeded choice among equals) until gates pass; every forced placement is
   recorded in `split_report.txt` with reason. With 12 labels at ≥638 records
   each, the greedy pass is expected to pass unaided; the repair path exists
   so infeasibility can never silently degrade into an accepted bad split.

### 4.4 Source balance

Source is **not** a stratification key (it would fight the label pass), but
per-source 80/10/10 proportions are verified post-hoc and must stay within
±3 percentage points per source per split — expected to pass trivially at
n≈31 k with the label pass. The 15-record `ptb` source (≈12/1–2/1–2) is
exempted from the ±3 pp gate by documented exception and reported exactly.
If any gate fails, the notebook does not silently accept the split: it raises,
and the documented fallback is source-blocked assignment (same algorithm run
per source, then concatenated).

### 4.5 Where split manifests are saved (before any image exists)

```text
dataset/splits/
├── train_records.csv   # record_id, dup_group_id, patient_id,
├── val_records.csv     # patient_recovery_method, source_dataset,
└── test_records.csv    # v1_original_record_id, target_labels (+12 multi-hot),
                        # subset_output_path, v1_waveform_sha256,
                        # upstream_filename_hr, ecg_id, strat_fold
```

Plus audit sidecars (also inside the TAR, see §14):

```text
dataset/splits/
├── duplicate_groups.csv             # 1,261 multi-record union groups:
│                                    # members, edge types (patient/hash/both),
│                                    # sources, labels
├── conflicting_duplicate_groups.csv # 31 groups: per-member original labels +
│                                    # dx codes, taxonomy (SUBSET_SUPERSET /
│                                    # DISJOINT / PARTIAL), mapping notes,
│                                    # verdict (all annotation_disagreement_
│                                    # unresolved), metric exclusion flag
├── patient_recovery_report.txt      # method counts, coverage, patient counts,
│                                    # 567 unrecoverable HR + reason, per-source
│                                    # recoverability table
└── split_report.txt                 # counts, 12×3 label table, source table,
                                    # forced-placement log (§4.3.4, if any)
```

---

## 5. How leakage will be prevented

1. **Split-before-render, structurally:** `splits/*.csv` are written and
   validated before the first kit invocation; the render/variant stages take
   the split assignment as input and write into per-subset directories
   (`train/` + the four eval subsets), so a variant physically cannot cross
   splits — or cross the clean/stress separation within a split.
2. **Union-grouped assignment** (§4.1): same-patient records and identical
   waveforms (incl. cross-source copies) share one split by construction.
3. **No shared filenames across splits:** output names embed `record_id`
   (§14.3); staged scratch names embed the split.
4. **No pixel-level source hints:** no printed header with source-identifying
   filenames (§1.2); identical render + augmentation distributions for every
   record regardless of source or split (variant RNG is seeded on
   `(record_id, variant_id)` only, never on label/source/split; same for the
   seeded train font, §6.7).
5. **Leakage validation** (§15) asserts the three-way PASS block on
   `record_id`, `dup_group_id`, `v1_waveform_sha256`, **and recovered
   `patient_id`**: 0 overlap train/val, train/test, val/test on every key.

---

## 6. ECG rendering configuration (verified against kit source)

All settings below were checked against `ecg_plot.py` / `extract_leads.py` /
`config.yaml`, not assumed from upstream docs.

### 6.1 Layout — 3×4 + Lead II rhythm strip, conventional order

- `--num_columns 4`, `paper_len 10.0` → **2.5 s per lead panel**, verified:
  `lead_length_in_seconds = configs['paper_len']/columns`.
- Column→time mapping (`format_4_by_3`, generalised `_column_index` patch):
  col 0 = I/II/III @ 0–2.5 s, col 1 = aVR/aVL/aVF @ 2.5–5 s,
  col 2 = V1/V2/V3 @ 5–7.5 s, col 3 = V4/V5/V6 @ 7.5–10 s.
- `leadNames_12` plots bottom-up, so the **visual** row order is the
  clinical standard:

```text
I      | aVR | V1 | V4
II     | aVL | V2 | V5
III    | aVF | V3 | V6
```

- The notebook's preview gate (§12) asserts this order on a rendered sample
  (lead-name text positions) rather than trusting the analysis.

### 6.2 Paper speed — 25 mm/s (fixed by the kit)

Hardcoded grid geometry (`x_grid_size = 0.2 s` per 5 mm major division) with
`25mm/s` printed on the sheet footer. Not configurable; recorded, not changed.

### 6.3 Gain — 10 mm/mV (fixed by the kit)

Hardcoded (`y_grid_size = 0.5 mV` per 5 mm) with `10mm/mV` printed on the
sheet. Amplitude is never normalised: voltage criteria (R-wave amplitude,
ST level) survive byte-identical from the standardized mV signal.

### 6.4 Grid — 1 mm minor / 5 mm major, always present

Major gridlines every 0.2 s × 0.5 mV, minor via `AutoMinorLocator(5)` =
1 mm × 0.1 mV. `--random_grid_present 1` (Bernoulli p=1) forces the grid on
every sheet. Colour: master rendered in the kit's grey `bw` style
(major 102 / minor 191, trace <60 — the exact levels the LUT recolouring in
§10 relies on), then recoloured per variant.

### 6.5 DPI — 200 for the master render

- 200 DPI satisfies the "≥ 200" requirement and is the kit's own default.
- Sheet is 11 × 8.5 in (landscape letter) → master **2200 × 1700 px**.
- Master spatial Nyquist at 200 DPI / 25 mm/s: 7.874 px/mm × 25 mm/s / 2 ≈
  **98.4 Hz. This figure applies to the 200-DPI master ONLY (v1.1 amendment
  5).** Rendering uses no additional signal-domain filtering (§9), so content
  above ~98 Hz cannot be represented on the master and folds against the grid
  — acceptable: diagnostic ECG content sits far below this, and 200 DPI (vs
  the old pipeline's 150) buys headroom precisely because we do not low-pass.
  The **final 1024×768 image carries ~90.4 DPI effective paper resolution**
  (≈44.5 Hz spatial Nyquist, §7); the plan never describes the final image
  as retaining the 98.4 Hz figure.
- 300 DPI was considered and rejected: it changes nothing in the final
  768×1024 pixels (downsample target is fixed) while ×2.25-ing master storage
  and render time.

### 6.6 Lead II rhythm strip — continuous full 10 s

`--full_mode II`: `frame['fullII'] = record[start : start + rate*10]`, i.e.
the entire standardized window, including the 15 native-1 kHz records (the
kit segments by `rate × seconds`, so no resampling is needed or performed).
Rhythm label `II` printed above the strip (patched). Calibration pulse drawn
at the strip start when enabled.

### 6.7 Calibration pulse, lead names, header, fonts

- Calibration pulse: ON (`--calibration_pulse 1`, Bernoulli p=1,
  0.2 s DC pulse at each row start and the strip start).
- Lead names: ON (`I, II, III, aVR, aVL, aVF, V1–V6`, rhythm `II`),
  printed **above** each trace per the applied patch (matches clinical carts).
- Printed ID/Age/Sex header: **OFF** (§1.2).
- Typeface (v1.1 amendment 8): **clean images use deterministic fixed
  `Arial.ttf` in every split** (pinned via `ECG_FIXED_FONT`), so the canonical
  clean view is bit-deterministic. **Augmented train variants use seeded
  per-record font diversity**: each train record draws one font from the
  kit's `Fonts/` set via the per-image RNG keyed on `record_id` only
  (never label/source/split), rendered as a second train-only master from
  which its v01–v03 derive. `render_font` is recorded per image and the
  §10.5 audit is extended to fonts (no font may concentrate in any label
  beyond chance). Cost: one extra master render pass over train records
  (~+45–60 min on 32 vCPUs, §13.5); font masters are transient per chunk
  (deleted after variants derive) and never enter `master/` or the TAR.
  (Upstream picks a random font per record even with all augmentation flags
  off, and the kit's batch `-se` seed is dead code — see `stage3_render.py`
  docstring — so flags + pinned/seeded fonts, not the seed, carry
  determinism.)
- Ink widths: stock (`ECG_LINE_WIDTH=0.75`, `ECG_GRID_LINE_WIDTH=0.5`,
  patch-tunable). Final trace ≈ 1 px at 768×1024 — physically correct for
  ~4.1 px/mm; preview (§12) must confirm legibility before the full run.
- Row geometry: patched layout env `ECG_ROW_GAP=4.4`, `ECG_RHYTHM_GAP=5.5`,
  `ECG_BOTTOM_RESERVE=1.9` (plot units, 1 unit = 10 mm) — the validated
  full-sheet layout from `image_pipeline`.

---

## 7. Image dimensions and resizing strategy

### 7.1 Native aspect ratio (measured from the renderer, not guessed)

Master: 2200 × 1700 px → aspect **1.2941** (W/H).
Requested model input: 1024 × 768 → aspect **1.3333**.
Mismatch is **3.0%** — small, handled by padding, not stretching.

### 7.2 Verdict on 768×1024: appropriate, keep it

No alternative is proposed: the aspects are close enough that
aspect-preserving fit needs only ~30 px total padding (~3% of width).
A "nearby alternative" (e.g. 992×768, the exact-fit width) would save 30
columns of padding at the cost of a non-standard model input size — not worth
it. No smaller size is proposed (per instructions, compute savings alone do
not justify downsampling diagnostic content).

### 7.3 Resize procedure (identical for every image: clean, augmented, stress)

```text
2200×1700 master PNG (grey) ──► variant pipeline (§10, still 2200×1700)
        │
        ▼
LANCZOS downsample to height 768  →  994×768 (aspect preserved exactly)
        │
        ▼
symmetric pad 15 px left + right  →  1024×768 (W×H), i.e. H=768, W=1024
pad colour = median of sheet corners (paper colour, stage-4 method)
        │
        ▼
final PNG, asserted 1024×768, asserted aspect preserved (no stretch)
```

- Downsampling happens **after** augmentation so lighting/texture/noise are
  applied at print resolution and optically averaged down — the physically
  correct order (what a real lens/scan does).
- **Effective final resolution (v1.1 amendment 5):** the 768/1700 = 0.4518×
  LANCZOS downsample gives **~90.4 DPI effective paper resolution**
  (200 × 0.4518), i.e. ~3.56 px/mm, ~89 px/s at 25 mm/s, and a final-image
  spatial Nyquist of **~44.5 Hz**. Nominal grid periods at final size:
  5 mm major ≈ 17.79 px, 1 mm minor ≈ 3.56 px. LANCZOS provides
  image-domain antialiased resampling; the ≥2× oversample from the 200-DPI
  master is what keeps the downsample clean. The 98.4 Hz figure is never
  attached to the final image.
- Geometric augmentation precedes the downsample; its spatial-scale effect is
  **measured empirically** (§10.7), not asserted analytically.
- No crop is ever applied after rendering, so no trace, label, pulse, or
  footer pixel can be lost to framing.

---

## 8. Master images vs training images + storage decision

### 8.1 Roles

```text
master/               2200×1700 grey PNG, 1 per record (31,109)
                      = the single source of truth; every variant derives from it
training_768x1024/    final model-ready PNGs (≈112,000) in train/val/test/
```

One render per record (not one per variant) is both a quality choice
(variants differ only in the controlled post-process) and a ~3.4× render-time
saving.

### 8.2 Storage estimate (measured analogies, stated honestly)

| Artifact | Count | Bytes each (measured) | Total |
|---|---|---|---|
| Masters (2200×1700 grey PNG) | 31,109 | ~0.35 MB | **~11 GB** |
| Training PNGs (1024×768 RGB) | ≈111,980 (incl. stress views, §10.6) | clean ~0.98 / aug ~1.57 MB | **~150 GB** |
| Scratch WFDB staging (transient, per chunk) | — | — | < 5 GB |
| Train font masters (transient, per chunk) | ≤ chunk train records | ~0.35 MB | < 1 GB live |
| TAR (`training_768x1024` + metadata, uncompressed) | 1 | ≈ training total | **~150 GB** |
| Peak EBS (masters + training + TAR + headroom) | — | — | **≈ 320 GB** |

Sizes are MEASURED on the 84-image preview (§20.5), not estimated: textured
1024×768 PNGs run ~1.6× larger than the pre-implementation guess (sensor
noise + paper texture dominate PNG entropy), while grey masters run ~3×
smaller. The §12 disk gate re-projects from measured bytes at runtime.

Basis: the old pipeline measured 1.2 MB/image @200 DPI JPEG and 2.5 MB for a
clean sheet at high quality; PNG runs larger than JPEG-q85 but our training
images are 4.7× fewer pixels than the master. Ranges are wide because paper
texture + sensor noise dominate PNG size unpredictably — the notebook's smoke
test (§12/§13) measures real per-image bytes on ~50 records and re-projects
totals before the full run, aborting if free disk < 1.5× projection.

### 8.3 Decision (explicit, per instructions)

- **Keep `master/` on EBS during the build** (needed for resume + QC + any
  re-varianting), but **exclude it from the training TAR**. Rationale: masters
  are exactly regenerable from (waveforms + bundle SHA + seed), and including
  them would ~1.5× the archive for files training never reads.
- After S3 upload succeeds, optionally archive masters as a **second TAR**
  (`cardiosentry_v1_10s_masters_200dpi.tar`, ~25–35 GB) to the same S3 prefix
  for provenance. Default: yes, upload it too (cheap insurance, same code
  path). This is listed as an approval item in §19.
- EC2 disk requirement: **≥ 400 GB free** on the build volume (gp3),
  revised up from 250 GB on measured PNG sizes (§20.5).
  The §12 cell re-projects from measured bytes and gates the full build.

### 8.4 Format notes

- PNG everywhere (per instructions). PNGs are already entropy-coded, so
  `.tar.gz` would burn CPU for ~0% gain — plain `.tar` (§16).
- JPEG-q85 would roughly halve bytes but is lossy on exactly the high-frequency
  content (1 mm grid, 1 px trace, ST morphology) the model must read; rejected
  per the mandated PNG output.

---

## 9. Signal processing / filtering

### 9.1 Previously used filter — reported exactly before any decision

`image_pipeline/stage2_transcode.py` + `config.py` (`BANDPASS`,
`_bandpass_sos`):

```text
high-pass cutoff:  0.05 Hz
low-pass cutoff:   40 Hz (overridable via ECG_LOWPASS_HZ)
filter type/order: Butterworth band-pass, order 2 (SOS)
notch frequency:   none
phase:             zero-phase (scipy sosfiltfilt, forward-backward)
post-step:         median subtraction per lead (DC removal)
application:       identical to every record (anti-confound rule)
```

### 9.2 Proposed: NO additional signal-domain filtering for V1

The notebook applies **no additional signal-domain filtering** (v1.1
amendment 6 wording): it will not filter, resample, stretch, repeat, pad,
re-gain, or morphologically alter the standardized waveforms. Rendering
consumes the `wfdb`-read mV signal exactly as standardized. This plan does
NOT claim the pipeline performs "no filtering whatsoever": rasterizing the
trace to pixels and LANCZOS-downsampling the master (§7.3) necessarily
perform spatial resampling — that is image-domain antialiasing, not a change
to the signal, and the two are never conflated. Reasons for no additional
signal-domain filtering:

1. The instructions forbid new filtering without a clear reason, and the
   waveforms are already final: filtering now would make the images diverge
   from the validated standardized dataset.
2. The old 40 Hz default was paired with that build's 150 DPI sheet
   (image Nyquist 73.8 Hz) and with a mains-hum confound concentrated in one
   v3 source. Neither condition holds here: 200 DPI gives 98.4 Hz of headroom
   (§6.5), and any residual hum is rendered faithfully and identically for all
   records rather than shaping the signal.
3. "Identical treatment" is trivially satisfied by no treatment; a filter
   would need its own audit trail in every manifest row.
4. No median/DC subtraction either: baseline offset is part of the signal,
   rows have headroom, and the preview (§12) will show whether wander ever
   threatens legibility.

**Escape hatch (explain-first):** if the preview reveals hum/wander that
hurts trace legibility, the fallback is the exact §9.1 filter, applied
identically to all records and recorded per-row — but that change requires a
new approval, not a silent edit.

---

## 10. Variant generation strategy

### 10.1 Counts (proposed)

| Subset | Records (approx) | Variants per record | Images (approx) |
|---|---|---|---|
| train | 24,887 | **4** = 1 clean + 3 augmented (`clean, v01-v03`) | ~99,550 |
| val_clean | 3,111 | **1** = clean only | ~3,111 |
| val_stress | 3,111 | **1** = deterministic stress only (§10.6) | ~3,111 |
| test_clean | 3,111 | **1** = clean only | ~3,111 |
| test_stress | 3,111 | **1** = deterministic stress only (§10.6) | ~3,111 |
| **total** | 31,109 | — | **~112,000** |

- **One clean/base image is always retained** for every record in every split:
  deterministic red grid, no geometry/lighting/texture, fixed Arial.
- Random augmented variants exist **only for train** (evaluation-integrity:
  extra val/test variants would reweight those ECGs inside every metric).
  The val/test stress view is a single deterministic second view per ECG,
  shipped in its own subset (`val_stress` / `test_stress`) and evaluated
  **separately**, never pooled with clean (§11). Each clean/stress pair
  belongs to the same record and the same split by construction (the split
  is record-level, §4; the pair is rendered per record, §13.13).
- 3 augmented + 1 clean for train matches the validated `image_pipeline`
  ratio while bounding the corpus to ~112k images / ~48-75 GB.
  Raising train to 5-6 variants is a one-line change but x1.25-1.5 storage;
  listed as an approval item (§19).
- Variants are generated **ahead of time** (stored PNGs), not dynamically:
  dynamic augmentation would reintroduce run-to-run metric wobble and prevent
  the exact byte-level validation (§15) and TAR freezing (§16) the task
  requires.

### 10.2 Deterministic seeds

- Per-image RNG: `blake2b("(record_id)|(variant_id)") XOR SEED` then a numpy
  Generator (stage-4 method; immune to `PYTHONHASHSEED` randomisation).
- Clean variant: fixed RNG stream (constant draw), bit-reproducible.
- Train font draw: same RNG family keyed on `(record_id, "font")` (§6.7).
- Stress variant: fixed recipe + fixed RNG stream (constant draw) — the same
  deterministic view on every run (§10.6).
- The kit's own `-se` seed is dead in batch mode (§6.7); determinism comes
  from forced flags + pinned/seeded fonts + our seeded post-process.
  Re-running the notebook over the same split reproduces identical bytes
  (resume path verifies this per chunk).

### 10.3 Effect pipeline (per augmented variant, at 2200x1700, then §7.3)

Adapted from the validated `stage4_augment.py` (bounds unchanged — they were
tuned until sheets read as "printout", not "poster" or "crumpled trash"):

1. **Grid recolour (LUT):** grey master to one of
   red (w .38) / orange (w .22) / pink (w .20) / grey (w .20), sampled from
   real printouts; per-image saturation x[0.72, 1.02]; trace (level ≤55) kept
   neutral so recolouring never tints the waveform. Clean variant: red,
   fixed saturation 1.0.
**Step 0 — protective shift (augmented + stress only, before tilt/keystone):**
fixed 20 px up-left translation at master scale with paper-coloured
backfill (§20.7). Closes the pathological worst case (max tilt +
keystone aligned biting ≤11 px off a strip end/footer in ~0.015% of
views); deterministic, changes no scale, visually imperceptible (9 px
at final size). Clean views are NOT shifted (canonical framing).
2. **Tilt:** uniform ±3.0 degrees, expand + centre-crop on paper-coloured
   canvas (no black corners, no clipping — avoids the kit's Affine defects).
3. **Perspective (keystone):** p=0.7, magnitude 0.4-1.8% of W/H, bicubic.
4. **Paper texture:** unconditional mild multiply field from a random
   wrinkle photo (random crop/flip/rotate), strength 0.30-0.62, clipped
   [0.86, 1.10].
5. **Creases:** p=0.62, 1-3 folds, derivative-of-Gaussian lighting profile,
   amplitude 0.06-0.13, clipped [0.88, 1.14] — never hard enough to break a
   trace — plus 1-2 low-frequency bow terms.
6. **Illumination:** directional ramp x[0.10, 0.22] + corner falloff
   x[0.06, 0.18] + optional (p=0.65) single-edge shadow.
7. **White balance:** neutral-centred sigma=0.012 per channel
   (explicitly NOT the kit's bimodal orange/blue cast).
8. **Exposure** x[0.92, 1.06] about black, **contrast** x[0.94, 1.08] about
   the mean.
9. **Paper renormalisation:** rescale so the 95th-percentile luminance lands
   in [216, 248] — bounds the product of (4)-(8) so sheets never render as
   dark card.
10. **Sensor noise:** Gaussian sigma in [1.5, 4.5] (0-255 units) —
    deliberately low: at 768x1024 the ST segment is a few pixels tall and
    heavy noise would bury the morphology PVC/LQT/TInv depend on.
11. **Defocus:** p=0.55, Gaussian radius 0.3-0.9 px, applied last (a real lens
    blurs scene + shading + noise together).

Deliberate deviation from stage 4: **no added 18 px margin** — the exact
1024x768 frame is produced by the §7.3 pad instead (same paper colour).

### 10.4 Clinical-invariance constraints (hard rules, empirically verified)

- Timing/amplitude scale: NO analytic percentage is claimed (v1.1 amendment
  7 — the earlier rotation-only figure did not bound the perspective
  transform). Instead the spatial-scale effect of tilt + keystone is
  **measured on real outputs** by the §10.7 grid-spacing validation at
  preview time and on a production sample, with the measured distortion
  recorded in `distortion_report.json`. No time-stretch, repeat, padding,
  re-gain, or morphological signal change exists anywhere in the pipeline.
- Lighting/contrast: multiplicative about paper white and renormalised into
  [216, 248], so trace-to-grid contrast is preserved while absolute ink
  position never moves.
- Lead order / rhythm identity / labels: untouched — augmentation is
  pixel-domain only, downstream of the frozen waveform-to-master render.
- Banned: rotation >3°, perspective >1.8%, any crop, any blur >0.9 px,
  compression artefacts (PNG), lighting that pushes paper outside [216, 248],
  text/QR overlays, B/W-only conversion of the clean image.
- **Tighten-if-needed rule:** if the preview distortion report (§12) shows
  local grid-spacing deviations that are clinically meaningful (gate: any
  window deviating >5% from nominal, or clean-vs-derived mismatch >4% —
  see §10.7), augmentation limits are tightened (rotation/perspective first)
  and the preview re-run. The full build never starts on a failing report.

### 10.5 Correlation audit

Every image row records `grid_palette` + `rotation_deg` + `render_font`; the
notebook emits `grid_palette_by_label.csv` + `grid_palette_audit.json` and a
font-by-label audit (FCMAE pattern extended, §6.7) and asserts no palette or
font is concentrated in any label beyond chance — the model must not learn
the paper instead of the pathology.

### 10.6 Deterministic artifact-stress view for val/test (v1.1 amendment 4)

Each val/test ECG additionally gets exactly one `__stress.png`
(`variant_id=stress`, `variant_type=stress`), derived from the same Arial
master as its clean sibling:

- **Fixed recipe** (same for every record, no per-record draw): orange grid
  palette, tilt −2.0° (right side down), keystone at 1.0% magnitude, paper
  texture on (strength 0.5), one fixed-position crease pair, fixed
  illumination direction/strength, exposure ×0.96, contrast ×1.03, sensor
  noise σ=3.0, defocus radius 0.6 px. Fixed RNG stream → bit-reproducible.
  (Tilt/keystone reduced from +2.5°/1.2% after the preview showed the
  left-positioned footer legend clipping; §20.7. The stress view stays
  meaningfully harder than clean on every other axis.)
- **Purpose:** a held-constant "hard but fair" view for robustness probing.
  Because the recipe is identical across records, the stress set measures
  model sensitivity to realistic artifacts without any sampling confound.
- **Metric separation:** stress images carry `primary_metric=0` and are
  evaluated as a separate `stress` slice only. The primary metric uses
  exactly one image per ECG (clean, non-conflicting). Pooling clean+stress
  of one ECG into one metric is forbidden and asserted against in §15.

### 10.7 Empirical local grid-spacing validation (v1.1 amendment 7)

`v1_support/gridcheck.py` measures the grid period directly on output
pixels by 2D-FFT spectral peaks; no analytic distortion claim is made
anywhere:

- **Method:** on the final 1024×768 image, take 9 windows (4 quadrants +
  4 edge-centres + centre, each 200×200 px). In each window, Hann-windowed
  2D-FFT on grayscale; find the two perpendicular grid-axis peaks in the
  annulus [0.8×, 1.25×] of nominal frequency, with sub-bin parabolic
  refinement. Rotation-tolerant by construction (tilt rotates peaks; radius
  is the scale measurement); trace/text contribute broadband noise only.
- **Measured feature: the MINOR (1 mm) grid** (nominal 3.558 px final /
  7.874 px master; major-equivalent = 5×). Validated reason (§20.4): the
  major peak sits in cardiac-harmonic clutter (QRS-train harmonics at
  r≈9/12 rival the r=11.24 major peak), while the minor peak (r=56) is
  isolated and ~18× stronger. Same printed transform, so distortion affects
  both identically.
- **Reported per image:** per-window axis periods + angles, deviation from
  nominal (%), max window deviation, clean-vs-derived mismatch per window.
  Aggregated to `distortion_report.json` (preview: all images; production:
  stratified sample of ≥500 + every stress image's clean sibling pair).
- **Gates:** FAIL (→ tighten limits, re-preview, never proceed) if any
  window deviates >5% from nominal, or any clean-vs-derived window mismatch
  exceeds 4% (calibrated post-preview from 3%; §20.4), or period estimation
  fails in >1 window (indicates grid illegibility, itself a defect).

---

## 11. Validation / test augmentation policy

Val and test contain **exactly one clean deterministic image per ECG**
(`<record_id>__clean.png`) **plus exactly one deterministic stress image**
(`<record_id>__stress.png`, §10.6). No random augmented variants in eval
splits. The pair members are separated into four named subsets —
`val_clean`, `val_stress`, `test_clean`, `test_stress` — each with its own
image directory and its own `metadata.csv` (§14.1); every metadata row
carries a `subset` column naming its subset (§14.2). There is no
undifferentiated val/test image set anywhere in the outputs. Metric
discipline:

- **Primary metric:** clean images of non-conflicting groups only
  (`primary_metric=1` ⟺ subset ∈ {val_clean, test_clean} AND
  group not in the 31 conflicting). One image per ECG; stable, comparable,
  exactly reproducible.
- **Stress metric:** stress subsets (`val_stress`, `test_stress`) reported
  as a separate slice only, never pooled with clean.
- **Excluded from primary:** stress views + conflicting-group eval images
  (`primary_metric=0` with `metric_note`). Excluded rows are still shipped
  in metadata (flagged) for transparency, never silently dropped.

Rationale: duplicated ECGs inside an eval split make every metric depend on
which records happened to draw more variants. (Training gets the diversity;
evaluation gets one canonical view plus one separately-scored stress view.)

---

## 12. Preview-before-full-generation plan

Before the full build, the notebook renders a preview set and **displays the
images inline**, then stops at a human gate:

- Records: at least 1 per requested label
  (NORM, AF, PAC, PVC, IAVB, LBBB, RBBB, LAE, TInv, LQT, PRWP),
  **plus** one native-1 kHz `ptb` record (fs edge case), one triple-label
  record, and one member of a duplicate group — about 14 records.
- Per preview record: `clean` + `stress` + 4 augmented samples spanning the
  effect space (tilt-heavy, lighting-heavy, alternate grid colour,
  texture/crease), plus an alternate-font augmented sample proving the §6.7
  font path.
- Reported alongside (measured, not asserted): actual pixel dims at each
  stage, master DPI, layout/lead order check, paper speed/gain footer text,
  rhythm-strip sample span (= rate x 10), resize/pad arithmetic, variant RNG
  draws, per-image byte sizes (feeds the §8.2 re-projection), the §10.7
  distortion report (per-window grid periods + gates), and a zoomed crop
  proving 1 mm grid + trace legibility at 768x1024.
- The full build cell refuses to run until the preview-approval flag is set
  (explicit `APPROVE_PREVIEW = True` edit) AND the distortion gates pass, so
  an unattended run can never skip the visual check or start on a failing
  spatial-scale report.

---

## 13. Notebook structure (18 sections, restartable, low-RAM)

Mapped 1:1 to the requested sections; restartability + streaming are
architectural, not afterthoughts:

1. **Configuration** — all knobs (paths, seed, ratio, DPI, variant counts,
   S3 URI, instance knobs) in one cell; writes `generation_config.json`.
2. **Environment / dependency check** — venv, package versions, thread-pinning
   env, EBS free-space gate, code-bundle SHA verify + extract,
   `is_patched()` gate, effective-settings echo.
3. **Inspect existing ECG-Image-Kit code** — prints kit file SHAs, patch
   markers, and the exact render flags that will be used.
4. **Inspect earlier notebook reference** — summarises §2 deltas in-notebook
   (prevents "cargo-cult" settings).
5. **Load accepted-record manifest** — §3 steps 1–5 (counts, cross-check,
   waveform gate, SHA check, label parse).
6. **Inspect patient/source identifiers** — demonstrates no patient IDs exist;
   computes the 1,261 union groups + 31 conflicting-label groups; writes
   the audit CSVs.
7. **Create train/val/test split** — §4 algorithm; writes `splits/*.csv` +
   `split_report.txt`. Refuses to overwrite existing manifests.
8. **Validate split integrity** — counts sum to 31,109; 12x3 label table +
   gates; source-balance gates; group-atomicity check.
9. **Configure ECG renderer** — stage scratch `.dat` for preview records
   (wfdb transcode, no filter); echo effective kit flags + env.
10. **Generate clean preview images** — §12 clean set.
11. **Generate preview variants** — §12 augmented set.
12. **Display and validate previews** — inline images + measured report +
    `APPROVE_PREVIEW` human gate + disk re-projection gate.
13. **Full image generation** — chunked (200 records/chunk, atomic
    `index_parts/chunk_XXXXX.csv` per FCMAE pattern), `ProcessPoolExecutor`
    over `cpu_count`, per-image skip-if-exists, progress bars, `failures.csv`
    (any failure aborts the chunk with a loud row — never a silent skip).
    Streams record-by-record; peak RAM ≈ workers x 0.5 GB (measured flat,
    §20.8). Refuses to start unless workers x 1 GB fits in 85% of
    MemAvailable (RAM-fit guard). Each chunk reports per-stage timings
    (render/variants/save) in the progress line.
14. **Validate generated images** — §15 checks.
15. **Generate metadata/manifests** — §14 files + palette audit.
16. **Package dataset into TAR** — §16.
17. **Upload TAR to S3** — §17.
18. **Final validation and summary** — §18 report.

### 13.5 EC2 sizing + time estimate

- CPU-only (`CUDA_VISIBLE_DEVICES=`). Target: **c8a.4xlarge
  (16 vCPU / 32 GB)**, 16 workers, **≥ 400 GB free gp3** (§8.3).
  Workers are single-threaded (OMP/MKL/NUMEXPR pinned to 1, §13.1), so
  16 workers subscribe all 16 threads; more workers cannot help a
  CPU-bound pool, fewer leave cores idle.
- Wall-time (**measured**, §20.8 — supersedes the earlier scaled guess):
  9.0 s/record on a weak mobile i3 (conservative upper bound; Genoa
  cores are faster) → §13 ≈ 5–7 h on 16 vCPUs; validation+pack ≈ 1 h;
  S3 upload ≈ 1–2 h → **≈ 7–10 h total**, likely ~8 h. The first
  chunks print live per-stage timings, so the real rate is visible
  within the first half hour. All resume-safe.

---

## 14. Output directory + metadata/manifest structures

### 14.1 Directory layout

```text
dataset/
├── train/
│   ├── images/                  # <record_id>__clean.png + __v01..v03.png
│   └── metadata.csv
├── val_clean/
│   ├── images/                  # <record_id>__clean.png (val records)
│   └── metadata.csv
├── val_stress/
│   ├── images/                  # <record_id>__stress.png (val records)
│   └── metadata.csv
├── test_clean/
│   ├── images/                  # <record_id>__clean.png (test records)
│   └── metadata.csv
├── test_stress/
│   ├── images/                  # <record_id>__stress.png (test records)
│   └── metadata.csv
├── splits/
│   ├── train_records.csv        # §4.5 (union-grouped, pre-render)
│   ├── val_records.csv
│   ├── test_records.csv
│   ├── duplicate_groups.csv
│   ├── conflicting_duplicate_groups.csv
│   ├── patient_recovery_report.txt
│   └── split_report.txt
├── full_metadata.csv            # all images, all splits
├── generation_config.json       # every knob + seed + bundle SHA + kit SHAs
├── grid_palette_by_label.csv    # §10.5 audit
├── grid_palette_audit.json
├── font_by_label.csv            # §6.7/§10.5 font audit
├── distortion_report.json       # §10.7 production-sample grid validation
└── failures.csv                 # empty on success; any row aborts packaging
```

Build-adjacent (NOT in the training TAR):

```text
outputs/
├── master/                      # 31,109 grey masters (optional second TAR)
├── staged_scratch/              # transient WFDB .dat, deleted per chunk
├── index_parts/chunk_*.csv      # resume checkpoints
├── preview/                     # §12 images (excluded from TAR)
└── cardiosentry_v1_10s_768x1024_multivariant.tar (+ .sha256)
```

### 14.2 Metadata schema (per-subset `metadata.csv` + `full_metadata.csv`)

```text
record_id, source_record_id (= record_id; V1 has no deeper source key),
patient_id (ptbxl:<id> where recovered, else empty — §0.2),
patient_recovery_method (filename/content/unrecoverable_incomplete_upstream/
  no_upstream_identity), upstream_filename_hr, ecg_id, strat_fold,
dup_group_id, variant_id (clean/v01/v02/v03/stress),
variant_type (clean/augmented/stress),
primary_metric (1 ⟺ eval-clean subset + non-conflicting group),
metric_note (empty | conflicting_duplicate_labels:<verdict> |
  stress_view | train_split),
split (train/val/test), subset (train/val_clean/val_stress/test_clean/
  test_stress — derived: train for train rows, else <split>_<variant_id>),
image_path (relative, e.g. val_clean/images/<record_id>__clean.png),
NORM AF IAVB LBBB RBBB PAC PVC LAFB LAE TInv LQT PRWP (multi-hot 0/1),
source_dataset, grid_palette, rotation_deg, render_font,
image_sha256, image_bytes, master_sha256
```

`full_metadata.csv` = concatenation of the five per-subset files with the
same columns (subset column distinguishes). All paths relative.

### 14.3 Variant naming

```text
<record_id>__clean.png     (every split; primary eval view in val/test)
<record_id>__v01.png       (train only)
<record_id>__v02.png       (train only)
<record_id>__v03.png       (train only)
<record_id>__stress.png    (val/test only; separate stress metric, §10.6)
```

Deterministic and traceable; `record_id` itself is `source__original`
(e.g. `chapman_shaoxing__JS00001__v01.png`). Parsed from the right
(`rsplit('__', 1)`) since record IDs contain `__`.

---

## 15. Validation approach (gates, not warnings)

Order: leakage → counts → files → pixels → labels → provenance.
Any failure aborts before packaging; `failures.csv` names every offender.

1. **Leakage:** 0 `record_id` / `dup_group_id` / `v1_waveform_sha256` /
   recovered-`patient_id` overlap across the three splits; every variant's
   split == its record's split; prints the required three-line PASS block
   (per key).
2. **Counts:** accepted source records == 31,109; split record counts sum to
   31,109; image count == train x 4 + val x 2 + test x 2 exactly; no
   unresolved / Lead-II-unconfirmed IDs present (membership test vs manifests).
3. **File integrity:** every metadata row → existing PNG; every PNG → exactly
   one row (no orphans); every PNG opens (`PIL.verify` + full `load()`);
   `image_sha256` recomputed and matched.
4. **Pixels:** every image exactly 1024x768; aspect-preservation proof
   (recorded resize arithmetic + the §10.7 empirical grid-spacing validation
   on a stratified sample of ≥500: nominal 17.79 px major period, gates
   >5% nominal-deviation / >4% clean-vs-derived mismatch / >1 failed window);
   clean + stress images byte-identical across a re-run spot check.
5. **Labels:** every row's 12 multi-hot values equal the accepted manifest's
   parse for that `record_id` (join check, not trust).
6. **Provenance:** source waveform files untouched — re-verify a random 500
   `subset_waveform_sha256` entries at the end (full re-hash was already done
   at load, §3.4); assert `generation_config.json` matches the running config.
7. **Eval-subset separation + metric separation:** every row's `subset`
   equals the value derived from (`split`, `variant_id`); every val/test
   record owns exactly one `<split>_clean` + one `<split>_stress` row —
   same record, same split — and per-subset image counts equal the split
   record counts exactly; train rows all carry subset `train`;
   `primary_metric=1` rows are exactly (subset ∈ {val_clean, test_clean}
   AND non-conflicting); conflicting-group eval rows and all stress rows
   carry `primary_metric=0` with correct `metric_note`; assert no pooling
   (each `record_id` contributes ≤1 `primary_metric=1` row, and no
   undifferentiated val/test image set exists).

---

## 16. TAR packaging

- Filename: `cardiosentry_v1_10s_768x1024_multivariant.tar` (plain `.tar` —
  §8.4; deterministic order via `LC_ALL=C sort`, fixed mtime for
  reproducibility, numeric owners).
- Contents: exactly `train/ val_clean/ val_stress/ test_clean/
  test_stress/ splits/ full_metadata.csv generation_config.json
  grid_palette_by_label.csv grid_palette_audit.json font_by_label.csv
  distortion_report.json failures.csv`
  (empty `failures.csv` included as a success witness).
- Excluded: `master/`, `preview/`, `index_parts/`, `staged_scratch/`,
  notebook checkpoints, kit work files, derived WFDB copies the kit drops
  next to renders (only `*.png` are ever collected from render dirs).
- Sidecar: `<tar>.sha256` (hex + filename, `sha256sum`-compatible).
- Pre-pack gate: all §15 checks green; TAR listing verified against
  `full_metadata.csv` (member count + names); local size reported.
- Optional second archive: `cardiosentry_v1_10s_masters_200dpi.tar` (§8.3).

---

## 17. S3 upload approach

- Destination:

```text
s3://cardiosentry-fcmae-758821198776-us-east-1-an/datasets/V1/cardiosentry_v1_10s_768x1024_multivariant.tar
(+ .sha256, + optionally the masters TAR and its .sha256)
```

- Auth: boto3 default credential chain → EC2 IAM instance role.
  **No keys in the notebook** (a preflight cell asserts no `AWS_SECRET*`
  env var is set from anything but the role chain).
- Method: `s3.upload_file` with multipart `TransferConfig`
  (threshold 64 MB, progress callback) and **`ChecksumAlgorithm=CRC64NVME`**
  (v1.1 amendment 10): S3 computes and enforces per-part checksums during a
  multipart upload and stores the composed full-object checksum — corruption
  in flight is rejected by S3 itself, not merely detected afterwards.
  `sha256` hex still travels in object `Metadata` as a provenance label
  (plus the `.sha256` sidecar object), and `ContentType: application/x-tar`.
  Preflight asserts the boto3/botocore build supports CRC64NVME; if not, the
  notebook aborts with an upgrade instruction rather than silently
  downgrading integrity.
- Pre-upload: TAR + `.sha256` exist; local SHA-256 recomputed and matched;
  all §15 validations recorded green in this run.
- Post-upload (required): `head_object` → size equals local bytes AND a
  full-object `ChecksumCRC64NVME` is present; prints final URI + size +
  checksum. Missing/mismatched → loud failure, no "success" report.
- Post-upload (independent verification, one of the following is required —
  HEAD + user-supplied `Metadata.sha256` alone is explicitly NOT described
  as independent byte-integrity verification):
  (a) if `awscrt` is importable, recompute the composed full-object
  CRC64NVME locally with identical part boundaries and compare to HEAD's
  `ChecksumCRC64NVME` (preferred: no re-download, cryptographically
  independent of the transfer); else (b) streaming `get_object` re-hash of
  the full object against the local SHA-256 (flag `VERIFY_S3_BYTES`,
  re-downloads ~48–75 GB). The notebook reports which path was taken.
- Resume: skip-and-reuse any remote object whose HEAD already matches
  (size + full-object checksum), so a re-run never re-uploads 70 GB.

---

## 18. Final report fields

The notebook's last section prints (and saves to `final_report.txt`):

```text
Source accepted records:            31109
Split method:                       grouped iterative multi-label stratification
                                    (union-find: patient_id + v1_waveform_sha256;
                                    29564 groups; no blanket rare-combo forcing)
Split seed:                         20260918
Patients recovered:                 11132 records w/ ptbxl patient_id
                                    (6456 filename + 4676 content; 567 HR
                                    unrecoverable, 19410 no-upstream-identity)
Train / Val / Test source records:  ~24887 / 3111 / 3111 (exact post-split)
Train variants/record:              4 (clean+v01..v03)
Eval subsets:                       val_clean + val_stress + test_clean +
                                    test_stress (1 image/record/subset)
Total / Train / Val_clean /         ~112000 / ~99550 / ~3111 /
  Val_stress / Test_clean /           ~3111 / ~3111 / ~3111
  Test_stress images:
Primary-metric images:              val_clean + test_clean non-conflicting
                                    only (exact count)
Images failed / skipped:            0 / 0 (any nonzero aborts packaging)
Master DPI:                         200 (2200x1700; ~98.4 Hz master only)
Final image dimensions:             1024x768 (WxH; ~90.4 DPI effective)
Resize/padding method:              LANCZOS to 994x768 + 15px paper-colour pad/side
Paper speed / Gain:                 25 mm/s / 10 mm/mV
Layout:                             3x4 (2.5 s/lead) + full 10 s Lead II strip
Rhythm strip:                       II, continuous 10 s (rate x 10 samples)
Output format:                      PNG
Filtering:                          no additional signal-domain filtering
                                    (spatial rasterization/resampling only)
Variant settings:                   §10.3 bounds + seed + palette weights +
                                    seeded train fonts + fixed stress recipe
Grid-distortion validation:         §10.7 report summary (max deviations)
Metadata rows:                      == image count (join-verified)
Leakage validation:                 3-line PASS block per key (§15.1)
TAR path / size:                    outputs/*.tar / measured bytes + sha256
S3 destination / upload status:     URI / CRC64NVME-verified (path a or b, §17)
Overall validation:                 PASS (all §15 gates green)
```

No model training is started; the notebook ends here.

---

## 19. Decisions requiring your approval (nothing silent)

| # | v1.1 resolution (approved) |
|---|---|
| 1 | Union-find split on recovered patient_id + hash (29,564 groups; §0.2/§4.1) |
| 2 | 80/10/10, seed 20260918, pure greedy + feasibility-only logged repair (§4.3) |
| 3 | Train 4 (1 clean + 3 aug); val/test clean + deterministic stress in four separated subsets, separate metric (§10.1/§10.6/§11) |
| 4 | DPI 200 master, PNG (§6.5) |
| 5 | 768x1024 kept (15 px/side pad, no stretch); final ≈90.4 DPI effective (§7) |
| 6 | No additional signal-domain filtering; spatial resampling acknowledged (§9) |
| 7 | No printed header; clean fixed Arial; seeded font diversity for augmented train only (§6.7) |
| 8 | Masters outside training TAR; optional masters TAR default-upload (§8/§16) |
| 9 | Plain `.tar` + `.sha256` (§16) |
| 10 | CRC64NVME upload checksum + independent verification path a/b (§17) |

Plan v1.1 approved; implementation (`v1_image_generation.ipynb` +
`v1_support/` + code bundle) proceeds in
`prototype-code/dataset_generation_code/`, with the preview gate run locally
for visual sign-off before the EC2 handoff.

---

## 20. Implementation notes (post-preview calibration record)

Everything below was found by running the preview, with measured evidence.
No approved clinical bound (geometry, filtering, variants, metric discipline)
was changed; calibrations are documented here, not made silently.

### 21.1 Split greedy uses relative deficits (no repair needed)

The first greedy implementation (absolute deficits) starved mid-size labels
in eval splits (LAE 89 vs want 125) and needed 354 logged repair moves.
Root cause: absolute deficits always favour the 8×-larger train split early.
Fix (`split.py`): maximise total RELATIVE deficit. Final split (seed
20260918, union groups): **train 24,881 / val 3,115 / test 3,113**,
all 12 labels within a few records of ideal (e.g. LAE 996/123/127,
PRWP 510/64/64), source gates pass, **0 repair moves**. The repair path
remains as the feasibility-only fallback (§4.3.4).

### 21.2 Distortion instrument: spectral minor-grid measurement

Three iterations, each driven by a measured failure on real renders:
1. Colour-mask + autocorrelation locked onto a spurious ~14 px peak
   (antialiased grid halos at 200 DPI) and missed the true period.
2. Strict major-only mask fixed the harmonic but trace-edge pixels
   (reddish antialiased QRS edges) passed the chroma test and corrupted
   profiles; tilted images smeared 1D profiles beyond use.
3. Final: Hann-windowed 2D-FFT per window on grayscale, two perpendicular
   peaks in the nominal annulus, sub-bin refinement. Rotation-tolerant;
   trace/text are broadband noise.

### 21.3 Minor (1 mm) grid is the measured feature

Full-spectrum analysis showed the major peak (r=11.24) rivalled by
cardiac-harmonic clutter (QRS-train 2nd/3rd harmonics at r≈9/12), while
the minor peak (r=56, 3.558 px) is isolated and ~18× stronger. Both grids
are printed by the same transform, so distortion affects them identically;
reports give minor periods with major-equivalent = 5×.

### 21.4 Mismatch gate calibrated 3% → 4% (evidence)

84-image preview distribution (756 windows, 630 pairs):
clean-window deviation p50/p90/max = 0.19 / 0.23 / 0.26% (instrument floor);
derived-window deviation p50/p90/max = 1.09 / 2.22 / 3.00%;
clean-vs-derived mismatch p50/p90/max = 0.89 / 2.03 / 2.77%.
The ~2.8% extreme is genuine keystone effect at the approved 1.8% bound
(instrument floor is 10× smaller), so a 3% gate would flake on production
extremes (4,500 windows vs 630 here). 4% keeps ~40% headroom while still
catching any accidental stretch ≥4%. Nominal gate stays 5% (max observed
3.00%). 9-test regression suite: `v1_support/tests/test_gridcheck.py`.

### 21.5 Storage revised on measured bytes

Preview means: grey master 0.35 MB (~11 GB total, was 25–35 est.);
clean 0.98 MB, augmented 1.57 MB (~150 GB training PNGs + ~150 GB TAR,
was 48–75 est. — noise/texture dominate PNG entropy). Peak EBS ≈ 320 GB
→ EC2 requirement raised to **≥ 400 GB free**; S3 upload re-estimated
1–2 h; total 9–14 h. The §12 gate re-projects from measured bytes, so
future drift is caught at runtime, not in this table.

### 21.6 Minor pixel-level fixes from preview inspection

- `paper_colour`: median → 90th percentile of corner patches (200-DPI
  antialiased halos dragged the median to (243,222,219); p90 recovers
  true paper white for pad/rotation-fill colours).
- PNG saves use `optimize=True` (deterministic, ~5% smaller).
- Notebook §2: S3 packages install-deferred to §17 with retries (a proxy
  hiccup at §2 must not kill a build that needs S3 hours later); notebook
  §13 disk floor raised to 300 GB.

### 20.7 Clipping margin analysis + protective shift

Measured dark-ink (lum<60) margins on preview masters (2200×1700, spines
excluded): top ≥218, left ≥90, right ≥91, footer-text bottom 32–40,
rhythm-trace bottom ≥71 (min over 14 records). Worst-case geometry
(±3° tilt + 1.8% keystone aligned) displaces edge content by ≤82 px
bottom / ≤36 px sides: the footer legend (constant "25mm/s / 10mm/mV",
zero per-record information) could clip in extreme views, and a
pathological stack could bite ≤11 px off one rhythm-strip end in
~0.015% of augmented train images. Fix: a FIXED 20 px up-left protective
translation (paper backfill) before any tilt/keystone in augmented and
stress views only — bottom/right margins become 91/111 px, so even the
pathological stack stays ≥4 px inside the frame. No scale change
(translation only), deterministic (stress unaffected), imperceptible
(9 px at final size). Clean views unshifted. A per-image automated
no-clip gate was evaluated and rejected: tilted axes spines legitimately
cross every border test, so border-touch detection cannot distinguish
frame lines from clipped traces; safety follows from margins + bounded
geometry instead, verified visually at ±2.88° extremes in §12. The stress recipe was
adjusted (+2.5°/1.2% → −2.0°/1.0%) after its footer legend clipped;
verified intact afterwards. Residual accepted risk: in ~0.05% of random
augmented TRAIN views (extreme tilt+keystone alignment on min-margin
records), a few pixels of the constant footer legend may clip — train-only,
constant text, zero information; clinical ink stays ≥4 px inside in all
cases by the margin analysis above.

### 20.8 c8a.4xlarge sizing measurement (v1.4)

Chunk worker profiled in-process on 4- and 12-record mixed chunks
(12-rec: 9 train + 2 val + 1 test): wall scales 36.0 s → 107.5 s
(2.99× for 3× records — perfectly linear, per-chunk overhead
negligible, CHUNK_SIZE=200 kept). Per-record cost 9.0 s on an
i3-1005G1 (weak mobile part — a conservative upper bound; Genoa
cores are faster). Stage split: kit render 44% (arial 22.3 s +
font pass 24.9 s), variants 46% (49.1 s), PNG save 8% (8.2 s),
resize 2%. Peak tree RSS 0.43 GB, flat vs chunk size (streaming
design) → 16 workers ≈ 7 GB on a 32 GB box. PNG `optimize=True`
micro-benchmark: 0.241 vs 0.177 s/img for +0.8% bytes — a 2.5%
saving, not worth changing approved output bytes. Serial §14–§16
(validation sample + TAR) considered for parallelization and
deferred: ≈ 5% of runtime, and the gate path stays single-threaded
by design (auditable).

## 21. Next steps (implementation complete; EC2 run pending)

1. ~~Write `v1_support/` + `v1_image_generation.ipynb` + code bundle~~ DONE.
2. ~~Run sections 1–12 locally through the preview gate~~ DONE (rehearsal
   exit 0; 84 previews inspected; distortion PASS; §20 calibrations
   recorded). Preview PNGs + `distortion_report_preview.json` are in
   `outputs/preview/` for your own visual sign-off.
3. ~~Hand off EC2 notes~~ DONE (`EC2_NOTES.md`). Remaining: copy the
   directory + input dataset to EC2, run §§1–12, approve the preview,
   then run §§13–18 (full build → TAR → S3).
4. v1.2: eval subsets separated (`val_clean`/`val_stress`/`test_clean`/
   `test_stress`); notebook + support package updated, rehearsal §§1–12
   re-run green. Train logic unchanged, pending a later decision.
5. v1.3: EC2 pulls the input tarball from S3 when no local copy exists.
6. v1.4: c8a.4xlarge sizing — per-chunk timings, §13 RAM-fit guard,
   measured ≈ 7–10 h estimate (step 3 still pending: the EC2 run).
7. v1.5: §1 project finder (cwd-independent `v1_support` import).
8. v1.6: render-package install moved ahead of the `v1_support` import.
9. v1.7: `wfdb>=4.3` floor (pandas-3 clash); broken installs detected +
   upgraded in place.
10. v1.8: §1 fresh-interpreter probe + restart-after-replace rule (stale
    `sys.modules` made same-process re-import mix old/new wfdb files).
11. v1.9: friendly S3 credential/permission errors; IAM policy gained
    output `GetObject` for HEAD-resume.
12. v1.10: `subset_output_path` rebased onto runtime `subset_dir`
    (absolute CSV paths broke every record on EC2).
13. v1.11: §12 inline thumbnails (full-size inline output stalled EC2).
