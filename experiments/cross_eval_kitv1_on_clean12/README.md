# kit-v1 baseline on the clean12 validation split (Colab, evaluation only)

Open **`CrossEval_KitV1_on_Clean12_Colab.ipynb`** in Colab on an **L4 or T4** runtime,
add the AWS keys to Colab Secrets, and choose Runtime → Run all. Upload only
the `.ipynb`; the library code is inlined.

## What it does

1. **Load both checkpoints.** It loads `best.pt` of
   `runs/convnext_v1_base_kitv1/convnext_v1_base_kitv1_clean_g6e2xlarge_seed42`
   (21-label head) and of `runs/convnext_v1_base_clean12_g6e2xlarge_seed42`
   (12-label head). Each uses the weight source (EMA or raw) its training
   selected, as recorded in the checkpoint.
2. **Rebuild the clean12 validation split.** It uses the run's own
   `patient_split.csv`, and the canonical hash must be `4eba80c7…`. That gives
   11,301 records.
3. **Exclude kit-v1 training patients.** Records whose `patient_key` is in the
   kit-v1 TRAIN split (from kit-v1's `record_manifest.csv`, sha256 `b3da84a9…`)
   are dropped. With the local copies of both manifests this is **5,126
   records (3,999 patients)**, leaving **6,175**; the notebook prints its own
   count.
4. **Fetch only those PNGs.** It downloads only the needed images (~6 GB) as
   byte ranges of the plain 104 GiB clean12 TAR (`tar_ranges.py`). The archive
   is never downloaded.
5. **Run both models on one tensor per batch.** Images are decoded and
   normalized with the clean12 training code, copied verbatim.
6. **Map and score.** kit-v1 outputs are mapped to the 12 clean12 labels.
   Per-label AP is computed for both models, plus the macro over the 11
   pathologies (NORM excluded).
7. **Check the reproduction.** This notebook's clean12 AP must match the
   clean12 run's saved `validation_predictions.csv` within 0.005. Otherwise
   nothing is written.
8. **Write the summary to Google Drive.** It writes
   `cross_eval_kitv1_on_clean12.csv` and a provenance `.json` (summary numbers
   only) to `MyDrive/CardioSentry/new_runs/cross_eval_kitv1_on_clean12/`.
   Nothing is written to S3. An existing result is never overwritten unless
   `CS_OVERWRITE_OUTPUT=1`.

**Expected time:** about 15–30 minutes on an L4, a little longer on a T4 (float16 there). Indexing the two TARs with
small range requests is most of it. S3 egress is about 10 GB.

**IAM:** attach the read-only `colab_s3_cross_eval_policy.json` to the Colab
IAM user. It grants read access to the two archives and the six run files
used.

## Files

| File | Purpose |
|---|---|
| `CrossEval_KitV1_on_Clean12_Colab.ipynb` | The notebook (generated; do not edit by hand) |
| `make_notebook.py` | Builds the notebook |
| `tar_ranges.py` | Random access to members of a plain TAR via byte ranges (inlined verbatim) |
| `cross_eval_lib.py` | Manifest join, split hash, exclusion, label mapping, AP table, never-overwriting summary writer (inlined verbatim) |
| `validate_notebook.py` | CPU checks, no AWS or GPU |
| `colab_s3_cross_eval_policy.json` | Read-only, least-privilege IAM policy for the Colab user |

## Verification

```bash
python3 make_notebook.py
../../.venv/bin/python validate_notebook.py
```

All 8 checks pass (2026-10-03):
- **Notebook:** it compiles, and the library cells are verbatim copies of the
  two modules.
- **Image pipeline:** the cell is verbatim clean12 training code.
- **TAR index:** it equals Python's `tarfile` across 6 archive formats (Python
  and GNU tar; pax, gnu, ustar) and 10 segment splits. It also catches a
  corrupted header.
- **S3 reader:** it pins the ETag, retries, and detects a changed object.
- **Real local data:** the split hash and the exclusion count (5,126).
- **End to end on CPU.** Every notebook cell runs against a fake S3, with tiny
  ConvNeXt checkpoints and synthetic archives. Checks:
  - the EMA/raw selection;
  - the exclusion;
  - the reproduction guard, which fires on wrong saved predictions;
  - the CSV schema;
  - writing to the Drive folder, never to S3, and refusing to overwrite.

**Not exercised locally:** real S3 latency and throughput, the real archives'
headers, Google Drive itself, and the GPU.

## Reading the result

- **Different training labels.** The kit-v1 model was trained on noisy 21-label
  targets, and here it is scored on clean12 labels. AP needs no threshold, so
  the kit-v1 thresholds play no part.
- **Optimistic clean12 AP.** These validation records chose the clean12 run's
  best epoch, so its AP here is slightly optimistic. The test split selected
  nothing.
- **Other overlap.** Patients in kit-v1's *validation* split (603 records) stay
  in by default. `CS_EXCLUDE_KITV1_SPLITS=train,validation` removes them too.
