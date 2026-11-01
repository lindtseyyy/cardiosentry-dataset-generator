# ConvNeXt-V2 Base on HEEDB clean12 ECG sheets — Google Colab (A100)

Open **CardioSentry_ConvNeXtV2_Base_Clean12_Colab.ipynb** in Google Colab on an
A100 runtime and choose Runtime → Run all. The notebook is self-contained:
upload only the `.ipynb`. The Python files here build and check it locally.

- **Dataset:** read from S3, read-only:
  `s3://cardiosentry/dataset/clean_ecg_signal_images_clean12.tar` and its `.sha256`.
- **Outputs:** written to Google Drive, at
  `MyDrive/CardioSentry/new_runs/<CS_RUN_NAME>/` (default name
  `convnextv2_base_clean12_colab_a100_seed42`).

## Where it comes from

| part | source |
|---|---|
| dataset audit, 12 labels, patient split (SHA-256 `4eba80c7…`) | `../convnext_v1_base_clean12_v1/` (unchanged, so V1 and V2 are scored on the same patients) |
| model, mixed-precision GRN, collapse guard, conservative V2 recipe | `../convnextv2_base_kitv1_clean_v1/` |
| new | Colab/Drive plumbing, streamed S3 extraction, rare-label repeat-factor sampling, patient-bootstrap CIs |

The generator was built by a three-way merge: the V1→V2 changes from the kit-v1
notebooks, applied on top of the clean12 V1 generator, then the Colab changes.

## Model and recipe

The model is `convnextv2_base.fcmae_ft_in22k_in1k_384`: about 88 M parameters,
GRN in every block, no LayerScale. It takes the native 768×1024 input.

The recipe is the kit-v1 V2 Base one. That recipe trained normally on kit-v1:
validation pathology Macro AP went 0.42 → 0.61 → 0.68 over its first three
epochs, with no collapse.
- **Learning rate:** peak head LR 1e-4, 3-epoch warm-up, then cosine decay to 1% over 20 epochs.
- **Layer-wise LR decay:** 0.925 per block, so the stem gets ≈ 0.056× the head LR.
- **Regularization:** weight decay 0.05, drop-path 0.2.
- **Batch and EMA:** effective batch 64, EMA 0.9998 with raw/EMA selection.
- **Early stopping:** patience 5, from epoch 10.
- **Safeguards:** a collapse guard and a mixed-precision GRN.

Licence of the pretrained weights: CC-BY-NC 4.0.

## What changed to help the rare labels

On the clean12 training split, 3AVB is 1.36% of records and 2AVB 3.22%. Every
other label is 8–15%.

- **Repeat-factor sampling** (Gupta et al., LVIS 2019), on by default
  (`CS_RARE_SAMPLING=1`, threshold `CS_RARE_SAMPLING_THRESHOLD=0.10`):
  - **Per label:** `r = max(1, sqrt(0.10 / frequency))`. That is 3AVB 2.72, 2AVB 1.76, STEMI 1.12, and every other label 1.
  - **Per image:** the largest `r` among its labels.
  - **Per epoch:** each image is drawn `floor(r)` times, plus one extra draw with probability `r − floor(r)` (systematic sampling over a seeded shuffle). An epoch therefore always has exactly 95,902 draws, 5.7% more than the 90,732 rendered training images.
  - **Order:** depends only on `(CS_SEED, epoch)`, so a resumed run replays it exactly.
  - **Augmentation:** repeats of one image get different brightness/contrast draws, because the draw is keyed by the occurrence, not the row.
- **`pos_weight` on the resampled counts.** The sqrt(neg/pos) weights are computed from the expected per-epoch counts, so rarity is not compensated twice. 3AVB 8.53 → 5.27, 2AVB 5.48 → 4.20; the others barely move. Net per-epoch gradient weight on positives is about ×1.6 for 3AVB and ×1.3 for 2AVB; other labels stay within ±3%.
- **Collapse reference:** the constant-predictor loss used by the collapse guard is computed at the prevalence the sampler actually feeds.
- **Patient-bootstrap 95% CIs:**
  - **What:** test AP for each label and the pathology Macro AP, written to `test_bootstrap_ci.csv`, `report.md` and `result.json`.
  - **How:** 1,000 resamples of the 9,602 test patients, with all of a patient's ECGs moving together; about a minute.
  - **Why:** 3AVB has only 154 test positives, so its AP must be read with its interval.

`CS_RARE_SAMPLING=0` restores plain uniform epochs with the original weights.
It is part of the run fingerprint, so use a new `CS_RUN_NAME` for it.

## Colab specifics

**Setup (once):**
1. Create an IAM user in the account that owns `cardiosentry` and attach
   `colab_s3_read_policy.json` to it. It grants read access only:
   - `s3:GetObject` on the archive, its `.sha256`, and `dataset/splits/*`;
   - `s3:ListBucket` on `dataset/*`.
2. Create an access key for that user.
3. In Colab, open **Secrets** (the key icon) and add `AWS_ACCESS_KEY_ID` and
   `AWS_SECRET_ACCESS_KEY`. Optionally add `AWS_DEFAULT_REGION=us-east-1`.
   Turn on notebook access for each.

If the IAM user is in a different AWS account, the bucket also needs the
statements in `../convnext_v1_base_clean12_v1/cross_account_bucket_policy.json`,
limited to the read statements.

**Disk and download:**
- **Streamed, verified, never stored:** section 3 streams the 104.5 GiB archive with 16 parallel 32 MiB ranged GETs straight into `tar -x`, hashing every byte. The archive itself is never written to disk.
- **Disk needed:** only the extracted images, about 104.5 GiB plus an 8 GiB margin. The cell checks free space first.
- **On failure:** if the SHA-256 or ETag doesn't match, or `tar` fails, the extracted tree is deleted and the cell raises.
- **New runtimes re-download:** each new Colab runtime streams the archive again, about 112 GB of S3 egress, billed to the bucket owner (around USD 10 at standard rates). Avoid unnecessary runtime restarts.

**Drive:**
- **Write path:** every run file is written locally first, then copied to the Drive run folder as `*.partial`, renamed, and size-checked.
- **When:** `best.pt` on every new best, and `last.pt` plus history and log after every epoch.
- **Resume:** after a disconnect, reconnect and Run all with the same run name. `last.pt` is copied back from Drive and training continues from the last finished epoch.
- **Unreachable Drive:** a Drive folder that can't be reached is never treated as a fresh run.

**Runtime release:** the last cell flushes Drive and **releases the Colab
runtime**, so it stops using compute units. It runs only if every cell before
it succeeded. Set `CS_RELEASE_RUNTIME=0` to keep the runtime.

**Hugging Face cache:** the cache stays on the local disk; the ~350 MB weights
download in seconds. `CS_HF_HOME=/content/drive/MyDrive/CardioSentry/hf_cache`
moves it to Drive.

## Runtime

There is no measured A100 timing yet. For reference, on an L40S:
- **V1 Base, kit-v1:** about 3,650 s per epoch, compiled.
- **V2 Base, kit-v1:** about 4,020 s per epoch for ~155k training images.

clean12 has 95.9k draws per epoch, so expect very roughly 40 minutes per epoch
on an A100, or 9–14 h for 20 epochs. Early stopping can end it sooner.
Section 8 prints the measured estimate before the long training cell starts.
Colab Pro+ background execution, or keeping the tab open, avoids idle
disconnects; resume covers the rest.

## Overrides

Set them in a cell before the config cell, for example
`import os; os.environ["CS_SEED"] = "43"`.

```text
CS_RUN_NAME                    default convnextv2_base_clean12_colab_a100_seed<CS_SEED>
CS_SEED=42                     training seed; does not change the split
CS_RARE_SAMPLING=1             0 = uniform epochs (new run name)
CS_RARE_SAMPLING_THRESHOLD=0.10
CS_BOOTSTRAP_ROUNDS=1000
CS_BASE_LR=1e-4                e.g. 5e-5 after a collapse-guard stop (new run name)
CS_RELEASE_RUNTIME=1           0 keeps the Colab runtime after the last cell
CS_DRIVE_RUNS_DIR              default /content/drive/MyDrive/CardioSentry/new_runs
CS_SPLIT_FILE                  reuse a specific patient_split.csv (local, Drive, or s3://)
CS_S3_DOWNLOAD_WORKERS=16      CS_S3_DOWNLOAD_CHUNK_MIB=32
CS_INTRA_EPOCH_SYNC_MINUTES=0  >0 writes mid-epoch snapshots (never used for resume)
CS_TORCH_COMPILE=1  CS_NUM_WORKERS  CS_POS_WEIGHT_MODE=sqrt|full|none  CS_DETERMINISTIC=0
```

## Regeneration and local verification

```bash
python3 make_notebook.py
../../.venv/bin/python validate_notebook.py   # torch, timm 1.0.28, scikit-learn, boto3
```

The CPU validator passes all 16 checks:
- **Inherited V2 checks:**
  - real V2 Base, with stock-timm keys and outputs, native 768×1024 inference, and backward;
  - GRN equivalence, and both collapse guards;
  - the data pipeline, compile selection, OOM probe fallback, and raw/EMA selection;
  - resume after interruption, and the inference export.
- **Patient split:** synthetic data, plus the real clean12 manifest, which reproduces SHA `4eba80c7…`.
- **New: sampler:** fixed length, epoch-keyed, mean draws equal to `r` on a clean12-sized problem, occurrence decoding, and distinct augmentation for repeats.
- **New: bootstrap:** reproducible, equal to explicit patient resampling, and 1.0 for perfect scores.
- **New: Drive persistence:** atomic copy, background copy, restore-missing-only, sync manifest, and refusal of an unmounted Drive.
- **New: streamed extraction against a fake S3:** in-order reassembly and byte-exact extraction; a hash mismatch, an ETag change, or a missing sidecar is refused and the tree deleted.

**End-to-end run of cells 2–7.** Run separately on CPU against the real clean12
manifest, with a fake S3 serving an archive of one tiny PNG per record (two
simulated render failures) and a temporary folder as Drive:
- the stream was verified and extracted;
- the audit passed, and the split reproduced `4eba80c7…`;
- an epoch has 95,902 draws, with the `pos_weight` values above;
- the split files and the sample grid were copied to "Drive";
- a rerun skipped the download;
- a wiped runtime re-streamed and restored the split from Drive;
- a shared S3 split was loaded and verified.

The final-cell bootstrap block also ran on the real test split (9,602 patients),
at 13 s per 200 rounds.

**Not exercised locally:** the A100 itself, real S3 and Drive I/O, and
full-corpus training.
