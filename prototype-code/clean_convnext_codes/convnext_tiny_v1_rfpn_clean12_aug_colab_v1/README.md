# ConvNeXt-V1 Tiny + Residual FPN on HEEDB clean12 + augmented sheets (Google Colab, A100)

Open **CardioSentry_ConvNeXtV1_Tiny_RFPN_Clean12_Aug_Colab.ipynb** in Colab (A100,
High-RAM) and choose Runtime → Run all. Upload only the `.ipynb`.

It is `../convnext_tiny_v1_rfpn_clean12_colab_v1/` (same model, recipe, split,
seed, harness and Drive setup) trained on **two sheets per record**:

| view | archive in `<shared folder>/new_dataset/` | source |
|---|---|---|
| `clean` | `clean_ecg_signal_images_clean12.tar` (+ `.sha256`) | `../clean12_ecg_signal_images_ec2/` |
| `augmented` | `clean_ecg_signal_images_clean12_augmented.tar` (+ `.sha256`) | `../clean12_augmented_ecg_images_ec2/` |

The two archives stay separate on Drive; both must be in the same folder. The
notebook streams and verifies each one, then joins the views on `record_path`.
`CS_DATASET_TAR_NAME` / `CS_AUG_DATASET_TAR_NAME` change the file names,
`CS_DATASET_DRIVE_DIR` the folder.

## What changes against the clean-only run

| part | change |
|---|---|
| training images | clean + augmented sheet of every training record, each once per epoch (~181k sheets, 2× the steps per epoch) |
| labels, patients, split | from the clean archive's audited `record_manifest.csv`; the augmented archive's copy must agree record by record (section 4a). Both views of a record are in its patient's split (`4eba80c7…`). |
| pos_weight | unchanged: both views carry the same labels |
| validation (checkpoint, raw/EMA, thresholds) | both views pooled |
| test | pooled (headline), clean view, augmented view; each with its own patient bootstrap (`test_bootstrap_*_{clean,augmented}.*`, `per_view_metrics.csv`) |
| comparison (section 11) | clean-only references pair with the **clean view** only; each model's clean − augmented gap is paired too |
| default run group | `convnext_tiny_v1_rfpn_clean12_aug_colab_a100_seed42` |
| default references | `convnext_tiny_v1_rfpn_clean12_colab_a100_seed42/convnext_tiny_rfpn` (headline: effect of adding augmented sheets, on clean sheets) and `convnext_tiny_v1_v2_clean12_colab_a100_seed42/convnext_tiny` |

## Disk: lossless WebP on Colab

Both archives as PNG need ~235-265 GiB; a Colab A100 runtime has ~193 GiB free
(the clean-only runs logged 88.8 GiB free after the 104.5 GiB clean archive).
With `CS_IMAGE_STORAGE=auto` (default) section 3 therefore stores every sheet as
**lossless WebP** while it streams (`CS_TRANSCODE_WORKERS`, default vCPUs − 1,
spawned processes). Each sheet is decoded with libpng as training decodes it,
encoded, decoded back with torchvision's WebP decoder, and must be
pixel-identical, or the cell stops. Measured on 8 sample sheets per view
(Pillow 12.3, torchvision 0.28, one local core):

| view | WebP settings | size vs PNG | decode time vs PNG |
|---|---|---|---|
| clean | `method=0, quality=75` | 59% | 31 vs 60 ms (−48%) |
| augmented | `method=1, quality=0` | 68% | 51 vs 74 ms (−31%) |

`CS_IMAGE_STORAGE=png` keeps the PNGs (needs the larger disk); `webp` forces the
conversion. Storage is not part of the fingerprint (the pixels are identical),
so a run may resume on either.

Not measured: the conversion's throughput on a Colab A100 runtime (local
encode cost ~0.11-0.20 s per sheet per core, so ~11 workers should keep up with
the Drive stream; section 3 prints MiB/s), and the real augmented archive's size.

## Time budget

Staging two archives (~1-1.5 h on a fresh runtime, estimate: the WebP encode,
~0.16-0.20 s per sheet per core locally, limits it below the Drive stream) plus ~2× the
clean-only epoch (measured 16 min) → ~30 min per epoch, ~10 h for 20 epochs
[INFERENCE from the clean-only run]. The benchmark cell prints the measured
projection before training. Resume works as in the clean-only notebook.

## Regeneration and local verification

```bash
python3 make_notebook.py                         # inlines fusion_necks.py
../../../.venv/bin/python validate_notebook.py   # CPU, a few minutes
```

The validator runs the notebook's own code on CPU with random weights,
synthetic sheets (clean as PNG, augmented as lossless WebP), a fake Drive API
and a temporary "Drive". Beyond the clean-only checks it covers:

- the WebP worker on 8 real clean/augmented sample sheets (pixel-identical through the training dataset);
- section 3 with both archives: API and mount mode, PNG and WebP storage, `auto`
  choosing PNG on a large disk and WebP on a small one, skip on re-run, a
  checksum mismatch on one archive (the other kept), a tar member outside the
  top folder, a damaged Drive copy;
- section 4a: failed/missing augmented sheets, a byte-different but equal record
  manifest, a disagreeing label, an unknown record, a duplicate;
- section 4c: both views of a record in one split, mapped to their stored
  files; pos_weight equal to the clean-only one;
- per-view test metrics equal to the pooled predictions restricted to a view;
- section 11's like-for-like pairing (a clean-only reference pairs with the
  clean view at exactly 0 on a copy, never with the augmented view).

It does not cover A100 speed and memory, `torch.compile`, the real Drive API,
or full-corpus training.
