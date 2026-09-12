# CardioSentry FCMAE Tiny baseline

Open **CardioSentry_FCMAE_Tiny.ipynb** in Colab. It embeds all required source
files and the current V2 downstream holdout snapshot, so only the notebook needs
to be uploaded to `MyDrive/CardioSentry/codes/fcmae_tiny_v1/`.

The baseline uses ConvNeXt-V2 Tiny, the official ImageNet-1K self-supervised
FCMAE encoder, 768-high × 1024-wide RGB inputs, 32-pixel patches, uniform 60%
masking, and normalized masked-pixel reconstruction. The released checkpoint
contains **196 encoder tensors and no decoder**; the decoder is initialized
randomly. This is continued/domain-adaptive pretraining, not training from scratch.

## Run

1. Select a CUDA Colab runtime. The notebook builds an isolated Python 3.10,
   PyTorch 2.1.2/CUDA 11.8, MinkowskiEngine environment. Reserve 20 GiB for the
   environment, plus the archive, extracted images, and resized-image cache.
2. Run the configuration and setup cells. Input defaults match the diagnostic
   image generator's archive under `experiments/fcmae_synthetic_images_diagnostic_v1/`.
3. The default `RUN_BENCHMARK=True`, `BENCHMARK_ONLY=True` runs section 5A and
   skips training. Run once on each allocated T4/L4/A100 with identical settings;
   enter Colab's displayed `COMPUTE_UNITS_PER_HOUR` (or leave `None` for time only).
   Reports go to `runs/fcmae_tiny_v1_benchmarks/<GPU_timestamp>/` on Drive.
   Share `summary.md` and `report.json`; the parent `comparison.csv` combines runs.
   Compare matching comparison keys. Rates and hardware speeds are never assumed.
4. Set `RUN_BENCHMARK=False`, `BENCHMARK_ONLY=False` and rerun configuration.
   Run the pilot at the real target resolution and inspect masks, reconstruction,
   memory, runtime, and the split audit. `RUN_FULL=False` is the default.
5. Set `RUN_FULL=True` for the full run after the pilot. The included holdout
   snapshot corresponds to the current supervised V2 release. Regenerate/replace
   it if downstream splits change. Use a new run directory for changed settings.

The initial 100 epochs, 5 warmup epochs, batch 2 × accumulation 16, and learning
rate 2e-5 are starting choices, not established ECG-optimal hyperparameters.
The code keeps official sparse convolutions and sparse GRN. Sparse GRN aggregates
active sites across a microbatch; accumulation does not reproduce that statistic
for a larger microbatch. FP16 is configurable and must pass the GPU pilot.

The benchmark uses a disposable model with 2 warmup optimizer updates and 3
rounds of 6 updates, plus validation and temporary Drive checkpoint/export writes.
It projects 1, 20, and 100 epochs at defaults, including checkpoint overhead and
separately extrapolated cache preparation. Observed ranges are not runtime bounds.
Benchmarking itself consumes session time; its measured cost is recorded separately.
Future restarts, idle time, recurring worker startup and unmeasured Drive sync
delays are excluded. It does not modify training checkpoints or preserve its weights.

## Outputs

`runs/fcmae_tiny_v1_768_seed42/{pilot,full}/` contains:

- `checkpoints/last.pt`: full resume state every 100 optimizer updates or about
  15 minutes, whichever comes first at an update boundary; also before validation
  and after each completed epoch.
- `checkpoints/previous.pt`: rolling preceding snapshot, used for recovery if
  `last.pt` is missing or unreadable.
- `checkpoints/checkpoint_epoch_XXXX.pt`: preserved full resume states every
  10 epochs and at the final epoch, including epoch 20.
- `encoders/encoder_initial.pt`: matched control before ECG pretraining.
- `encoders/encoder_best_ssl.pt`: best reconstruction checkpoint, not downstream-selected.
- `encoders/encoder_epoch_XXXX.pt`: periodic dense encoder exports.
- `previews/`: reconstruction PNGs.
- `metrics.csv`, split CSVs, exclusion audit and provenance at the run root.

All of these are written directly to the mounted Google Drive run folder during
training; no manual upload of trained weights is required. No trained weights
exist until the notebook is executed. Preserve full milestone checkpoints when
you need to resume from a particular epoch, rather than an encoder-only export.

### Resume within an epoch

The trainer saves the committed batch cursor, epoch permutation, model/decoder,
optimizer, scaler, RNG states, global update count, and accumulated epoch metrics.
Snapshots occur only after an optimizer update and gradient clearing. DataLoader
prefetch cannot advance the saved cursor. Restored workers load only remaining
batches; deterministic image transforms and a separate loader RNG prevent a new
iterator from changing the model's masking RNG stream. Random worker-side
augmentations are not supported by this guarantee.

The default time trigger waits for the next update boundary; it is not an exact
15-minute deadline. Set `checkpoint_every_updates` or `checkpoint_every_minutes`
to zero to disable either trigger. Both zero retain pre-validation and completed-
epoch saves. Writes introduce a pause, with duration dependent on Drive throughput.
Temporary-file serialization/flush and rename protect the previous checkpoint
against interrupted writes, but a mounted Drive operation cannot guarantee cloud
sync after an immediate runtime loss. Verify saved files before deliberately
ending a session. Only work since the last successful snapshot needs replay.

Format version 2 is deliberately incompatible with the previous epoch-only
notebook fingerprint. Existing runs must retain their original notebook; no
automatic migration or relaxed configuration checks are applied.

Exports use official dense ConvNeXt-V2 parameter names. Load into the vendored
`models.convnextv2.convnextv2_tiny`; only pooled norm and classifier head should
be missing. A timm model requires an explicit key conversion. Downstream
classification and final model selection are separate from this notebook.

Padding pixels do not contribute to target normalization or loss. Fully padded
patches are excluded from mask sampling; partially valid patches retain their
valid pixels. The all-visible sparse/dense parity check uses one image to match
the original sparse and dense GRN aggregation domains. No ordinary dense-CNN
fallback is used. Reconstruction displays use target patch statistics to reverse
normalization and are labeled accordingly.

Known patient IDs are grouped and excluded against downstream holdouts. Missing
patient IDs use record groups; cross-source patient linkage is not established.
The snapshot has 3,925 V2 validation/test records; its original metadata hash and
source namespace mapping are recorded in `holdout_provenance.json` and
`make_holdouts.py`. It contains identifiers/split membership, not diagnosis labels.

A local comparison against the current 25,109-record raw FCMAE manifest found
zero exact held-out record matches but **85 records sharing known patient IDs
with V2 validation/test patients**. The notebook excludes these patient matches
from its actual rendered-image index; its final exclusion count can be smaller
after the generator's fidelity filtering. No source corpus files were changed.

## Verification and maintenance

From the repository root:

```bash
.venv/bin/python prototype-code/fcmae_tiny_v1/test_data_utils.py
.venv/bin/python prototype-code/fcmae_tiny_v1/test_checkpoint_utils.py
.venv/bin/python prototype-code/fcmae_tiny_v1/test_training_state.py
.venv/bin/python prototype-code/fcmae_tiny_v1/test_benchmark_report.py
.venv/bin/python prototype-code/fcmae_tiny_v1/test_checkpoint_utils.py --checkpoint /path/to/convnextv2_tiny_1k_224_fcmae.pt
.venv/bin/python prototype-code/fcmae_tiny_v1/make_notebook.py
```

Regenerate the holdout snapshot with `make_holdouts.py` only when the source V2
metadata changes, then rebuild the notebook. Companion modules are the editable
source of truth; the notebook embeds their current contents.

Local verification covered CPU data/masking/loss tests and exact sparse/dense
tensor-layout roundtrip for all 196 tensors in the actual official checkpoint.
The production epoch loop was also tested on CPU against uninterrupted training:
model and AdamW states, RNG draws, sample order and loss totals match exactly after
mid-epoch resume, including worker prefetch and the short final accumulation group.
Fault-injection tests cover failed writes and recovery from an unreadable latest
checkpoint. These tests do not establish bitwise equivalence across CUDA hardware.
This machine has no available CUDA GPU. The isolated environment build, actual
sparse convolution execution, GPU export parity, and full-size optimizer step
remain to be exercised by the notebook's pilot. No pretraining run has been
performed or claimed.

## Upstream source and license

Vendored model code is unchanged from Meta ConvNeXt-V2 commit
`2553895753323c6fe0b2bf390683f5ea358a42b9`. The notebook installs the exact fork
`shwoo93/MinkowskiEngine@bbc30ef581ea6deb505976b663f5fc2358a83749`.
Adaptations are in `rect_fcmae.py`, `data_utils.py`, and `checkpoint_utils.py`.
The MIT code license and CC-BY-NC pretrained-weight terms are preserved in
`vendor/LICENSE`. Weights are downloaded at runtime, not committed.

Official checkpoint SHA-256 verified during notebook development:
`b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef`.

Sources: [official repository](https://github.com/facebookresearch/ConvNeXt-V2),
[ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808).

### CUDA compiler mismatch during setup

Setup revision 2 uses NVIDIA's `cuda-11.8.0` release channel with strict priority
and explicitly pins `cuda-nvcc=11.8.89`. It reconciles existing partial conda
environments and checks the exact compiler before building MinkowskiEngine.
Upload the regenerated notebook and run sections 1, 2, then 3; section 2 must
materialize the updated setup script. Section 3 displays “CUDA pin fix v2”.
The CUDA build still requires verification on Colab hardware.

Setup revision 3 additionally inserts `ATen/ATen.h` before the CUDA headers in
MinkowskiEngine's `src/spmm.cu`, resolving the incomplete `at::Tensor` header
ordering reported by the PyTorch 2.1.2/CUDA compiler. The patch is idempotent,
checks the expected source anchors, and is recorded in `runtime_ready.json`.
The upstream revision remains pinned; this is an explicit local build adaptation.
Re-upload and rerun sections 1–3; section 3 displays “CUDA + Tensor header fixes v3”.

Setup revision 4 fixes `collections.Sequence` imports in all five affected
MinkowskiEngine Python modules, using `collections.abc.Sequence` while preserving
`collections.namedtuple`. The reported revision-3 build completed compilation and
installation but failed on this Python 3.10 import. Revision 4 also disables the
upstream unconditional build-directory deletion so Ninja can reuse valid compiled
objects on retries in the same VM. Fresh sessions still require compilation.
The notebook section 3 heading now reads “Runtime compatibility fixes v4”.

Setup revision 5 fixes the final reproducibility export for micromamba 2.x.
Explicit lock generation now uses `micromamba env export --explicit` instead of
the unsupported conda-style `micromamba list --explicit` command. The reported
revision-4 run had already completed the CUDA environment and MinkowskiEngine
build before failing at this final metadata step, so rerunning sections 2–3 in
the same Colab VM can reuse the existing environment and Ninja build objects.
The notebook section 3 heading now reads “Runtime compatibility fixes v5”.

### L4 sparse preflight output mismatch

The original hidden-content preflight compared two complete MinkowskiEngine
CUDA forwards at a tolerance of `1e-5`. MinkowskiEngine 0.5.4 sparse CUDA
convolutions use atomic accumulation and can produce small run-to-run numerical
differences, so that comparison could fail on an L4 even when both forwards
received identical active features. The strict equality check now runs at the
deterministic isolation boundary immediately before dense stem features are
converted to a sparse tensor. This continues to prove that changed masked pixels
cannot enter the sparse encoder while allowing the real sparse forward, dense
encoder export parity, and backward-gradient checks to run independently.
