# ConvNeXt-V1 Tiny + Residual FPN on HEEDB clean12 (Google Colab, A100)

Open **CardioSentry_ConvNeXtV1_Tiny_RFPN_Clean12_Colab.ipynb** in Colab (A100,
High-RAM) and choose Runtime → Run all. Upload only the `.ipynb`.

**Drive setup.** The `CardioSentry` folder, including the ~104.5 GiB archive,
lives in Account A's Drive and is shared with Account B (Editor). Account B
runs Colab. Do this once, as Account B, in drive.google.com: Shared with me →
right-click the folder → Organize → Add shortcut → My Drive. Nothing is copied.
The config cell finds the folder under `/content/drive/.shortcut-targets-by-id/`.
It then runs a small read/write access test. The archive is read in place, and
outputs go to `<shared folder>/new_runs/`.

| variable | use |
|---|---|
| `CS_SHARED_FOLDER_ID` | the folder id from its URL; picks the folder unambiguously |
| `CS_SHARED_FOLDER_NAME` | the folder name, when it is not `CardioSentry` |
| `CS_DRIVE_PROJECT_ROOT` | an explicit mounted path |

It trains `convnext_tiny.fb_in22k_ft_in1k_384` with the Residual FPN
(`CS_FUSION=rfpn` of the V1 Base fusion notebooks). This fills the missing
cell of this grid:

| | plain | + RFPN |
|---|---|---|
| V1 Base | `../convnext_v1_base_clean12_v1/` | `../convnext_v1_base_msff_clean12_colab_v1/` |
| V1 Tiny | `../convnext_tiny_v1_v2_clean12_colab_v1/` (`convnext_tiny`) | this folder |

## Where it comes from

| part | source |
|---|---|
| Drive dataset streaming, split, harness, resume, multi-model run group, comparison, benchmark-only mode | `../convnext_tiny_v1_v2_clean12_colab_v1/` (the generator started as a copy). The V2 model and `MixedPrecisionGRN` were removed. |
| Recipe | the plain V1 Tiny one, unchanged: LR 2e-4, WD 0.02, warm-up 2, layer decay 0.86, drop-path 0.1, head init 0.001, rare sampling off |
| Neck | `fusion_necks.py`, byte-identical to the V1 Base RFPN notebooks' copy. Only `rfpn` (FPN neck + residual head, 256 channels) is used. |

## The model

- **Neck:** an FPN reads stages 2–4. Those are 192/384/768 channels at strides
  8/16/32, projected to 256 each.
- **Residual head:** each level is pooled, then the three are concatenated and
  passed through LayerNorm and a zero-initialised Linear. Its logits are added
  to the timm head's.
- **Step 0:** the model is exactly the plain V1 Tiny (checked).
- **Learning rates:**
  - the backbone keeps the plain V1 Tiny's per-block LR profile (20 levels,
    stem at 0.057×);
  - the neck and fusion classifier train at the peak LR.

Measured by `validate_notebook.py` at 768×1024:

| model | parameters | GMACs |
|---|---|---|
| V1 Tiny + RFPN | 29.96 M (2.13 M fusion) | 80.4 |
| V1 Tiny | 27.83 M | 69.8 |

The neck costs +15% compute on Tiny, against +4.5% on Base. Its stride-8 3×3
conv at 256 channels is a fixed cost, and Tiny's backbone is smaller.

## A100 tuning

Inherited from the Tiny notebook: BF16 AMP, TF32, channels-last, fused AdamW,
`torch.compile` (kept when ≥2% faster), the largest micro-batch under 90% of
VRAM (works on both 40 and 80 GB cards), and uint8 batches normalized on the GPU.

A Tiny model can train about as fast as Colab's 12 vCPUs decode 1024×768 PNGs,
so three changes target the input side:

| change | effect |
|---|---|
| libpng decode (`torchvision.io.decode_png`) instead of PIL + no-op resize | ~15% less CPU per sheet (45 vs 53 ms on one local core); pixel-identical on the 12 real clean12 sample sheets; off-size sheets keep the PIL resize path |
| raw and EMA validation in one pass | validation decodes each image once per epoch instead of twice |
| worker tuner also tries `cpu_count − 1` (11 on Colab A100) | the tuner picks the fastest of 4/6/8/10/11 |

The epoch-time estimate in section 9 still counts two validation passes, so on a
decode-bound A100 it is an upper bound. The first `training_history.csv` row is
the measured epoch time.

## Outputs and comparison

`<shared folder>/new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42/` (files owned by Account B):

- `convnext_tiny_rfpn/`: one complete run. `inference.pt` records `fusion_neck`,
  `fusion_head`, `fusion_channels` and `fusion_stage_indices`. To rebuild it,
  create the timm model, then wrap it with `ConvNeXtMultiScaleFusion`.
- `comparison.*`: section 11 pairs this run with the references on the same
  patient bootstrap resamples. The default `CS_REFERENCE_RUNS` paths are under
  `new_runs/`:
  - `convnext_tiny_v1_v2_clean12_colab_a100_seed42/convnext_tiny`: plain V1 Tiny,
    so the RFPN − plain row isolates the neck;
  - `convnext_v1_base_rfpn_clean12_colab_a100_seed42`: V1 Base + RFPN;
  - `convnextv1_base_clean12_a100_seed42`: plain V1 Base.

  A missing or mismatched reference is skipped.

## Overrides

- `CS_ARCHITECTURES=convnext_tiny_rfpn,convnext_tiny` also trains the plain V1
  Tiny in the same session (optional in-notebook baseline).
- `CS_<SETTING>_CONVNEXT_TINY_RFPN` overrides one recipe value. The settings
  are `BASE_LR`, `WEIGHT_DECAY`, `WARMUP_EPOCHS`, `LAYER_DECAY`, `DROP_PATH`,
  `HEAD_INIT_SCALE` and `RARE_SAMPLING`. Each is in the fingerprint, so use a
  new `CS_RUN_GROUP`.
- Resume, failures, `CS_BENCHMARK_ONLY` and the dataset options work as in
  `../convnext_tiny_v1_v2_clean12_colab_v1/README.md`.

## Regeneration and local verification

```bash
python3 make_notebook.py                         # inlines fusion_necks.py
../../../.venv/bin/python validate_notebook.py   # CPU, a few minutes
```

The validator runs the notebook's own code on CPU, with random weights,
synthetic PNGs, a fake Drive API and a temporary "Drive". It checks:

- the fusion cell equals `fusion_necks.py`;
- the recipe and its overrides;
- the RFPN model:
  - stock timm keys under `backbone.`;
  - output equal to the plain V1 Tiny at step 0;
  - backbone LR/WD profile equal to the plain model's, parameter by parameter;
  - fusion groups at the peak LR;
  - backward reaching every parameter;
- libpng decoding pixel-identical to PIL + resize on the real sample sheets in
  `../clean12_ecg_signal_images_ec2/samples/`, and the PIL fallback for an
  off-size palette PNG;
- one shared raw + EMA validation pass equal to two separate passes;
- section 3 dataset staging (API mode with injected faults, re-run skip, mount
  mode, SHA-256 mismatch, damaged Drive copy);
- an RFPN run interrupted in epoch 2 and resumed from Drive, with an export
  that strict-loads into timm + `ConvNeXtMultiScaleFusion`;
- the skip of a finished model, a failing baseline that does not stop the
  notebook, and the full baseline run;
- the comparison, with a nested `group/arch` reference that pairs at exactly 0
  and mismatched or missing references that are skipped;
- refusal of a changed recipe;
- benchmark-only mode;
- native 768×1024 forwards.

It does not cover A100 speed and memory, `torch.compile`, the real Drive API, or
full-corpus training.
