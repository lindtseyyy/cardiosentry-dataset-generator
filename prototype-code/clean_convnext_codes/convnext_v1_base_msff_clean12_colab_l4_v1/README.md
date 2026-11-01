# ConvNeXt-V1 Base + Residual FPN on HEEDB clean12 — Google Colab **L4**, compiled

Open **CardioSentry_ConvNeXtV1_Base_RFPN_Clean12_Colab_L4.ipynb** in Colab on an
**L4** runtime and choose Runtime → Run all. Upload only the `.ipynb`.

This is the L4 twin of `../convnext_v1_base_msff_clean12_colab_v1/` (the A100
notebook). Everything that affects the model is identical, so the L4 epoch time
compares directly with the A100 run:
- **model:** V1 Base + Residual FPN (`CS_FUSION=rfpn`);
- **data:** the same dataset and patient split (`4eba80c7…`);
- **recipe:** the same, including rare sampling off and an effective batch of 64.

## What differs from the A100 notebook

| | A100 notebook | this notebook |
|---|---|---|
| hardware profile | `colab-a100` | `colab-l4` |
| micro-batch probe | 32, 16, 8, … | 16, 8, 4, … (≈22 GB VRAM); still accumulates to 64 |
| eval-batch probe | 64, 32, … | 32, 16, … |
| `torch.compile` | used only if ≥2% faster than eager | **used whenever it compiles and fits** (`CS_PREFER_COMPILED=1`); a warning if it measured slower |
| timing report | — | L4 vs A100 table at the end of section 8, plus `timing_comparison.json` |
| benchmark-only | — | `CS_BENCHMARK_ONLY=1` stops after the report and releases the runtime |
| default run name | `…_colab_a100_seed42` | `convnext_v1_base_rfpn_clean12_colab_l4_seed42` |

**The timing report** (section 8) shows:
- train and eval images/s, eager vs compiled;
- the one-off compile time;
- the compile gain;
- per-epoch, 10-epoch and 20-epoch times for the L4 and the A100 reference;
- the slowdown factor.

The reference is `CS_REFERENCE_EPOCH_MINUTES=33`, your measured compiled A100
epoch. It is only meaningful if that A100 run used the same notebook settings:
V1 Base, `rfpn`, rare sampling off.

**The report is a steady-state projection.** It leaves out the compile, which
repeats on every new runtime, and the dataset download, about 104.5 GiB per new
runtime. The first finished epoch in `training_history.csv` is the measured
confirmation.

**Continuing the A100 run on an L4:** set `CS_RUN_NAME` to the A100 run's name.
GPU and micro-batch are runtime metadata, not part of the fingerprint.

## Usage

```python
import os
os.environ["CS_BENCHMARK_ONLY"] = "1"   # timing only; omit to train
```

Every other override is the same as in the A100 notebook (`CS_FUSION`,
`CS_SEED`, `CS_RARE_SAMPLING`, `CS_TORCH_COMPILE=0`, …).
`CS_PREFER_COMPILED=0` restores the "only if faster" rule.

## Verification

```bash
python3 make_notebook.py
../../.venv/bin/python validate_notebook.py
```

All 17 CPU checks pass, verified 2026-10-02.
- **Inherited:** the 16 checks of the A100 notebook.
- **Compile rule:** a working compiled model is kept even when slower, never when it failed or is over budget.
- **New: timing report check.** It runs the real section-8 report code on synthetic rows: ratio, compile gain, totals, JSON plus Drive copy, the slower-compiled warning, and the benchmark-only stop.

**Not exercised locally:** the L4 itself, real S3 and Drive I/O, compile time,
and full training.
