---
name: google-drive-directory-map
description: Use when generating a Jupyter notebook or saving any file, dataset, or output in the CardioSentry project folder (MyDrive/CardioSentry in Google Drive) — tells you which of its five top-level folders each artifact belongs in.
whenToUse: Creating or saving notebooks, scripts, datasets, experiment artifacts, Hugging Face downloads, or run results in the CardioSentry project folder.
---

# CardioSentry Directory Map

The CardioSentry project folder (`MyDrive/CardioSentry` in Google Drive) has
exactly five top-level folders. Everything the agent creates must land in one
of them — never at the CardioSentry root itself.

## Folder purposes

| Folder | Purpose | What goes in |
|---|---|---|
| `codes` | Source code and notebooks. | The generated `.ipynb` plus any `.py` scripts, in one project/topic subfolder each. |
| `datasets` | Raw, archived datasets. Read-only in practice. | Dataset archives (`.tar`, `.zip`, ...) with their checksum files next to them. Do not unpack or preprocess here. |
| `experiments` | Per-experiment working area. | Experiment configs, extracted/prepared data, intermediate plots, and per-experiment artifacts. One subfolder per experiment. |
| `hf_cache` | Hugging Face download cache. | Model and dataset downloads managed by Hugging Face. Never write here by hand. |
| `runs` | Finished training/evaluation runs. | Metrics (CSV/JSON), logs, predictions, and run summaries. One subfolder per run. |

## Directory map

```
MyDrive/CardioSentry
├── codes/
│   └── <project-or-topic>/
│       ├── <name>.ipynb        # notebooks the agent generates
│       └── <script>.py
├── datasets/
│   ├── <dataset>.tar|.zip|...  # raw archives, never unpacked here
│   └── <archive>.sha256        # checksums next to their archives
├── experiments/
│   └── <experiment>/
│       ├── config.*            # parameters/configuration
│       ├── data/               # experiment-local prepared data
│       └── plots/              # intermediate figures
├── hf_cache/                   # HF_HOME target (models, datasets, tokenizers)
└── runs/
    └── <run-name>/
        ├── metrics.csv|.json   # final numbers
        ├── predictions.csv     # model outputs
        └── logs/               # run logs
```

## Placement rules for a generated notebook

1. Save the notebook itself in `codes/<project-or-topic>/`.
2. Read inputs from `datasets/` (never reach into `hf_cache/` directly).
3. Write experiment-specific intermediate artifacts to `experiments/<experiment>/`.
4. Write final run results to `runs/<run-name>/`.
5. Point Hugging Face at the cache: set `HF_HOME` (or `HF_HUB_CACHE`) to `hf_cache/` and let the library download — do not create or move cache files yourself.

## Observed contents (this workspace)

- `codes/` — Python scripts (e.g. `compare.py`); generated notebooks go beside them.
- `datasets/` — large archives (`prototype_1152.tar`, `external_testset.tar`) with SHA-256 checksum files.
- `experiments/` — ECG experiment artifacts: rendered ECG plots (PNG), layout JSON, per-record data.
- `hf_cache/` — Hugging Face model blobs (100 MB+ binary files) with metadata.
- `runs/` — model evaluation results (e.g. `resnet50_hires`): prediction CSVs and JSON summaries.

## Mount note

This session's working directory is the CardioSentry root. When the folder is
mounted via GVfs, its five subfolders appear as opaque Drive IDs instead of
names. Verified ID → name mapping (2026-08-23):

| Drive ID | Folder |
|---|---|
| `1R2ncyGSlDwkeRt653Xa_2ZoQT4uPxgPg` | `codes` |
| `1VCfZAoC7ZtClMY2h76EEZW01w4PrrzDB` | `datasets` |
| `1mpDwEBItrIPNfm5sEWXcUh4sw06kqvca` | `experiments` |
| `1zExXJGz2ws-qR5iw4GjlTmyJHdvK5xBl` | `hf_cache` |
| `1BkqB_sn-6bN6jQxgf8ue9SBjiZqFphaY` | `runs` |

If the mount is read-only, save artifacts locally and tell the user where they
should be uploaded; never scatter files at the CardioSentry root.
