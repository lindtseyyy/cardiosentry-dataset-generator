# What to upload, and where

Mirror this tree into `MyDrive/CardioSentry`:

```
MyDrive/CardioSentry/
├── codes/synthetic_image_generation/
│   └── cardiosentry_generate_images.ipynb     <- open this in Colab
└── datasets/
    ├── cardiosentry_pipeline.tar.gz           <- 12 MB, patched pipeline + kit
    ├── cardiosentry_pipeline.tar.gz.sha256
    └── harmonized.tar                         <- 4.4 GB, build it yourself (below)
```

## Building harmonized.tar

```bash
cd dataset-combination-code/ecg_dataset_pipeline/data/releases/v3
tar -cf harmonized.tar harmonized
sha256sum harmonized.tar > harmonized.tar.sha256
```

Upload the `.tar`, not the folder. 74,662 loose files through Drive would take
hours and the notebook reads them from local disk anyway.

## The notebook writes to

| path | contents |
|---|---|
| `experiments/synthetic_images_v1/` | `images_{train,val,test}.tar` + checksums |
| `runs/synthetic_images_v1/` | `index.csv`, `settings.json`, stage logs |

Nothing is written to `datasets/` (read-only in practice) or the CardioSentry root.
