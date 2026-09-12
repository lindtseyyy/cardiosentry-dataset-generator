# ConvNeXt-V2 Tiny master-corpus fine-tuning on Google Colab

Open `CardioSentry_ConvNeXtV2_Tiny_Master_Colab_L4_High_RAM.ipynb` in Colab,
select **Runtime → Change runtime type → L4 GPU**, enable the **High-RAM** system
profile, and run the notebook from top to bottom.

The model, labels, loss, 768×1024 resolution, optimizer, layer-wise LR decay,
effective batch size, warm-up/cosine schedule, early stopping, checkpoint metric,
threshold policy, and all evaluation reports are identical to the EC2 version.

## Expected Google Drive inputs

The defaults match the output layout of `cardiosentry_generate_images.ipynb`:

```text
MyDrive/CardioSentry/
├── experiments/synthetic_images_v1/
│   ├── images_train_00.tar ... images_train_05.tar
│   ├── images_val.tar
│   ├── images_test.tar
│   └── <one .sha256 file per archive>
└── runs/synthetic_images_v1/
    └── index.csv                         optional; preferred when available
```

The notebook also checks `MyDrive/CardioSentry/datasets/synthetic_images_v1/`
for a reorganized archive folder. If `index.csv` is absent, place the exact v3
`harmonized_ecg_metadata.csv` beside the archives. The metadata-reconstruction
fallback and its strict SHA/count/identity verification are unchanged.

Override a detected path before running the configuration cell when necessary:

```python
import os
os.environ["CS_ARCHIVE_DIR"] = "/content/drive/MyDrive/CardioSentry/experiments/synthetic_images_v1"
os.environ["CS_MASTER_METADATA_CSV"] = "/content/drive/MyDrive/CardioSentry/experiments/synthetic_images_v1/harmonized_ecg_metadata.csv"
```

## Storage and persistence

Archives remain in Drive. The approximately 119.5 GiB extracted corpus is
written to fast, ephemeral local storage by default:

```text
/content/cardiosentry_scratch/convnextv2_tiny_master_v1/data/images/
```

The notebook checks for roughly 152 GiB of free local disk before extraction.
Extracted images disappear when Colab replaces the VM and must then be extracted
again. Recovery state and final results persist directly in Drive:

```text
MyDrive/CardioSentry/experiments/convnextv2_tiny_master_v1/data/
MyDrive/CardioSentry/runs/convnextv2_tiny_master_v1/convnextv2_tiny_master_seed42_colab_l4/
MyDrive/CardioSentry/hf_cache/
```

`last.pt`, `best.pt`, and `training_history.csv` are written to Drive after each
epoch, so a later runtime can resume after re-extracting the images.

## L4-specific runtime behavior

- BF16 autocast is selected when supported by the assigned GPU; FP16 and a
  gradient scaler are used as the fallback.
- The largest safe micro-batch is measured using a real optimizer step at the
  full 768×1024 resolution. Gradient accumulation preserves effective batch 32.
- Channels-last, TF32, pinned memory, persistent workers, and local image reads
  remain enabled for throughput.
- The assigned GPU, VRAM, system RAM, CPU count, and local disk are checked at
  runtime. A non-L4 assignment produces a warning and still uses automatic batch
  sizing.

## Optional S3 copy

Google Drive is the default persistence layer, so S3 is disabled. To additionally
copy run artifacts to the existing bucket, configure AWS credentials in the
runtime and set this before the configuration cell:

```python
import os
os.environ["CS_S3_UPLOAD"] = "1"
```

The extracted image corpus is never uploaded to S3 by this notebook.
