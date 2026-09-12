"""Build the Google Colab L4/High-RAM copy of the master training notebook.

The EC2 notebook remains the source of truth for model/training/evaluation
logic. This builder changes only runtime, storage, persistence, and mixed-
precision details that differ on a managed Colab L4 runtime.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "CardioSentry_ConvNeXtV2_Tiny_Master_EC2.ipynb"
OUTPUT = HERE / "CardioSentry_ConvNeXtV2_Tiny_Master_Colab_L4_High_RAM.ipynb"


def cell_text(cell: dict) -> str:
    return "".join(cell["source"])


def set_cell_text(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(True)


def find_cell(notebook: dict, needle: str) -> dict:
    matches = [cell for cell in notebook["cells"] if needle in cell_text(cell)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cell containing {needle!r}; found {len(matches)}")
    return matches[0]


def replace_once(value: str, old: str, new: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one occurrence of {old!r}; found {count}")
    return value.replace(old, new, 1)


if not SOURCE.is_file():
    raise FileNotFoundError(
        f"Generate the EC2 source notebook first with {HERE / 'make_notebook.py'}"
    )

notebook = copy.deepcopy(json.loads(SOURCE.read_text()))

# Title and hardware-aware overview.
intro = find_cell(notebook, "# CardioSentry — ConvNeXt-V2 Tiny")
text = cell_text(intro)
text = replace_once(
    text,
    "AWS `g5.2xlarge` (8 vCPU, 32 GiB RAM, one NVIDIA A10G with about 22 GiB usable\n"
    "accelerator memory) and does not depend on Google Colab.",
    "Google Colab with an NVIDIA L4 and the High-RAM system profile. The runtime\n"
    "preflight checks the actual GPU, VRAM, RAM, CPU count, and ephemeral disk\n"
    "instead of assuming that Colab resources are fixed.",
)
text = replace_once(
    text,
    "| acceleration | FP16 AMP, TF32, channels-last, pinned memory, persistent workers, automatic largest-safe micro-batch probe |",
    "| acceleration | BF16 AMP on L4 (FP16 fallback), TF32, channels-last, pinned memory, persistent workers, automatic largest-safe micro-batch probe |",
)
text = replace_once(
    text,
    "The default CUDA configuration is seeded but performance-oriented, not bitwise\n",
    "The default Colab CUDA configuration is seeded but performance-oriented, not bitwise\n",
)
set_cell_text(intro, text)

# Replace the EC2 storage description with the actual generator/Drive layout.
data_cell = find_cell(notebook, "## 0. Data expected on EC2")
set_cell_text(data_cell, r"""## 0. Data expected in Google Drive

Before running, choose **Runtime → Change runtime type → L4 GPU**, then select
the **High-RAM** system profile. The next cell mounts Google Drive at
`/content/drive`.

The image-generation notebook stored the corpus archives under:

```text
MyDrive/CardioSentry/experiments/synthetic_images_v1/
├── images_train_00.tar ... images_train_05.tar
├── images_val.tar
├── images_test.tar
└── <one matching .sha256 file for every archive>
```

It originally stored `index.csv` separately under
`MyDrive/CardioSentry/runs/synthetic_images_v1/`. The configuration searches
that original location and the archive folder. If the index is unavailable,
place the exact v3 `harmonized_ecg_metadata.csv` in the archive folder or set
`CS_MASTER_METADATA_CSV` before running the configuration cell.

The fallback reconstruction still requires the exact metadata SHA-256 and
performs the same record/image/class/source, variant, split, target, and
bidirectional file-coverage audits as the EC2 notebook.

## Colab storage strategy

- Archives remain in Google Drive and are never copied back or uploaded again.
- Images are extracted to `/content/cardiosentry_scratch/` for substantially
  faster training than reading 128,608 images through the Drive mount.
- Reconstructed-index evidence, checkpoints, predictions, reports, and plots
  are written directly to the proper `experiments/` and `runs/` folders in
  Google Drive.
- Local extracted images are ephemeral. A new Colab VM must extract them again,
  while `last.pt` in Drive allows training to resume afterward.

The extraction preflight requires roughly 152 GiB of free local disk. Colab
resources vary; the notebook stops before extraction if the assigned VM is too
small. You may override `CS_EXTRACT_ROOT`, but using Drive for extracted images
will usually reduce training throughput substantially.

## Persistence

Google Drive is the default and sufficient persistence layer. AWS S3 upload is
disabled by default in this Colab copy because Colab has no EC2 instance role.
It remains available as an explicit opt-in by setting `CS_S3_UPLOAD=1` and valid
AWS credentials before the configuration cell. When enabled, the same bucket,
verification, retry, and upload-manifest behavior as the EC2 notebook is used.
""")

# Mount Drive before importing the training dependencies. Do not replace the
# Colab-provided CUDA/PyTorch build.
install_cell = find_cell(notebook, "# 1. Install only the Python packages")
set_cell_text(install_cell, r"""# 1. Mount Google Drive and install only the packages the harness needs.
from google.colab import drive
drive.mount("/content/drive", force_remount=False)

# Keep Colab's CUDA-compatible PyTorch build; do not pip-install torch here.
%pip install -q "timm>=1.0.15,<1.1" "scikit-learn>=1.4" "pandas>=2.1" \
    "pillow>=10" "matplotlib>=3.8" "seaborn>=0.13" "tqdm>=4.66" "psutil>=5.9" \
    "boto3>=1.34"
""")

config_cell = find_cell(notebook, "# 2. Configuration and EC2/GPU preflight")
text = cell_text(config_cell)
text = replace_once(
    text,
    "# 2. Configuration and EC2/GPU preflight — edit paths here if needed.",
    "# 2. Google Colab L4/High-RAM configuration and hardware preflight.",
)
old_paths = '''# Environment variables make the notebook portable without editing it.
PROJECT_ROOT = Path(os.environ.get(
    "CARDIOSENTRY_PROJECT_ROOT", str(Path.home() / "CardioSentry")
)).expanduser().resolve()
ARCHIVE_DIR = Path(os.environ.get(
    "CS_ARCHIVE_DIR", str(PROJECT_ROOT / "datasets" / "synthetic_images_v1")
)).expanduser().resolve()
PROVIDED_INDEX_CSV = Path(os.environ.get(
    "CS_INDEX_CSV", str(ARCHIVE_DIR / "index.csv")
)).expanduser().resolve()
MASTER_METADATA_CSV = Path(os.environ.get(
    "CS_MASTER_METADATA_CSV", str(ARCHIVE_DIR / "harmonized_ecg_metadata.csv")
)).expanduser().resolve()
EXTRACT_ROOT = Path(os.environ.get(
    "CS_EXTRACT_ROOT", str(PROJECT_ROOT / "experiments" / "convnextv2_tiny_master_v1" / "data")
)).expanduser().resolve()
RECONSTRUCTED_INDEX_CSV = Path(os.environ.get(
    "CS_RECONSTRUCTED_INDEX_CSV", str(EXTRACT_ROOT / "reconstructed_index.csv")
)).expanduser().resolve()
RECONSTRUCTION_REPORT = EXTRACT_ROOT / "index_reconstruction_report.json"
RUN_ROOT = Path(os.environ.get(
    "CS_RUN_ROOT", str(PROJECT_ROOT / "runs" / "convnextv2_tiny_master_v1")
)).expanduser().resolve()
HF_HOME = Path(os.environ.get(
    "HF_HOME", str(PROJECT_ROOT / "hf_cache")
)).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_HOME)

RUN_NAME = os.environ.get("CS_RUN_NAME", "convnextv2_tiny_master_seed42")
RUN_DIR = RUN_ROOT / RUN_NAME
for path in (EXTRACT_ROOT, RUN_DIR, HF_HOME):
    path.mkdir(parents=True, exist_ok=True)
'''
new_paths = '''# Drive holds durable inputs/results; /content is high-throughput scratch.
PROJECT_ROOT = Path(os.environ.get(
    "CARDIOSENTRY_PROJECT_ROOT", "/content/drive/MyDrive/CardioSentry"
)).expanduser().resolve()

def first_existing_path(environment_name, candidates):
    configured = os.environ.get(environment_name)
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()

ARCHIVE_DIR = first_existing_path("CS_ARCHIVE_DIR", [
    PROJECT_ROOT / "experiments" / "synthetic_images_v1",
    PROJECT_ROOT / "datasets" / "synthetic_images_v1",
])
PROVIDED_INDEX_CSV = first_existing_path("CS_INDEX_CSV", [
    PROJECT_ROOT / "runs" / "synthetic_images_v1" / "index.csv",
    ARCHIVE_DIR / "index.csv",
])
MASTER_METADATA_CSV = first_existing_path("CS_MASTER_METADATA_CSV", [
    ARCHIVE_DIR / "harmonized_ecg_metadata.csv",
    PROJECT_ROOT / "datasets" / "synthetic_images_v1" / "harmonized_ecg_metadata.csv",
    PROJECT_ROOT / "codes" / "dataset-combination-code" / "ecg_dataset_pipeline" / "data" / "releases" / "v3" / "harmonized_ecg_metadata.csv",
])
EXPERIMENT_DATA_DIR = (
    PROJECT_ROOT / "experiments" / "convnextv2_tiny_master_v1" / "data"
).resolve()
EXTRACT_ROOT = Path(os.environ.get(
    "CS_EXTRACT_ROOT", "/content/cardiosentry_scratch/convnextv2_tiny_master_v1/data"
)).expanduser().resolve()
RECONSTRUCTED_INDEX_CSV = Path(os.environ.get(
    "CS_RECONSTRUCTED_INDEX_CSV", str(EXPERIMENT_DATA_DIR / "reconstructed_index.csv")
)).expanduser().resolve()
RECONSTRUCTION_REPORT = EXPERIMENT_DATA_DIR / "index_reconstruction_report.json"
RUN_ROOT = Path(os.environ.get(
    "CS_RUN_ROOT", str(PROJECT_ROOT / "runs" / "convnextv2_tiny_master_v1")
)).expanduser().resolve()
HF_HOME = Path(os.environ.get(
    "HF_HOME", str(PROJECT_ROOT / "hf_cache")
)).expanduser().resolve()
os.environ["HF_HOME"] = str(HF_HOME)

RUN_NAME = os.environ.get("CS_RUN_NAME", "convnextv2_tiny_master_seed42_colab_l4")
RUN_DIR = RUN_ROOT / RUN_NAME
for path in (EXPERIMENT_DATA_DIR, EXTRACT_ROOT, RUN_DIR, HF_HOME):
    path.mkdir(parents=True, exist_ok=True)
'''
text = replace_once(text, old_paths, new_paths)
text = replace_once(
    text,
    "# Persistent S3 destinations. Disable only for an intentional local-only run by\n"
    "# exporting CS_S3_UPLOAD=0 before starting the notebook kernel.\n"
    "S3_UPLOAD_ENABLED = env_flag(\"CS_S3_UPLOAD\", \"1\")",
    "# Drive is persistent in Colab. S3 is optional and requires explicit credentials.\n"
    "S3_UPLOAD_ENABLED = env_flag(\"CS_S3_UPLOAD\", \"0\")",
)
text = replace_once(
    text,
    'print("extract root:      ", EXTRACT_ROOT)\n',
    'print("local extract root:", EXTRACT_ROOT)\nprint("experiment evidence:", EXPERIMENT_DATA_DIR)\n',
)
text = replace_once(
    text,
    'f"Cannot write and verify {s3_uri(preflight_key)}. Attach an EC2 IAM role "\n'
    '            "with s3:PutObject and s3:GetObject for this prefix, or set CS_S3_UPLOAD=0 "\n'
    '            "for an intentional local-only run."',
    'f"Cannot write and verify {s3_uri(preflight_key)}. Configure valid AWS credentials "\n'
    '            "with s3:PutObject and s3:GetObject for this prefix, or leave "\n'
    '            "CS_S3_UPLOAD=0 and use Google Drive persistence."',
)
text = replace_once(
    text,
    '"CUDA is unavailable. Use a GPU-enabled AWS Deep Learning AMI/kernel and "\n'
    '        "verify the NVIDIA driver before training."',
    '"CUDA is unavailable. In Colab choose Runtime > Change runtime type > L4 GPU, "\n'
    '        "then reconnect before running this cell."',
)
text = replace_once(
    text,
    'print(f"actual GPU VRAM:    {gib(gpu.total_memory):.1f} GiB")\n',
    'print(f"actual GPU VRAM:    {gib(gpu.total_memory):.1f} GiB")\n'
    'if "L4" not in gpu.name.upper():\n'
    '    warnings.warn(f"Expected an NVIDIA L4, but Colab assigned {gpu.name}. The automatic batch probe will adapt.")\n',
)
text = replace_once(
    text,
    'warnings.warn("Less than 28 GiB RAM is visible; check the EC2 instance configuration.")',
    'warnings.warn("Less than 28 GiB RAM is visible; verify that the Colab High-RAM profile is active.")',
)
text = replace_once(
    text,
    'DEVICE = torch.device("cuda:0")\n',
    'DEVICE = torch.device("cuda:0")\n'
    'AMP_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16\n'
    'USE_GRAD_SCALER = USE_AMP and AMP_DTYPE == torch.float16\n'
    'print("AMP dtype:          ", str(AMP_DTYPE).removeprefix("torch."))\n',
)
set_cell_text(config_cell, text)

audit_cell = find_cell(notebook, "the EC2 extraction root")
set_cell_text(
    audit_cell,
    replace_once(cell_text(audit_cell), "the EC2 extraction root", "the Colab local extraction root"),
)

probe_cell = find_cell(notebook, "nominal A10G memory")
text = replace_once(
    cell_text(probe_cell),
    "This is safer than assuming the\nnominal A10G memory and lets the same notebook use a larger GPU later.",
    "This measures the assigned L4 (or fallback GPU) directly instead of assuming\nthat Colab hardware and available VRAM are fixed.",
)
set_cell_text(probe_cell, text)

amp_cell = find_cell(notebook, "def make_grad_scaler")
text = cell_text(amp_cell)
text = replace_once(
    text,
    'return torch.amp.GradScaler("cuda", enabled=USE_AMP)',
    'return torch.amp.GradScaler("cuda", enabled=USE_GRAD_SCALER)',
)
text = replace_once(
    text,
    "return torch.cuda.amp.GradScaler(enabled=USE_AMP)",
    "return torch.cuda.amp.GradScaler(enabled=USE_GRAD_SCALER)",
)
text = replace_once(text, "dtype=torch.float16", "dtype=AMP_DTYPE")
set_cell_text(amp_cell, text)

training_cell = find_cell(notebook, '"amp": "FP16"')
set_cell_text(
    training_cell,
    replace_once(
        cell_text(training_cell),
        '"amp": "FP16",',
        '"amp": str(AMP_DTYPE).removeprefix("torch."),',
    ),
)

outputs_cell = find_cell(notebook, "## Persistent outputs")
text = cell_text(outputs_cell)
text = text.replace(
    "<CS_RUN_ROOT>/convnextv2_tiny_master_seed42/",
    "/content/drive/MyDrive/CardioSentry/runs/convnextv2_tiny_master_v1/convnextv2_tiny_master_seed42_colab_l4/",
)
text = replace_once(
    text,
    "With the default configuration, the complete directory above is also persisted\n"
    "to:\n\n"
    "```text\n"
    "s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/runs/convnextv2_tiny_master_v1/convnextv2_tiny_master_seed42/\n"
    "```",
    "The complete directory above is written directly to Google Drive. S3 is not\n"
    "used unless `CS_S3_UPLOAD=1` is explicitly configured. If enabled, the default\n"
    "optional destination is:\n\n"
    "```text\n"
    "s3://cardiosentry-fcmae-758821198776-us-east-1-an/CardioSentry/runs/convnextv2_tiny_master_v1/convnextv2_tiny_master_seed42_colab_l4/\n"
    "```",
)
set_cell_text(outputs_cell, text)

notebook.setdefault("metadata", {})["accelerator"] = "GPU"
notebook["metadata"]["colab"] = {
    "gpuType": "L4",
    "provenance": [],
}

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

OUTPUT.write_text(json.dumps(notebook, indent=1))
print(f"wrote {OUTPUT} ({len(notebook['cells'])} cells)")
