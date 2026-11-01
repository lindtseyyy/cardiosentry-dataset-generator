"""Build the CardioSentry ECG-gate Colab notebook (ConvNeXt-V1 Tiny + RFPN clean12, frozen, + gate head)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CardioSentry_ConvNeXtV1_Tiny_RFPN_ECG_Gate_Colab.ipynb"
# Single source of truth for the fusion model: the file the base run was trained with.
FUSION_SOURCE = (HERE.parent / "convnext_tiny_v1_rfpn_clean12_colab_v1" / "fusion_necks.py").read_text()


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.strip("\n").splitlines(True)}


cells = [
    md(r"""
# CardioSentry — ECG / not-ECG gate on the ConvNeXt-V1 Tiny + RFPN clean12 model

Adds a binary **gate head** ("is this an ECG sheet?") to the fine-tuned
`convnext_tiny_v1_rfpn_clean12_colab_a100_seed42` model, so one forward pass
gives both the gate score and the 12 diagnostic labels.

**What is trained:** only the gate head (LayerNorm → Linear 1536→256 → GELU →
Dropout → Linear 256→1) on features the fine-tuned model already computes:
the ConvNeXt head's pooled, normalised stage-4 vector (768) and the RFPN's
pooled neck levels (3 × 256). The backbone, neck and both classifier heads are
**frozen**, so the 12 diagnostic logits are bit-identical to the original
model; section 4 checks this and section 8 re-checks it after training.

**Run:** Colab, A100 (any GPU works; the backbone is only run forward),
Runtime → Run all. Upload only this `.ipynb`.

**Drive (shared folder, as in the clean12 notebooks):** sign in as the account
the `CardioSentry` folder is shared with; it needs a shortcut to the folder in
its My Drive (Shared with me → right-click → Organize → Add shortcut → My
Drive). Inputs are read in place, nothing has to be uploaded besides the
dataset tar:

| input | path inside the shared folder |
|---|---|
| gate dataset | `new_dataset/ecg_gate_v1.tar` + `ecg_gate_v1.tar.sha256` |
| fine-tuned model | `new_runs/convnext_tiny_v1_rfpn_clean12_colab_a100_seed42/convnext_tiny_rfpn/inference.pt` (SHA-256 pinned) |

Outputs go to `new_runs/convnext_tiny_v1_rfpn_clean12_ecg_gate_v1_seed42/`:
`inference_with_gate.pt` (the original export, `model` untouched, plus
`gate_head` and its threshold/metadata), `gate_head.pt`, metrics, per-image
predictions, training history, an error grid and `report.md`.

**Input geometry.** Every image becomes the 768×1024 sheet the classifier
expects, by one of two app-like paths (training picks one at random, every
evaluation reports both):

- `fit`: aspect-preserving fit + paper-colour pad (the app's `fit_height_pad`);
- `rectify`: stretched to the rectify canvas (1650×1275, what the app's
  rectify does to a full-frame quad), 18 px margin cropped, then `fit`.

**Threshold.** Chosen on validation (FP32, both geometries pooled) as the
highest score that still accepts ≥ `CS_GATE_TARGET_RECALL` (default 0.99) of
real ECGs: wrongly rejecting an ECG is the costlier gate error.

Overrides (environment variables, set in the first code cell if needed):
`CS_SHARED_FOLDER_ID`, `CS_SHARED_FOLDER_NAME`, `CS_DRIVE_PROJECT_ROOT`,
`CS_DATASET_DRIVE_DIR`, `CS_GATE_TAR_NAME`, `CS_BASE_RUN_DIR`, `CS_RUN_GROUP`,
`CS_EPOCHS`, `CS_BATCH_SIZE`, `CS_LR`, `CS_WEIGHT_DECAY`, `CS_SEED`,
`CS_GATE_TARGET_RECALL`, `CS_NUM_WORKERS`, `CS_OVERWRITE`, `CS_RELEASE_RUNTIME`.
"""),
    md("## 1. Setup"),
    code(r"""
import copy, csv, hashlib, io, json, math, os, platform, random, shutil, tarfile, time, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

def env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off", ""}

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

SEED = int(os.environ.get("CS_SEED", "42"))
EPOCHS = int(os.environ.get("CS_EPOCHS", "12"))
BATCH_SIZE = int(os.environ.get("CS_BATCH_SIZE", "32"))
LR = float(os.environ.get("CS_LR", "1e-3"))
WEIGHT_DECAY = float(os.environ.get("CS_WEIGHT_DECAY", "0.05"))
WARMUP_EPOCHS = 1
TARGET_RECALL = float(os.environ.get("CS_GATE_TARGET_RECALL", "0.99"))
NUM_WORKERS = int(os.environ.get("CS_NUM_WORKERS", str(max(1, min(11, (os.cpu_count() or 2) - 1)))))
OVERWRITE = env_flag("CS_OVERWRITE", "0")
RELEASE_RUNTIME_WHEN_DONE = env_flag("CS_RELEASE_RUNTIME", "1")
TIMM_VERSION_EXPECTED = "1.0.28"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else None
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
if timm.__version__ != TIMM_VERSION_EXPECTED:
    warnings.warn(f"timm {timm.__version__} != {TIMM_VERSION_EXPECTED} (the base run's version)")
print(f"device {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu'}), "
      f"AMP {AMP_DTYPE}, torch {torch.__version__}, timm {timm.__version__}, workers {NUM_WORKERS}")
"""),
    md(r"""
## 2. Google Drive: shared folder, inputs, run folder

Same shared-folder lookup as the clean12 training notebooks
(`/content/drive/.shortcut-targets-by-id/<id>/<name>` first, so the shortcut's
name and any same-named folder of this account do not matter), then a small
read/write probe of the run folder.
"""),
    code(r"""
DRIVE_MOUNT_POINT = Path("/content/drive")
if IN_COLAB and not (DRIVE_MOUNT_POINT / "MyDrive").is_dir():
    from google.colab import drive
    drive.mount(str(DRIVE_MOUNT_POINT))  # sign in as the account the folder is shared with
SHARED_FOLDER_NAME = os.environ.get("CS_SHARED_FOLDER_NAME", "CardioSentry").strip()
SHARED_FOLDER_ID = os.environ.get("CS_SHARED_FOLDER_ID", "").strip()

def shortcut_target_id(path, mount_point):
    '''The shared folder's Drive id when `path` resolves into .shortcut-targets-by-id, else "".'''
    targets = Path(mount_point) / ".shortcut-targets-by-id"
    try:
        return Path(path).resolve().relative_to(targets.resolve()).parts[0]
    except (ValueError, IndexError, OSError):
        return ""

def resolve_shared_folder(mount_point, folder_name, folder_id="", explicit_root=""):
    '''(mounted path, Drive folder id or "", how it was found) of the shared folder.'''
    mount_point = Path(mount_point)
    if explicit_root:
        root = Path(explicit_root).expanduser()
        if not root.is_dir():
            raise RuntimeError(f"CS_DRIVE_PROJECT_ROOT={root} is not a folder on the mount.")
        return root, folder_id or shortcut_target_id(root, mount_point), "CS_DRIVE_PROJECT_ROOT"
    my_drive = mount_point / "MyDrive"
    targets = mount_point / ".shortcut-targets-by-id"
    shortcut = my_drive / folder_name
    shortcut_exists = shortcut.is_dir()  # also makes the mount list the shortcut's target
    help_text = (
        f"This Colab account needs a shortcut to the shared folder in its My Drive: drive.google.com -> "
        f"Shared with me -> right-click '{folder_name}' -> Organize -> Add shortcut -> My Drive. Then re-run "
        "this cell. Set CS_SHARED_FOLDER_NAME if the folder has another name, CS_SHARED_FOLDER_ID (the id "
        "in its URL) to pick it unambiguously, or CS_DRIVE_PROJECT_ROOT to a mounted path.")
    if folder_id:
        by_id = targets / folder_id
        found = sorted(p for p in by_id.iterdir() if p.is_dir()) if by_id.is_dir() else []
        if len(found) == 1:
            return found[0], folder_id, f"shortcut target {folder_id}"
        if shortcut_exists and shortcut_target_id(shortcut, mount_point) == folder_id:
            return shortcut, folder_id, "shortcut in My Drive"
        raise RuntimeError(f"No shortcut target for folder id {folder_id} under {targets}. " + help_text)
    if shortcut_exists and shortcut_target_id(shortcut, mount_point):
        return shortcut, shortcut_target_id(shortcut, mount_point), "shortcut in My Drive"
    by_name = sorted(p for p in targets.glob(f"*/{folder_name}") if p.is_dir()) if targets.is_dir() else []
    if len(by_name) == 1:
        if shortcut_exists:
            warnings.warn(f"{shortcut} is this account's own folder, not the shared one; "
                          f"using the shared folder's shortcut target {by_name[0]}.")
        return by_name[0], shortcut_target_id(by_name[0], mount_point), "shortcut target (by name)"
    if len(by_name) > 1:
        raise RuntimeError(f"{len(by_name)} shortcut targets are named {folder_name!r}: "
                           f"{[str(p) for p in by_name]}. Set CS_SHARED_FOLDER_ID.")
    if shortcut_exists:
        warnings.warn(f"{shortcut} is not recognisably a shortcut; using it as the shared folder.")
        return shortcut, "", "My Drive folder"
    raise RuntimeError(f"Shared folder {folder_name!r} not found under {my_drive}. " + help_text)

CARDIOSENTRY_DRIVE_ROOT, SHARED_FOLDER_DRIVE_ID, _how = resolve_shared_folder(
    DRIVE_MOUNT_POINT, SHARED_FOLDER_NAME, SHARED_FOLDER_ID, os.environ.get("CS_DRIVE_PROJECT_ROOT", "").strip())
print(f"shared Drive folder: {CARDIOSENTRY_DRIVE_ROOT} ({_how})")

DATASET_DRIVE_DIR = Path(os.environ.get("CS_DATASET_DRIVE_DIR", str(CARDIOSENTRY_DRIVE_ROOT / "new_dataset")))
GATE_TAR_NAME = os.environ.get("CS_GATE_TAR_NAME", "ecg_gate_v1.tar")
DATASET_TAR = DATASET_DRIVE_DIR / GATE_TAR_NAME
DATASET_SHA_FILE = DATASET_DRIVE_DIR / (GATE_TAR_NAME + ".sha256")
BASE_RUN_DIR = Path(os.environ.get("CS_BASE_RUN_DIR", str(
    CARDIOSENTRY_DRIVE_ROOT / "new_runs" / "convnext_tiny_v1_rfpn_clean12_colab_a100_seed42" / "convnext_tiny_rfpn")))
BASE_INFERENCE = BASE_RUN_DIR / "inference.pt"
# sha256 of the run's inference.pt, from its drive_sync_manifest.json (fingerprint 177e77611da26805).
BASE_INFERENCE_SHA256 = os.environ.get(
    "CS_BASE_INFERENCE_SHA256", "ab1b6bf5c62c25930d01a9e670138ad1c17a58e52cbd6ed5cdfb6e48701fc2d5")
RUN_GROUP = os.environ.get("CS_RUN_GROUP", "convnext_tiny_v1_rfpn_clean12_ecg_gate_v1_seed42")
DRIVE_RUN_DIR = CARDIOSENTRY_DRIVE_ROOT / "new_runs" / RUN_GROUP
SCRATCH = Path(os.environ.get("CS_SCRATCH_DIR", "/content/cs_scratch"))
RUN_DIR = SCRATCH / "runs" / RUN_GROUP
DATA_DIR = SCRATCH / "ecg_gate"

for path, what in ((DATASET_TAR, "gate dataset tar"), (DATASET_SHA_FILE, "its .sha256"),
                   (BASE_INFERENCE, "fine-tuned model inference.pt")):
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
if (DRIVE_RUN_DIR / "gate_complete.json").is_file() and not OVERWRITE:
    raise RuntimeError(f"{DRIVE_RUN_DIR} already holds a finished gate run. Set CS_OVERWRITE=1 to replace it, "
                       "or CS_RUN_GROUP to a new name.")

def probe_write(folder):
    '''Write/rename/read/delete a 1 MiB file in `folder` (proves Editor access through the mount).'''
    folder.mkdir(parents=True, exist_ok=True)
    payload = os.urandom(2**20)
    probe = folder / f".cs_access_test_{platform.node()}_{os.getpid()}"
    partial = probe.with_name(probe.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, probe)
    ok = probe.read_bytes() == payload
    probe.unlink()
    if not ok:
        raise RuntimeError(f"read-back of {probe} differs from what was written")

try:
    probe_write(DRIVE_RUN_DIR)
except OSError as error:
    raise RuntimeError(f"Cannot write to {DRIVE_RUN_DIR} ({error}); the account needs Editor access.") from error
if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
print("dataset:", DATASET_TAR, f"({DATASET_TAR.stat().st_size / 2**30:.2f} GiB)")
print("base model:", BASE_INFERENCE)
print("outputs:", DRIVE_RUN_DIR)
"""),
    md(r"""
## 3. Stage the gate dataset

Copies the tar from Drive to local disk, checks its SHA-256 against the
sidecar, extracts it and checks the manifest (every image present, labels and
splits as expected, no group in two splits).
"""),
    code(r"""
def sha256_file(path, chunk=16 * 2**20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()

expected_tar_sha = DATASET_SHA_FILE.read_text().split()[0].strip().lower()
local_tar = SCRATCH / GATE_TAR_NAME
SCRATCH.mkdir(parents=True, exist_ok=True)
if not (local_tar.is_file() and local_tar.stat().st_size == DATASET_TAR.stat().st_size
        and sha256_file(local_tar) == expected_tar_sha):
    started = time.time()
    shutil.copyfile(DATASET_TAR, local_tar)
    print(f"copied {local_tar.stat().st_size / 2**30:.2f} GiB from Drive in {time.time() - started:.0f}s")
    actual = sha256_file(local_tar)
    if actual != expected_tar_sha:
        raise RuntimeError(f"{GATE_TAR_NAME} SHA-256 {actual} != sidecar {expected_tar_sha}; re-upload the tar.")
DATASET_SHA256 = expected_tar_sha
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir(parents=True)
with tarfile.open(local_tar) as tf:
    tf.extractall(DATA_DIR, filter="data")
manifest_sha = sha256_file(DATA_DIR / "manifest.csv")
if manifest_sha != (DATA_DIR / "manifest.csv.sha256").read_text().split()[0]:
    raise RuntimeError("manifest.csv does not match manifest.csv.sha256")

manifest = pd.read_csv(DATA_DIR / "manifest.csv")
missing = [p for p in manifest["path"] if not (DATA_DIR / p).is_file()]
if missing:
    raise RuntimeError(f"{len(missing)} manifest images missing, e.g. {missing[:3]}")
if set(manifest["label"]) != {"ecg", "not_ecg"} or set(manifest["split"]) != {"train", "validation", "test"}:
    raise RuntimeError(f"unexpected labels/splits: {set(manifest['label'])} {set(manifest['split'])}")
leaky = manifest.groupby("group_id")["split"].nunique()
if (leaky > 1).any():
    raise RuntimeError(f"groups in more than one split: {list(leaky[leaky > 1].index[:5])}")
manifest["y"] = (manifest["label"] == "ecg").astype(np.float32)
max_images = int(os.environ.get("CS_MAX_IMAGES_PER_SPLIT", "0"))  # smoke tests only
if max_images:
    manifest = (manifest.groupby(["split", "label"], group_keys=False)
                .apply(lambda d: d.sample(min(len(d), max_images), random_state=SEED)).reset_index(drop=True))
    warnings.warn(f"CS_MAX_IMAGES_PER_SPLIT={max_images}: using a {len(manifest)}-image subset")
print(pd.crosstab([manifest["label"], manifest["source"]], manifest["split"], margins=True))
"""),
    md(r"""
## 4. Model: the fine-tuned ConvNeXt-V1 Tiny + RFPN, frozen, plus the gate head

`fusion_necks.py` below is the file the base run was trained with (inlined by
`make_notebook.py`). The export is loaded with `strict=True`; then the
wrapper's 12-label logits are checked to be **exactly equal** to the original
model's on real images.
"""),
    code(FUSION_SOURCE),
    code(r"""
base_ckpt = torch.load(BASE_INFERENCE, map_location="cpu", weights_only=False)
base_sha = sha256_file(BASE_INFERENCE)
if base_sha != BASE_INFERENCE_SHA256:
    raise RuntimeError(f"{BASE_INFERENCE} SHA-256 {base_sha} != pinned {BASE_INFERENCE_SHA256}")
for key, expected in (("fusion_neck", "fpn"), ("fusion_head", "residual"), ("fusion_channels", 256)):
    if base_ckpt[key] != expected:
        raise RuntimeError(f"inference.pt {key}={base_ckpt[key]!r}, expected {expected!r}")
if tuple(base_ckpt["fusion_stage_indices"]) != FUSION_STAGE_INDICES:
    raise RuntimeError("fusion stage indices differ from fusion_necks.py")
LABELS = list(base_ckpt["label_order"])
INPUT_HW = tuple(base_ckpt["input_size_hw"])
MEAN = torch.tensor(base_ckpt["normalization_mean"]).view(1, 3, 1, 1)
STD = torch.tensor(base_ckpt["normalization_std"]).view(1, 3, 1, 1)
if INPUT_HW != (768, 1024):
    raise RuntimeError(f"unexpected input size {INPUT_HW}")

def build_base_model(ckpt):
    backbone = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(ckpt["label_order"]))
    model = ConvNeXtMultiScaleFusion(backbone, ckpt["fusion_neck"], ckpt["fusion_head"],
                                     len(ckpt["label_order"]), channels=ckpt["fusion_channels"])
    model.load_state_dict(ckpt["model"], strict=True)
    return model.eval()

GATE_FEATURE_SPEC = ("concat[backbone.forward_head(norm_pre(stage4), pre_logits=True) (768, LayerNorm'd GAP), "
                     "GAP(FPN P3,P4,P5) (3x256, fp32)] = 1536")
GATE_HIDDEN, GATE_DROPOUT = 256, 0.2

class GateHead(nn.Module):
    def __init__(self, in_features=1536, hidden=GATE_HIDDEN, dropout=GATE_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, features):
        return self.fc2(self.drop(F.gelu(self.fc1(self.norm(features))))).squeeze(1)

class GatedRFPN(nn.Module):
    '''The frozen fine-tuned model plus a gate head; forward -> (12 diagnostic logits, gate logit).'''
    def __init__(self, base, gate):
        super().__init__()
        self.base = base
        self.gate = gate

    def forward_features(self, x):
        '''Exactly ConvNeXtMultiScaleFusion.forward for the logits, plus the gate's input features.'''
        base, backbone = self.base, self.base.backbone
        x = backbone.stem(x)
        stages = []
        for stage in backbone.stages:
            x = stage(x)
            stages.append(x)
        maps = base.neck([stages[i] for i in FUSION_STAGE_INDICES])
        pooled = torch.cat([torch.mean(m, dim=(2, 3), dtype=torch.float32) for m in maps], dim=1)
        fused_logits = base.fuse_fc(base.fuse_norm(pooled))
        x = backbone.norm_pre(x)
        logits = backbone.forward_head(x) + fused_logits
        pre_logits = backbone.forward_head(x, pre_logits=True)
        return logits, torch.cat([pre_logits.float(), pooled.float()], dim=1)

    def forward(self, x):
        logits, features = self.forward_features(x)
        return logits, self.gate(features.float())

def state_sha(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

base_model = build_base_model(base_ckpt)
for p in base_model.parameters():
    p.requires_grad_(False)
torch.manual_seed(SEED)
gate_head = GateHead()
model = GatedRFPN(base_model, gate_head).to(DEVICE)
if DEVICE.type == "cuda":
    model.base.to(memory_format=torch.channels_last)
BASE_STATE_SHA = state_sha(model.base)
print(f"base: {sum(p.numel() for p in model.base.parameters()) / 1e6:.2f} M params (frozen); "
      f"gate head: {sum(p.numel() for p in gate_head.parameters()) / 1e3:.1f} k params (trained)")
"""),
    md(r"""
## 5. Input pipeline

Both geometries end in the 768×1024 RGB sheet the classifier was trained on.
Train-time augmentation is applied to both classes alike (crop, small
rotation, colour, greyscale, blur, JPEG), so none of them marks a class.
Batches stay `uint8` until they reach the GPU, where they are normalised.
"""),
    code(r"""
RECTIFY_CANVAS_WH = (1650, 1275)   # the app's rectify canvas for this model family
RECTIFY_MARGIN_PX = 18             # cropped by the app before fit_height_pad
GEOMETRIES = ("fit", "rectify")

def paper_colour_p90(img):
    '''The app's / clean12 render worker's pad colour: p90 of the four corner patches.'''
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    k = max(8, min(h, w) // 40)
    patches = np.concatenate([arr[:k, :k].reshape(-1, 3), arr[:k, -k:].reshape(-1, 3),
                              arr[-k:, :k].reshape(-1, 3), arr[-k:, -k:].reshape(-1, 3)])
    return tuple(int(v) for v in np.percentile(patches, 90, axis=0))

def fit_height_pad(img, hw=INPUT_HW):
    '''The app's fit_height_pad: aspect-preserving LANCZOS fit into (h, w) + centred paper-colour pad.'''
    h, w = hw
    scale = min(h / img.height, w / img.width)
    new_w, new_h = min(w, round(img.width * scale)), min(h, round(img.height * scale))
    small = img.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    if (new_w, new_h) == (w, h):
        return small
    canvas = Image.new("RGB", (w, h), paper_colour_p90(small))
    canvas.paste(small, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas

def to_model_sheet(img, geometry):
    img = img.convert("RGB")
    if geometry == "rectify":
        m = RECTIFY_MARGIN_PX
        img = img.resize(RECTIFY_CANVAS_WH, Image.Resampling.BILINEAR)
        img = img.crop((m, m, img.width - m, img.height - m))
    elif geometry != "fit":
        raise ValueError(geometry)
    return fit_height_pad(img)

def jpeg_roundtrip(img, quality):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

COLOUR = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.02)

def augment_source(img, rng):
    if rng.random() < 0.5:   # crop 80-100% of the area, aspect +-10%
        area = rng.uniform(0.8, 1.0)
        aspect = math.exp(rng.uniform(math.log(0.9), math.log(1.1)))
        cw = min(img.width, round(img.width * math.sqrt(area * aspect)))
        ch = min(img.height, round(img.height * math.sqrt(area / aspect)))
        x0, y0 = rng.randint(0, img.width - cw), rng.randint(0, img.height - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-4, 4), resample=Image.Resampling.BILINEAR, expand=False,
                         fillcolor=paper_colour_p90(img))
    return img

def augment_sheet(img, rng):
    if rng.random() < 0.8:
        img = COLOUR(img)
    if rng.random() < 0.2:
        img = TF.to_grayscale(img, num_output_channels=3)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.5)))
    if rng.random() < 0.3:
        img = jpeg_roundtrip(img, rng.randint(50, 95))
    return img

class GateDataset(Dataset):
    def __init__(self, frame, geometry=None, train=False):
        self.paths = [str(DATA_DIR / p) for p in frame["path"]]
        self.y = frame["y"].to_numpy(np.float32, copy=True)
        self.geometry, self.train = geometry, train
        self.epoch = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = random.Random(hash((SEED, self.epoch, i)) if self.train else 0)
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        geometry = self.geometry or rng.choice(GEOMETRIES)
        if self.train:
            img = augment_source(img, rng)
        sheet = to_model_sheet(img, geometry)
        if self.train:
            sheet = augment_sheet(sheet, rng)
        if sheet.size != (INPUT_HW[1], INPUT_HW[0]):
            raise RuntimeError(f"sheet {sheet.size} != {INPUT_HW[::-1]}")
        return TF.pil_to_tensor(sheet), torch.tensor(self.y[i]), i

def loader(dataset, shuffle):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
                      pin_memory=DEVICE.type == "cuda", persistent_workers=NUM_WORKERS > 0 and shuffle,
                      generator=torch.Generator().manual_seed(SEED) if shuffle else None,
                      prefetch_factor=4 if NUM_WORKERS > 0 else None)

def to_input(batch_uint8):
    x = batch_uint8.to(DEVICE, non_blocking=True).float().div_(255)
    x = (x - MEAN.to(DEVICE)) / STD.to(DEVICE)
    return x.contiguous(memory_format=torch.channels_last) if DEVICE.type == "cuda" else x

frames = {split: manifest[manifest["split"] == split].reset_index(drop=True)
          for split in ("train", "validation", "test")}
train_set = GateDataset(frames["train"], train=True)
train_loader = loader(train_set, shuffle=True)
print({k: len(v) for k, v in frames.items()})
"""),
    code(r"""
# Parity: the wrapper's 12 logits are the original model's, bit for bit, on real images.
_probe = GateDataset(frames["validation"].groupby("label").head(2).reset_index(drop=True), geometry="fit")
_x = to_input(torch.stack([_probe[i][0] for i in range(len(_probe))]))
with torch.no_grad():
    _ref = model.base(_x)
    _logits, _ = model(_x)
if not torch.equal(_ref, _logits):
    raise RuntimeError(f"wrapper logits differ from the base model: max |d| = {(_ref - _logits).abs().max():.3g}")
print(f"parity OK: wrapper logits == base model logits on {len(_probe)} validation images (exact)")
del _probe, _x, _ref, _logits
"""),
    md(r"""
## 6. Train the gate head

The frozen model runs under `torch.no_grad` (BF16 autocast on the GPU);
only the head gets gradients. Validation features are deterministic, so they
are extracted once per geometry and reused every epoch. The epoch with the
lowest validation loss (both geometries pooled) is kept.
"""),
    code(r"""
def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP_DTYPE is not None else torch.autocast("cpu", enabled=False)

@torch.no_grad()
def extract(frame, geometry, amp=True):
    '''(gate features [N,1536] fp32 on CPU, 12-label logits [N,12], labels [N]) in manifest order.'''
    feats, logits, ys = [None] * len(frame), [None] * len(frame), frame["y"].to_numpy(np.float32, copy=True)
    model.eval()
    for xb, _yb, idx in loader(GateDataset(frame, geometry=geometry), shuffle=False):
        x = to_input(xb)
        if amp:
            with autocast():
                lg, ft = model.forward_features(x)
        else:
            lg, ft = model.forward_features(x)
        for j, i in enumerate(idx.tolist()):
            feats[i], logits[i] = ft[j].float().cpu(), lg[j].float().cpu()
    return torch.stack(feats), torch.stack(logits), torch.from_numpy(ys)

started = time.time()
val_cache = {g: extract(frames["validation"], g) for g in GEOMETRIES}
print(f"validation features cached for {GEOMETRIES} in {time.time() - started:.0f}s")

def head_eval(feats, ys):
    gate_head.eval()
    with torch.no_grad():
        logit = gate_head(feats.to(DEVICE)).float().cpu()
    loss = F.binary_cross_entropy_with_logits(logit, ys).item()
    prob = torch.sigmoid(logit).numpy()
    return loss, prob

optimizer = torch.optim.AdamW(gate_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = len(train_loader)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warmup_steps if s < warmup_steps
                                              else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))))
history, best = [], {"val_loss": float("inf")}
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_set.epoch = epoch
    model.base.eval()
    gate_head.train()
    losses, correct, seen = [], 0, 0
    for xb, yb, _ in train_loader:
        x, y = to_input(xb), yb.to(DEVICE)
        with torch.no_grad(), autocast():
            _, feats = model.forward_features(x)
        logit = gate_head(feats.float())
        loss = F.binary_cross_entropy_with_logits(logit, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        correct += ((logit > 0).float() == y).sum().item()
        seen += len(y)
    row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_acc": correct / seen,
           "lr": scheduler.get_last_lr()[0]}
    val_losses = []
    for g in GEOMETRIES:
        feats, _, ys = val_cache[g]
        loss, prob = head_eval(feats, ys)
        val_losses.append(loss)
        row[f"val_loss_{g}"] = loss
        row[f"val_auroc_{g}"] = roc_auc_score(ys.numpy(), prob) if len(set(ys.tolist())) == 2 else float("nan")
        row[f"val_acc_{g}"] = float(((prob >= 0.5) == ys.numpy().astype(bool)).mean())
    row["val_loss"] = float(np.mean(val_losses))
    row["epoch_seconds"] = round(time.time() - t0, 1)
    history.append(row)
    if row["val_loss"] < best["val_loss"]:
        best = {"val_loss": row["val_loss"], "epoch": epoch, "state": copy.deepcopy(gate_head.state_dict())}
    print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

gate_head.load_state_dict(best["state"])
gate_head.eval()
pd.DataFrame(history).to_csv(RUN_DIR / "training_history.csv", index=False)
if state_sha(model.base) != BASE_STATE_SHA:
    raise RuntimeError("the frozen base model changed during training")
print(f"best epoch {best['epoch']} (val loss {best['val_loss']:.4f}); base weights unchanged")
"""),
    md(r"""
## 7. Evaluate (FP32, as the CPU app runs it) and choose the threshold

Validation sets the threshold; test is reported once, per geometry, per
source and per subsource. For an ECG source the acceptance rate is its recall;
for a non-ECG source it is the false-accept rate.
"""),
    code(r"""
def score(frame, geometry):
    feats, logits12, ys = extract(frame, geometry, amp=False)
    _, prob = head_eval(feats, ys)
    return prob, logits12, ys.numpy()

scores = {(s, g): score(frames[s], g) for s in ("validation", "test") for g in GEOMETRIES}

val_pos = np.concatenate([scores[("validation", g)][0][scores[("validation", g)][2] == 1] for g in GEOMETRIES])
k = int(math.floor((1 - TARGET_RECALL) * len(val_pos)))
THRESHOLD = float(np.sort(val_pos)[k])
print(f"threshold {THRESHOLD:.4f}: accepts >= {TARGET_RECALL:.0%} of validation ECGs (both geometries pooled)")

def summarise(prob, ys, threshold):
    pred = prob >= threshold
    pos, neg = ys == 1, ys == 0
    return {"n": int(len(ys)), "auroc": float(roc_auc_score(ys, prob)), "ap": float(average_precision_score(ys, prob)),
            "ecg_recall": float(pred[pos].mean()), "false_accept_rate": float(pred[neg].mean()),
            "accuracy": float((pred == pos).mean()),
            "false_rejects": int((~pred & pos).sum()), "false_accepts": int((pred & neg).sum())}

metrics = {"threshold": THRESHOLD, "target_recall": TARGET_RECALL, "best_epoch": best["epoch"], "splits": {}}
rows = []
for (split, g), (prob, logits12, ys) in scores.items():
    metrics["splits"][f"{split}/{g}"] = {"at_threshold": summarise(prob, ys, THRESHOLD),
                                         "at_0.5": summarise(prob, ys, 0.5)}
    frame = frames[split]
    for i in range(len(frame)):
        rows.append({"path": frame.at[i, "path"], "split": split, "geometry": g, "label": frame.at[i, "label"],
                     "source": frame.at[i, "source"], "subsource": frame.at[i, "subsource"],
                     "gate_prob": float(prob[i]), "accepted": bool(prob[i] >= THRESHOLD),
                     **{f"prob_{lab}": float(torch.sigmoid(logits12[i, j])) for j, lab in enumerate(LABELS)}})
pred_df = pd.DataFrame(rows)
pred_df.to_csv(RUN_DIR / "gate_predictions.csv", index=False)

overview = pd.DataFrame({k: v["at_threshold"] for k, v in metrics["splits"].items()}).T
print(overview.round(4))
test_df = pred_df[pred_df["split"] == "test"]
per_source = (test_df.groupby(["label", "source", "geometry"])["accepted"].agg(["count", "mean"])
              .rename(columns={"mean": "accept_rate"}).unstack("geometry"))
per_sub = (test_df.groupby(["label", "source", "subsource", "geometry"])["accepted"].mean()
           .unstack("geometry").rename(columns=lambda g: f"accept_{g}"))
per_source.to_csv(RUN_DIR / "test_per_source.csv")
per_sub.to_csv(RUN_DIR / "test_per_subsource.csv")
metrics["test_per_source_accept_rate"] = {f"{l}/{s}/{g}": float(v) for (l, s, g), v in
                                          test_df.groupby(["label", "source", "geometry"])["accepted"].mean().items()}
print(per_source.round(3))
"""),
    code(r"""
import matplotlib.pyplot as plt

errors = test_df[(test_df["accepted"]) != (test_df["label"] == "ecg")].sort_values("gate_prob")
shown = errors.drop_duplicates("path").head(24)
if len(shown):
    cols = 6
    rows_n = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 2.6))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), shown.iterrows()):
        with Image.open(DATA_DIR / r["path"]) as im:
            ax.imshow(to_model_sheet(im, r["geometry"]))
        kind = "false reject" if r["label"] == "ecg" else "false accept"
        ax.set_title(f"{kind}\n{r['source']}/{r['subsource']}\n{r['geometry']} p={r['gate_prob']:.3f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(RUN_DIR / "test_errors.png", dpi=110)
    plt.show()
print(f"test errors at threshold: {len(errors)} rows ({errors['path'].nunique()} images) "
      f"over {len(test_df)} image x geometry rows")
"""),
    md(r"""
## 8. Export and verify

`inference_with_gate.pt` is the original `inference.pt` with its `model`
state untouched (the app's strict loader keeps working) plus `gate_head` and
its metadata. It is reloaded from disk into a fresh model and checked against
the in-memory one before anything is copied to Drive.
"""),
    code(r"""
gate_meta = {
    "gate_head_spec": {"in_features": 1536, "hidden": GATE_HIDDEN, "dropout": GATE_DROPOUT,
                       "layers": "LayerNorm -> Linear -> GELU -> Dropout -> Linear(1)", "positive_class": "ecg"},
    "gate_feature_spec": GATE_FEATURE_SPEC,
    "gate_threshold": THRESHOLD,
    "gate_threshold_policy": f"highest validation score keeping ECG recall >= {TARGET_RECALL} (FP32, both geometries)",
    "gate_geometries": {"fit": "fit_height_pad (LANCZOS fit + p90 paper pad) to 768x1024",
                        "rectify": f"BILINEAR resize to {RECTIFY_CANVAS_WH}, crop {RECTIFY_MARGIN_PX}px, then fit"},
    "gate_dataset_sha256": DATASET_SHA256,
    "gate_base_inference_sha256": BASE_INFERENCE_SHA256,
    "gate_best_epoch": best["epoch"],
    "gate_run_group": RUN_GROUP,
}
gate_state = {k: v.detach().cpu() for k, v in gate_head.state_dict().items()}
torch.save({"gate_head": gate_state, **gate_meta}, RUN_DIR / "gate_head.pt")
combined = dict(base_ckpt)
combined.update({"gate_head": gate_state, **gate_meta})
torch.save(combined, RUN_DIR / "inference_with_gate.pt")

# Reload into fresh modules and compare with the trained one on a few test images (CPU, FP32).
reloaded_ckpt = torch.load(RUN_DIR / "inference_with_gate.pt", map_location="cpu", weights_only=False)
fresh_gate = GateHead(reloaded_ckpt["gate_head_spec"]["in_features"], reloaded_ckpt["gate_head_spec"]["hidden"],
                      reloaded_ckpt["gate_head_spec"]["dropout"])
fresh_gate.load_state_dict(reloaded_ckpt["gate_head"], strict=True)
fresh = GatedRFPN(build_base_model(reloaded_ckpt), fresh_gate.eval()).eval()
check = GateDataset(frames["test"].groupby("label").head(2).reset_index(drop=True), geometry="fit")
x = torch.stack([check[i][0] for i in range(len(check))]).float().div(255)
x = (x - MEAN) / STD
cpu_model = GatedRFPN(copy.deepcopy(model.base).float().cpu().to(memory_format=torch.contiguous_format),
                      copy.deepcopy(gate_head).cpu()).eval()
with torch.no_grad():
    a_logits, a_gate = fresh(x)
    b_logits, b_gate = cpu_model(x)
    original = build_base_model(base_ckpt)(x)
if not (torch.equal(a_logits, original) and torch.allclose(a_gate, b_gate, atol=1e-5)):
    raise RuntimeError("reloaded export differs from the trained model / original logits")
print("export verified: diagnostic logits == original inference.pt; gate logits match the trained head")

with open(RUN_DIR / "gate_metrics.json", "w") as fh:
    json.dump({**metrics, **{k: v for k, v in gate_meta.items()}, "history": history}, fh, indent=2)
"""),
    code(r"""
t = metrics["splits"]
lines = [f"# ECG gate on ConvNeXt-V1 Tiny + RFPN clean12 ({RUN_GROUP})", "",
         f"- base model: `{BASE_INFERENCE}` (sha256 `{BASE_INFERENCE_SHA256[:16]}…`), frozen; diagnostic logits unchanged",
         f"- dataset: `{GATE_TAR_NAME}` (sha256 `{DATASET_SHA256[:16]}…`), "
         f"{len(frames['train'])}/{len(frames['validation'])}/{len(frames['test'])} train/val/test images",
         f"- gate head: {GATE_FEATURE_SPEC}; best epoch {best['epoch']} of {EPOCHS}",
         f"- threshold {THRESHOLD:.4f} (validation ECG recall >= {TARGET_RECALL})", "",
         "| split/geometry | AUROC | AP | ECG recall | false-accept rate | false rejects | false accepts |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for key, v in t.items():
    m = v["at_threshold"]
    lines.append(f"| {key} | {m['auroc']:.4f} | {m['ap']:.4f} | {m['ecg_recall']:.4f} | "
                 f"{m['false_accept_rate']:.4f} | {m['false_rejects']} | {m['false_accepts']} |")
lines += ["", "## Test acceptance rate per source", "", "```", per_source.round(3).to_string(), "```", ""]
(RUN_DIR / "report.md").write_text("\n".join(lines) + "\n")
(RUN_DIR / "config.json").write_text(json.dumps({
    "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
    "warmup_epochs": WARMUP_EPOCHS, "target_recall": TARGET_RECALL, "amp": str(AMP_DTYPE),
    "device": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "cpu",
    "torch": torch.__version__, "timm": timm.__version__, "dataset": str(DATASET_TAR),
    "base_inference": str(BASE_INFERENCE)}, indent=2) + "\n")
print("\n".join(lines))
"""),
    md("## 9. Copy to Drive"),
    code(r"""
def copy_to_drive(local_path):
    destination = DRIVE_RUN_DIR / local_path.relative_to(RUN_DIR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    shutil.copyfile(local_path, partial)
    try:
        os.replace(partial, destination)
    except OSError:  # some FUSE mounts refuse rename-over-existing
        destination.unlink(missing_ok=True)
        os.replace(partial, destination)
    if destination.stat().st_size != local_path.stat().st_size:
        raise RuntimeError(f"Drive size check failed for {destination}")
    return destination

artifacts = []
for path in sorted(p for p in RUN_DIR.rglob("*") if p.is_file()):
    copy_to_drive(path)
    artifacts.append({"file": str(path.relative_to(RUN_DIR)), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
complete = RUN_DIR / "gate_complete.json"
complete.write_text(json.dumps({"finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "artifacts": artifacts}, indent=2) + "\n")
copy_to_drive(complete)
for a in artifacts:
    print(f"saved {a['file']} ({a['bytes'] / 2**20:.1f} MiB)")
print("run folder:", DRIVE_RUN_DIR)
if IN_COLAB:
    from google.colab import drive
    drive.flush_and_unmount()
    print("Google Drive flushed")
    if RELEASE_RUNTIME_WHEN_DONE:
        from google.colab import runtime
        print("releasing the Colab runtime ...")
        runtime.unassign()
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {OUTPUT} ({len(cells)} cells)")
