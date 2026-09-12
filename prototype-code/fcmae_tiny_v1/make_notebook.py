"""Build a self-contained Colab notebook from reviewable companion modules."""
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.strip()+'\n'))


def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.strip()+'\n', execution_count=None, outputs=[]))


md('''# CardioSentry — ConvNeXt-V2 Tiny domain-adaptive FCMAE

**Baseline v1:** ImageNet-1K self-supervised FCMAE initialization → continued
ECG-image FCMAE → exported encoder for later supervised fine-tuning.

| Setting | Baseline |
|---|---|
| Image | RGB, **768 height × 1024 width** |
| Resize | Preserve aspect ratio, center-pad; no flips or random crops |
| Encoder | Official ConvNeXt-V2 Tiny sparse encoder |
| Mask | Uniform random 60% of patches containing valid image pixels |
| Patch grid | 32-pixel patches; 24 × 32 grid |
| Objective | Masked patch MSE, normalized targets, padding pixels excluded |
| Decoder | Official depth 1, width 512; randomly initialized (release is encoder-only) |
| Optimization | AdamW, warmup/cosine, configurable accumulation and FP16 |

This is a **rectangular/padding adaptation** of Meta's FCMAE, not an unmodified
reproduction. It retains the actual Minkowski sparse encoder and its GRN
behavior. The original sparse GRN aggregates active sites across a microbatch;
gradient accumulation does not make that statistic equivalent to a larger
microbatch. Record both batch size and accumulation in comparisons.

The notebook contains its supporting Python files; upload this `.ipynb` alone
to `MyDrive/CardioSentry/codes/fcmae_tiny_v1/`. Run cells in order on a CUDA Colab
runtime (L4/A100 preferred; capacity must be measured by the pilot).
It builds a separate Python 3.10/CUDA 11.8 runtime for the legacy sparse engine.
Allow at least 20 GiB for this environment **in addition to the image archive
and extracted images**. The notebook kernel itself is not replaced.

**Execution status:** CPU utility tests are supplied. CUDA compilation, the real
checkpoint, sparse parity, and the full-size training step are validated by the
runtime/pilot cells on your GPU. Do not interpret an unexecuted notebook as a
successful GPU run.
''')
md('''## 1. Mount Drive and configure inputs

The archive path matches the current diagnostic-fidelity image generator.
Generated masters remain unchanged. Extraction and batch reads use Colab's
local disk; periodic resumable checkpoints and metrics go to Drive.

Start with `RUN_FULL = False`. The pilot uses a small subset at the **actual
768 × 1024 input resolution**, and does not authorize claims about model quality.
The notebook embeds a canonical downstream validation/test exclusion CSV from
the current supervised V2 release. Its source hash is in `holdout_provenance.json`.
Rebuild it with `make_holdouts.py` if downstream splits change, or supply your
replacement through `HOLDOUT_CSV`.
Required columns: `source_dataset`, `canonical_record_id`, `patient_id`,
`patient_id_available` (1 or 0). Include every downstream validation/test record;
use the same canonical source namespace as the pretraining index. Unknown patient
IDs are reported rather than claimed to be patient-disjoint. Existing record
overlap removal in the corpus is not a substitute for this audit.

`SSL validation` below means a held-out subset of the eligible pretraining
corpus. It is separate from downstream validation/test. No pathology labels
are passed to the model.
''')
code('''from pathlib import Path
import os, json, subprocess, shutil, sys, time
from google.colab import drive
drive.mount('/content/drive')

DRIVE = Path('/content/drive/MyDrive/CardioSentry')
ARCHIVE = DRIVE/'experiments/fcmae_synthetic_images_diagnostic_v1/fcmae-synthetic-images-diagnostic-v1.tar'
ARCHIVE_SHA = Path(str(ARCHIVE)+'.sha256')
WORK = Path('/content/cardiosentry_fcmae_tiny_v1')
CODE = WORK/'code'
RUNTIME = WORK/'runtime'
EXPERIMENT = DRIVE/'experiments/fcmae_tiny_v1'
RUN = DRIVE/'runs/fcmae_tiny_v1_768_seed42'
HOLDOUT_CSV = CODE/'downstream_holdouts_v2.csv'  # Embedded current V2 snapshot; replace if splits change.
RUN_FULL = False
RUN_BENCHMARK = True
BENCHMARK_ONLY = True  # Keep True while comparing GPU costs; later set False for the pilot/training.
COMPUTE_UNITS_PER_HOUR = None  # Enter the current rate shown in Colab; 0 only for an actually free session.
PREPARATION_TIMINGS = {}
PILOT_PASSED = False
PYTHON_OVERRIDE = None  # Optional existing compatible Python executable.
for path in [WORK, CODE, EXPERIMENT, RUN]:
    path.mkdir(parents=True, exist_ok=True)
os.environ['HF_HOME'] = str(DRIVE/'hf_cache')

# Initial engineering schedule, not an experimentally established optimum.
# Use a new RUN directory when changing these settings.
CONFIG = dict(input_size=[768, 1024], seed=42, batch_size=2, accumulation=16,
              workers=2, epochs=100, warmup_epochs=5, learning_rate=2e-5,
              amp=True, clip_grad=1.0, ssl_val_fraction=.05, save_every=10,
              checkpoint_every_updates=100, checkpoint_every_minutes=15,
              run_dir=str(RUN), holdout_csv=str(HOLDOUT_CSV) if HOLDOUT_CSV else None)
assert all(n % 32 == 0 for n in CONFIG['input_size'])
print('Effective batch:', CONFIG['batch_size']*CONFIG['accumulation'])
print('Full run enabled:', RUN_FULL)
print('Benchmark only:', BENCHMARK_ONLY)
subprocess.run(['nvidia-smi'], check=True)
''')
md('''## 2. Materialize the pinned code

The embedded official model files are from ConvNeXt-V2 commit
`2553895753323c6fe0b2bf390683f5ea358a42b9`. The exact sparse-engine fork revision is
`shwoo93/MinkowskiEngine@bbc30ef581ea6deb505976b663f5fc2358a83749`.
The adaptations live in separate modules; original model files and license
are preserved. Rebuild this notebook with `make_notebook.py` after editing a
companion module.
''')
files = ['data_utils.py', 'rect_fcmae.py', 'checkpoint_utils.py', 'train.py', 'setup_runtime.py', 'test_data_utils.py', 'test_checkpoint_utils.py',
         'training_state.py', 'test_training_state.py',
         'benchmark.py', 'benchmark_report.py', 'test_benchmark_report.py',
         'downstream_holdouts_v2.csv', 'holdout_provenance.json']
files += [str(p.relative_to(HERE)) for p in sorted((HERE/'vendor').rglob('*')) if p.is_file() and '__pycache__' not in str(p)]
payload = {p: base64.b64encode((HERE/p).read_bytes()).decode() for p in files}
hashes = {p: hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in files}
code('import base64, hashlib\nPAYLOAD = '+repr(payload)+'\nSOURCE_HASHES = '+repr(hashes)+'''
for name, encoded in PAYLOAD.items():
    content = base64.b64decode(encoded)
    assert hashlib.sha256(content).hexdigest() == SOURCE_HASHES[name]
    target = CODE/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
(EXPERIMENT/'source_hashes.json').write_text(json.dumps(SOURCE_HASHES, indent=2))
print('Materialized', len(PAYLOAD), 'source/license files')
''')
md('''## 3. Build the isolated sparse-training runtime — Runtime compatibility fixes v5

This can take several minutes and downloads a compiler/CUDA toolkit. Build logs
are streamed. A successful import is only an environment check; the pilot later
tests actual sparse convolutions and dense encoder export. There is **no fallback
to input-only masking or to a different initialization** if setup fails.

The primary package versions are pinned; the resolved conda package URLs and
`pip freeze` are saved for reproducing the exact successful environment. Colab
images/drivers change, so retain any compiler log if this cell fails.
''')
code('''stage_started = time.perf_counter()
setup_log = RUN/'runtime_setup.log'
if PYTHON_OVERRIDE:
    TRAIN_PYTHON = str(PYTHON_OVERRIDE)
else:
    with open(setup_log, 'a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(CODE/'setup_runtime.py'), str(RUNTIME)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end=''); log.write(line); log.flush()
        if process.wait() != 0:
            raise RuntimeError(f'Sparse runtime setup failed. See {setup_log}')
    TRAIN_PYTHON = str(RUNTIME/'env/bin/python')

TRAIN_ENV = dict(os.environ, OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
if not PYTHON_OVERRIDE:
    TRAIN_ENV['PATH'] = str(RUNTIME/'env/bin')+os.pathsep+os.environ['PATH']
    TRAIN_ENV['LD_LIBRARY_PATH'] = str(RUNTIME/'env/lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH','')
    TRAIN_ENV['CUDA_HOME'] = str(RUNTIME/'env')
    for name in ['conda-explicit.txt', 'pip-freeze.txt', 'runtime_ready.json']:
        shutil.copy2(RUNTIME/name, RUN/name)
subprocess.run([TRAIN_PYTHON, '-c', 'import torch, MinkowskiEngine; print(torch.__version__, torch.version.cuda, MinkowskiEngine.__version__); assert torch.cuda.is_available()'],
               env=TRAIN_ENV, check=True)
subprocess.run([TRAIN_PYTHON, str(CODE/'test_data_utils.py')], env=TRAIN_ENV, check=True)
subprocess.run([TRAIN_PYTHON, str(CODE/'test_training_state.py')], env=TRAIN_ENV, check=True)
subprocess.run([TRAIN_PYTHON, str(CODE/'test_benchmark_report.py')], env=TRAIN_ENV, check=True)
PREPARATION_TIMINGS['runtime_setup_and_tests_seconds'] = time.perf_counter()-stage_started
''')
md('''## 4. Verify and extract the generated image archive

The checksum is verified before extraction. Only regular files/directories are
accepted; absolute paths, traversal paths, links, and devices are rejected. The
index and all images are kept together. If resuming in a fresh Colab session,
this cell restages the data without changing saved split assignments.
''')
code('''import tarfile
stage_started = time.perf_counter()
def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            digest.update(block)
    return digest.hexdigest()

assert ARCHIVE.is_file() and ARCHIVE_SHA.is_file(), 'Image generation/archive is not ready at configured path'
expected_sha = ARCHIVE_SHA.read_text().split()[0].lower()
assert len(expected_sha) == 64 and all(c in '0123456789abcdef' for c in expected_sha)
local_archive = WORK/ARCHIVE.name
if not local_archive.exists() or file_sha256(local_archive) != expected_sha:
    staging = local_archive.with_suffix('.copying')
    shutil.copyfile(ARCHIVE, staging)
    assert file_sha256(staging) == expected_sha, 'Archive checksum mismatch'
    staging.replace(local_archive)

DATA = WORK/'images_release'
marker = DATA/'.archive_sha256'
if not marker.exists() or marker.read_text().strip() != expected_sha:
    if DATA.exists():
        raise RuntimeError('Incomplete or different extraction exists: choose a new WORK directory or inspect it before removing it')
    stage = WORK/'images_release_extracting'
    if stage.exists():
        raise RuntimeError('Previous extraction was interrupted; inspect/remove images_release_extracting and retry')
    stage.mkdir()
    with tarfile.open(local_archive) as tf:
        members = tf.getmembers()
        required_bytes = sum(m.size for m in members if m.isfile())
        assert shutil.disk_usage(WORK).free > required_bytes*1.1, 'Insufficient extraction space'
        for member in members:
            destination = (stage/member.name).resolve()
            if Path(member.name).is_absolute() or not destination.is_relative_to(stage.resolve()):
                raise ValueError('Unsafe archive path: '+member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError('Unsupported archive member: '+member.name)
        for member in members:
            destination = stage/member.name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as source, open(destination, 'wb') as target:
                    shutil.copyfileobj(source, target)
    (stage/'.archive_sha256').write_text(expected_sha)
    stage.rename(DATA)
indexes = [p for p in DATA.rglob('index.csv') if (p.parent/'images').is_dir()]
assert len(indexes) == 1, f'Expected one image release index; found {indexes}'
INDEX = indexes[0]
CONFIG.update(index=str(INDEX), image_root=str(INDEX.parent), archive_sha256=expected_sha,
              resize_cache=str(WORK/'resize_cache'/expected_sha))
print('Image index:', INDEX)
PREPARATION_TIMINGS['archive_copy_verification_extraction_seconds'] = time.perf_counter()-stage_started
''')
md('''## 5. Download the self-supervised ImageNet-1K checkpoint

This URL is the **FCMAE** Tiny checkpoint in Meta's self-supervised table, not
the 1K classifier or 22K fine-tuned checkpoint. The loader rejects missing,
unexpected, or incompatible tensors rather than silently leaving layers random.
The official release contains 196 encoder tensors; the reconstruction decoder
is newly initialized. The downloaded SHA-256 is recorded and checked against
the file inspected while building this notebook (not a publisher-supplied digest).

Meta's code is MIT; the provided ImageNet weights are CC-BY-NC, as stated in the
bundled license. Record that provenance with downstream artifacts.
''')
code('''import urllib.request
stage_started = time.perf_counter()
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_tiny_1k_224_fcmae.pt'
checkpoint = WORK/'convnextv2_tiny_1k_224_fcmae.pt'
if not checkpoint.exists():
    temporary = checkpoint.with_suffix('.downloading')
    with urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as response, open(temporary, 'wb') as target:
        shutil.copyfileobj(response, target)
    temporary.replace(checkpoint)
CONFIG.update(checkpoint=str(checkpoint), checkpoint_url=CHECKPOINT_URL)
verified_checkpoint_sha = 'b25eadb89ad634b5c925ea04cdacebafe3e462486baf9482dc2d832bb9af57ef'
assert file_sha256(checkpoint) == verified_checkpoint_sha, 'Checkpoint differs from inspected official release'
subprocess.run([TRAIN_PYTHON, str(CODE/'test_checkpoint_utils.py'), '--checkpoint', str(checkpoint)],
               env=TRAIN_ENV, check=True)
config_path = WORK/'config.json'
config_path.write_text(json.dumps(CONFIG, indent=2))
print('Checkpoint SHA-256:', file_sha256(checkpoint))
print(json.dumps(CONFIG, indent=2))
PREPARATION_TIMINGS['checkpoint_download_verification_seconds'] = time.perf_counter()-stage_started
''')
md('''## 5A. Benchmark the assigned GPU and save a cost report

Run this on **each GPU you actually obtain** (T4, L4, A100). It cannot measure
unavailable GPUs. Keep `BENCHMARK_ONLY=True` while comparing: subsequent pilot
and full-training cells will be skipped, even with Run all.

Enter the **compute units/hour currently displayed in Colab** in the configuration
cell. No prices are assumed. If left as `None`, timing reports still work and
cost columns remain blank. A rate of 0 means an actually free allocated session,
not guaranteed free access or a guarantee of any session duration.

This runs the real sparse model at the configured resolution/batch/precision:
2 warmup updates, then 3 measured rounds of 6 updates (192 images per round at
the defaults), plus validation and actual mounted-Drive write timings. Expect
several minutes or longer depending on hardware. The GPU synchronization points
ensure elapsed time includes completed GPU work, not just queued operations.

Cache preparation is timed separately using a fresh sample cache. Full-dataset
cache cost is extrapolated; it is not a directly measured first full epoch.
Training uses a disposable model, separate from saved pretraining checkpoints.
The reports include memory, initialization/preflight status, settings, sample
identities, runtime versions, and projections for 1, 20, and the configured total
epochs (100 by default). Repeated-round ranges are **not confidence intervals**.
Costs include estimated checkpoint overhead and one observed preparation session;
they exclude future reconnect/setup repetitions and idle time.

Reports go to `MyDrive/CardioSentry/runs/fcmae_tiny_v1_benchmarks/<GPU_timestamp>/`.
Send **`summary.md` and `report.json`** after each GPU, or the combined
`comparison.csv` after all available GPUs. Compare only matching comparison keys;
do not change batch size, precision, resolution, data, or code between GPU tests.
An out-of-memory or compatibility failure is recorded as a failure, not assigned
an invented runtime estimate. The isolated environment must first build successfully.

Once a GPU is chosen, set `RUN_BENCHMARK=False` and `BENCHMARK_ONLY=False`,
rerun the configuration cell, then run the pilot before enabling `RUN_FULL`.
''')
code('''from datetime import datetime, timezone
import re, uuid, math
from IPython.display import display, Markdown

BENCHMARK_ROOT = DRIVE/'runs/fcmae_tiny_v1_benchmarks'
BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
if RUN_BENCHMARK:
    assert COMPUTE_UNITS_PER_HOUR is None or (
        isinstance(COMPUTE_UNITS_PER_HOUR, (int,float)) and math.isfinite(COMPUTE_UNITS_PER_HOUR)
        and COMPUTE_UNITS_PER_HOUR >= 0), 'Enter a nonnegative rate or None'
    gpu_name = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'], text=True).splitlines()[0]
    label = re.sub(r'[^A-Za-z0-9_-]+','_',gpu_name)
    benchmark_dir = BENCHMARK_ROOT/(label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    benchmark_dir.mkdir()
    benchmark_options = dict(units_per_hour=COMPUTE_UNITS_PER_HOUR,
                             preparation_timings=dict(PREPARATION_TIMINGS),
                             warmup_updates=2, measured_updates=6, rounds=3, validation_images=64)
    options_path = benchmark_dir/'options.json'
    options_path.write_text(json.dumps(benchmark_options, indent=2))
    command = [TRAIN_PYTHON,'-u',str(CODE/'benchmark.py'),'--config',str(config_path),
               '--options',str(options_path),'--output',str(benchmark_dir)]
    with open(benchmark_dir/'console.log','w') as log:
        process = subprocess.Popen(command,cwd=CODE,env=TRAIN_ENV,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                print(line,end=''); log.write(line); log.flush()
            benchmark_exit = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if (benchmark_dir/'summary.md').exists():
        display(Markdown((benchmark_dir/'summary.md').read_text()))
    print('Send these files:',benchmark_dir/'summary.md',benchmark_dir/'report.json')
else:
    benchmark_exit = 0
    print('Benchmark disabled; loading existing reports only.')

# The report helper uses only Python's standard library; no kernel packages change.
import importlib.util
spec = importlib.util.spec_from_file_location('fcmae_benchmark_report',CODE/'benchmark_report.py')
report_helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(report_helpers)
import pandas as pd
comparison = pd.DataFrame(report_helpers.comparison_rows(BENCHMARK_ROOT))
if not comparison.empty:
    comparison.to_csv(BENCHMARK_ROOT/'comparison.csv',index=False)
    display(comparison.drop(columns=['report']))
    print('Compare rows only within the same comparison_key; no automatic cross-setting ranking.')
if benchmark_exit:
    raise RuntimeError('Benchmark failed. Send its report.json and console.log before changing training settings.')
''')
md('''## 6. Run the pilot and inspect reconstruction

The pilot runs the real Tiny model with the configured input size and accumulation.
It checks rectangular output bounds, independence from hidden pixel values,
finite gradients, and exported dense-encoder parity on an all-visible batch of
one. The latter also checks sparse kernel ordering during checkpoint conversion.

It then trains for a short epoch, validates with repeatable masks, and writes
timing, memory, loss, and reconstruction panels under `runs/.../pilot/`.
Resized, unpadded PNGs are cached lazily on local disk, avoiding repeated
600-DPI decoding in later epochs. Reserve additional disk space for this cache.
The reconstruction panel uses **ground-truth patch mean/variance for display**
because the objective predicts normalized pixels; it is not an independent
recovery of absolute brightness or contrast.

If memory is insufficient, lower `batch_size`, adjust accumulation as needed,
and use a new run directory. If sparse FP16 is incompatible with your GPU,
set `amp=False` and rerun in a new run directory. Neither change is automatic.
''')
code('''def launch_training(pilot):
    command = [TRAIN_PYTHON, '-u', str(CODE/'train.py'), '--config', str(config_path)]
    if pilot:
        command.append('--pilot')
    log_path = RUN/('pilot_console.log' if pilot else 'full_console.log')
    with open(log_path, 'a') as log:
        process = subprocess.Popen(command, cwd=CODE, env=TRAIN_ENV,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end=''); log.write(line); log.flush()
            if process.wait() != 0:
                raise RuntimeError(f'Training failed; inspect {log_path}')
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise

if not BENCHMARK_ONLY:
    launch_training(pilot=True)
    PILOT_PASSED = True
    from IPython.display import display, Image as DisplayImage
    print((RUN/'pilot/split_audit.json').read_text())
    print((RUN/'pilot/metrics.csv').read_text())
    for figure in sorted((RUN/'pilot/previews').glob('reconstruction_*.png')):
        display(DisplayImage(filename=str(figure)))
else:
    print('Pilot skipped for GPU cost comparison. Set BENCHMARK_ONLY=False when ready to train.')
''')
md('''## 7. Full continued pretraining

After inspecting the pilot, set `RUN_FULL=True` in the configuration cell and
supply `HOLDOUT_CSV`, then rerun the configuration-dependent cells. A full run
requires the exclusion audit; a pilot may run without it and is not a reportable
experiment. The runtime refuses mismatched resume configurations.

All weights are saved directly to mounted Google Drive under
`MyDrive/CardioSentry/runs/fcmae_tiny_v1_768_seed42/`, with separate `pilot/` and
`full/` runs. Within each run:

- `checkpoints/last.pt`: latest full training state, saved every 100 optimizer
  updates or approximately 15 minutes (whichever occurs first, at the next
  completed update), before validation, and after each completed epoch.
- `checkpoints/previous.pt`: preceding snapshot, used automatically if `last.pt`
  is absent or unreadable. This is a rolling backup, not an accumulating archive.
- `checkpoints/checkpoint_epoch_XXXX.pt`: full resumable state every 10 epochs
  and at the final epoch, so epoch 20 remains available after training advances.
- `encoders/`: initial, best-SSL, and periodic encoder-only weights.
- `previews/`: reconstruction images. Metrics and audits remain at the run root.

`checkpoints/last.pt` contains model, optimizer, scaler, RNG states, image order,
the next unprocessed batch, and running epoch metrics. It resumes **within the
epoch** from the last saved update. At batch 2 × accumulation 16, 100 updates
usually cover 3,200 images; the time trigger can save sooner. Saving occurs only
after an optimizer update, with no partially accumulated gradients to restore.
Unsaved work since the last snapshot is replayed after a disconnect. If training
finished but validation was interrupted, only validation/artifact writing repeats.

Checkpoints are serialized to a temporary file, flushed, then renamed; the
previous snapshot is retained. A mounted Drive write is not a guarantee that
remote synchronization survives an immediate hard shutdown. Wait for the saved
message and verify the checkpoint appears in Drive before intentionally ending
a session. There is a write-time pause; GPU/Drive timing must be measured.

The sampler records **consumed** batches, not prefetched ones, and loader RNG is
separate from model/masking RNG. This assumes the current deterministic image
preprocessing; adding random worker augmentations requires extending this design.
Set an interval to 0 to disable that trigger, or both to 0 for epoch-boundary
saving only. Old checkpoints from the previous notebook format must be resumed
with that original notebook; this version refuses implicit migration.
The original sparse engine may still have nondeterministic GPU operations;
saved seeds do not imply bitwise reproducibility.

The 100-epoch schedule and 2e-5 learning rate are initial engineering choices.
Do not infer convergence from the scheduled length. SSL reconstruction loss is
a diagnostic; downstream validation selects the useful encoder checkpoint.
''')
code('''if RUN_FULL and not BENCHMARK_ONLY:
    assert globals().get('PILOT_PASSED'), 'Complete the pilot first'
    assert HOLDOUT_CSV is not None and Path(HOLDOUT_CSV).is_file(), 'Configure the canonical downstream holdout CSV'
    launch_training(pilot=False)
    print('Full pretraining complete:', RUN/'full')
else:
    print('Full training skipped: RUN_FULL is False or BENCHMARK_ONLY is True.')
''')
md('''## 8. Review metrics and use the encoder later

`encoder_initial.pt` is the matched ImageNet-1K FCMAE control before ECG updates.
`encoder_best_ssl.pt` is selected by SSL reconstruction only.
`encoder_epoch_XXXX.pt` files support downstream checkpoint comparisons.

Exports use **official dense ConvNeXt-V2 keys**, not timm's renamed keys. In the
same environment, use `models.convnextv2.convnextv2_tiny(num_classes=...)`, then
load `checkpoint['model']` with `strict=False`. Assert that only
`norm.weight`, `norm.bias`, `head.weight`, and `head.bias` are missing; initialize
those identically for the matched control and adapted model. The sparse encoder
has no pooled classification norm/head. A timm model needs an explicit key map.

Apply the same aspect-preserving input preprocessing during downstream
fine-tuning. Evaluate per-label metrics for STEMI, AF and LVH on downstream
validation; reserve test patients for final evaluation. This notebook does not
train the downstream classifier or claim that reconstruction improves diagnosis.
''')
code('''import pandas as pd
import matplotlib.pyplot as plt
if not BENCHMARK_ONLY:
    result = RUN/('full' if RUN_FULL else 'pilot')
    metrics = pd.read_csv(result/'metrics.csv')
    display(metrics)
    ax = metrics.plot(x='epoch', y=['train_loss', 'ssl_val_loss'], marker='o', figsize=(8,4))
    ax.set_ylabel('Masked normalized-pixel MSE'); ax.grid(alpha=.2)
    plt.show()
    print('Artifacts:')
    for path in sorted(result.rglob('*')):
        if path.is_file():
            print(path.relative_to(result))
else:
    print('Benchmark-only session: use the comparison table and benchmark reports above.')
''')
md('''## References and method boundaries

- [Official ConvNeXt-V2 / FCMAE](https://github.com/facebookresearch/ConvNeXt-V2)
- [ConvNeXt V2 paper](https://arxiv.org/abs/2301.00808)
- [HeartBEiT](https://www.nature.com/articles/s41746-023-00840-9)
- [ECG waveform/scalogram MAE](https://pmc.ncbi.nlm.nih.gov/articles/PMC12238358/)

Later controlled experiments: 1152 × 1536 inputs; waveform-stratified masking;
lead-local temporal masking; loss weighting. None is enabled here. The archive
does not need waveform masks for this uniform baseline. Preserve master images
and rendering provenance for those later experiments.
''')

for i, cell in enumerate(cells):
    cell['id'] = f'fcmae-{i:03d}'
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                         language_info=dict(name='python'), accelerator='GPU',
                                         colab=dict(name='CardioSentry_FCMAE_Tiny.ipynb', provenance=[])),
                nbformat=4, nbformat_minor=5)
path = HERE/'CardioSentry_FCMAE_Tiny.ipynb'
path.write_text(json.dumps(notebook, indent=1)+'\n')
print(path)
