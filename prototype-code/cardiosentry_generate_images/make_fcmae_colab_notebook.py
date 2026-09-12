"""Build the self-contained Colab notebook for FCMAE ECG image generation."""
from __future__ import annotations

import base64
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GENERATOR = HERE / "fcmae_synthetic_image_generator.py"
PATCHER = (
    ROOT
    / "dataset-combination-code"
    / "ecg_dataset_pipeline"
    / "image_pipeline"
    / "patch_kit.py"
)
OUTPUT = HERE / "fcmae_generate_synthetic_images_colab.ipynb"
PINNED_KIT_COMMIT = "27b90f56896c9fc78b05a83ca14844ea2637aa0b"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


generator_b64 = base64.b64encode(GENERATOR.read_bytes()).decode("ascii")
patcher_b64 = base64.b64encode(PATCHER.read_bytes()).decode("ascii")

cells = [
    md("""# FCMAE synthetic ECG image corpus — Google Colab

This notebook consumes the standalone raw-signal archive from Google Drive and
creates exactly **one synthetic paper ECG image per manifest record**.

Expected Drive layout:

```text
MyDrive/CardioSentry/
├── codes/fcmae_synthetic_image_generation/     # put this notebook here
├── datasets/
│   ├── fcmae-pretraining-v1-standalone.tar
│   └── fcmae-pretraining-v1-standalone.tar.sha256
├── experiments/fcmae_synthetic_images_diagnostic_v1/  # output image archive
└── runs/fcmae_synthetic_images_diagnostic_v1/          # audits and settings
```

The raw archive remains packed in `datasets/`. It is extracted to Colab's local
`/content` disk and rendering happens locally. Every completed render chunk is
also written to Drive as one verified checkpoint tar, so a runtime loss does not
erase an overnight build. A later, separate cell consolidates those results into
the final archive; checkpoint shards are removed only after that archive has
been uploaded and independently verified.

### Rendering policy

- One image per ECG—no train/validation variant expansion.
- No printed ID, age, sex, name, or date header.
- Native 257/500/1000 Hz sampling is preserved; there is no blanket resampling.
- The old 0.05–40 Hz pass is removed. No extra filter is applied to the
  distributed 257/500 Hz sources. SPHDB is explicitly not filtered twice.
- PTB alone is conditioned from its documented 0–1 kHz hardware bandwidth to
  0.05–150 Hz, zero-phase, before its 10-second segment is selected.
- No median centering, amplitude normalization, or amplitude clipping.
- 600 DPI at 25 mm/s gives 590.6 horizontal pixels/s (295.3 Hz image Nyquist),
  enough to represent the requested 150 Hz diagnostic band.
- Source and applied filter provenance is written per image and per source.
- No rotation, perspective, texture, folds, lighting, color cast, noise, blur,
  crop, or JPEG augmentation.
- Only the grid color changes: red, pink, orange, or grey.
- Grid color assignment is a seeded hash of the anonymous record ID. Source,
  diagnoses, and morphology fields are never assignment inputs.

The L4 GPU is deliberately unused. The L4 High-RAM runtime is selected for its
CPU and memory allocation. The notebook automatically uses every CPU available
to the process, while limiting each renderer to one BLAS/OpenMP thread to avoid
oversubscription. There are no worker-count or packaging switches to configure:
upload the two input files, select L4 High-RAM, and choose **Runtime → Run all**.
"""),
    code("""# 0. Mount Drive and configure the L4 High-RAM runtime
import os

# Set these before importing NumPy/SciPy. Each renderer process gets one
# numerical thread instead of recursively creating another full-size pool.
os.environ.update({
    'CUDA_VISIBLE_DEVICES': '',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
})

from google.colab import drive
drive.mount('/content/drive')

import hashlib, importlib, json, shutil, subprocess, sys, time
from pathlib import Path

DRIVE = Path('/content/drive/MyDrive/CardioSentry')
DATASETS = DRIVE / 'datasets'
CODES = DRIVE / 'codes' / 'fcmae_synthetic_image_generation'
EXPDIR = DRIVE / 'experiments' / 'fcmae_synthetic_images_diagnostic_v1'
RUNDIR = DRIVE / 'runs' / 'fcmae_synthetic_images_diagnostic_v1'
CHECKPOINT_DIR = EXPDIR / 'render_checkpoints_v2'
RENDER_MARKER = RUNDIR / 'render_complete.json'

RAW_TAR = DATASETS / 'fcmae-pretraining-v1-standalone.tar'
RAW_SHA = DATASETS / 'fcmae-pretraining-v1-standalone.tar.sha256'
WORK = Path('/content/fcmae_image_generation')
INPUT_DIR = WORK / 'input'
CODE_DIR = WORK / 'code'
KIT_ROOT = WORK / 'ecg-image-kit'
LOCAL_OUTPUT = WORK / 'synthetic_images_diagnostic_v2'
SMOKE_OUTPUT = WORK / 'smoke_sample_diagnostic_v2'

VCPUS = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
         else (os.cpu_count() or 1))
WORKERS = VCPUS
CHUNK_SIZE = 200       # 24,942 qualified records -> 125 jobs; enough tail parallelism

for path in (CODES, EXPDIR, RUNDIR, WORK, CODE_DIR):
    path.mkdir(parents=True, exist_ok=True)

print(f'vCPUs visible: {VCPUS}; renderer workers: {WORKERS}')
print('GPU hidden:   ', os.environ['CUDA_VISIBLE_DEVICES'] == '')
print('raw archive:  ', RAW_TAR)
print('local work:   ', WORK)
print('checkpoints:  ', CHECKPOINT_DIR)
"""),
    md("""## 1. Preflight and raw-archive integrity

Both the tar and its checksum sidecar should be uploaded to `datasets/`. This
cell hashes the Drive file before spending hours rendering it.
"""),
    code("""assert RAW_TAR.is_file(), f'Missing {RAW_TAR}'
assert RAW_SHA.is_file(), f'Missing {RAW_SHA}'

free_gib = shutil.disk_usage('/content').free / 2**30
assert free_gib >= 25, f'Only {free_gib:.1f} GiB free; use a fresh High-RAM runtime.'
if VCPUS < 12:
    print(f'WARNING: only {VCPUS} vCPUs are visible; the build will run with {WORKERS} workers.')

expected = RAW_SHA.read_text().split()[0]
h = hashlib.sha256()
with RAW_TAR.open('rb') as fh:
    for block in iter(lambda: fh.read(1 << 24), b''):
        h.update(block)
actual = h.hexdigest()
assert actual == expected, f'Raw archive checksum mismatch: {actual} != {expected}'
print(f'archive OK: {RAW_TAR.stat().st_size / 2**30:.2f} GiB, sha256 {actual}')
print(f'local free: {free_gib:.1f} GiB')
"""),
    md("""## 2. Install the renderer and unpack the embedded pipeline code

The notebook embeds the tested generator and patcher, so no second project-code
archive is needed. It shallow-clones the upstream ECG Image Kit at the pinned
commit used during local validation, then applies the same idempotent patches.

The renderer uses Colab's native NumPy/SciPy stack. It does **not** downgrade
NumPy in the live kernel: doing that on Python 3.13 can mix incompatible binary
wheels and cause a numpy.dtype size changed error. The kit's unused legacy
imgaug import is removed by the embedded patcher instead.
"""),
    code(f"""# Install only genuinely missing packages. In particular, never replace
# NumPy/SciPy/Pandas underneath the running Colab kernel.
import importlib.util
module_requirements = {{
    'wfdb': 'wfdb',
    'h5py': 'h5py',
    'PIL': 'pillow',
    'scipy': 'scipy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'yaml': 'pyyaml',
    'qrcode': 'qrcode',
}}
missing_requirements = [
    requirement for module, requirement in module_requirements.items()
    if importlib.util.find_spec(module) is None
]
if missing_requirements:
    print('installing missing packages:', ', '.join(missing_requirements))
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q', *missing_requirements
    ], check=True)
else:
    print('all renderer packages are already present')

# Validate the compiled scientific stack in a fresh interpreter before the
# long build. This gives a direct error here if the runtime was already damaged
# by an earlier NumPy replacement.
subprocess.run([
    sys.executable, '-c',
    'import numpy, scipy, pandas, h5py; '
    'print("scientific stack:", numpy.__version__, scipy.__version__, '
    'pandas.__version__, h5py.__version__)'
], check=True)

if not (KIT_ROOT / '.git').is_dir():
    subprocess.run([
        'git', 'clone', '--filter=blob:none',
        'https://github.com/alphanumericslab/ecg-image-kit.git', str(KIT_ROOT)
    ], check=True)
subprocess.run(['git', '-C', str(KIT_ROOT), 'checkout', '{PINNED_KIT_COMMIT}'], check=True)

import base64
(CODE_DIR / 'fcmae_synthetic_image_generator.py').write_bytes(
    base64.b64decode('{generator_b64}')
)
(CODE_DIR / 'patch_kit.py').write_bytes(base64.b64decode('{patcher_b64}'))
(CODE_DIR / 'config.py').write_text(
    "import os\\nfrom pathlib import Path\\n"
    "KIT = Path(os.environ['ECGKIT_DIR']) / 'codes' / 'ecg-image-generator'\\n"
    "LEAD_NAME_DY = 1.15\\n"
)

patch_env = dict(os.environ)
patch_env['ECGKIT_DIR'] = str(KIT_ROOT)
subprocess.run([sys.executable, str(CODE_DIR / 'patch_kit.py')],
               cwd=CODE_DIR, env=patch_env, check=True)

sys.path.insert(0, str(CODE_DIR))
import fcmae_synthetic_image_generator as G
importlib.reload(G)
print('renderer and embedded generator ready')
"""),
    md("""## 3. Extract the standalone raw corpus to local disk

Extraction is resumable: an existing complete manifest and `records/` tree are
reused. Original provenance paths inside the manifest are retained, while all
signal reads use the standalone `*_ref_path` columns.
"""),
    code("""def locate_dataset_root(root):
    matches = [p.parent for p in root.rglob('manifest.csv') if (p.parent / 'records').is_dir()]
    assert len(matches) == 1, f'Expected one dataset root, found {matches}'
    return matches[0]

try:
    DATASET_ROOT = locate_dataset_root(INPUT_DIR)
    print('reusing extracted corpus:', DATASET_ROOT)
except (AssertionError, FileNotFoundError):
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True)
    started = time.time()
    subprocess.run(['tar', '-xf', str(RAW_TAR), '-C', str(INPUT_DIR)], check=True)
    DATASET_ROOT = locate_dataset_root(INPUT_DIR)
    print(f'extracted in {(time.time() - started) / 60:.1f} min:', DATASET_ROOT)

preflight = G.verify_preflight(DATASET_ROOT, KIT_ROOT, Path(sys.executable))
print(json.dumps(preflight, indent=2))

# The outer tar digest proves the uploaded archive is intact. This second pass
# proves every materialized source/header and release metadata file still
# matches the native SHA-256 ledger produced during corpus construction.
subprocess.run(
    ['sha256sum', '--check', '--quiet', 'checksums.sha256'],
    cwd=DATASET_ROOT,
    check=True,
)
print('all standalone release checksums OK')
"""),
    md("""## 4. Audit waveform fidelity, then plan one image per qualified ECG

This reads the exact deterministic render segment from every source file. It
rejects rather than repairs missing samples, ADC rail hits, flat/empty leads,
or excursions beyond ±20 mV. The complete audit and exclusion reasons travel
with the image release. Short but otherwise valid source records remain
eligible; unavailable time is shown as a blank paper gap.

The palette assignment is made only after this source-quality gate and before
the smoke subset is selected. Cramér's V is an audit of finite-sample
association, not an assignment input.
"""),
    code("""import pandas as pd

FIDELITY_AUDIT, FIDELITY_SUMMARY = G.audit_waveform_fidelity(
    DATASET_ROOT,
    workers=WORKERS,
)
eligible_ids = FIDELITY_AUDIT.loc[
    FIDELITY_AUDIT.fidelity_eligible == 1,
    'fcmae_record_id',
]
manifest = G.load_generation_manifest(
    DATASET_ROOT,
    eligible_record_ids=eligible_ids,
)
EXPECTED_RAW_RECORDS = 25_109
EXPECTED_IMAGES = 24_942

assert FIDELITY_SUMMARY['records_audited'] == EXPECTED_RAW_RECORDS
assert FIDELITY_SUMMARY['records_eligible'] == EXPECTED_IMAGES
assert FIDELITY_SUMMARY['records_excluded'] == 167
print('Fidelity summary:')
print(json.dumps(FIDELITY_SUMMARY, indent=2))
print('\\nSource/filter policy:')
display(G.source_fidelity_policy_table())

source_by_palette, audit = G.palette_audit(manifest)
association = pd.DataFrame(audit['association_tests'])

print(f'{EXPECTED_RAW_RECORDS:,} raw ECG records -> '
      f'{EXPECTED_IMAGES:,} fidelity-qualified ECGs -> '
      f'{EXPECTED_IMAGES:,} planned PNG images')
print('\\nPalette totals:')
print(manifest.grid_palette.value_counts().to_string())
print('\\nSource by palette:')
print(source_by_palette.to_string())
print('\\nAssociation audit:')
print(association.to_string(index=False))

assert manifest.fcmae_record_id.is_unique
assert manifest.image_id.is_unique
assert manifest.grid_palette.value_counts().max() - manifest.grid_palette.value_counts().min() <= 1
"""),
    md("""## 5. Smoke test and visual inspection

Two records are used. The first exercises the SPHDB HDF5 path. The second is a
high-amplitude Chapman ECG whose valid trace enters the nominal header region;
it is a regression test ensuring waveform pixels are never mistaken for a
printed demographic header. Both images should show the standard 3×4 layout
plus lead-II rhythm strip, without demographic text or photographic augmentation.
"""),
    code("""from IPython.display import display
from PIL import Image as PILImage

SMOKE_PASSED = False
smoke = G.run_generation(
    dataset_root=DATASET_ROOT,
    output_root=SMOKE_OUTPUT,
    kit_root=KIT_ROOT,
    kit_python=Path(sys.executable),
    record_ids=['SPHDB_A00277', 'CHAPMAN_JS14206'],
    workers=2,
    chunk_size=1,
    fidelity_audit=FIDELITY_AUDIT,
    fidelity_summary=FIDELITY_SUMMARY,
)
cols = [
    'image_id', 'fcmae_record_id', 'source_dataset', 'grid_palette',
    'native_sampling_frequency_hz', 'source_filter_status',
    'applied_filter_type', 'applied_filter_high_hz', 'resampling_applied',
    'header_printed', 'header_band_minimum', 'segment_start_seconds', 'image_path'
]
print(smoke[cols].to_string(index=False))
assert len(smoke) == 2
assert (smoke.header_printed == 0).all()
assert (smoke.variation_policy == 'grid_colour_only').all()
assert all((SMOKE_OUTPUT / path).is_file() for path in smoke.image_path)
high_amplitude_regression = smoke.loc[
    smoke.fcmae_record_id == 'CHAPMAN_JS14206'
].iloc[0]
assert high_amplitude_regression.header_band_minimum < 95
SMOKE_PASSED = True
sample_paths = [SMOKE_OUTPUT / path for path in smoke.image_path]
for sample_path in sample_paths:
    display(PILImage.open(sample_path))

# Estimate storage from the actual 600-DPI smoke output. No size switch is
# exposed: the diagnostic-fidelity policy owns the resolution.
mean_sample_bytes = sum(path.stat().st_size for path in sample_paths) / len(sample_paths)
estimated_image_gib = mean_sample_bytes * EXPECTED_IMAGES / 2**30
required_local_gib = estimated_image_gib * 1.75 + 8
local_free_gib = shutil.disk_usage('/content').free / 2**30
drive_free_gib = shutil.disk_usage(DRIVE).free / 2**30
print(f'estimated image payload: {estimated_image_gib:.1f} GiB')
print(f'local free / required:    {local_free_gib:.1f} / {required_local_gib:.1f} GiB')
print(f'Drive free:               {drive_free_gib:.1f} GiB')
assert local_free_gib >= required_local_gib, 'Not enough local disk for the diagnostic build.'
assert drive_free_gib >= estimated_image_gib * 2.20, (
    'Not enough Google Drive space for recoverable checkpoints plus the final archive.'
)
"""),
    md("""## 6. Restore checkpoints and render the full corpus

No settings are required in this section. A failed smoke test prevents it from
starting—even if a notebook client continues executing queued cells after an
error. Existing verified Drive shards are restored first. During rendering,
every completely successful chunk is saved as an atomic tar plus SHA-256 under
`experiments/fcmae_synthetic_images_diagnostic_v1/render_checkpoints_v2/`.

If the runtime disconnects or a later chunk fails, start a fresh runtime and
run the notebook again. Completed shards are restored and skipped; only missing
chunks render again. Progress reports generated, processed, failed, elapsed,
and estimated remaining counts after every chunk.
"""),
    code("""assert globals().get('SMOKE_PASSED') is True, (
    'Smoke test did not pass; full generation and Drive upload are blocked.'
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
restore_state = G.restore_chunk_checkpoints(CHECKPOINT_DIR, LOCAL_OUTPUT)
print('checkpoint restore:', json.dumps(restore_state, indent=2))

started = time.time()
try:
    full_index = G.run_generation(
        dataset_root=DATASET_ROOT,
        output_root=LOCAL_OUTPUT,
        kit_root=KIT_ROOT,
        kit_python=Path(sys.executable),
        workers=WORKERS,
        chunk_size=CHUNK_SIZE,
        keep_work=False,
        fidelity_audit=FIDELITY_AUDIT,
        fidelity_summary=FIDELITY_SUMMARY,
        checkpoint_dir=CHECKPOINT_DIR,
    )
except Exception:
    # Preserve the small diagnostic files even when one chunk fails. Successful
    # image chunks have already been persisted independently by the generator.
    for name in (
        'failures.csv', 'index.csv', 'generation_config.json',
        'waveform_fidelity_summary.json', 'source_fidelity_policy.csv',
    ):
        source = LOCAL_OUTPUT / name
        if source.is_file():
            shutil.copy2(source, RUNDIR / ('incomplete_' + name))
    print('Render interrupted or incomplete. Verified successful chunks remain at:')
    print(CHECKPOINT_DIR)
    print('Restart the runtime and run this notebook again; completed chunks will be restored.')
    raise

# Hard gate: packaging cannot run unless the complete one-to-one corpus exists.
missing = [p for p in full_index.image_path if not (LOCAL_OUTPUT / p).is_file()]
assert len(full_index) == len(manifest) == EXPECTED_IMAGES
assert full_index.fcmae_record_id.nunique() == len(manifest)
assert full_index.image_path.nunique() == len(manifest)
assert not missing, missing[:5]
assert (full_index.header_printed == 0).all()
assert (full_index.variation_policy == 'grid_colour_only').all()
build_hours = (time.time() - started) / 3600
print(f'full build verified in {build_hours:.2f} h')

render_state = {
    'status': 'render_complete',
    'records': int(len(full_index)),
    'images': int(len(full_index)),
    'build_hours_this_pass': round(build_hours, 3),
    'checkpoint_directory': str(CHECKPOINT_DIR),
    'generation_policy_sha256': json.loads(
        (LOCAL_OUTPUT / 'generation_config.json').read_text()
    )['generation_policy_sha256'],
}
RENDER_MARKER.write_text(json.dumps(render_state, indent=2) + '\\n')
print('RENDER COMPLETE — persistent chunk checkpoints are safe on Drive')
print(RENDER_MARKER)
"""),
    md("""## 7. Consolidate, upload, and independently verify the final archive

This is intentionally separate from rendering. It cannot start unless the
24,942-image corpus and the persistent render-complete marker both exist.

The cell streams one final tar to Drive, displays upload progress every minute,
remounts Drive, and hashes the stored file independently. Checkpoint shards are
deleted only after the final archive and success marker have both been verified.
If packaging is interrupted, rerun the notebook: section 6 restores the shards
without regenerating their images, and this cell retries consolidation.
"""),
    code("""assert RENDER_MARKER.is_file(), 'Persistent render-complete marker is missing.'
assert 'full_index' in globals(), 'Run section 6 to restore/verify the local images first.'

archive = EXPDIR / 'fcmae-synthetic-images-diagnostic-v1.tar'
checksum = EXPDIR / 'fcmae-synthetic-images-diagnostic-v1.tar.sha256'
success_marker = RUNDIR / '_SUCCESS.json'

members = [
    'images', 'index.csv', 'generation_config.json', 'grid_palette_assignment.csv',
    'grid_palette_by_source.csv', 'grid_palette_audit.json', 'failures.csv',
    'waveform_fidelity_audit.csv', 'waveform_fidelity_exclusions.csv',
    'waveform_fidelity_summary.json', 'source_fidelity_policy.csv'
]
partial = EXPDIR / (archive.name + '.partial')
if partial.exists():
    partial.unlink()

# One Drive write and one simultaneous hash; no loose images touch Drive.
command = (
    'set -o pipefail; tar -cf - -C ' + repr(str(LOCAL_OUTPUT)) + ' ' +
    ' '.join(members) + ' | tee ' + repr(str(partial)) + ' | sha256sum'
)
print('Packaging and uploading final archive...')
pack_started = time.time()
process = subprocess.Popen(
    ['bash', '-c', command],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
next_report = 60
while process.poll() is None:
    time.sleep(15)
    elapsed_seconds = time.time() - pack_started
    if elapsed_seconds >= next_report:
        uploaded_gib = partial.stat().st_size / 2**30 if partial.exists() else 0
        print(f'  {elapsed_seconds / 60:.0f} min: {uploaded_gib:.1f} GiB written to Drive')
        next_report += 60
stdout, stderr = process.communicate()
if process.returncode:
    print(stderr[-2000:])
    raise RuntimeError('Drive packaging failed; the final archive was not replaced.')
digest = stdout.split()[0]
os.replace(partial, archive)
checksum.write_text(f'{digest}  {archive.name}\\n')

for name in members[1:]:
    shutil.copy2(LOCAL_OUTPUT / name, RUNDIR / name)
(RUNDIR / 'raw_input_archive.txt').write_text(
    f'{RAW_TAR.name}\\nsha256 {expected}\\nraw_records {EXPECTED_RAW_RECORDS}\\n'
    f'fidelity_qualified_records {EXPECTED_IMAGES}\\n'
)

# Flush pending Drive writes, remount, then hash the stored bytes—not merely
# the stream sent to Drive.
drive.flush_and_unmount()
drive.mount('/content/drive')
stored = hashlib.sha256()
stored_bytes = 0
stored_total = archive.stat().st_size
next_verify_report = 1 << 30
with archive.open('rb') as fh:
    for block in iter(lambda: fh.read(1 << 24), b''):
        stored.update(block)
        stored_bytes += len(block)
        if stored_bytes >= next_verify_report:
            print(
                f'  verified {stored_bytes / 2**30:.0f}/'
                f'{stored_total / 2**30:.1f} GiB from Drive'
            )
            next_verify_report += 1 << 30
assert stored.hexdigest() == digest, 'Uploaded Drive archive checksum mismatch'

success = {
    'status': 'complete',
    'records': int(len(full_index)),
    'images': int(len(full_index)),
    'raw_records_audited': int(FIDELITY_SUMMARY['records_audited']),
    'fidelity_excluded_records': int(FIDELITY_SUMMARY['records_excluded']),
    'fidelity_policy': FIDELITY_SUMMARY['policy'],
    'workers': int(WORKERS),
    'chunk_size': int(CHUNK_SIZE),
    'build_hours': round(build_hours, 3),
    'raw_archive': RAW_TAR.name,
    'raw_sha256': expected,
    'image_archive': archive.name,
    'image_archive_sha256': digest,
    'image_archive_bytes': int(archive.stat().st_size),
}
success_marker.write_text(json.dumps(success, indent=2) + '\\n')

assert json.loads(success_marker.read_text())['image_archive_sha256'] == digest

# The final archive and success marker are now independently verified. Only
# now is it safe to remove the temporary duplicate checkpoint shards.
removed_checkpoints = 0
cleanup_warning = ''
try:
    if CHECKPOINT_DIR.exists():
        for path in list(CHECKPOINT_DIR.iterdir()):
            if path.is_file():
                path.unlink()
                removed_checkpoints += 1
        if not any(CHECKPOINT_DIR.iterdir()):
            CHECKPOINT_DIR.rmdir()
except Exception as exc:
    cleanup_warning = f'{type(exc).__name__}: {exc}'

print('DONE — generated, uploaded, and independently verified')
print(f'images:   {len(full_index):,}')
print(f'archive:  {archive} ({archive.stat().st_size / 2**30:.2f} GiB)')
print(f'sha256:   {digest}')
print(f'metadata: {RUNDIR}')
print(f'cleaned {removed_checkpoints} superseded checkpoint files after verification')
if cleanup_warning:
    print('WARNING: final archive is safe, but checkpoint cleanup was incomplete:', cleanup_warning)
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
