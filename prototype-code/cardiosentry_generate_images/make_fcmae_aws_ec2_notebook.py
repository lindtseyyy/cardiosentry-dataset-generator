"""Build the self-contained AWS EC2 notebook for FCMAE ECG image generation."""
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
OUTPUT = HERE / "fcmae_generate_synthetic_images_aws_ec2.ipynb"
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
    md("""# FCMAE synthetic ECG image corpus — AWS EC2 (Ubuntu)

This notebook consumes the standalone raw-signal archive from S3 and creates
exactly **one synthetic paper ECG image per fidelity-qualified manifest record**.
It is designed for a `c8a.4xlarge` with a 300-GiB gp3 EBS root volume, and it
also works from a Google Colab frontend connected to the EC2 Jupyter runtime.
It does not import or depend on `google.colab`.

All downloads, extraction, ECG-Image-Kit work files, chunk indexes, PNGs,
validation, packaging, and hashing happen on the EC2 local EBS filesystem. S3
is used only to download the raw input archive and upload the verified final
release. Authentication uses boto3's normal credential chain; on EC2 this
should be an IAM instance role. **Do not put access keys in this notebook.**

### Restart behavior

The full output directory is persistent on EBS. Each completely successful
200-record chunk writes an atomic `index_parts/chunk_XXXXX.csv`; on restart the
generator validates those rows and image files, skips completed records, and
renders only missing work. Packaging and S3 upload are separate cells, so a
packaging or upload problem cannot erase or force regeneration of completed
images.

### Rendering policy (unchanged from the tested Colab notebook)

- Exactly 24,942 qualified ECGs and 24,942 PNGs; one image per ECG.
- No printed ID, age, sex, name, or date header.
- Native 257/500/1000 Hz sampling is preserved; there is no blanket resampling.
- No 40 Hz default low-pass. No extra filter is applied to the distributed
  257/500 Hz sources, and SPHDB is explicitly not filtered twice.
- PTB alone is conditioned from its documented 0–1 kHz hardware bandwidth to
  0.05–150 Hz, zero-phase, before its 10-second segment is selected.
- No median centering, amplitude normalization, or amplitude clipping.
- 600 DPI at 25 mm/s is retained for diagnostic morphology fidelity.
- Only grid color changes: red, pink, orange, or grey.
- Grid assignment remains a seeded hash of the anonymous record ID. Source,
  diagnosis, and morphology never influence the grid color.
- No photographic or geometric augmentation is introduced.

Run the cells in order. Re-running from the top is safe after interruption.
"""),
    code("""# 0. Configuration and EC2 preflight — edit only this block as needed
import os

# Set these before importing NumPy/SciPy. Every renderer process gets one
# numerical thread rather than recursively creating its own thread pool.
os.environ.update({
    'CUDA_VISIBLE_DEVICES': '',
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
})

import hashlib, importlib, json, shutil, socket, subprocess, sys, threading, time
from pathlib import Path

# Use the current Jupyter/kernel user's home by default. This becomes
# /home/ubuntu/CardioSentry on Ubuntu AMIs and /home/ec2-user/CardioSentry on
# Amazon Linux, while still allowing an explicit environment override.
PROJECT_ROOT = Path(
    os.environ.get(
        'CARDIOSENTRY_PROJECT_ROOT',
        str(Path.home() / 'CardioSentry'),
    )
).expanduser().resolve()
DATASETS_DIR = PROJECT_ROOT / 'datasets'
WORK_DIR = PROJECT_ROOT / 'work' / 'fcmae_synthetic_images_diagnostic_v1'
OUTPUT_DIR = PROJECT_ROOT / 'experiments' / 'fcmae_synthetic_images_diagnostic_v1'
RUNS_DIR = PROJECT_ROOT / 'runs' / 'fcmae_synthetic_images_diagnostic_v1'

S3_BUCKET = 'cardiosentry-fcmae-758821198776-us-east-1-an'
S3_INPUT_PREFIX = 'input'
S3_OUTPUT_PREFIX = 'output'
S3_RUN_NAME = 'fcmae_synthetic_images_diagnostic_v1'
AWS_REGION = 'us-east-1'

RAW_TAR_NAME = 'fcmae-pretraining-v1-standalone.tar'
RAW_SHA_NAME = RAW_TAR_NAME + '.sha256'
FINAL_ARCHIVE_NAME = 'fcmae-synthetic-images-diagnostic-v1.tar'

VCPUS = (
    len(os.sched_getaffinity(0))
    if hasattr(os, "sched_getaffinity")
    else (os.cpu_count() or 1)
)
WORKERS = VCPUS
CHUNK_SIZE = 200  # Initial default; edit here only if deliberately changing it.

DPI = 600
EXPECTED_RAW_RECORDS = 25_109
EXPECTED_IMAGES = 24_942

RAW_TAR = DATASETS_DIR / RAW_TAR_NAME
RAW_SHA = DATASETS_DIR / RAW_SHA_NAME
INPUT_DIR = WORK_DIR / 'input'
CODE_DIR = WORK_DIR / 'code'
KIT_ROOT = WORK_DIR / 'ecg-image-kit'
LOCAL_OUTPUT = OUTPUT_DIR / 'rendered_corpus'
SMOKE_OUTPUT = WORK_DIR / 'smoke_sample_diagnostic_v2'
RENDER_MARKER = RUNS_DIR / 'render_complete.json'
RELEASE_DIR = OUTPUT_DIR / 'release'
ARCHIVE = RELEASE_DIR / FINAL_ARCHIVE_NAME
CHECKSUM = RELEASE_DIR / (FINAL_ARCHIVE_NAME + '.sha256')
SUCCESS_MARKER = RUNS_DIR / '_SUCCESS.json'

S3_INPUT_BASE = S3_INPUT_PREFIX.strip('/')
S3_OUTPUT_BASE = '/'.join(
    part for part in (S3_OUTPUT_PREFIX.strip('/'), S3_RUN_NAME.strip('/')) if part
)
S3_RAW_TAR_KEY = f'{S3_INPUT_BASE}/{RAW_TAR_NAME}'
S3_RAW_SHA_KEY = f'{S3_INPUT_BASE}/{RAW_SHA_NAME}'

for path in (PROJECT_ROOT, DATASETS_DIR, WORK_DIR, CODE_DIR, OUTPUT_DIR, RUNS_DIR, RELEASE_DIR):
    path.mkdir(parents=True, exist_ok=True)

def total_ram_gib():
    try:
        fields = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            fields[key] = int(value.strip().split()[0])
        return fields['MemTotal'] / 2**20
    except Exception:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        return pages * page_size / 2**30

workspace_disk = shutil.disk_usage(PROJECT_ROOT)
print('hostname:             ', socket.gethostname())
print('CPU count:            ', VCPUS)
print(f'RAM:                   {total_ram_gib():.1f} GiB')
print(f'available local disk:  {workspace_disk.free / 2**30:.1f} GiB')
print('project root:         ', PROJECT_ROOT)
print('local datasets:       ', DATASETS_DIR)
print('local work:           ', WORK_DIR)
print('local output:         ', OUTPUT_DIR)
print('runs/log directory:   ', RUNS_DIR)
print('workers:              ', WORKERS)
print('chunk size:           ', CHUNK_SIZE)
print('DPI:                  ', DPI)
print(f'expected image count:  {EXPECTED_IMAGES:,}')
print('S3 source:            ', f's3://{S3_BUCKET}/{S3_INPUT_BASE}/')
print('S3 destination:       ', f's3://{S3_BUCKET}/{S3_OUTPUT_BASE}/')
print('BLAS/OpenMP limits:   ', {
    name: os.environ[name] for name in (
        'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'
    )
})

assert WORKERS >= 1
assert CHUNK_SIZE >= 1
if VCPUS != 16:
    print(f'NOTE: this process sees {VCPUS} CPUs; c8a.4xlarge normally exposes about 16.')
if workspace_disk.total < 250 * 2**30:
    print('WARNING: this filesystem is smaller than the requested 300-GiB EBS volume.')
"""),
    md("""## 1. Install the renderer and unpack the embedded pipeline code

The notebook embeds the same tested generator and patcher as the Colab version.
It clones ECG-Image-Kit at the same pinned commit and applies the same idempotent
patches. Only missing Python packages are installed; the live NumPy/SciPy stack
is never downgraded or replaced unnecessarily.
"""),
    code(f"""# boto3 uses the IAM instance role through its default credential chain.
import importlib.util
module_requirements = {{
    'boto3': 'boto3',
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
    print('all renderer and S3 packages are already present')

subprocess.run([
    sys.executable, '-c',
    'import numpy, scipy, pandas, h5py; '
    'print("scientific stack:", numpy.__version__, scipy.__version__, '
    'pandas.__version__, h5py.__version__)'
], check=True)

for executable in ('git', 'tar', 'sha256sum'):
    assert shutil.which(executable), f'Missing required system executable: {{executable}}'

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
assert G.DPI == DPI == 600
assert tuple(G.DIAGNOSTIC_BAND_HZ) == (0.05, 150.0)
print('renderer and embedded generator ready')
print('generator sha256:', hashlib.sha256(
    (CODE_DIR / 'fcmae_synthetic_image_generator.py').read_bytes()
).hexdigest())
"""),
    md("""## 2. Download the standalone raw corpus from S3 to local EBS

The checksum sidecar and tar are downloaded to `DATASETS_DIR`; rendering never
reads from S3. Existing local files of the correct remote size are reused, then
the tar is independently hashed against its sidecar. boto3 obtains credentials
from the EC2 IAM role—there are no keys in this notebook.
"""),
    code("""import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError

s3 = boto3.client('s3', region_name=AWS_REGION)
transfer_config = TransferConfig(
    multipart_threshold=64 * 2**20,
    multipart_chunksize=64 * 2**20,
    max_concurrency=min(16, WORKERS),
    use_threads=True,
)

identity = boto3.client('sts', region_name=AWS_REGION).get_caller_identity()
print('AWS caller ARN:', identity['Arn'])

class Progress:
    def __init__(self, label, total):
        self.label = label
        self.total = total
        self.seen = 0
        self.last_print = 0.0
        self.lock = threading.Lock()

    def __call__(self, amount):
        with self.lock:
            self.seen += amount
            now = time.time()
            if now - self.last_print >= 15 or self.seen >= self.total:
                print(
                    f'  {self.label}: {self.seen / 2**30:.2f}/'
                    f'{self.total / 2**30:.2f} GiB '
                    f'({100 * self.seen / max(1, self.total):.1f}%)'
                )
                self.last_print = now

def download_s3_object(key, destination, force=False):
    head = s3.head_object(Bucket=S3_BUCKET, Key=key)
    remote_size = int(head['ContentLength'])
    if not force and destination.is_file() and destination.stat().st_size == remote_size:
        print('reusing local S3 object:', destination)
        return head
    partial = destination.with_name(destination.name + '.partial')
    if partial.exists():
        partial.unlink()
    print(f'downloading s3://{S3_BUCKET}/{key} -> {destination}')
    s3.download_file(
        S3_BUCKET, key, str(partial),
        Config=transfer_config,
        Callback=Progress(destination.name, remote_size),
    )
    assert partial.stat().st_size == remote_size
    os.replace(partial, destination)
    return head

# The tiny sidecar is always refreshed so a same-size S3 update cannot be
# hidden by a stale local copy. The much larger tar is reused only after its
# locally computed digest matches this freshly downloaded sidecar.
download_s3_object(S3_RAW_SHA_KEY, RAW_SHA, force=True)
download_s3_object(S3_RAW_TAR_KEY, RAW_TAR)

expected = RAW_SHA.read_text().split()[0].lower()
assert len(expected) == 64 and all(c in '0123456789abcdef' for c in expected)
h = hashlib.sha256()
hashed_bytes = 0
next_report = 1 << 30
with RAW_TAR.open('rb') as fh:
    for block in iter(lambda: fh.read(1 << 24), b''):
        h.update(block)
        hashed_bytes += len(block)
        if hashed_bytes >= next_report:
            print(f'  locally hashed {hashed_bytes / 2**30:.0f} GiB')
            next_report += 1 << 30
actual = h.hexdigest()
assert actual == expected, f'Raw archive checksum mismatch: {actual} != {expected}'
print(f'raw archive OK: {RAW_TAR.stat().st_size / 2**30:.2f} GiB, sha256 {actual}')
"""),
    md("""## 3. Extract and validate the standalone corpus on local EBS

An existing complete extraction is reused. The outer digest proves the S3 tar
is intact; the internal checksum ledger then verifies every materialized signal,
header, and release metadata file before generation begins.
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
subprocess.run(
    ['sha256sum', '--check', '--quiet', 'checksums.sha256'],
    cwd=DATASET_ROOT,
    check=True,
)
print('all standalone release checksums OK')
"""),
    md("""## 4. Audit waveform fidelity, then plan one image per qualified ECG

This is the unchanged fidelity gate. It reads the deterministic render segment
from every source file and rejects rather than repairs missing samples, ADC rail
hits, flat/empty leads, or excursions beyond ±20 mV. Short but otherwise valid
records remain eligible, with unavailable time shown as a blank paper gap.
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
    md("""## 5. Smoke test and local capacity check

The same two records are retained. SPHDB exercises the HDF5 path. The
high-amplitude Chapman record is the regression case that ensures valid trace
pixels entering the nominal header region are never mistaken for printed
demographic text. The actual 600-DPI outputs drive the local capacity estimate.
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

mean_sample_bytes = sum(path.stat().st_size for path in sample_paths) / len(sample_paths)
estimated_image_gib = mean_sample_bytes * EXPECTED_IMAGES / 2**30
# Room for PNGs, final uncompressed tar, source tar/extraction, and headroom.
required_local_gib = estimated_image_gib * 2.20 + 15
local_disk = shutil.disk_usage(PROJECT_ROOT)
local_free_gib = local_disk.free / 2**30
print(f'estimated image payload: {estimated_image_gib:.1f} GiB')
print(f'local free / required:    {local_free_gib:.1f} / {required_local_gib:.1f} GiB')
assert local_free_gib >= required_local_gib, (
    'Not enough EBS space for local rendering, validation, and the final archive.'
)
"""),
    md("""## 6. Render and verify the full corpus on local gp3 EBS

This cell never writes to S3. A failed smoke test blocks the full run. Existing
valid local `index_parts` and PNGs are automatically skipped, so restarting the
notebook does not regenerate completed chunks. The generator's optional
`checkpoint_dir` argument creates duplicate tar shards for ephemeral runtimes;
it is deliberately not used here because `LOCAL_OUTPUT` already lives on
persistent EBS. Progress reports generated, processed, failed, elapsed, and
estimated remaining work after each chunk.
"""),
    code("""assert globals().get('SMOKE_PASSED') is True, (
    'Smoke test did not pass; full generation is blocked.'
)

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
    )
except Exception:
    for name in (
        'failures.csv', 'index.csv', 'generation_config.json',
        'waveform_fidelity_summary.json', 'source_fidelity_policy.csv',
    ):
        source = LOCAL_OUTPUT / name
        if source.is_file():
            shutil.copy2(source, RUNS_DIR / ('incomplete_' + name))
    print('Render interrupted or incomplete. Completed chunks and images remain on EBS at:')
    print(LOCAL_OUTPUT)
    print('Restart this notebook; valid completed records will be skipped.')
    raise

missing = [p for p in full_index.image_path if not (LOCAL_OUTPUT / p).is_file()]
assert len(full_index) == len(manifest) == EXPECTED_IMAGES
assert full_index.fcmae_record_id.nunique() == len(manifest)
assert full_index.image_path.nunique() == len(manifest)
assert not missing, missing[:5]
assert (full_index.header_printed == 0).all()
assert (full_index.variation_policy == 'grid_colour_only').all()
build_hours = (time.time() - started) / 3600
print(f'full build verified in {build_hours:.2f} h this pass')

render_state = {
    'status': 'render_complete',
    'records': int(len(full_index)),
    'images': int(len(full_index)),
    'build_hours_this_pass': round(build_hours, 3),
    'local_output': str(LOCAL_OUTPUT),
    'workers': int(WORKERS),
    'chunk_size': int(CHUNK_SIZE),
    'dpi': int(DPI),
    'generation_policy_sha256': json.loads(
        (LOCAL_OUTPUT / 'generation_config.json').read_text()
    )['generation_policy_sha256'],
}
RENDER_MARKER.write_text(json.dumps(render_state, indent=2) + '\\n')
print('RENDER COMPLETE — validated images and chunk state are persistent on local EBS')
print(RENDER_MARKER)
"""),
    md("""## 7. Create and hash the final archive locally

This cell is intentionally separate from rendering and cannot start until the
24,942-image one-to-one corpus passes all gates. The final tar and checksum are
written to local EBS first. No S3 filesystem or direct-to-S3 rendering is used.
"""),
    code("""assert RENDER_MARKER.is_file(), 'Local render-complete marker is missing.'
assert 'full_index' in globals(), 'Run section 6 to restore/verify local images first.'

members = [
    'images', 'index.csv', 'generation_config.json', 'grid_palette_assignment.csv',
    'grid_palette_by_source.csv', 'grid_palette_audit.json', 'failures.csv',
    'waveform_fidelity_audit.csv', 'waveform_fidelity_exclusions.csv',
    'waveform_fidelity_summary.json', 'source_fidelity_policy.csv'
]
for member in members:
    assert (LOCAL_OUTPUT / member).exists(), f'Missing release member: {member}'

def sha256_file(path, report=True):
    digest = hashlib.sha256()
    seen = 0
    next_report = 1 << 30
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 24), b''):
            digest.update(block)
            seen += len(block)
            if report and seen >= next_report:
                print(f'  hashed {seen / 2**30:.0f}/{path.stat().st_size / 2**30:.1f} GiB')
                next_report += 1 << 30
    return digest.hexdigest()

reuse_archive = False
if ARCHIVE.is_file() and CHECKSUM.is_file():
    recorded_digest = CHECKSUM.read_text().split()[0]
    print('checking existing local final archive before reuse...')
    reuse_archive = sha256_file(ARCHIVE) == recorded_digest

if not reuse_archive:
    partial = ARCHIVE.with_name(ARCHIVE.name + '.partial')
    if partial.exists():
        partial.unlink()
    print('creating final archive locally on EBS...')
    pack_started = time.time()
    process = subprocess.Popen(
        ['tar', '-cf', str(partial), '-C', str(LOCAL_OUTPUT), *members],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    next_report = 60
    while process.poll() is None:
        time.sleep(15)
        elapsed_seconds = time.time() - pack_started
        if elapsed_seconds >= next_report:
            written_gib = partial.stat().st_size / 2**30 if partial.exists() else 0
            print(f'  {elapsed_seconds / 60:.0f} min: {written_gib:.1f} GiB written locally')
            next_report += 60
    stderr = process.stderr.read() if process.stderr else ''
    if process.returncode:
        print(stderr[-2000:])
        raise RuntimeError('Local packaging failed; completed images remain intact.')
    os.replace(partial, ARCHIVE)
    print('hashing local final archive...')
    digest = sha256_file(ARCHIVE)
    CHECKSUM.write_text(f'{digest}  {ARCHIVE.name}\\n')
else:
    digest = CHECKSUM.read_text().split()[0]
    print('reusing verified local final archive')

for name in members[1:]:
    shutil.copy2(LOCAL_OUTPUT / name, RUNS_DIR / name)
(RUNS_DIR / 'raw_input_archive.txt').write_text(
    f'{RAW_TAR.name}\\nsha256 {expected}\\nraw_records {EXPECTED_RAW_RECORDS}\\n'
    f'fidelity_qualified_records {EXPECTED_IMAGES}\\n'
)

assert sha256_file(ARCHIVE, report=False) == digest
print(f'LOCAL ARCHIVE COMPLETE: {ARCHIVE} ({ARCHIVE.stat().st_size / 2**30:.2f} GiB)')
print('sha256:', digest)
"""),
    md("""## 8. Upload the verified release to S3 and verify stored bytes

The final archive, checksum, index, configuration, fidelity audits, failure log,
palette audits, and run metadata are uploaded beneath the configured S3 output
prefix. Existing objects with matching size and SHA-256 metadata are skipped.
Every uploaded object's bytes are then streamed back through SHA-256 validation;
`_SUCCESS.json` is created only after all verifications pass.
"""),
    code("""assert ARCHIVE.is_file() and CHECKSUM.is_file()
assert sha256_file(ARCHIVE, report=False) == digest

artifact_paths = [
    ARCHIVE,
    CHECKSUM,
    LOCAL_OUTPUT / 'index.csv',
    LOCAL_OUTPUT / 'generation_config.json',
    LOCAL_OUTPUT / 'grid_palette_assignment.csv',
    LOCAL_OUTPUT / 'grid_palette_by_source.csv',
    LOCAL_OUTPUT / 'grid_palette_audit.json',
    LOCAL_OUTPUT / 'failures.csv',
    LOCAL_OUTPUT / 'waveform_fidelity_audit.csv',
    LOCAL_OUTPUT / 'waveform_fidelity_exclusions.csv',
    LOCAL_OUTPUT / 'waveform_fidelity_summary.json',
    LOCAL_OUTPUT / 'source_fidelity_policy.csv',
    RENDER_MARKER,
    RUNS_DIR / 'raw_input_archive.txt',
]
assert all(path.is_file() for path in artifact_paths)

artifact_rows = []
for path in artifact_paths:
    local_digest = digest if path == ARCHIVE else sha256_file(path, report=False)
    artifact_rows.append({
        'local_path': str(path),
        'filename': path.name,
        'bytes': int(path.stat().st_size),
        'sha256': local_digest,
        's3_key': f'{S3_OUTPUT_BASE}/{path.name}',
    })

upload_manifest_path = RUNS_DIR / 's3_upload_manifest.json'
upload_manifest_path.write_text(json.dumps({
    'bucket': S3_BUCKET,
    'prefix': S3_OUTPUT_BASE,
    'artifacts': artifact_rows,
}, indent=2) + '\\n')
artifact_rows.append({
    'local_path': str(upload_manifest_path),
    'filename': upload_manifest_path.name,
    'bytes': int(upload_manifest_path.stat().st_size),
    'sha256': sha256_file(upload_manifest_path, report=False),
    's3_key': f'{S3_OUTPUT_BASE}/{upload_manifest_path.name}',
})

def remote_matches(row):
    try:
        head = s3.head_object(Bucket=S3_BUCKET, Key=row['s3_key'])
    except ClientError as exc:
        status = exc.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        if status == 404:
            return False
        raise
    return (
        int(head['ContentLength']) == row['bytes']
        and head.get('Metadata', {}).get('sha256') == row['sha256']
    )

for row in artifact_rows:
    if remote_matches(row):
        print('reusing matching S3 object:', row['s3_key'])
        continue
    print('uploading:', row['s3_key'])
    s3.upload_file(
        row['local_path'], S3_BUCKET, row['s3_key'],
        ExtraArgs={
            'Metadata': {'sha256': row['sha256']},
            'ContentType': 'application/x-tar' if row['filename'].endswith('.tar')
                           else 'application/octet-stream',
        },
        Config=transfer_config,
        Callback=Progress(row['filename'], row['bytes']),
    )
    assert remote_matches(row), f'S3 HEAD verification failed: {row["s3_key"]}'

def verify_s3_sha256(row):
    print('verifying stored S3 bytes:', row['s3_key'])
    response = s3.get_object(Bucket=S3_BUCKET, Key=row['s3_key'])
    remote_digest = hashlib.sha256()
    seen = 0
    next_report = 1 << 30
    body = response['Body']
    try:
        for block in iter(lambda: body.read(1 << 24), b''):
            remote_digest.update(block)
            seen += len(block)
            if seen >= next_report:
                print(f'  verified {seen / 2**30:.0f}/{row["bytes"] / 2**30:.1f} GiB from S3')
                next_report += 1 << 30
    finally:
        body.close()
    assert seen == row['bytes'], f'S3 byte-count mismatch for {row["s3_key"]}'
    assert remote_digest.hexdigest() == row['sha256'], (
        f'S3 SHA-256 mismatch for {row["s3_key"]}'
    )

for row in artifact_rows:
    verify_s3_sha256(row)

success = {
    'status': 'complete',
    'records': int(len(full_index)),
    'images': int(len(full_index)),
    'raw_records_audited': int(FIDELITY_SUMMARY['records_audited']),
    'fidelity_excluded_records': int(FIDELITY_SUMMARY['records_excluded']),
    'fidelity_policy': FIDELITY_SUMMARY['policy'],
    'workers': int(WORKERS),
    'chunk_size': int(CHUNK_SIZE),
    'dpi': int(DPI),
    'raw_archive': RAW_TAR.name,
    'raw_sha256': expected,
    'image_archive': ARCHIVE.name,
    'image_archive_sha256': digest,
    'image_archive_bytes': int(ARCHIVE.stat().st_size),
    's3_archive': f's3://{S3_BUCKET}/{S3_OUTPUT_BASE}/{ARCHIVE.name}',
    'verified_s3_artifacts': int(len(artifact_rows)),
}
SUCCESS_MARKER.write_text(json.dumps(success, indent=2) + '\\n')
success_digest = sha256_file(SUCCESS_MARKER, report=False)
success_row = {
    'local_path': str(SUCCESS_MARKER),
    'filename': SUCCESS_MARKER.name,
    'bytes': int(SUCCESS_MARKER.stat().st_size),
    'sha256': success_digest,
    's3_key': f'{S3_OUTPUT_BASE}/{SUCCESS_MARKER.name}',
}
s3.upload_file(
    success_row['local_path'], S3_BUCKET, success_row['s3_key'],
    ExtraArgs={'Metadata': {'sha256': success_digest}, 'ContentType': 'application/json'},
    Config=transfer_config,
)
assert remote_matches(success_row)
verify_s3_sha256(success_row)

print('DONE — generated locally, packaged locally, uploaded, and independently verified')
print(f'images:      {len(full_index):,}')
print(f'local tar:   {ARCHIVE} ({ARCHIVE.stat().st_size / 2**30:.2f} GiB)')
print(f'S3 archive:  s3://{S3_BUCKET}/{S3_OUTPUT_BASE}/{ARCHIVE.name}')
print(f'sha256:      {digest}')
print(f'run metadata: {RUNS_DIR}')
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(OUTPUT)
