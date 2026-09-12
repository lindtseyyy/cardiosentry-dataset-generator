"""Build an isolated CUDA 11.8/Python 3.10 sparse-training runtime on Colab."""
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


CUDA_CHANNEL = 'nvidia/label/cuda-11.8.0'
SETUP_VERSION = 5


def patch_engine_headers(engine):
    """PyTorch 2.1 CUDAUtils.h needs a complete at::Tensor before inclusion."""
    path = Path(engine)/'src/spmm.cu'
    source = path.read_text()
    anchor = '#include <ATen/cuda/CUDAContext.h>'
    replacement = '#include <ATen/ATen.h>  // CardioSentry: complete Tensor before CUDAUtils\n'+anchor
    if replacement in source:
        return
    if source.count(anchor) != 1 or '#include <ATen/cuda/CUDAUtils.h>' not in source:
        raise RuntimeError('Unexpected MinkowskiEngine spmm.cu headers; refusing compatibility patch')
    path.write_text(source.replace(anchor, replacement, 1))
    print('Applied spmm.cu complete-Tensor header fix (setup v3)', flush=True)


def patch_engine_python(engine):
    """Repair all five legacy ABC imports in the pinned Python package."""
    package = Path(engine)/'MinkowskiEngine'
    for name in ['MinkowskiCoordinateManager.py', 'MinkowskiKernelGenerator.py',
                 'MinkowskiCommon.py', 'MinkowskiTensorField.py', 'utils/quantization.py']:
        path = package/name
        source = path.read_text()
        old = ('from collections import Sequence, namedtuple' if name == 'MinkowskiKernelGenerator.py'
               else 'from collections import Sequence')
        new = ('from collections.abc import Sequence\nfrom collections import namedtuple'
               if name == 'MinkowskiKernelGenerator.py' else 'from collections.abc import Sequence')
        if new in source:
            continue
        if source.count(old) != 1:
            raise RuntimeError(f'Unexpected legacy imports in {path}; refusing compatibility patch')
        path.write_text(source.replace(old, new, 1))
    # Upstream deletes build/ on every install. Let Ninja reuse valid objects
    # and track changed source/header dependencies and compiler command lines.
    path = Path(engine)/'setup.py'
    source = path.read_text()
    old = 'run_command("rm", "-rf", "build")'
    new = '# CardioSentry: retain build objects for Ninja incremental rebuilds'
    if new not in source:
        if source.count(old) != 1:
            raise RuntimeError('Unexpected upstream build cleanup; refusing patch')
        path.write_text(source.replace(old, new, 1))
    print('Verified Python 3.10 imports in five files; incremental builds enabled (v4)', flush=True)


def check_cuda(env, build_env):
    """Check the exact compiler PyTorch will use, not nvidia-smi's driver ceiling."""
    nvcc = Path(env)/'bin/nvcc'
    output = subprocess.check_output([str(nvcc), '--version'], env=build_env, text=True)
    match = re.search(r'release\s+(\d+\.\d+)', output)
    if not match or match.group(1) != '11.8':
        raise RuntimeError(f'Expected CUDA compiler 11.8 at {nvcc}; got: {output}')
    print(output, flush=True)
    return match.group(1)


def setup(root):
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    env = root/'env'
    ready = root/'runtime_ready.json'
    if ready.exists() and json.loads(ready.read_text()).get('setup_version') == SETUP_VERSION:
        check_cuda(env, os.environ.copy())
        return str(env/'bin/python')
    if shutil.disk_usage(root).free < 20*2**30:
        raise RuntimeError('Reserve at least 20 GiB for the compiler/runtime, plus extracted dataset storage')
    archive = root/'micromamba.tar.bz2'
    if not archive.exists():
        urllib.request.urlretrieve('https://micro.mamba.pm/api/micromamba/linux-64/2.0.5', archive)
    with tarfile.open(archive) as tf:
        item = tf.getmember('bin/micromamba')
        if not item.isfile():
            raise ValueError('Unexpected micromamba archive')
        (root/'bin').mkdir(exist_ok=True)
        with tf.extractfile(item) as src, open(root/'bin/micromamba', 'wb') as dst:
            shutil.copyfileobj(src, dst)
    mamba = root/'bin/micromamba'; mamba.chmod(0o755)
    build_env = dict(os.environ, MAMBA_ROOT_PREFIX=str(root/'mamba'),
                     CUDA_HOME=str(env), CUDA_PATH=str(env), CUDACXX=str(env/'bin/nvcc'), MAX_JOBS='2', OMP_NUM_THREADS='4')
    def run(args):
        subprocess.run([str(a) for a in args], env=build_env, check=True)
    # A toolkit metapackage alone permits newer transitive CUDA components.
    # The release label plus strict priority selects the matching component set.
    # Install into an existing prefix too, repairing failed older setup attempts.
    action = 'install' if (env/'conda-meta').exists() else 'create'
    run([mamba, action, '-y', '-p', env, '--override-channels', '--strict-channel-priority',
         '-c', CUDA_CHANNEL, '-c', 'conda-forge',
         'python=3.10.14', 'pip=24.0', 'cuda-toolkit=11.8.0', 'cuda-nvcc=11.8.89',
         'openblas=0.3.27', 'gxx_linux-64=11.4.0', 'ninja=1.12.1'])
    py = env/'bin/python'
    build_env['PATH'] = str(env/'bin')+os.pathsep+os.environ['PATH']
    build_env['LD_LIBRARY_PATH'] = str(env/'lib')+os.pathsep+os.environ.get('LD_LIBRARY_PATH', '')
    build_env['CC'] = str(env/'bin/x86_64-conda-linux-gnu-cc')
    build_env['CXX'] = str(env/'bin/x86_64-conda-linux-gnu-c++')
    check_cuda(env, build_env)
    run([py, '-m', 'pip', 'install', 'setuptools==68.2.2', 'wheel==0.43.0', 'numpy==1.26.4'])
    run([py, '-m', 'pip', 'install', 'torch==2.1.2', 'torchvision==0.16.2',
         '--index-url', 'https://download.pytorch.org/whl/cu118'])
    run([py, '-m', 'pip', 'install', 'timm==0.9.16', 'pandas==2.2.2', 'Pillow==10.4.0',
         'matplotlib==3.8.4'])
    run([py, '-c', 'import torch; from torch.utils.cpp_extension import CUDA_HOME; '
         'import os; assert torch.version.cuda == \"11.8\"; '
         'assert CUDA_HOME == os.environ[\"CUDA_HOME\"]; print(\"Build CUDA_HOME:\", CUDA_HOME)'])
    engine = root/'MinkowskiEngine'
    revision = 'bbc30ef581ea6deb505976b663f5fc2358a83749'
    if not engine.exists():
        run(['git', 'clone', 'https://github.com/shwoo93/MinkowskiEngine.git', engine])
    run(['git', '-C', engine, 'checkout', '--detach', revision])
    patch_engine_headers(engine)
    patch_engine_python(engine)
    # The Meta-pinned fork, not the NVIDIA upstream branch. Setup options must
    # reach setup.py directly; wheel builds otherwise discard these flags.
    subprocess.run([str(py), 'setup.py', 'install', '--force_cuda', '--blas=openblas',
                    '--blas_include_dirs='+str(env/'include'), '--blas_library_dirs='+str(env/'lib')],
                   cwd=engine, env=build_env, check=True)
    run([py, '-c', 'import torch, MinkowskiEngine; assert torch.cuda.is_available(); print(torch.__version__, MinkowskiEngine.__version__)'])
    # micromamba 2.x exposes explicit lock export under ``env export``;
    # ``list --explicit`` is conda syntax and exits 109 after an otherwise
    # successful runtime build.
    lock = subprocess.check_output(
        [str(mamba), 'env', 'export', '-p', str(env), '--explicit'],
        env=build_env,
        text=True,
    )
    if '@EXPLICIT' not in lock.splitlines():
        raise RuntimeError('micromamba did not produce an explicit environment lock')
    (root/'conda-explicit.txt').write_text(lock)
    (root/'pip-freeze.txt').write_text(subprocess.check_output([str(py), '-m', 'pip', 'freeze'], env=build_env, text=True))
    ready.write_text(json.dumps(dict(python=str(py), engine_revision=revision,
                                      setup_version=SETUP_VERSION, cuda_channel=CUDA_CHANNEL,
                                      engine_patches=['spmm-complete-tensor-header', 'collections-abc-five-files',
                                                      'retain-ninja-build-objects']), indent=2))
    return str(py)


if __name__ == '__main__':
    import sys
    print(setup(sys.argv[1]))
