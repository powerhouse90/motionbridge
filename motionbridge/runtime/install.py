"""One-time setup of everything MotionBridge runs outside ComfyUI's own Python.

python -m motionbridge.runtime.install --home DIR --accept-licenses [--blender download]

Standard library only, so it runs from any ComfyUI Python. Every step checks before it
downloads and resumes interrupted downloads, so running it again is cheap. Layout of DIR:

    uv/           uv package manager (builds the GEM environment, fetches its Python)
    gem-x/        GEM-X source + third_party/sam-3d-body + third_party/soma, inputs/ = weights
    env/          GEM's own Python environment (kept apart from ComfyUI's packages)
    blender/      portable Blender, unless an installed one is used
    cache/        TORCH_HOME / HF_HOME for the GEM subprocess (YOLOX detector lands here)
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
WINDOWS = os.name == 'nt'

UV_VERSION = '0.12.5'
GEM_X = ('NVlabs/GEM-X', '32992550dba114c62243fb55e361311972dce8f9')
SAM_3D_BODY = ('facebookresearch/sam-3d-body', 'b5c765a0d89d789985e186d396315e7590887b94')
SOMA_X = ('NVlabs/SOMA-X', 'e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5')
WEIGHTS_REPO, WEIGHTS_REVISION = 'nvidia/GEM-X', '5b9a69d5ed4713ac4ba2a63b335e66db87279a3e'
# (file in the weights repo, place under gem-x/, bytes)
WEIGHTS = [
    ('gem_soma.ckpt', 'inputs/pretrained/gem_soma.ckpt', 541758499),
    ('sam3d_body.ckpt', 'inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt', 2109129346),
    ('model_config.yaml', 'inputs/checkpoints/sam-3d-body-dinov3/model_config.yaml', None),
    ('vitpose.pth', 'inputs/checkpoints/vitpose/vitpose.pth', 3388483384),
    ('mhr_model.pt', 'inputs/mhr_data/mhr_model.pt', 696110248),
    ('scale_comps.pth', 'inputs/soma_data/scale_comps.pth', None),
    ('scale_mean.pth', 'inputs/soma_data/scale_mean.pth', None),
]
# SOMA body model assets live in Git LFS; the source zip only holds pointers. Images are skipped.
SOMA_LFS = ['Anny/SOMA_wrap.obj', 'Anny/base_body.obj', 'GarmentMeasurements/SOMA_wrap.obj',
            'GarmentMeasurements/mean.obj', 'GarmentMeasurements/point.npz', 'MHR/SOMA_wrap_lod1.obj',
            'MHR/base_body_lod1.obj', 'MHR/base_body_lod6.obj', 'MHR/mhr_model_lod1.pt', 'MHR/mhr_model_lod6.pt',
            'SMPL/SOMA_wrap.obj', 'SMPL/base_body.obj', 'SMPL/smpl_anim.npy', 'SMPLX/SOMA_wrap.obj',
            'SMPLX/base_body.obj', 'SOMA_neutral.npz', 'correctives_model.pt', 'example_animation.npy']
BLENDER = ('5.0.1', 'https://download.blender.org/release/Blender5.0/blender-5.0.1-windows-x64.zip')
TORCH_INDEX = 'https://download.pytorch.org/whl/cu126'

LICENSES = """MotionBridge downloads third-party software and model weights. By accepting you agree to their terms:
  GEM-X code (Apache-2.0) and GEM weights (NVIDIA Open Model License)   https://huggingface.co/nvidia/GEM-X
  SAM 3D Body weights, redistributed in the GEM-X repository (SAM License) https://github.com/facebookresearch/sam-3d-body
  SOMA body model (Apache-2.0)                                             https://github.com/NVlabs/SOMA-X
  Blender, if downloaded (GPL)                                             https://www.blender.org/about/license/
  uv (Apache-2.0 / MIT) and the Python packages in gem_requirements.txt (their own licenses)"""


class Runtime:
    """Where everything lives. Nodes use this to find GEM's Python, GEM's root and Blender."""

    def __init__(self, home):
        self.home = Path(home).resolve()
        self.uv = self.home / 'uv' / ('uv.exe' if WINDOWS else 'uv')
        self.gem_root = self.home / 'gem-x'
        self.env = self.home / 'env'
        self.gem_python = self.env / ('Scripts/python.exe' if WINDOWS else 'bin/python')
        self.cache = self.home / 'cache'
        self.state_file = self.home / 'state.json'

    def state(self):
        return json.loads(self.state_file.read_text()) if self.state_file.exists() else {}

    def remember(self, **values):
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps({**self.state(), **values}, indent=1))

    def blender(self):
        """Blender chosen at install time: the downloaded portable one, or an installed one."""
        chosen = self.state().get('blender')
        if chosen and Path(chosen).is_file():
            return chosen
        found = find_installed_blender()
        if found:
            return found
        raise RuntimeError('Blender was not found. Run MotionBridge Setup (it can download Blender).')

    def gem_env(self):
        """Environment for GEM subprocesses: its own caches, no inherited Python paths."""
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
        env.update(TORCH_HOME=str(self.cache / 'torch'), HF_HOME=str(self.cache / 'hf'), PYTHONNOUSERSITE='1')
        return env

    def missing(self):
        """What setup still has to do, in plain words; empty when ready."""
        gaps = []
        if not (self.gem_root / 'gem' / '__init__.py').exists():
            gaps.append('GEM-X source code')
        for _, place, size in WEIGHTS:
            p = self.gem_root / place
            if not p.exists() or (size and p.stat().st_size != size):
                gaps.append(f'model file {Path(place).name}')
        if not (self.gem_root / 'inputs' / 'soma_assets' / 'SOMA_neutral.npz').exists():
            gaps.append('SOMA body model assets')
        if not self.state().get('env_ready') or not self.gem_python.exists():
            gaps.append('GEM Python environment')
        try:
            self.blender()
        except RuntimeError:
            gaps.append('Blender')
        return gaps


def find_installed_blender():
    candidates = [os.environ.get('BLENDER_PATH', ''), shutil.which('blender') or '']
    if WINDOWS:
        root = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'Blender Foundation'
        candidates += [str(p) for p in sorted(root.glob('Blender */blender.exe'), reverse=True)]
    else:
        candidates += ['/Applications/Blender.app/Contents/MacOS/Blender', '/usr/bin/blender']
    return next((str(Path(p)) for p in candidates if p and Path(p).is_file()), None)


def log(message):
    print(f'[MotionBridge setup] {message}', flush=True)


def download(url, dest, size=None, progress=None):
    """Resumable HTTP download to dest (via dest.part). Skips when dest already has the right size."""
    dest = Path(dest)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    done = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={'User-Agent': 'MotionBridge-setup',
                                                   **({'Range': f'bytes={done}-'} if done else {})})
    with urllib.request.urlopen(request, timeout=60) as response:
        if done and response.status != 206:  # server ignored the range: start over
            done = 0
        total = done + int(response.headers.get('Content-Length', 0) or 0)
        with part.open('ab' if done else 'wb') as f:
            last = -1
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if progress and total:
                    progress(done, total)
                percent = int(done * 100 / total) if total else 0
                if total > 50e6 and percent // 10 != last:
                    last = percent // 10
                    log(f'  {dest.name}: {percent}% of {total / 1e9:.2f} GB')
    if size is not None and part.stat().st_size != size:
        raise RuntimeError(f'{dest.name}: expected {size} bytes, got {part.stat().st_size}; run setup again to resume')
    part.replace(dest)
    return dest


def unpack_github(repo, commit, dest, downloads):
    """GitHub source zip at a fixed commit -> dest (top folder stripped)."""
    if dest.exists() and any(dest.iterdir()):
        return
    archive = download(f'https://codeload.github.com/{repo}/zip/{commit}',
                       downloads / f"{repo.replace('/', '_')}_{commit[:8]}.zip")
    with zipfile.ZipFile(archive) as z:
        top = z.namelist()[0].split('/')[0] + '/'
        for member in z.infolist():
            if member.is_dir() or not member.filename.startswith(top):
                continue
            target = dest / member.filename[len(top):]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, target.open('wb') as out:
                shutil.copyfileobj(src, out)


def link_dir(link, target):
    """Directory link without admin rights (junction on Windows); copies if linking is refused."""
    if link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        if WINDOWS:
            subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(target)], check=True, capture_output=True)
        else:
            link.symlink_to(target, target_is_directory=True)
    except (OSError, subprocess.CalledProcessError):
        shutil.copytree(target, link)


def run(command, rt, **kwargs):
    log('  $ ' + ' '.join(str(c) for c in command))
    env = {**rt.gem_env(), 'UV_PYTHON_INSTALL_DIR': str(rt.home / 'python'), 'UV_CACHE_DIR': str(rt.home / 'uv_cache')}
    result = subprocess.run([str(c) for c in command], env=env, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(f'command failed ({result.returncode}): {command[0]} ...\n'
                           + (result.stdout + result.stderr)[-3000:])
    return result.stdout


def install(home, accept_licenses=False, blender='auto', progress=None):
    """Bring home up to date. blender: 'auto' (use an installed Blender, else download), 'download',
    or a path to blender(.exe). progress(step, steps, text) is called between steps."""
    if not accept_licenses:
        raise PermissionError('Accept the model and software licenses first.\n' + LICENSES)
    if not WINDOWS or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('Automatic setup currently supports Windows x64 with an NVIDIA GPU. See README for manual setup.')
    rt = Runtime(home)
    downloads = rt.home / 'downloads'
    steps = ['uv', 'source', 'soma assets', 'weights', 'environment', 'detector', 'blender']

    def step(i, text):
        log(f'[{i + 1}/{len(steps)}] {text}')
        if progress:
            progress(i, len(steps), text)

    step(0, 'uv package manager')
    if not rt.uv.exists():
        archive = download(f'https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip',
                           downloads / f'uv-{UV_VERSION}.zip')
        with zipfile.ZipFile(archive) as z:
            z.extractall(rt.uv.parent)

    step(1, 'GEM-X source (+ SAM 3D Body, SOMA)')
    unpack_github(*GEM_X, rt.gem_root, downloads)
    unpack_github(*SAM_3D_BODY, rt.gem_root / 'third_party' / 'sam-3d-body', downloads)
    unpack_github(*SOMA_X, rt.gem_root / 'third_party' / 'soma', downloads)

    step(2, 'SOMA body model assets (~0.9 GB)')
    assets = rt.gem_root / 'third_party' / 'soma' / 'assets'
    for name in SOMA_LFS:
        target = assets / name
        if target.exists() and target.stat().st_size > 1024 and not target.read_bytes()[:40].startswith(b'version https://git-lfs'):
            continue
        target.unlink(missing_ok=True)
        download(f'https://media.githubusercontent.com/media/{SOMA_X[0]}/{SOMA_X[1]}/assets/{name}', target)
    link_dir(rt.gem_root / 'inputs' / 'soma_assets', assets)

    step(3, 'model weights from Hugging Face (~6.7 GB)')
    for name, place, size in WEIGHTS:
        download(f'https://huggingface.co/{WEIGHTS_REPO}/resolve/{WEIGHTS_REVISION}/{name}', rt.gem_root / place, size,
                 progress=(lambda d, t, n=name: progress(3, len(steps), f'{n} {d / 1e9:.1f}/{t / 1e9:.1f} GB'))
                 if progress else None)

    step(4, 'GEM Python environment (~6 GB, several minutes)')
    if not rt.state().get('env_ready') or not rt.gem_python.exists():
        if not rt.gem_python.exists():
            run([rt.uv, 'venv', rt.env, '--python', '3.12'], rt)
        run([rt.uv, 'pip', 'install', '--python', rt.gem_python, '-r', HERE / 'gem_requirements.txt',
             '--extra-index-url', TORCH_INDEX, '--index-strategy', 'unsafe-best-match'], rt)
        run([rt.uv, 'pip', 'install', '--python', rt.gem_python, '--no-deps', '-e', rt.gem_root / 'third_party' / 'soma',
             '-e', rt.gem_root], rt)
        run([rt.gem_python, '-c', 'import torch, gem, soma; assert torch.cuda.is_available(), "CUDA not available"; '
             'print(torch.__version__, torch.cuda.get_device_name(0))'], rt, cwd=rt.gem_root)
        rt.remember(env_ready=True)
        shutil.rmtree(rt.home / 'uv_cache', ignore_errors=True)  # the wheels are installed; the cache only doubles disk use

    step(5, 'person detector (YOLOX)')
    run([rt.gem_python, '-c', 'from gem.utils.yolox_detector import _download_yolox_onnx as d; print(d())'], rt,
        cwd=rt.gem_root)

    step(6, 'Blender')
    if blender not in ('auto', 'download'):
        if not Path(blender).is_file():
            raise FileNotFoundError(f'Blender not found at {blender}')
        rt.remember(blender=str(Path(blender)))
    elif blender == 'auto' and find_installed_blender():
        rt.remember(blender=find_installed_blender())
    else:
        exe = rt.home / 'blender' / f'blender-{BLENDER[0]}-windows-x64' / 'blender.exe'
        if not exe.exists():
            with zipfile.ZipFile(download(BLENDER[1], downloads / Path(BLENDER[1]).name)) as z:
                z.extractall(rt.home / 'blender')
        rt.remember(blender=str(exe))
    shutil.rmtree(downloads, ignore_errors=True)

    gaps = rt.missing()
    if gaps:
        raise RuntimeError('Setup finished but these are still missing: ' + ', '.join(gaps))
    log(f'ready: {rt.home}  (Blender: {rt.blender()})')
    return rt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--home', required=True, help='install folder, normally ComfyUI/models/motionbridge')
    p.add_argument('--accept-licenses', action='store_true', help='agree to the third-party licenses listed by --licenses')
    p.add_argument('--licenses', action='store_true', help='print the licenses and exit')
    p.add_argument('--blender', default='auto', help="'auto', 'download', or a path to an installed Blender")
    a = p.parse_args()
    if a.licenses:
        print(LICENSES)
        return
    try:
        install(a.home, a.accept_licenses, a.blender)
    except PermissionError as e:
        sys.exit(str(e))


if __name__ == '__main__':
    main()
