"""Video -> GEM motion_export.npz, one stage per subprocess, resumable.

python -m motionbridge.gem.extract --video clip.mp4 --out DIR   (paths from config.toml)
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from motionbridge import config as configuration

STAGES = ['detect', 'pose', 'features', 'motion', 'export']


def extract(video, out, gem_root, gem_python, focal_scale=1.0):
    video, out = Path(video).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    receipt = out / 'input.json'
    if receipt.exists() and json.loads(receipt.read_text())['sha256'] != digest:
        raise ValueError(f'{out} holds an extraction of different video bytes; choose a new folder')
    receipt.write_text(json.dumps({'video': video.as_posix(), 'sha256': digest}, indent=1))
    backend = Path(__file__).with_name('backend.py')
    for stage in STAGES:
        done = out / f'{stage}.done'
        if done.exists():
            continue
        print('GEM stage', stage, flush=True)
        with (out / f'{stage}.log').open('w') as log:
            result = subprocess.run([gem_python, str(backend), '--gem-root', gem_root, '--video', str(video),
                                     '--out', str(out), '--stage', stage, '--focal-scale', str(focal_scale)], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'GEM stage {stage} failed:\n' + (out / f'{stage}.log').read_text()[-2000:])
        done.write_text('complete\n')
    return out / 'motion_export.npz'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    paths = configuration.load()['paths']
    print(extract(a.video, a.out, paths['gem_root'], paths['gem_python']))


if __name__ == '__main__':
    main()
