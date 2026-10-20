"""Video -> GEM motion_export.npz, one stage per subprocess, resumable.

python -m motionbridge.gem.extract --video clip.mp4 --out DIR   (paths from config.toml)
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from motionbridge import config as configuration

STAGES = ['detect', 'pose', 'features', 'motion', 'export']


def extract(video, out, gem_root, gem_python, focal_scale=1.0, env=None, interrupted=None):
    """Run GEM on video into out/, resuming finished stages. env: environment for the GEM process.
    interrupted: called while a stage runs; raise from it to cancel (the stage process is killed)."""
    video, out = Path(video).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    receipt = out / 'input.json'
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old['sha256'] != digest or old.get('focal_scale', 1.0) != focal_scale:
            raise ValueError(f'{out} holds an extraction of different video bytes or lens; choose a new folder')
    receipt.write_text(json.dumps({'video': video.as_posix(), 'sha256': digest, 'focal_scale': focal_scale}, indent=1))
    backend = Path(__file__).with_name('backend.py')
    for stage in STAGES:
        done = out / f'{stage}.done'
        if done.exists():
            continue
        print('GEM stage', stage, flush=True)
        with (out / f'{stage}.log').open('w') as log:
            process = subprocess.Popen([str(gem_python), str(backend), '--gem-root', str(gem_root), '--video', str(video),
                                        '--out', str(out), '--stage', stage, '--focal-scale', str(focal_scale)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            try:
                while process.poll() is None:
                    if interrupted:
                        interrupted()
                    time.sleep(0.2)
            except BaseException:
                process.kill()
                process.wait()
                raise
        if process.returncode:
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
