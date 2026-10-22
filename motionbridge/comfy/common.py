"""Shared by the MotionBridge nodes: where the runtime lives, running Blender, caching by content."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import comfy.model_management
import folder_paths

from motionbridge.runtime.install import Runtime

PACKAGE = Path(__file__).resolve().parents[1]
BLENDER_SCRIPTS = PACKAGE / 'blender'
CHARACTER_FILES = ('.blend', '.glb', '.gltf', '.fbx')


def runtime(check=True):
    """The runtime folder: MOTIONBRIDGE_HOME, else ComfyUI/models/motionbridge."""
    home = os.environ.get('MOTIONBRIDGE_HOME') or Path(folder_paths.models_dir) / 'motionbridge'
    rt = Runtime(home)
    if check:
        gaps = rt.missing()
        if gaps:
            raise RuntimeError('MotionBridge is not set up yet (missing: ' + ', '.join(gaps) +
                               '). Run the "MotionBridge Setup" node once.')
    return rt


def output_folder(*parts):
    folder = Path(folder_paths.get_output_directory()).joinpath('motionbridge', *parts)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def digest(*items):
    """Content hash of files (by bytes) and plain values; used for cache folders."""
    h = hashlib.sha256()
    for item in items:
        if isinstance(item, Path) and item.is_file():
            with item.open('rb') as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    h.update(block)
        else:
            h.update(json.dumps(item, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]


def resolve_input(value, suffixes=CHARACTER_FILES):
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = Path(folder_paths.get_input_directory()) / p
    p = p.resolve()
    if not p.is_file():
        raise ValueError(f'File not found: {value}. Use the upload button or a path relative to ComfyUI/input.')
    if p.suffix.lower() not in suffixes:
        raise ValueError('Supported files: ' + ', '.join(s.upper().lstrip('.') for s in suffixes))
    return p


def run_blender(script, args, log_file, rt=None, on_line=None):
    """Blender in the background on one of our scripts. Cancel in ComfyUI stops it."""
    rt = rt or runtime()
    command = [rt.blender(), '--background', '--factory-startup', '--python-exit-code', '1',
               '--python', str(BLENDER_SCRIPTS / script), '--'] + [str(a) for a in args]
    run_process(command, log_file, f'Blender ({script})', on_line=on_line)


def run_process(command, log_file, label, env=None, on_line=None):
    """A subprocess with its output in log_file. Cancel in ComfyUI stops it.
    on_line(line) sees each output line as it arrives (progress reporting)."""
    log_file = Path(log_file)
    with log_file.open('w', encoding='utf-8') as log:
        process = subprocess.Popen([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        seen = 0
        try:
            while process.poll() is None:
                comfy.model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.2)
                if on_line:
                    text = log_file.read_text(encoding='utf-8', errors='replace')
                    for line in text[seen:].splitlines(keepends=True):
                        if line.endswith('\n'):
                            seen += len(line)
                            on_line(line.strip())
        except BaseException:
            process.kill()
            process.wait()
            raise
    if process.returncode:
        raise RuntimeError(f'{label} failed:\n' + log_file.read_text(encoding='utf-8', errors='replace')[-2500:])


def free_vram():
    """Unload ComfyUI's models (e.g. H3) so GEM and Blender get the GPU; 8 GB cards need this."""
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
