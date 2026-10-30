"""Generate the golden H3 takes through a running ComfyUI, then extract them with GEM.

python -m tests.h3_takes queue     submit every take that has no receipt yet
python -m tests.h3_takes collect   for finished takes: copy video, split panels, GEM, import to golden/motions
python -m tests.h3_takes status    show where each take is
python -m tests.h3_takes reimport  rebuild golden H3 motions from existing extractions
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from urllib.request import Request, urlopen

from motionbridge import config as configuration
import torch

from motionbridge.core import head_pose, soma
from motionbridge.gem.extract import extract

ROOT = configuration.ROOT
# Off: fitting GEM's head to ViTPose face points tilted heads further DOWN on the mesh (checked on
# the walk against the H3 frame). Kept for a later, face-model-based head pose.
HEAD_POSE = False
# H3 animates an orthographic reference render, but GEM assumes a ~53 degree lens, which hunched every
# take. 8x its focal length approximates the orthographic camera (tested 1x/4x/8x/16x on walk_diagonal).
FOCAL_SCALE = 8.0
EXTRACTION = 'extraction_f8'
PANELS = {'front + 3/4': ['front', 'threequarter'], 'front + 3/4 + hands': ['front', 'threequarter', 'hands']}


def api(config, path, data=None):
    request = Request(config['paths']['comfy_url'] + path, data=json.dumps(data).encode() if data is not None else None,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=60) as response:
        body = response.read()
        return json.loads(body) if body.strip() else {}  # /free answers with an empty body


SPEC = Path(os.environ.get('H3_TAKES', ROOT / 'golden' / 'h3_takes.toml'))  # another takes file (H3_TAKES=...)


def takes():
    spec = tomllib.loads(SPEC.read_text(encoding='utf-8'))
    return spec['defaults'], spec['takes']


def motions_dir():
    """Where imported motions go: the takes file's own import_dir, else golden/motions."""
    d = takes()[0].get('import_dir')
    return Path(d) if d else ROOT / 'golden' / 'motions'


def folder(config, take):
    return Path(config['paths']['runs']) / 'h3' / take['name']


def queue(config, only=None):
    defaults, all_takes = takes()
    all_takes = [t for t in all_takes if not only or t['name'] in only]
    template = json.loads((ROOT / 'workflows' / 'h3_dual_view_api.json').read_text())
    for take in all_takes:
        out = folder(config, take)
        if (out / 'queue_receipt.json').exists():
            print(f"{take['name']}: already queued")
            continue
        out.mkdir(parents=True, exist_ok=True)
        intro = defaults['three_view_intro'] if take['layout'].endswith('hands') else defaults['two_view_intro']
        prompt = json.loads(json.dumps(template))
        prompt['1']['inputs'].update(asset_file=defaults.get('asset_file') or config['paths']['performer'], layout=take['layout'])
        prompt['10']['inputs'].update(prompt=' '.join([intro, take['action'], defaults['outro']]),
                                      length=take.get('frames', defaults['frames']))
        prompt['4']['inputs']['unet_name'] = defaults['unet_name']
        if defaults.get('no_attention_patch'):  # an install without KJNodes' attention patch: model straight through
            patch = next(k for k, v in prompt.items() if v['class_type'] == 'MiniMaxH3MemoryEfficientSageAttentionPatch')
            src = prompt.pop(patch)['inputs']['model']
            for v in prompt.values():
                for k, x in v['inputs'].items():
                    if isinstance(x, list) and x and x[0] == patch:
                        v['inputs'][k] = src
        prompt['11']['inputs']['noise_seed'] = defaults['seed']
        prompt['19']['inputs']['filename_prefix'] = f"video/MotionBridge_{defaults.get('prefix', 'golden')}_{take['name']}"
        (out / 'prompt_api.json').write_text(json.dumps(prompt, indent=1))
        receipt = api(config, '/prompt', {'prompt': prompt, 'client_id': 'motionbridge-golden'})
        if receipt.get('node_errors'):
            # ComfyUI still runs the valid outputs (here just the reference render), so stop loudly.
            (out / 'rejected.json').write_text(json.dumps(receipt, indent=1))
            raise SystemExit(f"{take['name']}: ComfyUI rejected the graph: "
                             f"{json.dumps(receipt['node_errors'])[:600]}")
        (out / 'queue_receipt.json').write_text(json.dumps(receipt, indent=1))
        print(f"{take['name']}: queued {receipt['prompt_id']}")


def finished_video(config, take):
    pid = json.loads((folder(config, take) / 'queue_receipt.json').read_text())['prompt_id']
    history = api(config, '/history/' + pid)
    if pid not in history:
        return None
    job = history[pid]
    if job['status']['status_str'] != 'success':
        raise RuntimeError(f"{take['name']} failed in ComfyUI: {json.dumps(job['status'])[:800]}")
    files = [v for node in job['outputs'].values() for values in node.values() if isinstance(values, list)
             for v in values if isinstance(v, dict) and str(v.get('filename', '')).endswith('.mp4')]
    f = files[0]
    return Path(config['paths']['comfy_output']) / f.get('subfolder', '') / f['filename']


def collect(config):
    defaults, all_takes = takes()
    ready = []
    for take in all_takes:
        out = folder(config, take)
        if not (out / 'queue_receipt.json').exists():
            continue
        if not (out / 'raw_h3.mp4').exists():
            video = finished_video(config, take)
            if video is None:
                print(f"{take['name']}: still generating")
                continue
            shutil.copy2(video, out / 'raw_h3.mp4')
        ready.append(take)
    if not ready:
        return
    state = api(config, '/queue')
    if state['queue_running'] or state['queue_pending']:
        print('ComfyUI is still busy; GEM waits so both never share the GPU. Run collect again when it finishes.')
        return
    api(config, '/free', {'unload_models': True, 'free_memory': True})
    for take in ready:
        out = folder(config, take)
        views = PANELS[take['layout']]
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                'stream=width,height', '-of', 'json', str(out / 'raw_h3.mp4')],
                               capture_output=True, text=True, check=True)
        size = json.loads(probe.stdout)['streams'][0]
        width = size['width'] // len(views)
        for i, view in enumerate(views):
            panel = out / view / 'source.mp4'
            panel.parent.mkdir(exist_ok=True)
            if not panel.exists():
                subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(out / 'raw_h3.mp4'), '-vf',
                                f"crop={width}:{size['height']}:{i * width}:0", '-an', '-c:v', 'libx264', '-crf', '0',
                                '-pix_fmt', 'yuv420p', str(panel)], check=True)
            if view == 'hands':
                continue  # the hand panel feeds the finger pass later, not body extraction
            started = time.time()
            motion_file = extract(panel, out / view / EXTRACTION, config['paths']['gem_root'],
                                  config['paths']['gem_python'], FOCAL_SCALE)
            print(f"{take['name']}/{view}: GEM done in {time.time() - started:.0f} s")
        import_take(config, take)


def import_take(config, take):
    """Front-panel GEM motion -> golden motion, with the head pitch corrected against the 2D face."""
    extraction = folder(config, take) / 'front' / EXTRACTION
    motion = soma.from_gem_export(extraction / 'motion_export.npz', 24)
    if HEAD_POSE:
        keypoints = torch.load(extraction / 'capture' / 'preprocess' / 'vitpose.pt').numpy()
        motion, report = head_pose.refine(motion, extraction / 'motion_export.npz', keypoints)
        (folder(config, take) / 'head_pose.json').write_text(json.dumps(report, indent=1))
    motions_dir().mkdir(parents=True, exist_ok=True)
    motion.save(motions_dir() / f"h3_{take['name']}.npz")
    print(f"{take['name']}: imported {motions_dir() / ('h3_' + take['name'] + '.npz')}")


def reimport(config):
    """Rebuild every golden H3 motion from its existing extraction (no GPU)."""
    _, all_takes = takes()
    for take in all_takes:
        if (folder(config, take) / 'front' / EXTRACTION / 'motion_export.npz').exists():
            import_take(config, take)


def status(config):
    _, all_takes = takes()
    for take in all_takes:
        out = folder(config, take)
        stage = ('imported' if (motions_dir() / f"h3_{take['name']}.npz").exists() else
                 'video collected' if (out / 'raw_h3.mp4').exists() else
                 'queued' if (out / 'queue_receipt.json').exists() else 'not queued')
        print(f"{take['name']}: {stage}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['queue', 'collect', 'status', 'reimport'])
    p.add_argument('--only', nargs='*', help='take names (queue only)')
    a = p.parse_args()
    config = configuration.load(os.environ.get('MB_CONFIG'))  # another config.toml (MB_CONFIG=...)
    if a.command == 'queue':
        queue(config, a.only)
    else:
        {'collect': collect, 'status': status, 'reimport': reimport}[a.command](config)


if __name__ == '__main__':
    main()
