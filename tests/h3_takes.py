"""Generate the golden H3 takes through a running ComfyUI, then extract them with GEM.

python -m tests.h3_takes queue     submit every take that has no receipt yet
python -m tests.h3_takes collect   for finished takes: copy video, split panels, GEM, import to golden/motions
python -m tests.h3_takes status    show where each take is
"""
import argparse
import json
import shutil
import subprocess
import time
import tomllib
from pathlib import Path
from urllib.request import Request, urlopen

from motionbridge import config as configuration
from motionbridge.core import soma
from motionbridge.gem.extract import extract

ROOT = configuration.ROOT
PANELS = {'front + 3/4': ['front', 'threequarter'], 'front + 3/4 + hands': ['front', 'threequarter', 'hands']}


def api(config, path, data=None):
    request = Request(config['paths']['comfy_url'] + path, data=json.dumps(data).encode() if data is not None else None,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=60) as response:
        body = response.read()
        return json.loads(body) if body.strip() else {}  # /free answers with an empty body


def takes():
    spec = tomllib.loads((ROOT / 'golden' / 'h3_takes.toml').read_text(encoding='utf-8'))
    return spec['defaults'], spec['takes']


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
        prompt['1']['inputs'].update(asset_file=defaults['asset_file'], layout=take['layout'])
        prompt['10']['inputs'].update(prompt=' '.join([intro, take['action'], defaults['outro']]),
                                      length=defaults['frames'])
        prompt['4']['inputs']['unet_name'] = defaults['unet_name']
        prompt['11']['inputs']['noise_seed'] = defaults['seed']
        prompt['19']['inputs']['filename_prefix'] = f"video/MotionBridge_golden_{take['name']}"
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
            motion_file = extract(panel, out / view / 'extraction', config['paths']['gem_root'],
                                  config['paths']['gem_python'])
            print(f"{take['name']}/{view}: GEM done in {time.time() - started:.0f} s")
        motion = soma.from_gem_export(out / 'front' / 'extraction' / 'motion_export.npz', 24)
        motion.save(ROOT / 'golden' / 'motions' / f"h3_{take['name']}.npz")
        print(f"{take['name']}: imported golden/motions/h3_{take['name']}.npz")


def status(config):
    _, all_takes = takes()
    for take in all_takes:
        out = folder(config, take)
        stage = ('imported' if (ROOT / 'golden' / 'motions' / f"h3_{take['name']}.npz").exists() else
                 'video collected' if (out / 'raw_h3.mp4').exists() else
                 'queued' if (out / 'queue_receipt.json').exists() else 'not queued')
        print(f"{take['name']}: {stage}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['queue', 'collect', 'status'])
    p.add_argument('--only', nargs='*', help='take names (queue only)')
    a = p.parse_args()
    config = configuration.load()
    if a.command == 'queue':
        queue(config, a.only)
    else:
        {'collect': collect, 'status': status}[a.command](config)


if __name__ == '__main__':
    main()
