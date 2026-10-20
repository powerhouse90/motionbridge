"""Golden-set runner: every change is judged on the same motions and rigs.

python -m tests.golden import    copy GEM exports listed in config.toml into golden/motions
python -m tests.golden motions   metrics + contact sheet for every golden motion
python -m tests.golden rigs      inspect every rig listed in config.toml with Blender
python -m tests.golden all       motions + rigs
"""
import argparse
import json
import subprocess
from pathlib import Path

from motionbridge import config as configuration
from motionbridge.core import metrics, soma
from tests import render

ROOT = configuration.ROOT
MOTIONS = ROOT / 'golden' / 'motions'
SOMA_UP, SOMA_FORWARD = 1, 2


def import_motions(config):
    MOTIONS.mkdir(parents=True, exist_ok=True)
    if not config['motions']:
        raise SystemExit('No [[motions]] entries in config.toml')
    for m in config['motions']:
        motion = soma.from_gem_export(m['gem_export'], m['fps'])
        motion.save(MOTIONS / f"{m['name']}.npz")
        print(f"imported {m['name']}: {motion.frames} frames at {motion.fps} fps")


def measure_soma(motion):
    p = motion.positions
    feet = [motion.index(n) for n in ('LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase')]
    ground = metrics.ground_height(p, feet, SOMA_UP)
    result = {'frames': motion.frames, 'fps': motion.fps, 'ground_m': round(ground, 4),
              'jitter_m_s2': round(metrics.jitter(p, motion.fps), 3)}
    for side in ('Left', 'Right'):
        speed, contact = metrics.foot_slide(p, motion.index(side + 'ToeBase'), SOMA_UP, motion.fps, ground)
        result[side.lower() + '_foot_slide_m_s'] = round(speed, 4)
        result[side.lower() + '_foot_contact'] = round(contact, 3)
    direct, path = metrics.root_travel(p, motion.index('Hips'), SOMA_UP)
    result['root_travel_m'] = [round(direct, 3), round(path, 3)]
    return result, ground


def motions_report(config):
    files = sorted(MOTIONS.glob('*.npz'))
    if not files:
        raise SystemExit('No golden motions; run `python -m tests.golden import` first')
    out = Path(config['paths']['runs']) / 'golden'
    out.mkdir(parents=True, exist_ok=True)
    report, rows, fps = {}, [], 24
    for f in files:
        motion = soma.load(f)
        report[f.stem], ground = measure_soma(motion)
        rows.append(render.clip_frames(motion.positions, motion.parents, motion.names, SOMA_UP, SOMA_FORWARD,
                                       ground, f.stem))
        fps = motion.fps
    (out / 'motions.json').write_text(json.dumps(report, indent=1))
    render.write_video(out / 'motions.mp4', rows, fps)
    render.write_strip(out / 'motions.png', rows)
    for name, r in report.items():
        print(name, r)
    print(f'wrote {out / "motions.mp4"} and {out / "motions.png"}')


def rigs_report(config):
    out = Path(config['paths']['runs']) / 'golden' / 'rigs'
    out.mkdir(parents=True, exist_ok=True)
    script = ROOT / 'motionbridge' / 'blender' / 'dump_rig.py'
    for rig in config['rigs']:
        target = out / f"{rig['name']}.json"
        command = [config['paths']['blender'], '-b', '--factory-startup', '--python-exit-code', '1',
                   '--python', str(script), '--', '--file', rig['file'], '--out', str(target)]
        if rig.get('armature'):
            command += ['--armature', rig['armature']]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode:
            print(f"FAILED {rig['name']}:\n{(done.stdout + done.stderr)[-1500:]}")
            continue
        data = json.loads(target.read_text())
        heads = [b['head'] for b in data['bones']] + [b['tail'] for b in data['bones']]
        height = max(h[2] for h in heads) - min(h[2] for h in heads)
        print(f"{rig['name']}: armature {data['armature']!r}, {len(data['bones'])} bones "
              f"({sum(b['deform'] for b in data['bones'])} deform), skeleton height {height:.2f} m, "
              f"{len(data['meshes'])} skinned meshes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['import', 'motions', 'rigs', 'all'])
    p.add_argument('--config')
    a = p.parse_args()
    config = configuration.load(a.config)
    if a.command == 'import':
        import_motions(config)
    if a.command in ('motions', 'all'):
        motions_report(config)
    if a.command in ('rigs', 'all'):
        rigs_report(config)


if __name__ == '__main__':
    main()
