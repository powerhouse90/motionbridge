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

import cv2
import numpy as np

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


ROLE_COLORS = {'Left': render.LEFT, 'Right': render.RIGHT}
TRUNK, AUX, EXTRA, UNUSED = (235, 235, 235), (60, 200, 230), (110, 110, 110), (55, 55, 55)


def anonymize(dump):
    """Same rig with meaningless bone names, to prove the mapper does not depend on naming."""
    rename = {b['name']: f'b{i}' for i, b in enumerate(dump['bones'])}
    bones = [dict(b, name=rename[b['name']], parent=rename.get(b['parent']),
                  children=[rename[c] for c in b['children']]) for b in dump['bones']]
    return dict(dump, bones=bones), rename


def maps_report(config):
    from motionbridge.core import rigmap
    out = Path(config['paths']['runs']) / 'golden'
    (out / 'maps').mkdir(parents=True, exist_ok=True)
    panels, summary = [], {}
    for rig in config['rigs']:
        source = out / 'rigs' / f"{rig['name']}.json"
        if not source.is_file():
            print(f"{rig['name']}: no rig dump; run `python -m tests.golden rigs`")
            continue
        dump = json.loads(source.read_text())
        m = rigmap.build(dump)
        (out / 'maps' / f"{rig['name']}.json").write_text(json.dumps(m.to_json(), indent=1))

        blind_dump, rename = anonymize(dump)
        blind = rigmap.build(blind_dump)
        flat = lambda v: tuple(v) if isinstance(v, list) else v
        named_roles = {r: flat(v) for r, v in m.roles.items()}
        blind_roles = {r: flat([rename[x] for x in v] if isinstance(v, list) else rename[v])
                       for r, v in m.roles.items()}
        blind_got = {r: flat(v) for r, v in blind.roles.items()}
        agree = [r for r in named_roles if blind_got.get(r) == blind_roles[r]]
        # A wrong answer is acceptable only if the mapper said it was unsure.
        silent = [r for r in named_roles if r not in agree and blind.confidence.get(r, 0) >= 0.8]
        fingers = sum(1 for r in m.roles if 'Hand' in r and r[-4:] != 'Hand')
        low = {r: c for r, c in m.confidence.items() if c < 0.8}
        summary[rig['name']] = {'roles': len(m.roles), 'missing': m.missing(), 'finger_chains': fingers,
                                'name_blind_agreement': f'{len(agree)}/{len(named_roles)}',
                                'name_blind_differs': sorted(set(named_roles) - set(agree)),
                                'name_blind_silent_errors': silent,
                                'low_confidence': low, 'notes': m.notes}
        print(f"{rig['name']}: {len(m.roles)} roles, {fingers} finger chains, missing {m.missing() or 'none'}, "
              f"name-blind agreement {len(agree)}/{len(named_roles)} "
              f"({'OK' if not silent else 'SILENT ERRORS ' + ', '.join(silent)}), low confidence {low or 'none'}")
        for note in m.notes:
            print('   note:', note)

        role_of = {}
        for role, bones in m.roles.items():
            for b in (bones if isinstance(bones, list) else [bones]):
                side = next((s for s in ROLE_COLORS if role.startswith(s)), None)
                role_of[b] = ROLE_COLORS[side] if side else TRUNK
        for b in m.hips_aux:
            role_of[b] = AUX
        for b in m.extras:
            role_of.setdefault(b, EXTRA)
        segs = [(np.array(b['head']), np.array(b['tail'])) for b in dump['bones']]
        colors = [role_of.get(b['name'], UNUSED) for b in dump['bones']]
        mapped = [i for i, b in enumerate(dump['bones']) if colors[i] not in (EXTRA, UNUSED)]
        panels.append(render.segments_panel(segs, colors, m.left or [1, 0, 0], m.forward or [0, -1, 0],
                                            [0, 0, 1], m.ground, rig['name'], on_top=mapped))
    (out / 'maps.json').write_text(json.dumps(summary, indent=1))
    if panels:
        cv2.imwrite(str(out / 'maps.png'), np.hstack(panels))
        print(f'wrote {out / "maps.png"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['import', 'motions', 'rigs', 'map', 'all'])
    p.add_argument('--config')
    a = p.parse_args()
    config = configuration.load(a.config)
    if a.command == 'import':
        import_motions(config)
    if a.command in ('motions', 'all'):
        motions_report(config)
    if a.command in ('rigs', 'all'):
        rigs_report(config)
    if a.command in ('map', 'all'):
        maps_report(config)


if __name__ == '__main__':
    main()
