"""Golden-set runner: every change is judged on the same motions and rigs.

python -m tests.golden import    copy GEM exports listed in config.toml into golden/motions
python -m tests.golden motions   metrics + contact sheet for every golden motion
python -m tests.golden rigs      inspect every rig listed in config.toml with Blender
python -m tests.golden all       motions + rigs
python -m tests.golden compare h3_crouch [h3_jump ...]   GEM's own body next to every baked character,
                                 tracking profile and front views (run retarget and bake first)
"""
import argparse
import json
import subprocess
import sys
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


def rig_colors(dump, m):
    """Per-bone color: side for limbs, white trunk, cyan pelvis aux, gray extras, dark unused."""
    role_of = {}
    for role, bones in m.roles.items():
        for b in (bones if isinstance(bones, list) else [bones]):
            side = next((s for s in ROLE_COLORS if role.startswith(s)), None)
            role_of[b] = ROLE_COLORS[side] if side else TRUNK
    for b in m.hips_aux:
        role_of[b] = AUX
    for b in m.extras:
        role_of.setdefault(b, EXTRA)
    return [role_of.get(b['name'], UNUSED) for b in dump['bones']]


def retarget_report(config):
    from motionbridge.core import retarget, rigmap
    out = Path(config['paths']['runs']) / 'golden' / 'retarget'
    out.mkdir(parents=True, exist_ok=True)
    rigs = []
    for rig in config['rigs']:
        source = Path(config['paths']['runs']) / 'golden' / 'rigs' / f"{rig['name']}.json"
        if source.is_file():
            dump = json.loads(source.read_text())
            rigs.append((rig['name'], dump, rigmap.build(dump)))
    report = {}
    for f in sorted(MOTIONS.glob('*.npz')):
        motion = soma.load(f)
        world, left, fwd = retarget.soma_facing(motion)
        pos = world[..., :3, 3]
        p = [max(i, 0) for i in motion.parents]
        keep = [i for i, q in enumerate(motion.parents) if q > 0]
        _, ground = measure_soma(motion)
        columns = [render.animated_segments(pos[:, keep], pos[:, [p[i] for i in keep]],
                                            [render.side_colors(motion.names)[i] for i in keep],
                                            left, fwd, SOMA_UP_VEC, ground, f'{f.stem} SOMA')]
        report[f.stem] = {}
        for name, dump, m in rigs:
            before = retarget.retarget(motion, dump, m, feet_ik=False)
            result = retarget.retarget(motion, dump, m)
            result.save(out / f'{name}__{f.stem}.npz')
            before.save(out / f'{name}__{f.stem}_noik.npz')
            report[f.stem][name] = {'before_ik': measure_rig(before, m, dump, result.locks),
                                    'after_ik': measure_rig(result, m, dump, result.locks)}
            columns.append(render.animated_segments(result.heads, result.tails, rig_colors(dump, m), m.left,
                                                    m.forward, m.up, m.ground, name))
        render.write_grid(out / f'{f.stem}.png', columns)
        render.write_video(out / f'{f.stem}.mp4', [[np.hstack(x) for x in zip(*columns)]], motion.fps)
        print(f.stem)
        for n, r in report[f.stem].items():
            b, a = r['before_ik'], r['after_ik']
            print(f"   {n:17s} planted slide {b['planted_slide_m_s']:.3f} -> {a['planted_slide_m_s']:.3f} m/s "
                  f"(p95 {b['planted_slide_p95_m_s']:.3f} -> {a['planted_slide_p95_m_s']:.3f}), worst floor "
                  f"{b['planted_floor_worst_m']:+.3f} -> {a['planted_floor_worst_m']:+.3f} m  {'; '.join(a['notes'])}")
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    print(f'wrote sheets and videos to {out}')


SOMA_UP_VEC = [0., 1., 0.]
MESH_MOTIONS = ['h3_walk', 'h3_jump', 'h3_crouch', 'arthur_idle']


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (20, 20, 20), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def meshvideo_report(config, motions=None):
    """Textured characters, before vs after foot IK, next to the H3 source video. What to judge results on."""
    runs = Path(config['paths']['runs'])
    out = runs / 'golden' / 'meshvideo'
    script = ROOT / 'motionbridge' / 'blender' / 'render_preview.py'
    motions = motions or MESH_MOTIONS
    actions = [x for m in motions for x in (m, m + '_noik')]
    for rig in config['rigs']:
        blend = runs / 'golden' / 'bake' / f"{rig['name']}.blend"
        done = subprocess.run([config['paths']['blender'], '-b', '--python-exit-code', '1', '--python', str(script),
                               '--', '--blend', str(blend), '--actions', *actions, '--outdir',
                               str(out / rig['name'])], capture_output=True, text=True)
        if done.returncode:
            print(f"FAILED {rig['name']}:\n{(done.stdout + done.stderr)[-1500:]}")
            return
        print(f"rendered {rig['name']}")
    for motion in motions:
        source = runs / 'h3' / motion.removeprefix('h3_') / 'front' / 'source.mp4'
        capture, src = cv2.VideoCapture(str(source)), []
        while source.exists():
            ok, frame = capture.read()
            if not ok:
                break
            src.append(cv2.resize(frame, (360, 480)))
        length = len(list((out / config['rigs'][0]['name'] / motion).glob('*.png')))
        src = src or [np.full((480, 360, 3), 30, np.uint8)] * length  # no H3 source for this clip
        rows = []
        for suffix, title in (('_noik', 'no IK'), ('', 'foot IK')):
            frames = []
            for t in range(len(src)):
                tiles = [label(src[t].copy(), 'H3 source' if source.exists() else 'no source video')]
                for rig in config['rigs']:
                    png = out / rig['name'] / (motion + suffix) / f'{t + 1:04d}.png'
                    tiles.append(label(cv2.imread(str(png)), f"{rig['name']} - {title}"))
                frames.append(np.hstack(tiles))
            rows.append(frames)
        render.write_video(out / f'{motion}.mp4', rows, 24)
        print(f'wrote {out / (motion + ".mp4")}')


TILE = (360, 420)
# Profile squares to the pelvis (legs, posture); front squares to the shoulders, because a posed rest can
# turn the chest from the pelvis and a pelvis-facing camera then shows the arms turned with it.
TRACK = {'profile': 'hips', 'front': 'shoulders'}


def compare_report(config, motions):
    """The check every new clip goes through: GEM's body (no retargeting) beside each textured character,
    same tracking cameras, profile row over front row, H3 source on the left."""
    from motionbridge.core import rigmap
    runs = Path(config['paths']['runs'])
    out = runs / 'golden' / 'compare'
    blender = config['paths']['blender']
    for motion in motions:
        take = runs / 'h3' / motion.removeprefix('h3_') / 'front'
        export = take / 'extraction_f8' / 'motion_export.npz'
        rigs = [r for r in config['rigs'] if (runs / 'golden' / 'bake' / f"{r['name']}.blend").exists()]
        frames = out / f'gem_{motion}.npz'
        if export.exists():
            out.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, '-m', 'tests.render_soma_mesh', 'prepare', str(export), str(frames)],
                           check=True, cwd=ROOT)
        for view in ('profile', 'front'):
            if export.exists():
                subprocess.run([blender, '-b', '--python-exit-code', '1', '--python',
                                str(ROOT / 'tests' / 'render_soma_mesh.py'), '--', '--frames', str(frames),
                                '--outdir', str(out / 'gem' / f'{motion}_{view}'), '--view', view,
                                '--track', TRACK[view]],
                               check=True, capture_output=True)
            for rig in rigs:
                roles = rigmap.build(json.loads((runs / 'golden' / 'rigs' / f"{rig['name']}.json").read_text())).roles
                done = subprocess.run([blender, '-b', '--python-exit-code', '1', '--python',
                                       str(ROOT / 'motionbridge' / 'blender' / 'render_preview.py'), '--', '--blend',
                                       str(runs / 'golden' / 'bake' / f"{rig['name']}.blend"), '--actions', motion,
                                       '--outdir', str(out / rig['name']), '--view', view, '--size', *map(str, TILE),
                                       '--hips-pair', roles['LeftLeg'], roles['RightLeg'],
                                       *(['--square-pair', roles['LeftArm'], roles['RightArm']]
                                         if TRACK[view] == 'shoulders' else [])],
                                      capture_output=True, text=True)
                if done.returncode:
                    raise RuntimeError(f"{rig['name']} {view}:\n{(done.stdout + done.stderr)[-1500:]}")
            print(f'rendered {motion} {view}')
        capture, src = cv2.VideoCapture(str(take / 'source.mp4')), []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            src.append(cv2.resize(frame, TILE))
        rows = []
        for view in ('profile', 'front'):
            tiles = [(out / 'gem' / f'{motion}_{view}', f'GEM body ({view})')]
            tiles += [(out / r['name'] / f'{motion}_{view}', r['name']) for r in rigs]
            length = len(list(tiles[-1][0].glob('*.png')))
            row = []
            for t in range(length):
                cells = [label(src[min(t, len(src) - 1)].copy(), 'H3 source') if src else np.full((TILE[1], TILE[0], 3), 30, np.uint8)]
                for folder, title in tiles:
                    png = folder / f'{t + 1:04d}.png'
                    img = cv2.resize(cv2.imread(str(png)), TILE) if png.exists() else np.full((TILE[1], TILE[0], 3), 30, np.uint8)
                    cells.append(label(img, title))
                row.append(np.hstack(cells))
            rows.append(row)
        render.write_video(out / f'{motion}.mp4', rows, 24)
        print(f'wrote {out / (motion + ".mp4")}')


def bake_report(config):
    """Key every retarget result onto its real character in Blender and check it against the core."""
    runs = Path(config['paths']['runs']) / 'golden'
    out = runs / 'bake'
    out.mkdir(parents=True, exist_ok=True)
    script = ROOT / 'motionbridge' / 'blender' / 'apply_motion.py'
    for rig in config['rigs']:
        results = sorted((runs / 'retarget').glob(f"{rig['name']}__*.npz"))
        if not results:
            print(f"{rig['name']}: no retarget results; run `python -m tests.golden retarget`")
            continue
        command = [config['paths']['blender'], '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                   str(script), '--', '--file', rig['file'], '--out', str(out / f"{rig['name']}.blend"),
                   '--fbx', str(out / f"{rig['name']}.fbx"), '--report', str(out / f"{rig['name']}.json"),
                   '--motions'] + [str(r) for r in results]
        if rig.get('armature'):
            command += ['--armature', rig['armature']]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode:
            print(f"FAILED {rig['name']}:\n{(done.stdout + done.stderr)[-1500:]}")
            continue
        report = json.loads((out / f"{rig['name']}.json").read_text())
        worst = max(a['max_head_error_m'] for a in report['actions'].values())
        print(f"{rig['name']}: {len(report['actions'])} actions, worst Blender-vs-core head error "
              f"{worst * 1000:.4f} mm, rest unchanged: {report['rest_matrices_unchanged']}")


def measure_rig(result, m, dump, locks):
    """Planted-foot slide (m/s) on the frames SOMA marks as planted: ankle on heel frames, toe on toe
    frames. Also the worst floor error on those frames (m, + floating, - sinking) and jitter."""
    index = {n: i for i, n in enumerate(result.names)}
    heads = result.heads
    speeds, floor = [], []
    for side in ('Left', 'Right'):
        joints = {'heel': index[m.roles[side + 'Foot']], 'toe': index.get(m.roles.get(side + 'ToeBase'))}
        for t in range(1, len(heads)):
            lock = locks[side][t]
            if lock is None or lock != locks[side][t - 1] or joints.get(lock) is None:
                continue
            i = joints[lock]
            speeds.append(np.linalg.norm(heads[t, i, :2] - heads[t - 1, i, :2]) * result.fps)
            floor.append(heads[t, i, 2] - dump['bones'][i]['head'][2])
    worst_floor = max(floor, key=abs) if floor else 0.0
    return {'scale': round(result.scale, 3), 'planted_slide_m_s': round(float(np.mean(speeds)) if speeds else 0, 4),
            'planted_slide_p95_m_s': round(float(np.percentile(speeds, 95)) if speeds else 0, 4),
            'planted_floor_worst_m': round(float(worst_floor), 4),
            'jitter_m_s2': round(metrics.jitter(heads, result.fps), 3), 'notes': result.notes or []}


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

        segs = [(np.array(b['head']), np.array(b['tail'])) for b in dump['bones']]
        colors = rig_colors(dump, m)
        mapped = [i for i, b in enumerate(dump['bones']) if colors[i] not in (EXTRA, UNUSED)]
        panels.append(render.segments_panel(segs, colors, m.left or [1, 0, 0], m.forward or [0, -1, 0],
                                            [0, 0, 1], m.ground, rig['name'], on_top=mapped))
    (out / 'maps.json').write_text(json.dumps(summary, indent=1))
    if panels:
        cv2.imwrite(str(out / 'maps.png'), np.hstack(panels))
        print(f'wrote {out / "maps.png"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['import', 'motions', 'rigs', 'map', 'retarget', 'bake', 'meshvideo', 'compare', 'all'])
    p.add_argument('motions', nargs='*', help='compare: golden motion names')
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
    if a.command in ('retarget', 'all'):
        retarget_report(config)
    if a.command in ('bake', 'all'):
        bake_report(config)
    if a.command in ('meshvideo', 'all'):
        meshvideo_report(config)
    if a.command == 'compare':
        compare_report(config, a.motions)


if __name__ == '__main__':
    main()
