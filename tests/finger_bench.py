"""Finger benchmark: known finger motion on a rig, rendered from a first-person camera, tracked,
fitted back, and scored in degrees. Works on any rig the hand finder understands.

python -m tests.finger_bench NAME RIG_FILE [--armature A] [--side Right]
Needs runs/fp_rigs/dumps/NAME.json (blender/dump_rig.py) and the MediaPipe env in runs/fp_probe/.
Writes runs/finger_bench/NAME/: frames, tracking, report.json.
"""
import argparse, json, subprocess
from pathlib import Path

import numpy as np

from motionbridge import config as configuration
from motionbridge.core import hand_fit, hands

ROOT = configuration.ROOT
MP_PYTHON = ROOT / 'runs' / 'fp_probe' / 'mp_env' / 'Scripts' / 'python.exe'
MP_MODEL = ROOT / 'runs' / 'fp_probe' / 'hand_landmarker.task'
T = 96


def motion(model):
    """Known test motion: open, spread, fist, point, count to five, trigger squeezes; the wrist turns throughout."""
    F = [f for f in hands.FINGERS if f in model.hand.fingers]
    def pose(**fingers):
        x = np.zeros(model.hand.size)
        for i, f in enumerate(F):
            x[3 + 4 * i:7 + 4 * i] = fingers.get(f, fingers.get('rest', (0.15, 0, 0.15, 0.1)))
        return x
    fist, thumb_in = (1.3, 0, 1.5, 1.0), (0.6, 0.3, 0.7, 0.6)
    opened, thumb_out = (0.05, 0, 0.05, 0.0), (-0.2, -0.3, 0.0, 0.0)
    keys = [
        (0, pose()),
        (10, pose(Index=(0.05, -0.2, 0.05, 0), Middle=opened, Ring=(0.05, 0.15, 0.05, 0), Pinky=(0.05, 0.3, 0.05, 0), Thumb=thumb_out)),
        (22, pose(rest=fist, Thumb=thumb_in)), (30, pose(rest=fist, Thumb=thumb_in)),
        (38, pose(rest=fist, Index=opened, Thumb=thumb_in)), (44, pose(rest=fist, Index=opened, Thumb=thumb_in)),
        (50, pose(rest=fist, Index=opened, Middle=opened, Thumb=thumb_in)),
        (56, pose(rest=fist, Index=opened, Middle=opened, Ring=opened, Thumb=thumb_in)),
        (62, pose(rest=opened, Thumb=thumb_in)), (68, pose(rest=opened, Thumb=thumb_out)),
        (76, pose(rest=(1.2, 0, 1.3, 0.9), Index=(0.5, 0, 0.8, 0.5), Thumb=(0.5, 0.2, 0.4, 0.3))),
        (81, pose(rest=(1.2, 0, 1.3, 0.9), Index=(0.85, 0, 1.3, 0.85), Thumb=(0.5, 0.2, 0.4, 0.3))),
        (86, pose(rest=(1.2, 0, 1.3, 0.9), Index=(0.5, 0, 0.8, 0.5), Thumb=(0.5, 0.2, 0.4, 0.3))),
        (91, pose(rest=(1.2, 0, 1.3, 0.9), Index=(0.85, 0, 1.3, 0.85), Thumb=(0.5, 0.2, 0.4, 0.3))),
        (95, pose(rest=(1.2, 0, 1.3, 0.9), Index=(0.5, 0, 0.8, 0.5), Thumb=(0.5, 0.2, 0.4, 0.3))),
    ]
    x = np.zeros((T, model.hand.size))
    for (a, xa), (b, xb) in zip(keys, keys[1:]):
        for t in range(a, b + 1):
            s = (t - a) / (b - a); s = s * s * (3 - 2 * s)
            x[t] = xa + (xb - xa) * s
    t = np.arange(T)
    x[:, :3] = np.c_[0.35 * np.sin(2 * np.pi * t / 48), 0.25 * np.sin(2 * np.pi * t / 32), 0.3 * np.sin(2 * np.pi * t / 64)]
    return x


def bend_error(model, fitted, truth):
    """Mean absolute error per finger over its three bend values, degrees; plus the wrist."""
    out = {}
    for f, (k, _) in hand_fit.finger_slices(model).items():
        out[f] = float(np.degrees(np.abs(fitted[:, [k, k + 2, k + 3]] - truth[:, [k, k + 2, k + 3]]).mean()))
    out['wrist'] = float(np.degrees(np.linalg.norm(fitted[:, :3] - truth[:, :3], axis=1).mean()))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('name'); p.add_argument('rig_file')
    p.add_argument('--armature'); p.add_argument('--side', default='Right')
    a = p.parse_args()
    paths = configuration.load()['paths']
    out = ROOT / 'runs' / 'finger_bench' / a.name; out.mkdir(parents=True, exist_ok=True)
    dump = json.loads((ROOT / 'runs' / 'fp_rigs' / 'dumps' / f'{a.name}.json').read_text())
    sk = hands.Skeleton(dump)
    hand = next(h for h in hands.find_hands(dump) if h.side == a.side)
    model = hands.HandModel(sk, hand)
    truth = motion(model)
    bones = [hand.hand] + [n for j in hand.fingers.values() for n in j]
    basis = np.array([[model.basis(x)[sk.index[n]] for n in bones] for x in truth])
    np.save(out / 'truth_basis.npy', basis); np.save(out / 'truth_params.npy', truth)
    frames = out / 'frames'
    if not (frames / 'truth_uv.npy').exists():
        req = {'file': str(Path(a.rig_file).resolve()), 'armature': a.armature or dump['armature'], 'bones': bones,
               'basis': str(out / 'truth_basis.npy'), 'keypoints': hand.keypoint_bones(), 'outdir': str(frames),
               'size': [960, 540], 'side': a.side, 'palm_normal': model.palm_normal.tolist()}
        (out / 'request.json').write_text(json.dumps(req))
        r = subprocess.run([paths['blender'], '-b', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1',
                            '--python', str(ROOT / 'tests' / 'finger_bench_render.py'), '--', str(out / 'request.json')],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
    camera = json.loads((frames / 'camera.json').read_text())
    truth_uv = np.load(frames / 'truth_uv.npy')[..., :2]
    ours = np.array([hand_fit.project(model.keypoints(x), camera) for x in truth[::12]])
    print(f'{a.name}: our hand model vs Blender, max {np.abs(ours - truth_uv[::12]).max():.3f} px')
    span = np.linalg.norm(truth_uv[0, 0] - truth_uv[0, 12])
    # square crop around the hand, sized to its full span (a fist's outline is too tight for the detector)
    centre = (truth_uv.min(1) + truth_uv.max(1)) / 2
    half = 0.5 * np.maximum((truth_uv.max(1) - truth_uv.min(1)).max(1), span)[:, None] * 1.6
    np.save(out / 'crops.npy', np.c_[centre - half, centre + half])
    for kind, extra in (('whole', []), ('crop', [str(out / 'crops.npy')])):
        target = out / f'track_{kind}.npz'
        if not target.exists():
            r = subprocess.run([str(MP_PYTHON), str(ROOT / 'tests' / 'hand_track_mp.py'), str(frames), str(MP_MODEL), str(target)] + extra,
                               capture_output=True, text=True)
            print(f'  tracking ({kind}):', (r.stdout.strip().splitlines() or ['?'])[-1])
    report = {'hand_span_px': float(span)}
    for kind, anatomical in (('whole', False), ('crop', False), ('crop', True)):
        obs = np.load(out / f'track_{kind}.npz')['points'][..., :2]
        found = ~np.isnan(obs[:, 0, 0])
        key_err = np.nanmedian(np.linalg.norm(obs[found] - truth_uv[found], axis=-1))
        fitted, used = hand_fit.fit(model, obs, camera, anatomical=anatomical)
        np.save(out / f'fit_{kind}_{"anat" if anatomical else "free"}.npy', fitted)
        e = bend_error(model, fitted, truth)
        jitter = float(np.degrees(np.abs(np.diff(fitted[:, 3:], 2, axis=0)).mean()))
        label = f'{kind:5s} + {"anatomical" if anatomical else "free     "}'
        report[label] = {'frames_tracked': int(found.sum()), 'keypoint_px': float(key_err), 'bend_error_deg': e,
                         'finger_jitter_deg': jitter}
        print(f'  {label}: tracked {found.sum():2d}/{T}, keypoints {key_err:5.1f} px | bend error (deg) ' +
              ' '.join(f'{k} {v:4.1f}' for k, v in e.items()) + f' | jitter {jitter:.2f}')
    (out / 'report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
