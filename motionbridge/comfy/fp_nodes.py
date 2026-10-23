"""First-person nodes: Load FP Scene -> (H3) -> Track FP Hands -> Export FP Animation.

The scene is one .blend with FPS arms, the first-person camera and the props. Its start frame is H3's first
frame, so frame 0 of the video is known exactly; the tracker follows the hands, and the main prop, from
there. In capture mode (the default) the start frame is dressed for tracking: the arms in two colours and
the main prop see-through, so H3 keeps the hands apart and the prop's whole shape visible. Only that
throwaway frame changes; the rig and the exported files keep the scene's own look. Same conventions as
nodes.py: small dicts of paths between nodes, heavy work in subprocesses, caching by content.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui, InputImpl

from motionbridge.comfy.common import digest, free_vram, output_folder, resolve_input, run_blender, run_process, runtime
from motionbridge.comfy.nodes import frames_to_video

FPScene = io.Custom('MOTIONBRIDGE_FP_SCENE')
FPMotion = io.Custom('MOTIONBRIDGE_FP_MOTION')
GEM_SCRIPTS = Path(__file__).resolve().parents[1] / 'gem'


class LoadFPScene(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeLoadFPScene', display_name='Load FP Scene', category='MotionBridge/First Person',
            description='A first-person scene in one .blend: FPS arms (any rig with finger bones), the scene camera '
                        'as the player\'s view, and the props. Finds the hands and fingers by shape, picks the main prop '
                        '(the one a hand holds in the start pose), freezes the start pose, and renders the start frame: '
                        'feed it to H3 as the first frame. Other props stay in the scene for H3 but are not tracked.',
            inputs=[
                io.String.Input('scene_file', default='', tooltip='A .blend; use the upload button or a path relative to ComfyUI/input.'),
                io.Int.Input('width', default=1280, min=256, max=4096, step=16),
                io.Int.Input('height', default=720, min=256, max=4096, step=16),
                io.Boolean.Input('capture_mode', default=True,
                                 tooltip='Dress the start frame for tracking (arms in two colours, main prop see-through). '
                                         'Much better tracking; the exported animation keeps your look either way.'),
                io.String.Input('main_prop', default='', optional=True,
                                tooltip='Name of the prop to track. Default: the one a hand holds in the start pose.'),
                io.String.Input('armature_name', default='', advanced=True, optional=True,
                                tooltip='Only needed when the file holds more than one armature.'),
                io.String.Input('setup', optional=True, force_input=True,
                                tooltip='Connect MotionBridge Setup here so one Run installs first.'),
            ],
            outputs=[FPScene.Output(display_name='fp_scene'), io.Image.Output(display_name='start_frame'),
                     io.String.Output(display_name='report')])

    @classmethod
    def fingerprint_inputs(cls, scene_file, width, height, capture_mode=True, main_prop='', armature_name='', setup=None):
        try:
            return digest(resolve_input(scene_file, ('.blend',)), width, height, capture_mode, main_prop, armature_name)
        except (OSError, ValueError):
            return float('nan')

    @classmethod
    def execute(cls, scene_file, width, height, capture_mode=True, main_prop='', armature_name='', setup=None):
        from motionbridge.core import fp_track
        source = resolve_input(scene_file, ('.blend',))
        # keyed by the scripts that build the folder, not this file: a node edit mustn't redo every take downstream
        blender = Path(__file__).resolve().parents[1] / 'blender'
        folder = output_folder('fp_scene', f'{source.stem}_{digest(source, width, height, armature_name, blender / "fp_scene.py", blender / "fp_capture.py")}')
        if not (folder / 'rig.json').exists():
            args = ['--file', source, '--outdir', folder, '--size', width, height] + (['--armature', armature_name] if armature_name else [])
            run_blender('fp_scene.py', args, folder / 'fp_scene.log')
        dump = json.loads((folder / 'rig.json').read_text())
        rigs = []
        for side in ('Right', 'Left'):
            try:
                rigs.append(fp_track.Rig(dump, side))
            except ValueError:
                pass
        if not rigs:
            raise ValueError(f'No hand with finger bones found on "{dump["armature"]}".')
        names = [it['name'] for it in dump['items']]
        items = [fp_track.Item(folder / it['file']) for it in dump['items']]
        prop = fp_track.main_prop(rigs, items, names, main_prop.strip())
        lines = [f'{source.name}: armature "{dump["armature"]}", camera lens {dump["lens"]:.1f} mm, {width}x{height}']
        for rig in rigs:
            how = (f'moved by its IK control "{rig.sk.names[rig.ctrl]}" (the arm follows)' if rig.mode == 'control'
                   else f'placed by turning "{rig.sk.names[rig.arm[1]]}" and "{rig.sk.names[rig.arm[0]]}"')
            lines.append(f'  {rig.hand.side.lower()} hand "{rig.hand.hand}", {how}; fingers ' +
                         ', '.join(f'{f} ({" > ".join(j)})' for f, j in rig.hand.fingers.items()))
        if prop is None:
            lines.append('  main prop: none (no hand holds a prop in the start pose; name one in main_prop to track it)')
        else:
            sock = fp_track.socket(rigs, items[prop])
            held = f', held by the {rigs[sock[0]].hand.side.lower()} hand from the start' if sock else ', not held at the start'
            lines.append(f'  main prop "{names[prop]}" ({", ".join(dump["items"][prop]["objects"])}){held}')
        others = [n for k, n in enumerate(names) if k != prop]
        if others:
            lines.append(f'  left as placed (not tracked): {", ".join(others)}')
        log_text = (folder / 'fp_scene.log').read_text(encoding='utf-8', errors='replace') if (folder / 'fp_scene.log').exists() else ''
        lines += [f'  {line.strip()} (pack them in Blender: File > External Data > Pack Resources)'
                  for line in log_text.splitlines() if line.startswith('WARNING')]
        start_png = folder / 'start.png'
        if capture_mode:
            start_png = folder / f'start_capture_{prop}.png'
            if not start_png.exists():
                by_side = {r.hand.side: r.hand.hand for r in rigs}
                args = ['--scene', folder / 'scene.blend', '--out', start_png, '--right', by_side.get('Right', rigs[0].hand.hand)]
                args += ['--left', by_side['Left']] if 'Left' in by_side and 'Right' in by_side else []
                args += ['--prop', ','.join(dump['items'][prop]['objects'])] if prop is not None else []
                run_blender('fp_capture.py', args, folder / 'fp_capture.log')
            lines.append('  capture mode: start frame with orange right arm, blue left arm, see-through main prop')
        text = '\n'.join(lines)
        start = torch.from_numpy(np.asarray(Image.open(start_png).convert('RGB')).astype(np.float32) / 255)[None]
        scene = {'folder': str(folder), 'name': source.stem, 'prop': prop, 'capture': bool(capture_mode)}
        return io.NodeOutput(scene, start, text, ui=ui.PreviewText(text))


class Progress:
    """One ComfyUI progress bar across a node's stages (weighted by how long each takes), and a
    console line about every tenth of each stage, so long runs show where they are."""

    def __init__(self, stages):
        self.stages, self.done, self.shown = stages, {k: 0.0 for k in stages}, {k: -1 for k in stages}
        self.bar = comfy.utils.ProgressBar(1000)

    def __call__(self, stage, i, n, label):
        self.done[stage] = i / max(n, 1)
        total = sum(w * self.done[k] for k, w in self.stages.items()) / sum(self.stages.values())
        self.bar.update_absolute(int(total * 1000))
        tenth = int(self.done[stage] * 10)
        if tenth != self.shown[stage]:
            self.shown[stage] = tenth
            print(f'[MotionBridge] {label}: {int(i)}/{n}', flush=True)


def extract_progress(line, T, progress):
    """Progress from fp_extract's output lines: "masks 12", "hands 12"."""
    words = line.split()
    if len(words) == 2 and words[1].isdigit() and words[0] in ('masks', 'hands'):
        label = 'SAM 2 masks (arms and prop)' if words[0] == 'masks' else 'SAM 3D Body hands'
        progress(words[0], int(words[1]) + 1, T, label)


def write_frames(images, folder):
    folder.mkdir(parents=True, exist_ok=True)
    frames = (images.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    for t, f in enumerate(frames):
        Image.fromarray(f).save(folder / f'{t:05d}.jpg', quality=95)
    return frames.shape[2], frames.shape[1]


PIPELINES = ['points + grasp', 'points', 'classic']
DENSE_POINTS = 40000  # samples on the prop's surface: the thin parts (a grip) need them for depth and contact


def dense_points(tris, n=DENSE_POINTS, seed=0):
    """n points spread over triangles (N, 3, 3): 30 per triangle, then a random n of those."""
    rng = np.random.default_rng(seed)
    a, b = rng.random((2, len(tris), 30))
    m = a + b > 1
    a[m], b[m] = 1 - a[m], 1 - b[m]
    pts = (tris[:, 0, None] + a[..., None] * (tris[:, 1] - tris[:, 0])[:, None] + b[..., None] * (tris[:, 2] - tris[:, 0])[:, None]).reshape(-1, 3)
    return pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]


def arm_basis(sk, basis, arm_names, arm_world):
    """Upper arms and forearms given as world matrices per frame -> the same as bone basis, so the track keys
    them like any bone (pose = parent pose @ rest offset @ basis, as hands.Skeleton). basis: (T, B, 4, 4)."""
    out = basis.copy()
    idx = [sk.index[str(n)] for n in arm_names]
    for t in range(len(out)):
        pose = sk.pose(out[t])
        for k, i in enumerate(idx):  # parents first: fp_twist gives each upper arm before its forearm
            p = sk.parent[i]
            parent = np.eye(4) if p is None else pose[p]
            out[t, i] = np.linalg.inv(sk.offset[i]) @ np.linalg.inv(parent) @ arm_world[t, k]
            pose[i] = arm_world[t, k]
    return out


GRASP_RUNNER = """
import json, sys
sys.path.insert(0, sys.argv.pop(1))  # the pack's root: ComfyUI portable's Python ignores PYTHONPATH
import numpy as np
from motionbridge.core import fp_track as fp, fp_grasp
scene, extracted, inp, out, W, H, workers, strength = sys.argv[1:9]
dump = json.loads(open(scene + '/rig.json').read())
d = np.load(inp)
sides = [str(v) for v in d['sides']]
rigs = [fp.Rig(dump, v) for v in sides]
prop = fp.Item(scene + '/' + str(d['item_file']))
if 'points' in d.files:
    prop.points = d['points']
ex = np.load(extracted)
P, report = fp_grasp.grasp_take(rigs, prop, d['motion'], [ex['kp2d_' + v] for v in sides], [ex['kp3d_' + v] for v in sides],
                                [d['params_' + v] for v in sides], rigs[0].camera(int(W), int(H)), workers=int(workers),
                                strength=float(strength), log=lambda m: print(m, flush=True),
                                progress=lambda k, n: print('grasp', k, n, flush=True))
np.savez(out, **{'params_' + v: p for v, p in zip(sides, P)}, **{'holding_' + v: np.array(report['pass2'][i]) for i, v in enumerate(sides)})
print('GRASP ok', flush=True)
"""


def grasp_step(scene, dump, k, rigs, prop, motion, params, extracted, cam, strength, workers, folder, log, progress):
    """fp_grasp.grasp_take over the take -> (params per rig, fingers holding per frame per rig). workers 0: in this
    process (serial: slow). workers N: in a separate Python process with N workers of its own (a node cannot
    spawn workers itself: they would re-import ComfyUI's main script)."""
    from motionbridge.core import fp_grasp
    sides = [r.hand.side for r in rigs]
    if workers <= 0:
        ex = np.load(extracted)
        P, report = fp_grasp.grasp_take(rigs, prop, motion, [ex[f'kp2d_{v}'] for v in sides], [ex[f'kp3d_{v}'] for v in sides],
                                        params, cam, workers=0, strength=strength, log=log, progress=progress)
        return P, report['pass2']
    inp, out = folder / 'grasp_in.npz', folder / 'grasp_out.npz'
    np.savez(inp, sides=np.array(sides), item_file=dump['items'][k]['file'], points=prop.points, motion=motion,
             **{f'params_{v}': P for v, P in zip(sides, params)})
    root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([root] + [p for p in [os.environ.get('PYTHONPATH')] if p])}

    def on_line(line):
        w = line.split()
        if len(w) == 3 and w[0] == 'grasp' and w[1].isdigit() and w[2].isdigit():
            progress(int(w[1]), int(w[2]))
        elif line:
            log(line)
    run_process([sys.executable, '-c', GRASP_RUNNER, root, scene, extracted, inp, out, cam['W'], cam['H'], workers, strength],
                folder / 'fp_grasp.log', 'Grasp step', env=env, on_line=on_line)
    d = np.load(out)
    return [d[f'params_{v}'] for v in sides], [list(d[f'holding_{v}']) for v in sides]


def track_take(scene, dump, k, rigs, ex, extracted, frames_dir, folder, cam, smoothing, see_through, grasp, workers, roll,
               log, progress):
    """The points pipeline: the main prop placed from points followed on its texture (fp_points), the hands
    tracked against it (see-through in capture mode), the grasp step where a hand holds it (fp_grasp), the
    forearms rolled with the hands (fp_twist). Returns (track npz, report lines). Each step is cached by its
    inputs and the code that makes it."""
    import cv2
    from motionbridge.core import fp_track, fp_points, fp_grasp, fp_twist, fp_elbow, fp_hold
    sides = '_'.join(r.hand.side for r in rigs)
    T = len(ex[f'kp2d_{rigs[0].hand.side}'])
    lines = []
    prop = fp_track.Item(scene / dump['items'][k]['file']) if k is not None else None
    tris_file = scene / dump['items'][k].get('tris', f'item_{k}_tris.npy') if k is not None else None
    tris = np.load(tris_file).astype(np.float64) if tris_file is not None and tris_file.exists() else None
    if tris is not None and not len(tris):
        tris = None
    if tris is not None:
        prop.points = dense_points(tris)
    motion = smoothed = None
    scenery = [fp_track.Item(scene / it['file']) for j, it in enumerate(dump['items']) if j != k and it.get('file')]
    if prop is not None:
        cache = folder / f'prop_points_{digest(Path(fp_points.__file__), tris is not None)}.npz'
        if cache.exists():
            c = np.load(cache)
            motion, err, held = c['motion'], c['err'], c['held']
        else:
            frames = [cv2.imread(str(p)) for p in sorted(Path(frames_dir).glob('*.jpg'))]
            arms = np.zeros(ex['item_0'].shape, bool)
            for r in rigs:
                arms |= ex[f'arm_{r.hand.side}']
            motion, err, held, _ = fp_points.track_points(frames, ex['item_0'], arms, cam, tris=tris, item=prop, log=log,
                                                          progress=lambda i, n: progress('prop', i, n, 'following points on the main prop'))
            np.savez(cache, motion=motion, err=err, held=held)
        smoothed = fp_track.smooth_items([prop], [motion], smoothing)[0] if smoothing > 0 else motion
        # the props left as placed (a desk, a keyboard) are solid: a prop read too deep is slid out toward the camera
        smoothed = fp_hold.clear_prop(smoothed, prop, scenery, cam, log=lines.append)
        lines.append(f'main prop from points on its texture{"" if tris is not None else " (no triangles in the scene: ray-marched its signed distance)"}: '
                     f'{np.nanmedian(err):.1f} px off the video (median), fewest points held {int(held.min())}')
    props = [prop] if prop is not None else []
    prop_key = (cache.name, digest(Path(fp_hold.__file__))) if prop is not None else ''  # the hand fit depends on the prop's motion
    raw_file = folder / f'raw_points_{sides}_{digest(Path(fp_track.__file__), see_through, smoothing if prop is not None else 0, tris is not None, prop_key)}.npz'
    if raw_file.exists():
        raw = np.load(raw_file)
        params = [raw[f'params_{r.hand.side}'] for r in rigs]
        errs = [raw[f'errs_{r.hand.side}'] for r in rigs]
    else:
        params, errs = [], []
        for rig in rigs:
            side = rig.hand.side
            P, E = fp_track.track_hand(rig, props, [smoothed] if props else [], ex[f'kp2d_{side}'], ex[f'kp3d_{side}'], ex[f'arm_{side}'],
                                       cam, log=log, see_through=see_through,
                                       progress=lambda i, n, side=side: progress(f'fit_{side}', i, n, f'fitting the {side.lower()} hand, frame'))
            params.append(P)
            errs.append(E)
        np.savez(raw_file, **{f'params_{r.hand.side}': P for r, P in zip(rigs, params)}, **{f'errs_{r.hand.side}': E for r, E in zip(rigs, errs)})
    holding = None
    if grasp and prop is not None:
        cache = folder / f'grasp_{sides}_{digest(Path(fp_grasp.__file__), raw_file.name, smoothing)}.npz'
        if cache.exists():
            c = np.load(cache)
            final = [c[f'params_{r.hand.side}'] for r in rigs]
            holding = [list(c[f'holding_{r.hand.side}']) for r in rigs]
        else:
            final, holding = grasp_step(scene, dump, k, rigs, prop, smoothed, params, extracted, cam, smoothing, workers, folder, log,
                                        lambda i, n: progress('grasp', i, n, 'grasp step, frame solves'))
            np.savez(cache, **{f'params_{r.hand.side}': P for r, P in zip(rigs, final)},
                     **{f'holding_{r.hand.side}': np.array(h) for r, h in zip(rigs, holding)})
    elif smoothing > 0:
        final = [fp_track.smooth_hand(rig, props, [motion] if props else [], [smoothed] if props else [], P, smoothing, log=None)
                 for rig, P in zip(rigs, params)]
        if prop is not None:
            final = fp_track.settle_contacts(rigs, prop, smoothed, final, None, cam, log=log)
    else:
        final = list(params)
    # what the video can't show: a held hand stays on its grip, a free one is bridged between its good frames
    for i, (rig, E) in enumerate(zip(rigs, errs)):
        side = rig.hand.side
        rel = fp_hold.reliability(ex[f'kp2d_{side}'], E, cam['W'], cam['H'])
        spans = []
        if prop is not None:
            final[i], spans = fp_hold.hold_lock(rig, final[i], prop, smoothed, rel, ex[f'kp2d_{side}'], cam, log=lines.append,
                                              masks=ex[f'arm_{side}'])
            # a regrip the video reads worse than a grip this hand is known to make there (start pose, a well-read
            # hold): the known grip
            grips = fp_hold.known_grips(rig, final[i], prop, smoothed, rel, spans)
            final[i] = fp_hold.reuse_grips(rig, final[i], prop, smoothed, spans, grips, log=lines.append)
        final[i] = fp_hold.bridge(rig, final[i], rel, spans, log=lines.append, cam=cam)
        final[i] = fp_hold.clear_hand(rig, final[i], scenery, cam, log=lines.append)  # nor a hand through the desk
    basis = fp_track.combined_basis(rigs, final)
    arm_names = np.array([], str)
    if roll:
        # the arms to their masks: where the shoulder sits (a viewmodel pushed forward and down when aiming), then
        # the elbow's swing, then the forearm rolled with the hand
        masks = {r.hand.side: ex[f'arm_{r.hand.side}'] for r in rigs if f'arm_{r.hand.side}' in ex.files}
        offset = fp_elbow.fit_shoulders(rigs, basis, masks, cam, log=log)
        swing, _ = fp_elbow.fit_elbows(rigs, basis, masks, cam, log=log, offset=offset)
        arm_names, arm_world, rolls = fp_twist.forearm_roll(rigs, basis, log=log, swing=swing, offset=offset)
        if len(arm_names):
            basis = arm_basis(rigs[0].sk, basis, arm_names, arm_world)
            lines.append('forearm roll: ' + ', '.join(f'{side.lower()} median {np.median(np.abs(r)):.0f} deg, max {np.abs(r).max():.0f} deg'
                                                      for side, r in rolls.items()))
            lines.append('shoulders fitted to the arm masks: ' + ', '.join(
                f'{side.lower()} moved up to {np.linalg.norm(o, axis=1).max() * 100:.0f} cm' for side, o in offset.items()))
    for i, (rig, P, Q, E) in enumerate(zip(rigs, params, final, errs)):
        line = (f'{rig.hand.side.lower()} hand vs video: median {np.nanmedian(E) * 100:.1f}% of the hand\'s size, worst '
                f'{np.nanmax(E) * 100:.1f}% (frame {int(np.nanargmax(E))}); jitter {fp_track.jitter(rig, P):.1f} -> {fp_track.jitter(rig, Q):.1f} mm/frame^2')
        if holding is not None:
            h = np.array(holding[i])
            line += f'; holds the prop on {int((h > 0).sum())} frames, 4+ fingers on {int((h >= 4).sum())}'
        lines.append(line)
    # the camera: H3 may move it though asked not to; the props left as placed (a desk, a keyboard) don't move, so
    # their apparent motion is the camera's. The export moves the camera, and the arms and the prop seen from it.
    view = None
    scenery_tris = [np.load(scene / it['tris']) for j, it in enumerate(dump['items']) if j != k and it.get('tris')]
    if scenery_tris:
        vcache = folder / f'view_{digest(Path(fp_points.__file__), k)}.npz'
        if vcache.exists():
            c = np.load(vcache)
            view = c['view'] if c['moved'] else None
        else:
            from scipy import ndimage
            frames = [cv2.imread(str(p)) for p in sorted(Path(frames_dir).glob('*.jpg'))]
            exclude = np.zeros(ex[f'arm_{rigs[0].hand.side}'].shape, bool)
            for r in rigs:
                exclude |= ex[f'arm_{r.hand.side}']
            if prop is not None:
                exclude |= np.array([ndimage.binary_dilation(m, iterations=6) for m in ex['item_0']])
            res = fp_points.solve_view(frames, np.concatenate(scenery_tris), exclude, cam, log=lines.append)
            view = res[0] if res else None
            np.savez(vcache, view=view if view is not None else np.eye(4)[None], moved=view is not None)
        if view is not None:
            lines.append('camera moved in the video: solved from the props left as placed; the export moves the camera')
    track = folder / f'track_{sides}_{digest(raw_file.name, grasp, roll, smoothing, Path(fp_grasp.__file__), Path(fp_twist.__file__), Path(fp_elbow.__file__), Path(fp_hold.__file__), Path(fp_points.__file__))}.npz'
    np.savez(track, names=np.array(rigs[0].names), basis=basis, arm_names=arm_names,
             items=np.array([smoothed]) if prop is not None else np.zeros((0, T, 4, 4)),
             item_roots=np.array([dump['items'][k]['root']] if prop is not None else []),
             **({'view': view} if view is not None else {}))
    return track, lines


class TrackFPHands(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeTrackFPHands', display_name='Track FP Hands', category='MotionBridge/First Person',
            description='Pose the scene\'s arms (hands, wrists and every finger) and its main prop to the generated '
                        'video, frame by frame from the known start frame. SAM 2 masks each arm and the prop, SAM 3D Body '
                        'reads each hand; the hand holding the prop carries it, and fingers never pass through it.',
            inputs=[
                FPScene.Input('fp_scene'),
                io.Image.Input('images', optional=True, tooltip='Frames, e.g. straight from H3\'s VAE Decode.'),
                io.Video.Input('video', optional=True, tooltip='Or a video (Load Video).'),
                io.Float.Input('fps', default=24.0, min=1, max=120, tooltip='Frame rate of images (ignored for video).'),
                io.Float.Input('smoothing', default=0.7, min=0.0, max=1.0, step=0.05,
                               tooltip='0 = raw tracking. 0.7 removes the frame-to-frame shake without softening the grab.'),
                io.Combo.Input('hands', options=['both', 'right', 'left'], default='both',
                               tooltip='Which hands to track; a hand out of view in the start frame is left as it is.'),
                io.Combo.Input('pipeline', options=PIPELINES, default='classic', optional=True,
                               tooltip='points + grasp: the main prop placed from points on its texture, then fingers made to '
                                       'hold it where a hand holds it. points: no grasp step. classic: the earlier tracker '
                                       '(prop from its outline, carried by the hand that holds it at the start).'),
                io.Int.Input('grasp_workers', default=8, min=0, max=64, optional=True, advanced=True,
                             tooltip='Processes for the grasp step (it solves every held frame twice). 0 runs it inside '
                                     'ComfyUI, one frame at a time: slow (about 40 s a frame).'),
                io.Boolean.Input('forearm_roll', default=True, optional=True, advanced=True,
                                 tooltip='Turn each forearm along its length with the hand (pronation / supination), so the '
                                         'wrist only bends. Mutes the arm IK in the exported .blend.'),
            ],
            outputs=[FPMotion.Output(display_name='fp_motion'), io.String.Output(display_name='report')])

    @classmethod
    def execute(cls, fp_scene, fps, smoothing, hands='both', images=None, video=None, pipeline='classic', grasp_workers=8,
                forearm_roll=True):
        from motionbridge.core import fp_track
        if video is not None:
            components = video.get_components()
            images, fps = components.images, float(components.frame_rate)
        if images is None:
            raise ValueError('Connect images or a video.')
        scene = Path(fp_scene['folder'])
        sample = hashlib.sha256((images[::3, ::4, ::4].cpu().numpy() * 255).astype(np.uint8).tobytes()).hexdigest()
        dump = json.loads((scene / 'rig.json').read_text())
        W, H = images.shape[2], images.shape[1]
        if abs(W / H / (dump['W'] / dump['H']) - 1) > 0.06:  # H3 rounds sizes to multiples of 32; the camera absorbs that
            raise ValueError(f'The video is {W}x{H} but the scene was rendered at {dump["W"]}x{dump["H"]}: '
                             f'the frame shapes must match (H3 keeps the start frame\'s shape).')
        rigs = []
        for side in {'both': ('Right', 'Left'), 'right': ('Right',), 'left': ('Left',)}[hands]:
            try:
                rig = fp_track.Rig(dump, side)
            except ValueError:
                continue
            # tracked if SAM 2 can be prompted on it in the start frame: a hand mostly below the frame edge (a rifle
            # aimed forward) still counts, and comes into view later; off-screen frames are fp_hold's
            if len(rig.arm_prompts(rig.camera(W, H))) >= 2:
                rigs.append(rig)
        if not rigs:
            raise ValueError('No hand is in view in the start frame.')
        cam = rigs[0].camera(W, H)
        # the tracked prop and hands are part of the key: the extraction masks them, so a run with others can't reuse it
        folder = output_folder('fp_motion', digest(sample, list(images.shape), fps, scene.name, fp_scene.get('prop'),
                                                   [r.hand.side for r in rigs]))
        T = images.shape[0]
        k = fp_scene.get('prop')
        prop = fp_track.Item(scene / dump['items'][k]['file']) if k is not None else None
        stages = {'masks': 3, 'hands': 2 * len(rigs), 'prop': 1}
        stages.update({f'fit_{r.hand.side}': 4 for r in rigs})
        if pipeline == 'points + grasp' and prop is not None:
            stages['grasp'] = 6 * len(rigs) if grasp_workers > 0 else 60 * len(rigs)
        progress = Progress(stages)
        extracted = folder / 'extracted.npz'
        if not extracted.exists():
            write_frames(images, folder / 'frames')
            spans = {}
            for rig in rigs:
                kp = fp_track.project(rig.keypoints(np.zeros(rig.size)), cam)
                spans[rig.hand.side] = float(np.linalg.norm(kp[0] - kp[12]))
            visible = dump['items'][k]['visible'] if k is not None else []
            prompts = {'arms': {rig.hand.side: rig.arm_prompts(cam).tolist() for rig in rigs},
                       'items': [] if prop is None else [(np.array(visible[:8]) * [W / dump['W'], H / dump['H']]).tolist()
                                                         if visible else prop.prompts(cam).tolist()],
                       'focal': dump['lens'] / dump['sensor'] * W, 'spans': spans}
            (folder / 'prompts.json').write_text(json.dumps(prompts))
            rt = runtime()
            free_vram()
            run_process([rt.gem_python, GEM_SCRIPTS / 'fp_extract.py', '--gem-root', rt.gem_root,
                         '--sam2-checkpoint', rt.sam2_checkpoint, '--frames', folder / 'frames',
                         '--prompts', folder / 'prompts.json', '--out', extracted],
                        folder / 'fp_extract.log', 'SAM 2 / SAM 3D Body', env=rt.gem_env(), on_line=lambda line: extract_progress(line, T, progress))
        ex = np.load(extracted)
        log = []
        if pipeline != 'classic':
            track, lines = track_take(scene, dump, k, rigs, ex, extracted, folder / 'frames', folder, cam, smoothing,
                                      bool(fp_scene.get('capture', False)), pipeline == 'points + grasp', grasp_workers,
                                      forearm_roll, log.append, progress)
            head = f'{T} frames at {fps:g} fps on {dump["armature"]}; main prop: {dump["items"][k]["name"] if prop is not None else "none"}; {pipeline}'
            report = '\n'.join([head] + lines + log[-3:])
            motion = {'scene': fp_scene, 'track': str(track), 'frames': str(folder / 'frames'), 'fps': fps, 'name': fp_scene['name']}
            return io.NodeOutput(motion, report, ui=ui.PreviewText(report))
        props = [prop] if prop is not None else []
        raw_file = folder / f'raw_{"_".join(r.hand.side for r in rigs)}.npz'
        if raw_file.exists():
            raw = np.load(raw_file)
            motion = raw['prop'] if prop is not None else None
            params = [raw[f'params_{r.hand.side}'] for r in rigs]
            errs = [raw[f'errs_{r.hand.side}'] for r in rigs]
        else:
            motion = None
            if prop is not None:  # its outline first (the hands must not pass through it while they are tracked)
                motion = fp_track.track_item(prop, ex['item_0'], cam, log=log.append,
                                             progress=lambda i, n: progress('prop', i, n, 'tracking the main prop'))
            params, errs = [], []
            for rig in rigs:
                side = rig.hand.side
                P, E = fp_track.track_hand(rig, props, [motion] if props else [], ex[f'kp2d_{side}'], ex[f'kp3d_{side}'],
                                           ex[f'arm_{side}'], cam, log=log.append,
                                           progress=lambda i, n, side=side: progress(f'fit_{side}', i, n, f'fitting the {side.lower()} hand, frame'))
                params.append(P)
                errs.append(E)
            np.savez(raw_file, **({'prop': motion} if prop is not None else {}), **{f'params_{r.hand.side}': P for r, P in zip(rigs, params)},
                     **{f'errs_{r.hand.side}': E for r, E in zip(rigs, errs)})
        # the hand that holds the prop in the start frame decides where it is
        sock = fp_track.socket(rigs, prop) if prop is not None else None
        if sock:
            h, grip, fingers = sock
            motion = fp_track.prop_from_hand(rigs[h], params[h], grip)
            log.append(f'main prop held by the {rigs[h].hand.side.lower()} hand: it follows that hand\'s grip')
        if smoothing > 0:
            smoothed = fp_track.smooth_items(props, [motion], smoothing)[0] if prop is not None else None
            params_s = [fp_track.smooth_hand(rig, props, [motion] if props else [], [smoothed] if props else [], P, smoothing, log=None)
                        for rig, P in zip(rigs, params)]
        else:
            smoothed, params_s = motion, list(params)
        if sock:
            params_s[h] = fp_track.hold_prop(rigs[h], params_s[h], smoothed, grip, fingers)
        params_s = fp_track.settle_contacts(rigs, prop, smoothed, params_s, None, cam, log=log.append)
        basis, smoothed, eased = fp_track.ease_pops(rigs, fp_track.combined_basis(rigs, params_s), prop, smoothed)
        track = folder / f'track_{"_".join(r.hand.side for r in rigs)}_s{smoothing:g}.npz'
        np.savez(track, names=np.array(rigs[0].names), basis=basis,
                 items=np.array([smoothed]) if prop is not None else np.zeros((0, T, 4, 4)),
                 item_roots=np.array([dump['items'][k]['root']] if prop is not None else []))
        lines = [f'{T} frames at {fps:g} fps on {dump["armature"]}; main prop: {dump["items"][k]["name"] if prop is not None else "none"}']
        for rig, P, Ps, E in zip(rigs, params, params_s, errs):
            lines.append(f'{rig.hand.side.lower()} hand vs video: median {np.nanmedian(E) * 100:.1f}% of the hand\'s size, worst '
                         f'{np.nanmax(E) * 100:.1f}% (frame {int(np.nanargmax(E))}); jitter {fp_track.jitter(rig, P):.1f} -> '
                         f'{fp_track.jitter(rig, Ps):.1f} mm/frame^2')
        if eased:
            lines.append('pops eased: ' + ', '.join(eased))
        report = '\n'.join(lines + log[-3:])
        motion = {'scene': fp_scene, 'track': str(track), 'frames': str(folder / 'frames'), 'fps': fps,
                  'name': fp_scene['name']}
        return io.NodeOutput(motion, report, ui=ui.PreviewText(report))


class ExportFPAnimation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeExportFP', display_name='Export FP Animation', category='MotionBridge/First Person',
            description='Key the tracked hand and item onto the scene and save BLEND (arm IK kept, editable) + FBX '
                        '(+ GLB) with every bone baked. Renders a preview next to the source video.',
            is_output_node=True,
            inputs=[
                FPMotion.Input('fp_motion'),
                io.String.Input('filename_prefix', default='motionbridge/fp'),
                io.String.Input('action_name', default='fp_motion'),
                io.Boolean.Input('glb', default=True),
                io.Combo.Input('preview', options=['side by side', 'rig only', 'none'], default='side by side'),
            ],
            outputs=[io.Video.Output(display_name='preview'), io.String.Output(display_name='files')])

    @classmethod
    def execute(cls, fp_motion, filename_prefix, action_name, glb, preview):
        name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in action_name) or 'fp_motion'
        scene = Path(fp_motion['scene']['folder'])
        folder = output_folder('fp_export', digest(Path(fp_motion['track']), scene / 'scene.blend', name, glb, preview != 'none'))
        blend, fbx, glb_file = folder / f'{name}.blend', folder / f'{name}.fbx', folder / f'{name}.glb'
        frames = folder / 'preview'
        if not blend.exists():
            free_vram()
            args = ['--scene', scene / 'scene.blend', '--track', fp_motion['track'], '--out', blend, '--fbx', fbx,
                    '--action', name, '--fps', fp_motion['fps']] + (['--glb', glb_file] if glb else [])
            args += ['--preview', frames] if preview != 'none' else []
            T = len(sorted(Path(fp_motion['frames']).glob('*.jpg')))
            progress = Progress({'preview': 1})
            run_blender('fp_apply.py', args, folder / 'fp_apply.log', on_line=lambda line: progress(
                'preview', int(line.split()[1]) + 1, T, 'rendering the preview') if line.startswith('preview ') and line.split()[1].isdigit() else None)
        target_dir, stem, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        files = []
        for src in [blend, fbx] + ([glb_file] if glb else []):
            dst = Path(target_dir) / f'{stem}_{name}{src.suffix}'
            shutil.copy2(src, dst)
            files.append(dst)
        text = '\n'.join(str(f) for f in files)
        video_out, ui_out = None, ui.PreviewText(text)
        if preview != 'none':
            pngs = sorted(frames.glob('*.png'))
            rig_frames = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in pngs])
            if preview == 'side by side':
                src = sorted(Path(fp_motion['frames']).glob('*.jpg'))
                h, w = rig_frames.shape[1:3]
                video = np.stack([np.asarray(Image.open(p).convert('RGB').resize((w, h))) for p in src[:len(rig_frames)]])
                rig_frames = np.concatenate([video, rig_frames], axis=2)
            mp4 = Path(target_dir) / f'{stem}_{name}_preview.mp4'
            frames_to_video(torch.from_numpy(rig_frames.astype(np.float32) / 255), fp_motion['fps'], mp4, crf=18)
            video_out = InputImpl.VideoFromFile(str(mp4))
            ui_out = ui.PreviewVideo([ui.SavedResult(mp4.name, subfolder, io.FolderType.output)])
        return io.NodeOutput(video_out, text, ui=ui_out)


NODES = [LoadFPScene, TrackFPHands, ExportFPAnimation]
