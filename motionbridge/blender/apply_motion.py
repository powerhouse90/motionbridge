"""Key retarget results onto a character and verify them. Never saves over the source.

blender -b --python-exit-code 1 --python apply_motion.py -- --file CHAR --out OUT.blend
        --motions a.npz [b.npz ...] [--armature NAME] [--fbx OUT.fbx] [--glb OUT.glb] [--report report.json]

Each .npz (core.retarget.Result.save) becomes one action named after the file. After keying,
every bone head is read back from Blender and compared with the core's forward kinematics.
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import load, pick_armature, skins  # noqa: E402

CHECK_FRAMES = 8  # frames compared per action, spread over the clip


def key_action(rig, name, data):
    names = [str(n) for n in data['names']]
    quats, locs = data['quaternions'], data['locations']
    frames = len(quats)
    rig.animation_data_create()
    action = bpy.data.actions.new(name)
    action.use_fake_user = True  # keep every motion in the file, not only the active one
    rig.animation_data.action = action
    pose = rig.pose.bones
    for pb in pose:
        pb.rotation_mode = 'QUATERNION'
    for t in range(frames):
        for b, bone in enumerate(names):
            pb = pose[bone]
            pb.rotation_quaternion = quats[t, b].tolist()
            pb.location = locs[t, b].tolist()
            pb.scale = (1, 1, 1)
            pb.keyframe_insert('rotation_quaternion', frame=t + 1)
            pb.keyframe_insert('location', frame=t + 1)
    return frames


def verify(rig, data, frames):
    names = [str(n) for n in data['names']]
    worst = 0.0
    for t in np.linspace(0, frames - 1, CHECK_FRAMES).round().astype(int):
        bpy.context.scene.frame_set(int(t) + 1)
        bpy.context.view_layer.update()
        got = np.array([rig.matrix_world @ rig.pose.bones[n].head for n in names])
        worst = max(worst, float(np.abs(got - data['heads'][t]).max()))
    return worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--motions', nargs='+', required=True)
    p.add_argument('--armature')
    p.add_argument('--fbx')
    p.add_argument('--glb')
    p.add_argument('--report')
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    source = Path(a.file).resolve()
    out = Path(a.out).resolve()
    if out == source:
        raise ValueError('refusing to save over the source character file')
    load(source)
    rig = pick_armature(a.armature)
    rest_before = {b.name: [list(r) for r in b.matrix_local] for b in rig.data.bones}
    report = {'file': source.as_posix(), 'armature': rig.name, 'actions': {}}
    scene = bpy.context.scene
    for path in a.motions:
        data = np.load(path)
        name = Path(path).stem.split('__')[-1]
        frames = key_action(rig, name, data)
        scene.render.fps = int(round(float(data['fps'])))
        scene.frame_start, scene.frame_end = 1, frames
        error = verify(rig, data, frames)
        report['actions'][name] = {'frames': frames, 'max_head_error_m': error}
        print(f'apply_motion: {name}: {frames} frames, max head error {error * 1000:.4f} mm', flush=True)
    report['rest_matrices_unchanged'] = all(
        np.allclose([list(r) for r in b.matrix_local], rest_before[b.name]) for b in rig.data.bones)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), copy=True)
    if a.fbx or a.glb:
        bpy.ops.object.select_all(action='DESELECT')
        rig.select_set(True)
        for mesh in bpy.context.scene.objects:
            if mesh.type == 'MESH' and skins(mesh, rig):
                mesh.select_set(True)
        bpy.context.view_layer.objects.active = rig
    if a.fbx:
        bpy.ops.export_scene.fbx(filepath=str(Path(a.fbx).resolve()), use_selection=True, bake_anim=True,
                                 bake_anim_use_all_actions=True, bake_anim_use_nla_strips=False,
                                 add_leaf_bones=False)
    if a.glb:
        bpy.ops.export_scene.gltf(filepath=str(Path(a.glb).resolve()), export_format='GLB', use_selection=True,
                                  export_animations=True, export_animation_mode='ACTIONS')
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    print('apply_motion: saved', out, flush=True)


main()
