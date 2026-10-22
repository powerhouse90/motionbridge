"""Key a tracked first-person take onto the prepared scene and export it.

blender -b --python-exit-code 1 --python fp_apply.py -- --scene scene.blend --track track.npz --out OUT.blend
        [--fbx OUT.fbx] [--glb OUT.glb] [--action NAME] [--fps 24] [--preview DIR]

track.npz: names (bones), basis (T, B, 4, 4) per frame, items (K, T, 4, 4) each item's motion, item_roots (K);
optionally arm_names (A): arm bones whose basis in the track already places them (core.fp_twist: the forearm
rolls with the hand).
The BLEND keeps the arm's IK (the hand is keyed, the arm follows; still editable) unless the track names arm
bones: then the IK over those arms is muted, since IK would put the forearm back without its roll. FBX and GLB get every bone baked from what Blender shows, since those formats carry no IK. The item is
keyed on MB_<item> empties. --preview renders every frame from the scene camera.
"""
import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fbx')
    p.add_argument('--glb')
    p.add_argument('--action', default='fp_motion')
    p.add_argument('--fps', type=float, default=24)
    p.add_argument('--preview')
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.scene).resolve()))
    scene = bpy.context.scene
    d = np.load(a.track)
    names, basis, items = [str(n) for n in d['names']], d['basis'], d['items']
    T = len(basis)
    rig = next(o for o in scene.objects if o.type == 'ARMATURE' and all(n in o.pose.bones for n in names))
    roots = [bpy.data.objects[str(n)] for n in d['item_roots']]
    moving = [i for i in range(len(names)) if np.abs(basis[:, i] - basis[0, i]).max() > 1e-7]
    arms = [str(n) for n in d['arm_names']] if 'arm_names' in d.files else []
    for pb in rig.pose.bones:  # an IK chain over a keyed arm would pull it back
        for c in pb.constraints:
            if c.type == 'IK':
                chain = [pb] + list(pb.parent_recursive)
                chain = chain[:c.chain_count] if c.chain_count else chain
                if any(b.name in arms for b in chain):
                    c.mute = True
    scene.render.fps, scene.render.fps_base = round(a.fps), round(a.fps) / a.fps
    scene.frame_start, scene.frame_end = 1, T
    rig.animation_data_create()
    rig.animation_data.action = bpy.data.actions.new(a.action)
    for root in roots:
        root.animation_data_create()
        root.animation_data.action = bpy.data.actions.new(f'{a.action}_{root.name}')
        root.rotation_mode = 'QUATERNION'
    for t in range(T):
        for i in moving:
            pb = rig.pose.bones[names[i]]
            pb.rotation_mode = 'QUATERNION'
            pb.matrix_basis = Matrix(basis[t, i].tolist())
            pb.keyframe_insert('location', frame=t + 1)
            pb.keyframe_insert('rotation_quaternion', frame=t + 1)
        for root, M in zip(roots, items):
            root.matrix_world = Matrix(M[t].tolist())
            root.keyframe_insert('location', frame=t + 1)
            root.keyframe_insert('rotation_quaternion', frame=t + 1)
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.out).resolve()), copy=True)

    if a.preview:
        out = Path(a.preview)
        out.mkdir(parents=True, exist_ok=True)
        for t in range(T):
            scene.frame_set(t + 1)
            scene.render.filepath = str(out / f'{t:05d}.png')
            bpy.ops.render.render(write_still=True)
            print('preview', t, flush=True)

    if a.fbx or a.glb:
        # bake what Blender shows (IK included) into every bone, then export
        bpy.context.view_layer.objects.active = rig
        for o in scene.objects:
            o.select_set(o in (rig,))
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.select_all(action='SELECT')
        bpy.ops.nla.bake(frame_start=1, frame_end=T, only_selected=False, visual_keying=True,
                         clear_constraints=True, use_current_action=True, bake_types={'POSE'})
        bpy.ops.object.mode_set(mode='OBJECT')
        if a.fbx:
            bpy.ops.export_scene.fbx(filepath=str(Path(a.fbx).resolve()), use_selection=False, bake_anim=True,
                                     add_leaf_bones=False, bake_anim_use_nla_strips=False)
        if a.glb:
            bpy.ops.export_scene.gltf(filepath=str(Path(a.glb).resolve()), export_format='GLB', export_animations=True)
    print('FPAPPLY ok', T, 'frames,', len(moving), 'keyed bones' + (f' (IK muted over {len(arms)} rolled arm bones)' if arms else ''))


if __name__ == '__main__':
    main()
