"""Grip take -> start scene: pose the scene with the settled end of a tracked grab (the last frames
averaged: hands, fingers and the prop), so later animations can all start from that one grip.

blender -b --python-exit-code 1 --python fp_grip.py -- --scene scene.blend --track track.npz --out grip.blend
        [--frames 8] [--render grip.png]
track.npz: names, basis (T, B, 4, 4), items (K, T, 4, 4), item_roots (K) as Track FP Hands writes them.
"""
import argparse
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion


def average(mats):
    """Mean of 4x4 transforms: positions averaged, rotations as an averaged quaternion (signs aligned)."""
    qs = [Matrix(m.tolist()).to_quaternion() for m in mats]
    ref = qs[-1]
    acc = np.zeros(4)
    for q in qs:
        v = np.array(q)
        acc += v if np.dot(v, np.array(ref)) >= 0 else -v
    q = Quaternion(acc / np.linalg.norm(acc))
    scale = Matrix(mats[-1].tolist()).to_scale()
    return Matrix.LocRotScale(np.mean([m[:3, 3] for m in mats], axis=0).tolist(), q, scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--track', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--render')
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.scene).resolve()))
    d = np.load(a.track)
    names, basis = [str(n) for n in d['names']], d['basis'][-a.frames:]
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and all(n in o.pose.bones for n in names))
    rig.animation_data_clear()
    for i, n in enumerate(names):
        pb = rig.pose.bones[n]
        pb.rotation_mode = 'QUATERNION'
        pb.matrix_basis = average(basis[:, i])
    for root, M in zip(d['item_roots'], d['items']):
        obj = bpy.data.objects[str(root)]
        obj.animation_data_clear()
        obj.matrix_world = average(M[-a.frames:])
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.out).resolve()), copy=True)
    if a.render:
        bpy.context.scene.render.filepath = str(Path(a.render).resolve())
        bpy.ops.render.render(write_still=True)
    print('FPGRIP ok, averaged the last', len(basis), 'frames')


if __name__ == '__main__':
    main()
