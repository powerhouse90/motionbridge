"""Capture-mode start frame: the same scene and pose, dressed for the trackers rather than the audience.
The right arm orange, the left arm blue (each vertex by which side of the body it is on: the side of the
plane halfway between the two wrists), the main prop see-through. The user's files are not changed: this
renders from the prepared working copy without saving it.

blender -b --python-exit-code 1 --python fp_capture.py -- --scene scene.blend --out start.png
        --right HAND_BONE [--left HAND_BONE] --prop OBJ[,OBJ...] [--alpha 0.35]
"""
import argparse
import sys

import bpy
import numpy as np

RIGHT, LEFT = (1.0, 0.45, 0.05), (0.05, 0.25, 1.0)


def flat(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1)
    b.inputs['Roughness'].default_value = 0.55
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--left')
    p.add_argument('--prop', default='')
    p.add_argument('--alpha', type=float, default=0.35)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    bpy.ops.wm.open_mainfile(filepath=a.scene)
    scene = bpy.context.scene
    rig = next(o for o in scene.objects if o.type == 'ARMATURE' and a.right in o.pose.bones)
    right = flat('capture_right', RIGHT)
    left = flat('capture_left', LEFT)
    wr = np.array(rig.matrix_world @ rig.pose.bones[a.right].head)
    wl = np.array(rig.matrix_world @ rig.pose.bones[a.left].head) if a.left else None
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for o in scene.objects:
        if o.type != 'MESH' or not any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers):
            continue
        posed = o.evaluated_get(depsgraph).to_mesh()
        v = np.array([list(o.matrix_world @ x.co) for x in posed.vertices])
        if wl is None:
            side = np.ones(len(v), int)
        else:
            side = (((v - (wr + wl) / 2) @ (wr - wl)) > 0).astype(int)  # 1: right of the halfway plane
        o.data.materials.clear()
        o.data.materials.append(left)
        o.data.materials.append(right)
        for poly in o.data.polygons:
            poly.material_index = int(round(side[list(poly.vertices)].mean()))
    for name in [n for n in a.prop.split(',') if n]:
        o = bpy.data.objects[name]
        for slot in o.material_slots:
            if slot.material is None:
                continue
            m = slot.material.copy()
            slot.material = m
            if m.use_nodes and 'Principled BSDF' in m.node_tree.nodes:
                m.node_tree.nodes['Principled BSDF'].inputs['Alpha'].default_value = a.alpha
            m.surface_render_method = 'BLENDED'
    scene.render.filepath = a.out
    bpy.ops.render.render(write_still=True)
    print('FPCAPTURE ok')


if __name__ == '__main__':
    main()
