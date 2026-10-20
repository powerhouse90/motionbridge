"""Skinning check, no retargeting: rest pose with only the upper arms lowered by ANGLE degrees.

blender -b --python tests/render_arms_down.py -- --blend BAKED.blend --left BONE --right BONE --out still.png
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blend', required=True)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--angle', type=float, default=75)
    p.add_argument('--out', required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = str(Path(a.out).resolve())
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    scene = bpy.context.scene
    rig = next(o for o in scene.objects if o.type == 'ARMATURE' and o.animation_data)
    rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.location = (0, 0, 0)
    bpy.context.view_layer.update()
    for name, sign in ((a.left, -1), (a.right, 1)):
        pb = rig.pose.bones[name]
        # rotate about the world forward axis (-Y facing rigs): left arm down = negative about -Y
        world = rig.matrix_world @ pb.matrix
        rot = Matrix.Rotation(math.radians(a.angle) * sign, 4, Vector((0, 1, 0)))
        new = Matrix.Translation(world.translation) @ rot @ Matrix.Translation(-world.translation) @ world
        pb.matrix = rig.matrix_world.inverted() @ new
        bpy.context.view_layer.update()
    meshes = [o for o in scene.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)]
    for o in scene.objects:
        o.hide_render = o not in meshes
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = 600, 700
    sun = bpy.data.objects.new('key', bpy.data.lights.new('key', 'SUN'))
    sun.data.energy = 3.5
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new('view', bpy.data.cameras.new('view'))
    scene.collection.objects.link(cam)
    scene.camera = cam
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pts = []
    for m in meshes:
        ev = m.evaluated_get(depsgraph)
        me = ev.to_mesh()
        pts += [ev.matrix_world @ v.co for v in me.vertices]
        ev.to_mesh_clear()
    lo = Vector([min(q[i] for q in pts) for i in range(3)])
    hi = Vector([max(q[i] for q in pts) for i in range(3)])
    # upper body, front view
    centre = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z + (hi.z - lo.z) * 0.72))
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = (hi.z - lo.z) * 0.6
    cam.location = centre + Vector((0, -5, 0))
    cam.rotation_euler = (math.radians(90), 0, 0)
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)


main()
