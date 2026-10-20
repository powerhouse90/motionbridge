"""Render baked actions on the textured character over a checker floor (Eevee). Test tooling.

blender -b --python-exit-code 1 --python tests/render_mesh_video.py -- --blend BAKED.blend
        --actions h3_walk h3_walk_noik --outdir DIR [--size 360 480]

Writes DIR/<action>/0001.png ... One fixed camera per action, framed on the whole clip,
front three-quarter and slightly above so feet against the checker floor are easy to read.
"""
import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def skinned_meshes(rig):
    return [o for o in bpy.context.scene.objects if o.type == 'MESH' and any(
        m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)]


def mesh_points(meshes, step=40):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pts = []
    for m in meshes:
        ev = m.evaluated_get(depsgraph)
        deformed = ev.to_mesh()
        pts += [ev.matrix_world @ v.co for v in list(deformed.vertices)[::step]]
        ev.to_mesh_clear()
    return pts


def add_floor(height):
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, height))
    floor = bpy.context.active_object
    mat = bpy.data.materials.new('checker_floor')
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    checker = nodes.new('ShaderNodeTexChecker')
    checker.inputs['Scale'].default_value = 48  # 25 cm squares on a 12 m plane
    checker.inputs['Color1'].default_value = (0.55, 0.55, 0.55, 1)
    checker.inputs['Color2'].default_value = (0.25, 0.25, 0.25, 1)
    links.new(checker.outputs['Color'], nodes['Principled BSDF'].inputs['Base Color'])
    floor.data.materials.append(mat)
    return floor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blend', required=True)
    p.add_argument('--actions', nargs='+', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--size', nargs=2, type=int, default=[360, 480])
    p.add_argument('--view', choices=['front34', 'side', 'profile', 'front'], default='front34')
    p.add_argument('--hips-pair', nargs=2, metavar=('LEFT_THIGH', 'RIGHT_THIGH'),
                   help='profile view: thigh bones that give the body left-right axis each frame')
    p.add_argument('--square-pair', nargs=2, metavar=('LEFT', 'RIGHT'),
                   help='square the camera to this pair instead (e.g. upper arms, to judge the upper body)')
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    outdir = Path(a.outdir).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    scene = bpy.context.scene
    rig = next(o for o in scene.objects if o.type == 'ARMATURE' and o.animation_data)
    meshes = skinned_meshes(rig)
    for o in scene.objects:
        o.hide_render = o.type != 'MESH' or o not in meshes

    rig.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    floor_z = min(q.z for q in mesh_points(meshes, 1))
    rig.data.pose_position = 'POSE'
    add_floor(floor_z)

    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x, scene.render.resolution_y = a.size
    scene.render.image_settings.file_format = 'PNG'
    world = bpy.data.worlds.new('grey') if not scene.world else scene.world
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.12, 0.12, 0.13, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
    sun_data = bpy.data.lights.new('key', 'SUN')
    sun_data.energy = 3.5
    sun = bpy.data.objects.new('key', sun_data)
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(sun)
    cam_data = bpy.data.cameras.new('view')
    cam_data.lens = 50
    cam = bpy.data.objects.new('view', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    for action in a.actions:
        rig.animation_data.action = bpy.data.actions[action]
        start, end = (int(x) for x in bpy.data.actions[action].frame_range)
        scene.frame_start, scene.frame_end = start, end
        pts = []
        for f in range(start, end + 1, 6):
            scene.frame_set(f)
            pts += mesh_points(meshes)
        lo = Vector([min(q[i] for q in pts) for i in range(3)])
        hi = Vector([max(q[i] for q in pts) for i in range(3)])
        lo.z = floor_z
        centre = (lo + hi) / 2
        radius = (hi - lo).length / 2
        # Front three-quarter, 15 degrees above (or a level profile). The rigs here face -Y.
        yaw, pitch = (30, 15) if a.view == 'front34' else (90, 0)
        direction = Vector((math.sin(math.radians(yaw)), -math.cos(math.radians(yaw)), math.tan(math.radians(pitch))))
        direction.normalize()
        fov = 2 * math.atan(36 / 2 / cam_data.lens) * min(1, a.size[0] / a.size[1])
        cam.location = centre + direction * radius / math.sin(fov / 2) * 1.05
        cam.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
        target = outdir / (action if a.view == 'front34' else action + '_' + a.view)
        target.mkdir(parents=True, exist_ok=True)
        if a.view not in ('profile', 'front'):
            scene.render.filepath = str(target) + '/'
            bpy.ops.render.render(animation=True)
        else:
            # Tracking camera every frame, level, following the hips: at the right side (profile) or in front.
            distance = radius / math.sin(fov / 2) * 1.05
            for f in range(start, end + 1):
                scene.frame_set(f)
                l_thigh = rig.matrix_world @ rig.pose.bones[a.hips_pair[0]].head
                r_thigh = rig.matrix_world @ rig.pose.bones[a.hips_pair[1]].head
                pair = a.square_pair or a.hips_pair
                side = (rig.matrix_world @ rig.pose.bones[pair[1]].head) - (rig.matrix_world @ rig.pose.bones[pair[0]].head)
                side.z = 0
                side.normalize()
                pelvis = (l_thigh + r_thigh) / 2
                aim = Vector((pelvis.x, pelvis.y, centre.z))
                look = side if a.view == 'profile' else Vector((0, 0, 1)).cross(side)  # front: up x right = forward
                cam.location = aim + look * distance
                cam.rotation_euler = (-look).to_track_quat('-Z', 'Y').to_euler()
                scene.render.filepath = str(target / f'{f:04d}.png')
                bpy.ops.render.render(write_still=True)
        print(f'render_mesh_video: {action} frames {start}-{end} -> {target}', flush=True)


main()
