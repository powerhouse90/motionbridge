"""Render GEM's own SOMA body (no retargeting) with the same floor and camera as render_mesh_video.

Prepare frames first (outside Blender):  python -m tests.render_soma_mesh prepare EXPORT.npz OUT.npz
Then:  blender -b --python tests/render_soma_mesh.py -- --frames OUT.npz --outdir DIR [--view side]
"""
import math
import sys
from pathlib import Path

import numpy as np


def prepare(export, out):
    """SOMA world vertices -> Blender world (Z-up, facing -Y), turned like retarget's face_start."""
    from motionbridge.core import soma, retarget
    d = np.load(export)
    motion = soma.from_gem_export(export, 24)
    world, left, fwd = retarget.soma_facing(motion)
    # recover the yaw soma_facing applied, from the hips' turned vs original rotation
    yaw = world[0, motion.index('Hips'), :3, :3] @ np.linalg.inv(motion.world[0, motion.index('Hips'), :3, :3])
    to_blender = np.column_stack([left, -fwd, [0, 1, 0]]).T  # rows: Blender X(left), Y(back), Z(up)
    v = d['global_vertices'] @ yaw.T @ to_blender.T
    hips = motion.positions[:, [motion.index('LeftLeg'), motion.index('RightLeg')]] @ yaw.T @ to_blender.T
    shoulders = motion.positions[:, [motion.index('LeftArm'), motion.index('RightArm')]] @ yaw.T @ to_blender.T
    floor = np.percentile(v[..., 2].min(axis=1), 5)
    shift = v[0, :, :2].mean(axis=0)
    v[..., 2] -= floor
    v[..., :2] -= shift
    for pair in (hips, shoulders):
        pair[..., 2] -= floor
        pair[..., :2] -= shift
    np.savez_compressed(out, vertices=v.astype(np.float32), faces=d['faces'], hips=hips.astype(np.float32),
                        shoulders=shoulders.astype(np.float32))


def render(frames_file, outdir, view, track='hips'):
    import bpy
    from mathutils import Vector
    data = np.load(frames_file)
    verts, faces = data['vertices'], data['faces']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    mesh = bpy.data.meshes.new('soma')
    mesh.from_pydata(verts[0].tolist(), [], faces.tolist())
    body = bpy.data.objects.new('soma', mesh)
    scene.collection.objects.link(body)
    mat = bpy.data.materials.new('skin')
    mat.diffuse_color = (0.8, 0.62, 0.5, 1)
    mat.use_nodes = True
    mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.8, 0.62, 0.5, 1)
    body.data.materials.append(mat)
    for p in mesh.polygons:
        p.use_smooth = True
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    floor = bpy.context.active_object
    fm = bpy.data.materials.new('checker')
    fm.use_nodes = True
    checker = fm.node_tree.nodes.new('ShaderNodeTexChecker')
    checker.inputs['Scale'].default_value = 48
    checker.inputs['Color1'].default_value = (0.55, 0.55, 0.55, 1)
    checker.inputs['Color2'].default_value = (0.25, 0.25, 0.25, 1)
    fm.node_tree.links.new(checker.outputs['Color'], fm.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    floor.data.materials.append(fm)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 8
    scene.render.resolution_x, scene.render.resolution_y = 600, 700
    world = bpy.data.worlds.new('grey')
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.12, 0.12, 0.13, 1)
    sun = bpy.data.objects.new('key', bpy.data.lights.new('key', 'SUN'))
    sun.data.energy = 3.5
    sun.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new('view', bpy.data.cameras.new('view'))
    cam.data.lens = 50
    scene.collection.objects.link(cam)
    scene.camera = cam
    lo, hi = verts.reshape(-1, 3).min(0), verts.reshape(-1, 3).max(0)
    lo[2] = 0
    centre, radius = Vector((lo + hi) / 2), float(np.linalg.norm(hi - lo) / 2)
    yaw_deg, pitch = (30, 15) if view == 'front34' else (90, 0)
    direction = Vector((math.sin(math.radians(yaw_deg)), -math.cos(math.radians(yaw_deg)), math.tan(math.radians(pitch))))
    direction.normalize()
    fov = 2 * math.atan(36 / 2 / cam.data.lens) * min(1, 600 / 700)
    cam.location = centre + direction * radius / math.sin(fov / 2) * 1.05
    cam.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    distance = radius / math.sin(fov / 2) * 1.05
    for t in range(len(verts)):
        mesh.vertices.foreach_set('co', verts[t].ravel())
        mesh.update()
        if view in ('profile', 'front'):  # same tracking camera as render_mesh_video, level
            l_hip, r_hip = Vector(data[track][t, 0].tolist()), Vector(data[track][t, 1].tolist())
            side = r_hip - l_hip
            side.z = 0
            side.normalize()
            pelvis = (l_hip + r_hip) / 2
            aim = Vector((pelvis.x, pelvis.y, centre.z))
            look = side if view == 'profile' else Vector((0, 0, 1)).cross(side)  # front: up x right = forward
            cam.location = aim + look * distance
            cam.rotation_euler = (-look).to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = str(Path(outdir) / f'{t + 1:04d}.png')
        bpy.ops.render.render(write_still=True)


if __name__ == '__main__':
    if 'bpy' in sys.modules or '--' in sys.argv:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument('--frames', required=True)
        p.add_argument('--outdir', required=True)
        p.add_argument('--view', default='side')
        p.add_argument('--track', choices=['hips', 'shoulders'], default='hips',
                       help='which left-right pair the tracking camera squares up to')
        a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
        render(str(Path(a.frames).resolve()), str(Path(a.outdir).resolve()), a.view, a.track)
    else:
        prepare(sys.argv[2], sys.argv[3])
