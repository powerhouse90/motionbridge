"""Workbench stills of a baked character at chosen frames, front and side. Test tooling.

blender -b --python-exit-code 1 --python tests/render_mesh.py -- --blend BAKED.blend --action NAME
        --frames 1 37 73 --out still.png
"""
import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blend', required=True)
    p.add_argument('--action', required=True)
    p.add_argument('--frames', nargs='+', type=int, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    a.out = str(Path(a.out).resolve())  # Blender resolves relative paths against the opened file
    bpy.ops.wm.open_mainfile(filepath=str(Path(a.blend).resolve()))
    rig = next(o for o in bpy.context.scene.objects if o.type == 'ARMATURE' and o.animation_data)
    rig.animation_data.action = bpy.data.actions[a.action]
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_WORKBENCH'
    scene.display.shading.light = 'STUDIO'
    scene.display.shading.color_type = 'MATERIAL'
    scene.render.resolution_x, scene.render.resolution_y = 360, 480
    scene.render.film_transparent = False
    meshes = [o for o in scene.objects if o.type == 'MESH' and any(
        m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)]
    for o in scene.objects:  # hide everything but the character
        o.hide_render = o not in meshes
    cam_data = bpy.data.cameras.new('still')
    cam_data.type = 'ORTHO'
    cam = bpy.data.objects.new('still', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    import numpy as np
    tiles = []
    for frame in a.frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        pts = []
        for m in meshes:
            ev = m.evaluated_get(depsgraph)
            deformed = ev.to_mesh()
            pts += [ev.matrix_world @ v.co for v in list(deformed.vertices)[::25]]
            ev.to_mesh_clear()
        lo = Vector([min(q[i] for q in pts) for i in range(3)])
        hi = Vector([max(q[i] for q in pts) for i in range(3)])
        centre, size = (lo + hi) / 2, max(hi - lo) * 1.15
        cam_data.ortho_scale = size
        for view, direction in (('front', Vector((0, -1, 0))), ('side', Vector((1, 0, 0)))):
            cam.location = centre + direction * size * 3
            cam.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()
            path = Path(a.out).with_name(f'{Path(a.out).stem}_{frame}_{view}.png')
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            tiles.append(path)
    images = [bpy.data.images.load(str(t)) for t in tiles]
    w, h = images[0].size
    sheet = np.zeros((h, w * len(images), 4), np.float32)
    for i, im in enumerate(images):
        sheet[:, i * w:(i + 1) * w] = np.array(im.pixels[:], np.float32).reshape(h, w, 4)
    out = bpy.data.images.new('sheet', w * len(images), h)
    out.pixels = sheet.ravel()
    out.filepath_raw = a.out
    out.file_format = 'PNG'
    out.save()
    for t in tiles:
        t.unlink()


main()
