"""Blender side of the finger benchmark: pose a hand from given basis matrices, frame a first-person
camera on it, render, and report Blender's own screen positions of the 21 hand points.

blender -b --disable-autoexec --python tests/finger_bench_render.py -- REQUEST.json
REQUEST: {file, armature, bones: [names], basis: (T, B, 4, 4) npy path, keypoints: [[bone, head|tail]],
          outdir, size: [w, h]}
Constraints on the rig are muted (the benchmark keys every finger joint explicitly).
"""
import json, math, sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'motionbridge' / 'blender'))
from scene import load, pick_armature  # noqa: E402

req = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())
out = Path(req['outdir']); out.mkdir(parents=True, exist_ok=True)
load(Path(req['file']))
rig = pick_armature(req.get('armature'))
scene = bpy.context.scene
for o in scene.objects:
    if o.type == 'ARMATURE' and o.animation_data:
        o.animation_data.action = None
    # render only the character skinned to this rig: props in the source file would hide the fingers
    skinned = o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers)
    o.hide_render = o.type != 'LIGHT' and not skinned
for pb in rig.pose.bones:
    for c in pb.constraints:
        c.mute = True
    pb.rotation_mode = 'QUATERNION'
    pb.matrix_basis = Matrix.Identity(4)
basis = np.load(req['basis'])
names = req['bones']
T = len(basis)

def apply(t):
    for i, n in enumerate(names):
        rig.pose.bones[n].matrix_basis = Matrix(basis[t, i].tolist())
    bpy.context.view_layer.update()

def point(bone, end):
    pb = rig.pose.bones[bone]
    return rig.matrix_world @ (pb.head if end == 'head' else pb.tail)

# first-person camera framed on frame 0's hand: behind the wrist and over the back of the hand, the
# way FPS hands are seen (palm normal from the hand model; the back of the hand faces away from it)
apply(0)
kb = req['keypoints']
wrist, tip = point(*kb[0]), point(*kb[12])
L = (tip - wrist).length
d = (tip - wrist).normalized()
back = -Vector(req['palm_normal']).normalized()
up = back
inward = back.cross(d).normalized() * (1 if req.get('side', 'Right') == 'Right' else -1)
centre = (wrist + tip) / 2
for o in [o for o in scene.objects if o.type == 'CAMERA']:
    bpy.data.objects.remove(o)  # the file's own cameras (and markers bound to them) would take over the render
scene.timeline_markers.clear()
cam = bpy.data.objects.new('bench_cam', bpy.data.cameras.new('bench_cam'))
cam.data.lens, cam.data.sensor_width, cam.data.clip_start = 24, 36, L * 0.05
scene.collection.objects.link(cam); scene.camera = cam
cam.location = centre - d * 2.2 * L + back * 1.6 * L + inward * 0.6 * L
cam.rotation_euler = (centre - cam.location).to_track_quat('-Z', 'Y').to_euler()
scene.render.engine = 'BLENDER_EEVEE'
scene.eevee.taa_render_samples = 8
scene.render.resolution_x, scene.render.resolution_y = req['size']
scene.render.resolution_percentage = 100  # source files may render at a fraction of their resolution
world = bpy.data.worlds.new('bench'); scene.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.2, 0.21, 0.23, 1)
for rot, energy in (((50, 0, 30), 3.0), ((60, 0, 210), 1.5)):
    sun = bpy.data.objects.new('bench_sun', bpy.data.lights.new('bench_sun', 'SUN'))
    sun.data.energy = energy; sun.rotation_euler = [math.radians(a) for a in rot]
    scene.collection.objects.link(sun)
W, H = req['size']
uv = np.zeros((T, len(kb), 3))
for t in range(T):
    apply(t)
    for k, (b, e) in enumerate(kb):
        c = world_to_camera_view(scene, cam, point(b, e))
        uv[t, k] = (c.x * W, (1 - c.y) * H, c.z)
    scene.render.filepath = str(out / f'{t:04d}.png')
    bpy.ops.render.render(write_still=True)
np.save(out / 'truth_uv.npy', uv)
(out / 'camera.json').write_text(json.dumps({'world': [list(r) for r in cam.matrix_world], 'lens': cam.data.lens,
                                             'sensor': cam.data.sensor_width, 'W': W, 'H': H}))
print('BENCH rendered', T, 'frames')
