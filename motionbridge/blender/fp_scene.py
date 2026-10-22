"""Prepare a first-person scene: FPS arms + camera + the held item, all in one file.

blender -b --python-exit-code 1 --python fp_scene.py -- --file SCENE --outdir DIR [--armature NAME]
        [--items 'A,B;C'] [--size W H] [--lights auto|keep]

Writes into DIR (the source file is never saved):
  scene.blend   working copy: the start pose baked into the bones (stored animation and
                non-IK constraints removed, so nothing re-poses the hands); each item's objects
                parented to one empty, MB_<item>, that the export animates
  start.png     the start frame from the scene camera (H3's first frame)
  rig.json      bones (rest, basis, evaluated pose), rig and camera matrices, lens, image size, and
                per item: its objects, its empty, and points on it the camera sees (SAM 2 prompts)
  item_K.npz    item K's vertices (world) per object, and its signed distance grid (raw parity
                test; core.fp_track seals non-watertight meshes)
  item_K_tris.npy  item K's triangles (N, 3, 3), world, at the scene placement: where a point picked on
                its texture in the start frame lies on the model (core.fp_points)
Items are what moves on its own (a pistol, its slide, a magazine): every visible mesh no armature
deforms, grouped by its top-level parent, so parts under one empty move as one. --items names
groups instead: "A,B;C" is two items, A+B and C. Empties named MB_socket_R / MB_socket_L under an item
mark where a hand is meant to hold it (axes as the hand bone's); a hand already holding an item in the
start frame needs none.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import load, pick_armature  # noqa: E402

GRID_STEP = 0.003   # m, signed distance grid spacing
GRID_MARGIN = 0.05  # m around the item


def rows(m):
    return [list(r) for r in m]


def bake_start_pose(rig):
    """Freeze what the rig shows now into its bones: later frames are keyed explicitly."""
    pose = {pb.name: pb.matrix.copy() for pb in rig.pose.bones}
    rig.animation_data_clear()
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if c.type != 'IK':  # IK stays: the arm follows the hand's control
                c.mute = True
    order = sorted(rig.pose.bones, key=lambda b: len(b.parent_recursive))
    for pb in order:
        pb.matrix = pose[pb.name]
        bpy.context.view_layer.update()


def item_groups(rig, spec):
    """{name: [mesh objects]}: one group per top-level parent, or as named in spec."""
    if spec:
        groups = {}
        for g in spec.split(';'):
            names = [n.strip() for n in g.split(',') if n.strip()]
            found = [bpy.data.objects[n] for n in names]
            groups[names[0]] = [o for root in found for o in [root] + list(root.children_recursive) if o.type == 'MESH']
        return groups
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH' and not o.hide_render and o.visible_get()
              and not any(m.type == 'ARMATURE' for m in o.modifiers) and o.parent != rig]
    groups = {}
    for o in meshes:
        top = o
        while top.parent is not None and top.parent != rig:
            top = top.parent
        groups.setdefault(top.name, []).append(o)
    return groups


def item_sockets(objs):
    """Empties named MB_socket_R / MB_socket_L anywhere under the item's top-level object: where a hand
    is meant to hold it (the empty's axes match the hand bone's)."""
    top = objs[0]
    while top.parent is not None and top.parent.type != 'ARMATURE':
        top = top.parent
    out = []
    for o in [top] + list(top.children_recursive):
        if o.type == 'EMPTY' and o.name.startswith('MB_socket'):
            side = 'L' if o.name.rstrip('.0123456789').upper().endswith(('L', 'LEFT')) else 'R'
            out.append({'side': side, 'name': o.name, 'matrix': rows(o.matrix_world)})
    return out


def visible_points(objs, scene, size, count=40):
    """Screen points (pixels) on these objects that the camera sees directly."""
    from bpy_extras.object_utils import world_to_camera_view
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam = scene.camera
    origin = cam.matrix_world.translation
    names = {o.name for o in objs}
    pts = [o.matrix_world @ v.co for o in objs for v in o.data.vertices]
    rng = np.random.default_rng(0)
    out = []
    for i in rng.permutation(len(pts))[:4000]:
        p = pts[i]
        d = (p - origin)
        hit, loc, _, _, obj, _ = scene.ray_cast(depsgraph, origin, d.normalized(), distance=d.length + 1e-3)
        if hit and obj.name in names and (loc - p).length < 0.004:
            c = world_to_camera_view(scene, cam, p)
            if 0 < c.x < 1 and 0 < c.y < 1:
                out.append([c.x * size[0], (1 - c.y) * size[1]])
        if len(out) >= count:
            break
    return out


def signed_distance(objs):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bm = bmesh.new()
    for o in objs:
        mesh = o.evaluated_get(depsgraph).to_mesh()
        mesh.transform(o.matrix_world)
        bm.from_mesh(mesh)
    tree = BVHTree.FromBMesh(bm)
    verts = np.array([list(v.co) for v in bm.verts])
    lo, hi = verts.min(0) - GRID_MARGIN, verts.max(0) + GRID_MARGIN
    axes = [np.arange(lo[k], hi[k] + GRID_STEP, GRID_STEP) for k in range(3)]
    rays = [Vector((0.577, 0.577, 0.577)), Vector((-0.3, 0.8, -0.52)).normalized(), Vector((0.1, -0.2, 0.97)).normalized()]
    grid = np.zeros([len(a) for a in axes], np.float32)
    for i, x in enumerate(axes[0]):
        for j, y in enumerate(axes[1]):
            for k, z in enumerate(axes[2]):
                p = Vector((x, y, z))
                inside = 0
                for d in rays:  # odd number of crossings = inside (majority of three rays)
                    n, q = 0, p.copy()
                    while True:
                        hit = tree.ray_cast(q, d)
                        if hit[0] is None:
                            break
                        n, q = n + 1, hit[0] + d * 1e-5
                    inside += n % 2
                dist = tree.find_nearest(p)[3]
                grid[i, j, k] = -dist if inside >= 2 else dist
    return grid, axes


def default_lights(scene):
    world = bpy.data.worlds.new('MotionBridge_world')
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.22, 0.23, 0.25, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.6
    cam = scene.camera.matrix_world
    right, up, fwd = [cam.to_3x3() @ Vector(v) for v in ((1, 0, 0), (0, 1, 0), (0, 0, -1))]
    key = bpy.data.objects.new('MotionBridge_key', bpy.data.lights.new('MotionBridge_key', 'AREA'))
    key.data.energy, key.data.size = 60, 1.2
    key.location = cam.translation + right * 0.5 + up * 0.7 + fwd * 0.1
    key.rotation_euler = (cam.translation + fwd * 0.4 - key.location).to_track_quat('-Z', 'Y').to_euler()
    fill = bpy.data.objects.new('MotionBridge_fill', bpy.data.lights.new('MotionBridge_fill', 'SUN'))
    fill.data.energy = 1.2
    fill.rotation_euler = (math.radians(60), 0, math.radians(200))
    for o in (key, fill):
        scene.collection.objects.link(o)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--outdir', required=True)
    p.add_argument('--armature')
    p.add_argument('--items', default='')
    p.add_argument('--size', type=int, nargs=2, default=[1280, 720])
    p.add_argument('--lights', default='auto')
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    load(Path(a.file).resolve())
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError('the scene has no camera: add the first-person camera and make it the scene camera')
    scene.frame_set(scene.frame_start)
    rig = pick_armature(a.armature)
    bake_start_pose(rig)

    groups = item_groups(rig, a.items)
    if not groups:
        raise ValueError('no item found: put the held object (a mesh not deformed by the arms) in the scene, or name it')
    for k, (name, objs) in enumerate(groups.items()):
        root = bpy.data.objects.new(f'MB_{name}', None)
        scene.collection.objects.link(root)
        for o in objs:
            o.animation_data_clear()
        for o in objs:
            if o.parent is None or o.parent not in objs:
                world = o.matrix_world.copy()
                o.parent = root
                o.matrix_world = world
    bpy.context.view_layer.update()

    if a.lights == 'auto' and not any(o.type == 'LIGHT' for o in scene.objects):
        default_lights(scene)
    if scene.render.engine not in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT', 'CYCLES'):
        scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = a.size
    scene.render.resolution_percentage = 100
    scene.camera.data.sensor_fit = 'HORIZONTAL'  # the tracker's camera model: focal from width

    missing = [im.name for im in bpy.data.images if im.source == 'FILE' and not im.packed_file and not Path(bpy.path.abspath(im.filepath)).is_file()]
    if missing:
        print('WARNING: textures not found, the arms or items will render untextured:', ', '.join(missing))
    for im in bpy.data.images:  # the working copy lives elsewhere; relative texture paths would break
        if im.source == 'FILE' and not im.packed_file and im.name not in missing:
            im.pack()
    bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'), copy=True)
    scene.render.filepath = str(out / 'start.png')
    bpy.ops.render.render(write_still=True)

    cam = scene.camera
    dump = {'armature': rig.name, 'rig_world': rows(rig.matrix_world),
            'bones': [{'name': b.name, 'parent': b.parent.name if b.parent else None, 'rest': rows(b.matrix_local),
                       'length': b.length, 'basis': rows(rig.pose.bones[b.name].matrix_basis),
                       'pose': rows(rig.pose.bones[b.name].matrix)} for b in rig.data.bones],
            'cam_world': rows(cam.matrix_world), 'lens': cam.data.lens, 'sensor': cam.data.sensor_width,
            'W': a.size[0], 'H': a.size[1],
            'items': [{'name': name, 'objects': [o.name for o in objs], 'root': f'MB_{name}', 'file': f'item_{k}.npz',
                       'tris': f'item_{k}_tris.npy', 'visible': visible_points(objs, scene, a.size), 'sockets': item_sockets(objs)}
                      for k, (name, objs) in enumerate(groups.items())]}
    (out / 'rig.json').write_text(json.dumps(dump))

    depsgraph = bpy.context.evaluated_depsgraph_get()
    for k, objs in enumerate(groups.values()):
        verts, tris = {}, []
        for o in objs:
            mesh = o.evaluated_get(depsgraph).to_mesh()
            verts[o.name] = np.array([list(o.matrix_world @ v.co) for v in mesh.vertices], np.float32)
            mesh.calc_loop_triangles()
            idx = np.empty(len(mesh.loop_triangles) * 3, np.int32)
            mesh.loop_triangles.foreach_get('vertices', idx)
            tris.append(verts[o.name][idx.reshape(-1, 3)])
        np.save(out / f'item_{k}_tris.npy', np.concatenate(tris) if tris else np.zeros((0, 3, 3), np.float32))
        grid, axes = signed_distance(objs)
        np.savez_compressed(out / f'item_{k}.npz', names=np.array(list(verts)), sdf=grid, x=axes[0], y=axes[1], z=axes[2],
                            **{f'v_{i}': v for i, v in enumerate(verts.values())})
    print('FPSCENE ok', rig.name, len(groups), 'items:', ', '.join(f'{n} ({len(o)})' for n, o in groups.items()))


if __name__ == '__main__':
    main()
