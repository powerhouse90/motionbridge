"""Inspect a character file and write its armature as JSON. Never saves the source.

blender -b --python-exit-code 1 --python dump_rig.py -- --file CHAR --out rig.json [--armature NAME] [--frame N]

Positions are Blender world space (Z-up, meters) with every object transform applied. Each bone's 'start' is its
posed matrix at the start frame (the frame Render Starting Views draws, so the first frame of the video).
"""
import argparse
import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import load, pick_armature, skins  # noqa: E402


def vec(v):
    return [round(x, 6) for x in v]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--armature')
    p.add_argument('--frame', type=int, default=1)
    a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
    path = Path(a.file).resolve()
    load(path)
    rig = pick_armature(a.armature)
    bpy.context.scene.frame_set(a.frame)
    bpy.context.view_layer.update()
    to_world = rig.matrix_world
    # Vertices each bone actually moves: separates real bones from end markers and unused helpers.
    # Girth: median distance from a bone to the vertices it dominates (thin neck vs wide skull).
    weighted, distances = {}, {}
    segment = {b.name: (to_world @ b.head_local, to_world @ b.tail_local) for b in rig.data.bones}
    for o in bpy.data.objects:
        if o.type == 'MESH' and skins(o, rig):
            names = {g.index: g.name for g in o.vertex_groups}
            for v in o.data.vertices:
                best = None
                for g in v.groups:
                    if g.weight > 0.01 and g.group in names:
                        weighted[names[g.group]] = weighted.get(names[g.group], 0) + 1
                        if best is None or g.weight > best[1]:
                            best = (names[g.group], g.weight)
                if best and best[0] in segment:
                    start, end = segment[best[0]]
                    p = o.matrix_world @ v.co
                    axis = end - start
                    t = max(0.0, min(1.0, (p - start).dot(axis) / max(axis.length_squared, 1e-12)))
                    distances.setdefault(best[0], []).append((p - (start + axis * t)).length)
    girth = {n: sorted(d)[len(d) // 2] for n, d in distances.items()}
    bones = []
    for b in rig.data.bones:
        rest = to_world @ b.matrix_local
        start = to_world @ rig.pose.bones[b.name].matrix
        bones.append({'name': b.name, 'parent': b.parent.name if b.parent else None,
                      'head': vec(to_world @ b.head_local), 'tail': vec(to_world @ b.tail_local),
                      'rest': [vec(row) for row in rest], 'start': [vec(row) for row in start], 'deform': b.use_deform, 'weighted_vertices': weighted.get(b.name, 0),
                      'girth': round(girth.get(b.name, 0.0), 5),
                      'children': [c.name for c in b.children]})
    meshes = []
    for o in bpy.data.objects:
        if o.type != 'MESH':
            continue
        if not (skins(o, rig) or o.parent == rig):
            continue
        corners = [o.matrix_world @ v.co for v in o.data.vertices]
        lo = [min(c[i] for c in corners) for i in range(3)] if corners else [0, 0, 0]
        hi = [max(c[i] for c in corners) for i in range(3)] if corners else [0, 0, 0]
        meshes.append({'name': o.name, 'vertices': len(o.data.vertices), 'bounds': [vec(lo), vec(hi)],
                       'vertex_groups': len(o.vertex_groups),
                       'shape_keys': [k.name for k in o.data.shape_keys.key_blocks] if o.data.shape_keys else []})
    result = {'file': path.as_posix(), 'armature': rig.name, 'object_matrix': [vec(r) for r in to_world],
              'unit_scale': bpy.context.scene.unit_settings.scale_length,
              'other_armatures': [o.name for o in bpy.data.objects if o.type == 'ARMATURE' and o != rig],
              'bones': bones, 'meshes': meshes,
              # textures neither packed nor found on disk: the character would render untextured
              'missing_images': [i.name for i in bpy.data.images if i.source == 'FILE' and not i.packed_file
                                 and not Path(bpy.path.abspath(i.filepath)).is_file()]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f'dump_rig: {rig.name}, {len(bones)} bones, {len(meshes)} skinned meshes -> {a.out}')


main()
