"""Neutral start scene for a first-person grip take: the arms in a neutral pose on either side (hands low,
palms in, fingers relaxed, NOT placed on the prop) and one prop as the scene's main item, floating in the
middle or resting on a desk. Writes a packed .blend ready for Load FP Scene, and a preview render.

blender -b --disable-autoexec ARMS.blend --python fp_neutral.py -- --prop PROP --out scene.blend
    PROP: a .glb/.gltf/.fbx/.obj file, FILE.blend:Object, or builtin:NAME (with --props-lib props.py)
    [--name Prop] [--length M] [--axis 0|1|2] [--turn KIND,DEG] [--centre F,R,U] [--place float|surface]
    [--hands both|right|left] [--pitch DEG] [--texdir DIR ...] [--drop OBJ,...] [--preview out.png] [--size W H]

Rig: the OGA FPS arms by default (bones hand.R/L, hand.R/L.control, f_*.01.R/L, thumb.*); other rigs need
their bone names passed with --bones (see STATUS.md: not yet general).
"""
import argparse, importlib.util, math, os, sys
import bpy
import numpy as np
from mathutils import Matrix, Vector

UP = Vector((0, 0, 1))


def args():
    p = argparse.ArgumentParser()
    p.add_argument('--prop', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--name', default='')
    p.add_argument('--length', type=float, default=0.0, help='prop size (m) along --axis; 0: preset, else auto')
    p.add_argument('--axis', type=int, default=-1, help='model axis the length is measured on; -1: longest')
    p.add_argument('--turn', default='', help='KIND,DEG: yawed (model +X forward, turned left), upright '
                                              '(+Z up, +X turned from right toward camera), diagonal, screen, as_is')
    p.add_argument('--centre', default='', help='prop centre in the view: forward,right,up (m)')
    p.add_argument('--place', default='float', choices=['float', 'surface'])
    p.add_argument('--hands', default='both', choices=['both', 'right', 'left'])
    p.add_argument('--pitch', type=float, default=None, help='look down (deg); default 35 for surface, else 0')
    p.add_argument('--texdir', action='append', default=[], help='folders to look for missing textures by name')
    p.add_argument('--drop', default='', help='imported objects to remove (e.g. a spare magazine)')
    p.add_argument('--props-lib', default='')
    p.add_argument('--preview', default='')
    p.add_argument('--size', type=int, nargs=2, default=[1152, 640])
    p.add_argument('--bones', default='hand.{s},hand.{s}.control,f_middle.01.{s},f_index.01.{s},f_pinky.01.{s}')
    return p.parse_args(sys.argv[sys.argv.index('--') + 1:])


class View:
    """The camera's axes and the eye; at(forward, right, up) -> world point."""

    def __init__(self, cam):
        self.c0 = cam.matrix_world.translation.copy()
        self.r, self.u, self.f = [cam.matrix_world.to_3x3() @ Vector(v) for v in ((1, 0, 0), (0, 1, 0), (0, 0, -1))]

    def at(self, fw, rt, up):
        return self.c0 + self.f * fw + self.r * rt + self.u * up


def prepare_rig(rig, cam, pitch):
    """Rig to metres about the eye (the OGA file is in decimetres), optional look-down (camera and arms turn
    together, as FPS view arms do), finger copy-rotations muted so fingers are posed per joint."""
    rig.animation_data_clear()
    c0 = cam.matrix_world.translation.copy()
    if max(rig.dimensions) > 3:  # decimetre rig
        rig.matrix_world = Matrix.Translation(c0) @ Matrix.Scale(0.1, 4) @ Matrix.Translation(-c0) @ rig.matrix_world
    if pitch:
        right0 = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))
        P = Matrix.Translation(c0) @ Matrix.Rotation(-math.radians(pitch), 4, right0) @ Matrix.Translation(-c0)
        cam.matrix_world = P @ cam.matrix_world
        rig.matrix_world = P @ rig.matrix_world
    for pb in rig.pose.bones:
        for c in pb.constraints:
            if c.type == 'COPY_ROTATION':
                c.mute = True
    bpy.context.view_layer.update()


def neutral_pose(rig, view, bones):
    """Both hands low on either side, mirror images, palms turned in, fingers relaxed 15 deg."""
    def hand_frame(side):
        hand, _, mid, idx, pinky = [b.format(s=side) for b in bones]
        p = lambda n: rig.matrix_world @ rig.pose.bones[n].head
        w = p(hand)
        d = (p(mid) - w).normalized()
        n = (p(idx) - w).cross(p(pinky) - w).normalized()
        if side == 'L':
            n = -n
        n = (n - n.dot(d) * d).normalized()
        return w, Matrix((d, n, d.cross(n))).transposed()

    def place(side, wrist, d, n):
        pb = rig.pose.bones[bones[1].format(s=side)]
        w0, F0 = hand_frame(side)
        d = d.normalized(); n = (n - n.dot(d) * d).normalized()
        R = (Matrix((d, n, d.cross(n))).transposed() @ F0.transposed()).to_4x4()
        W1 = Matrix.Translation(wrist) @ R @ Matrix.Translation(-w0) @ rig.matrix_world @ pb.matrix
        pb.matrix = rig.matrix_world.inverted() @ W1
        bpy.context.view_layer.update()

    f, r, u = view.f, view.r, view.u
    place('R', view.at(0.36, 0.25, -0.2), f + u * 0.35 - r * 0.15, -r)
    place('L', view.at(0.36, -0.25, -0.2), f + u * 0.35 + r * 0.15, r)
    for pb in rig.pose.bones:
        if pb.name.startswith(('f_', 'thumb')):
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler = (math.radians(15 * (0.5 if pb.name.startswith('thumb') else 1)), 0, 0)
    bpy.context.view_layer.update()
    return [hand_frame('R')[0], hand_frame('L')[0]]


def import_prop(spec, root, drop, lib):
    """Meshes of the prop parented under root (kept at identity while importing: model frame)."""
    scene = bpy.context.scene
    before = set(scene.objects)
    if spec.startswith('builtin:'):
        lib.use(scene)
        lib.BUILDERS[spec.split(':', 1)[1]](root)
    else:
        path, _, obj = spec.partition('.blend:')
        ext = os.path.splitext(spec)[1].lower()
        if obj:
            with bpy.data.libraries.load(path + '.blend') as (src, dst):
                dst.objects = [obj]
            for o in dst.objects:
                scene.collection.objects.link(o)
        elif ext in ('.glb', '.gltf'):
            bpy.ops.import_scene.gltf(filepath=spec)
        elif ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=spec)
        elif ext == '.obj':
            bpy.ops.wm.obj_import(filepath=spec)
        else:
            raise ValueError(f'unsupported prop file: {spec}')
        new = [o for o in scene.objects if o not in before]
        for o in [o for o in new if o.name in drop]:
            new.remove(o); bpy.data.objects.remove(o)
        for o in [o for o in new if o.type in ('LIGHT', 'CAMERA')]:
            new.remove(o); bpy.data.objects.remove(o)
        for o in new:
            if o.parent is None or o.parent not in new:
                w = o.matrix_world.copy(); o.parent = root; o.matrix_world = w
    bpy.context.view_layer.update()
    return [o for o in root.children_recursive if o.type == 'MESH']


def facing(view, kind, deg):
    """Model axes -> world, by named presets relative to the view."""
    f, r, u = view.f, view.r, view.u
    t = math.radians(deg)
    if kind == 'yawed':      # +X forward turned left by deg (its side shows), +Z up
        x = (f * math.cos(t) - r * math.sin(t)).normalized(); return Matrix((x, u.cross(x), u)).transposed()
    if kind == 'upright':    # +Z up, +X from the right turned toward the camera by -deg
        x = (r * math.cos(t) + f * math.sin(t)).normalized(); return Matrix((x, u.cross(x), u)).transposed()
    if kind == 'diagonal':   # +X from the lower right to the upper left across the view, +Y to the right
        x = (-r * 0.7 + u * 0.6 + f * 0.3).normalized(); y = (-r - x * x.dot(-r)).normalized()
        return Matrix((x, y, x.cross(y))).transposed()
    if kind == 'screen':     # +Z toward the camera tipped back deg, turned 15 deg; +Y up
        n = (-f * math.cos(t) + u * math.sin(t)); n = (n * math.cos(0.26) + r * math.sin(0.26)).normalized()
        y = (u - n * n.dot(u)).normalized(); return Matrix((y.cross(n), y, n)).transposed()
    if kind == 'flat':       # lying on a level surface, +X across the view, +Y away
        fh = (f - UP * f.dot(UP)).normalized(); return Matrix((fh.cross(UP), fh, UP)).transposed()
    return Matrix.Identity(3)  # as_is


def points(parts):
    return np.array([list(o.matrix_world @ v.co) for o in parts for v in o.data.vertices])


def fit(root, parts, view, G, length, axis, centre):
    """Scale to length along axis (model frame), centre the bounding box at centre, turned by G."""
    root.matrix_world = Matrix.Identity(4); bpy.context.view_layer.update()
    P = points(parts); lo, hi = P.min(0), P.max(0)
    ext = hi - lo
    axis = int(np.argmax(ext)) if axis < 0 else axis
    scale = length / ext[axis]
    mid = Vector(((lo + hi) / 2).tolist())
    root.matrix_world = Matrix.Translation(view.at(*centre)) @ G.to_4x4() @ Matrix.Scale(scale, 4) @ Matrix.Translation(-mid)
    bpy.context.view_layer.update()
    return scale, ext


def arm_points(rig):
    dg = bpy.context.evaluated_depsgraph_get()
    out = []
    for o in bpy.context.scene.objects:
        if o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == rig for m in o.modifiers):
            me = o.evaluated_get(dg).to_mesh()
            out += [list(o.matrix_world @ v.co) for v in me.vertices]
    return np.array(out)


def clear_of_arms(root, parts, arms, view, gap=0.03, steps=20):
    """Move the prop away from the camera (along the view) until no arm vertex is within gap of its points."""
    arms = arms[::3]
    moved = 0.0
    for _ in range(steps):
        q = points(parts)[::7]
        dmin = min(np.sqrt(((q[i:i + 256, None] - arms[None]) ** 2).sum(-1)).min() for i in range(0, len(q), 256))
        if dmin >= gap:
            break
        root.matrix_world = Matrix.Translation(view.f * 0.02) @ root.matrix_world
        bpy.context.view_layer.update(); moved += 0.02
    return moved


def desk_under(parts, view, scene, lib_mat):
    """A desk slab right under the prop, its front edge 5 cm in front of the prop; level in the world."""
    P = points(parts)
    fh = (view.f - UP * view.f.dot(UP)).normalized(); rh = fh.cross(UP)
    c = Vector(P.mean(0).tolist())
    front = min(Vector(p).dot(fh) for p in P) - c.dot(fh)
    near = c + fh * (front - 0.05)
    kz = P[:, 2].min()
    root = bpy.data.objects.new('Desk', None); scene.collection.objects.link(root)
    corners = [near - rh * 0.55, near + rh * 0.55, near + rh * 0.55 + fh * 0.7, near - rh * 0.55 + fh * 0.7]
    v = []
    for z in (kz - 0.0305, kz - 0.0005):
        v += [(p.x, p.y, z) for p in corners]
    faces = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    me = bpy.data.meshes.new('Desk_top'); me.from_pydata(v, [], faces); me.update()
    o = bpy.data.objects.new('Desk_top', me); scene.collection.objects.link(o); o.parent = root
    o.data.materials.append(lib_mat)
    return root


def desk_material():
    m = bpy.data.materials.new('desk_top'); m.use_nodes = True; nt = m.node_tree
    b = nt.nodes['Principled BSDF']; b.inputs['Roughness'].default_value = 0.85
    t = nt.nodes.new('ShaderNodeTexNoise'); t.inputs['Scale'].default_value = 25
    cr = nt.nodes.new('ShaderNodeValToRGB')
    cr.color_ramp.elements[0].position, cr.color_ramp.elements[1].position = 0.35, 0.65
    cr.color_ramp.elements[0].color = (0.16, 0.115, 0.08, 1); cr.color_ramp.elements[1].color = (0.12, 0.085, 0.06, 1)
    nt.links.new(nt.nodes.new('ShaderNodeTexCoord').outputs['Object'], t.inputs['Vector'])
    nt.links.new(t.outputs['Fac'], cr.inputs['Fac']); nt.links.new(cr.outputs['Color'], b.inputs['Base Color'])
    return m


def light(scene, view):
    w = bpy.data.worlds.new('w'); scene.world = w; w.use_nodes = True
    w.node_tree.nodes['Background'].inputs['Color'].default_value = (0.22, 0.23, 0.25, 1)
    w.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.6
    key = bpy.data.objects.new('FP_key', bpy.data.lights.new('FP_key', 'AREA')); key.data.energy = 60; key.data.size = 1.2
    key.location = view.c0 + view.r * 0.5 + view.u * 0.7 + view.f * 0.1
    key.rotation_euler = (view.at(0.4, 0.05, -0.1) - key.location).to_track_quat('-Z', 'Y').to_euler()
    scene.collection.objects.link(key)
    fill = bpy.data.objects.new('FP_fill', bpy.data.lights.new('FP_fill', 'SUN')); fill.data.energy = 1.2
    fill.rotation_euler = (math.radians(60), 0, math.radians(200)); scene.collection.objects.link(fill)


def pack_images(texdirs):
    """Pack every image; a missing file is looked up by name in texdirs; still missing ones are reported
    (the scene still saves: Load FP Scene warns about them)."""
    missing = []
    for im in bpy.data.images:
        if im.source != 'FILE' or im.packed_file:
            continue
        path = bpy.path.abspath(im.filepath)
        if not os.path.isfile(path):
            for d in texdirs:
                for root, _, files in os.walk(d):
                    if os.path.basename(path) in files:
                        path = os.path.join(root, os.path.basename(path)); break
                if os.path.isfile(path):
                    im.filepath = path; im.reload(); break
        if os.path.isfile(path):
            im.pack()
        else:
            missing.append(im.name)
    return missing


def main():
    a = args()
    scene = bpy.context.scene
    cam = scene.camera
    rig = next(o for o in scene.objects if o.type == 'ARMATURE')
    for o in [o for o in scene.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o)
    lib = None
    if a.props_lib:
        spec = importlib.util.spec_from_file_location('fp_props', a.props_lib)
        lib = importlib.util.module_from_spec(spec); spec.loader.exec_module(lib)
    preset = lib.PRESETS.get(a.prop.split(':', 1)[1], {}) if a.prop.startswith('builtin:') and lib else {}
    pitch = a.pitch if a.pitch is not None else (35 if a.place == 'surface' else 0)
    prepare_rig(rig, cam, pitch)
    view = View(cam)
    bones = a.bones.split(',')
    wrists = neutral_pose(rig, view, bones)
    hand_len = (rig.matrix_world @ rig.pose.bones[bones[2].format(s='R')].head - wrists[0]).length  # wrist -> middle knuckle

    name = a.name or (a.prop.split(':', 1)[1].capitalize() if a.prop.startswith('builtin:') else
                      os.path.splitext(os.path.basename(a.prop.split('.blend:')[-1]))[0])
    root = bpy.data.objects.new(name, None); scene.collection.objects.link(root)
    parts = import_prop(a.prop, root, set(filter(None, a.drop.split(','))), lib)
    if not parts:
        raise ValueError(f'no meshes in {a.prop}')
    for o in parts:
        o['motionbridge_prop'] = name
    # size: given, preset, or from the hand (a hand-held prop about 2.2 hand-lengths long: wrist->knuckle ~ 9 cm)
    length = a.length or preset.get('length') or 2.2 * hand_len
    axis = a.axis if a.axis >= 0 else preset.get('axis', -1)
    if a.turn:
        kind, deg = (a.turn.split(',') + ['0'])[:2]
    elif a.place == 'surface':
        kind, deg = 'flat', 0
    else:
        kind, deg = preset.get('turn', ('as_is', 0))
    centre = [float(v) for v in a.centre.split(',')] if a.centre else list(preset.get('centre', (0.5, 0.0, -0.1) if a.place == 'surface' else (0.45, 0.0, -0.07)))
    if a.hands != 'both':  # the take is for one hand: the prop a little toward that hand
        centre[1] += 0.04 if a.hands == 'right' else -0.04
    scale, ext = fit(root, parts, view, facing(view, kind, float(deg)), length, axis, centre)
    moved = clear_of_arms(root, parts, arm_points(rig), view)
    if a.place == 'surface':
        desk_under(parts, view, scene, desk_material())
    light(scene, view)
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = a.size[0], a.size[1], 100
    cam.data.clip_start = 0.01
    scene.frame_set(1)
    missing = pack_images(a.texdir)
    bpy.ops.wm.save_as_mainfile(filepath=a.out, copy=True)
    if a.preview:
        scene.render.filepath = a.preview; bpy.ops.render.render(write_still=True)
    print(f'FPNEUTRAL ok prop "{name}" ({len(parts)} meshes) raw size {np.round(ext, 3).tolist()} scale {scale:.4f} '
          f'length {length:.3f} m, turned {kind} {deg}, pushed back {moved:.2f} m to clear the arms, '
          f'missing textures: {missing or "none"}')


if __name__ == '__main__':
    main()
