"""Loading a character file and finding its rig. Shared by the Blender scripts."""
import bpy


def load(path):
    suffix = path.suffix.lower()
    if suffix == '.blend':
        bpy.ops.wm.open_mainfile(filepath=str(path))
        return
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if suffix == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(path))
    elif suffix in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(path))
    else:
        raise ValueError(f'unsupported character file: {path.name}')


def pick_armature(name):
    armatures = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    if name:
        found = [o for o in armatures if o.name == name]
        if not found:
            raise ValueError(f'no armature named {name!r}; found {[o.name for o in armatures]}')
        return found[0]
    if len(armatures) == 1:
        return armatures[0]
    # Several armatures: the character's rig is the one that deforms a mesh.
    skinning = [o for o in armatures if any(skins(m, o) for m in bpy.data.objects if m.type == 'MESH')]
    if len(skinning) != 1:
        raise ValueError(f'cannot choose among armatures {[o.name for o in armatures]} '
                         f'({len(skinning)} deform a mesh); pass --armature')
    return skinning[0]


def skins(mesh, rig):
    return any(m.type == 'ARMATURE' and m.object == rig for m in mesh.modifiers)
