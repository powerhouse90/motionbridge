"""SOMA motion -> any mapped humanoid rig, keeping the rig's own bone lengths.

For every mapped bone b following SOMA joint j:

    world_rotation_b(t) = delta_j(t) @ align_b @ rest_rotation_b

delta_j is SOMA's world-space rotation away from its rest pose, expressed in the
rig's frame. align_b is a swing that turns the rig's rest bone onto SOMA's rest
segment, so a T-pose rig follows an A-pose source (and the reverse). Unmapped
bones (twist helpers, pelvis aux, extras) keep their rest pose relative to their
parent. Forward kinematics then rebuilds positions from the rig's own offsets,
and root travel is scaled by the hip-to-ankle length ratio.

Everything runs in armature space (Blender bone convention: Y along the bone).
The result holds per-bone local pose quaternions/locations for Blender to key,
plus world-space heads/tails for measuring and drawing.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

SOMA_UP = np.array([0., 1., 0.])
SPINE, NECK = ['Spine1', 'Spine2', 'Chest'], ['Neck1', 'Neck2']
LIMB = [('Shoulder', 'Shoulder', 'Arm'), ('Arm', 'Arm', 'ForeArm'), ('ForeArm', 'ForeArm', 'Hand'),
        ('Leg', 'Leg', 'Shin'), ('Shin', 'Shin', 'Foot'), ('Foot', 'Foot', 'ToeBase'),
        ('ToeBase', 'ToeBase', 'ToeEnd')]
FINGER_JOINTS = {'Thumb': ['Thumb1', 'Thumb2', 'Thumb3'], 'Index': ['Index1', 'Index2', 'Index3', 'Index4'],
                 'Middle': ['Middle1', 'Middle2', 'Middle3', 'Middle4'],
                 'Ring': ['Ring1', 'Ring2', 'Ring3', 'Ring4'], 'Pinky': ['Pinky1', 'Pinky2', 'Pinky3', 'Pinky4']}


@dataclass
class Result:
    names: list            # every bone in the rig, Blender order
    quaternions: np.ndarray  # (T, B, 4) local pose rotation, w x y z
    locations: np.ndarray    # (T, B, 3) local pose translation
    heads: np.ndarray        # (T, B, 3) world space
    tails: np.ndarray        # (T, B, 3) world space
    scale: float             # rig leg length / SOMA leg length
    fps: float
    pairs: dict              # bone -> SOMA joint it follows

    def save(self, path):
        np.savez_compressed(path, names=np.array(self.names), quaternions=self.quaternions.astype(np.float32),
                            locations=self.locations.astype(np.float32), heads=self.heads.astype(np.float32),
                            tails=self.tails.astype(np.float32), scale=self.scale, fps=self.fps)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _swing(a, b):
    """Smallest rotation matrix taking direction a onto direction b."""
    a, b = _unit(a), _unit(b)
    axis, c = np.cross(a, b), float(np.dot(a, b))
    if c < -0.999999:
        perp = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return Rotation.from_rotvec(_unit(perp) * np.pi).as_matrix()
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def _orthonormal(m):
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def _pairs(roles):
    """[(bone, soma joint, soma child joint, next rig bone or None, align?)]"""
    out = []

    def chain(bones, joints, end, after):
        n, m = len(bones), len(joints)
        for i, b in enumerate(bones):
            j = joints[min(m - 1, round(i * (m - 1) / max(n - 1, 1)))] if n > 1 else joints[0]
            k = (joints + [end])[joints.index(j) + 1]
            out.append((b, j, k, bones[i + 1] if i + 1 < n else after, True))

    spine, neck = roles.get('Spine', []), roles.get('Neck', [])
    out.append((roles['Hips'], 'Hips', 'Spine1', (spine or neck or [roles['Head']])[0], False))
    if spine:
        chain(spine, SPINE, 'Neck1', (neck or [roles['Head']])[0])
    if neck:
        chain(neck, NECK, 'Head', roles['Head'])
    out.append((roles['Head'], 'Head', 'HeadEnd', None, True))
    for side in ('Left', 'Right'):
        for role, j, k in LIMB:
            if side + role not in roles:
                continue
            following = {'Shoulder': 'Arm', 'Arm': 'ForeArm', 'ForeArm': 'Hand', 'Leg': 'Shin', 'Shin': 'Foot',
                         'Foot': 'ToeBase'}.get(role)
            out.append((roles[side + role], side + j, side + k, roles.get(side + following) if following else None,
                        True))
        if side + 'Hand' in roles:
            middle = roles.get(side + 'HandMiddle')
            out.append((roles[side + 'Hand'], side + 'Hand', side + 'HandMiddle1', middle[0] if middle else None,
                        True))
        for finger, joints in FINGER_JOINTS.items():
            bones = roles.get(f'{side}Hand{finger}')
            if not bones:
                continue
            joints = [f'{side}Hand{x}' for x in joints]
            end = f'{side}Hand{finger}End'
            offset = len(joints) - len(bones)  # align from the fingertip
            for i, b in enumerate(bones):
                j = joints[max(0, offset + i)]
                k = (joints + [end])[joints.index(j) + 1]
                out.append((b, j, k, bones[i + 1] if i + 1 < len(bones) else None, True))
    return out


def soma_facing(motion, face_start=True):
    """SOMA world matrices (optionally turned so frame 0 faces the rest front), plus rest left/forward."""
    r = motion.rest[:, :3, 3]
    j = motion.index
    fwd = (r[j('LeftToeBase')] + r[j('RightToeBase')]) - (r[j('LeftFoot')] + r[j('RightFoot')])
    fwd = _unit(fwd - SOMA_UP * fwd.dot(SOMA_UP))
    left = np.cross(SOMA_UP, fwd)
    world = motion.world.copy()
    if face_start:
        hips = j('Hips')
        f0 = world[0, hips, :3, :3] @ motion.rest[hips, :3, :3].T @ fwd
        f0 = f0 - SOMA_UP * f0.dot(SOMA_UP)
        if np.linalg.norm(f0) > 1e-6:
            yaw = _swing(f0, fwd)
            world[..., :3, :3] = yaw @ world[..., :3, :3]
            world[..., :3, 3] = world[..., :3, 3] @ yaw.T
    return world, left, fwd


def retarget(motion, dump, rigmap, face_start=True):
    """motion: SomaMotion; dump: dump_rig.py JSON; rigmap: RigMap for that dump."""
    bones = dump['bones']
    names = [b['name'] for b in bones]
    index = {n: i for i, n in enumerate(names)}
    parent = [index.get(b['parent']) for b in bones]
    order = []
    seen = set()
    while len(order) < len(bones):  # parents before children
        for i, p in enumerate(parent):
            if i not in seen and (p is None or p in seen):
                order.append(i)
                seen.add(i)

    # Armature space: undo the object transform, keep bone rotations orthonormal.
    obj = np.array(dump['object_matrix'])
    obj_inv, obj_rot = np.linalg.inv(obj), _orthonormal(obj[:3, :3])
    rest = np.array([obj_inv @ np.array(b['rest']) for b in bones])
    rest[:, :3, :3] = [_orthonormal(r) for r in rest[:, :3, :3]]
    to_arm = obj_rot.T
    up, fwd, left = (to_arm @ np.array(v) for v in (rigmap.up, rigmap.forward, rigmap.left))

    # SOMA -> rig frame. Facing comes from the rest pose (toes point forward).
    r = motion.rest[:, :3, 3]
    j = motion.index
    world, soma_left, soma_fwd = soma_facing(motion, face_start)
    convert = np.column_stack([left, fwd, up]) @ np.column_stack([soma_left, soma_fwd, SOMA_UP]).T

    T = motion.frames
    rotations = np.einsum('ab,tjbc,jdc,ed->tjae', convert, world[..., :3, :3], motion.rest[:, :3, :3], convert)
    soma_rest = r @ convert.T
    soma_pos = world[..., :3, 3] @ convert.T

    pairs = _pairs(rigmap.roles)
    target_rot = {}
    for bone, sj, sk, nxt, align in pairs:
        b = index[bone]
        bone_dir = (rest[index[nxt], :3, 3] - rest[b, :3, 3]) if nxt else rest[b, :3, 1]
        seg = soma_rest[j(sk)] - soma_rest[j(sj)]
        a = _swing(bone_dir, seg) if align and np.linalg.norm(bone_dir) > 1e-6 else np.eye(3)
        target_rot[b] = np.einsum('tab,bc->tac', rotations[:, j(sj)], a @ rest[b, :3, :3])

    # Root: hip-to-ankle ratio scales travel and height above the floor.
    roles = rigmap.roles
    def rig_leg(side):
        return rest[index[roles[side + 'Leg']], :3, 3].dot(up) - rest[index[roles[side + 'Foot']], :3, 3].dot(up)
    soma_leg = (soma_rest[j('LeftLeg')] - soma_rest[j('LeftFoot')]).dot(up)
    scale = (rig_leg('Left') + rig_leg('Right')) / 2 / soma_leg
    feet = [j(f'{s}{x}') for s in ('Left', 'Right') for x in ('Foot', 'ToeBase')]
    soma_ground = np.percentile((soma_pos[:, feet] @ up).min(axis=1), 5)
    # Hips height relative to standing: SOMA's hips joint sits higher above the thighs than most rigs'
    # hips bones, so only the change from standing is scaled; the rig keeps its own standing height.
    soma_stand = soma_rest[j('Hips')].dot(up) - min(soma_rest[feet] @ up)
    hips_b = index[roles['Hips']]
    hips_rest = rest[hips_b, :3, 3]
    soma_hips = soma_pos[:, j('Hips')]
    flat = soma_hips - soma_hips[0]
    flat -= np.outer(flat @ up, up)
    rise = (soma_hips @ up - soma_ground) - soma_stand
    root = hips_rest + scale * flat + np.outer(scale * rise, up)

    # Forward kinematics in armature space.
    pose = np.repeat(rest[None], T, axis=0)
    for i in order:
        p = parent[i]
        if p is not None:
            pose[:, i] = pose[:, p] @ (np.linalg.inv(rest[p]) @ rest[i])
        if i in target_rot:
            pose[:, i, :3, :3] = target_rot[i]
        if i == hips_b:
            pose[:, i, :3, 3] = root

    # Local pose (Blender basis): rest_local^-1 @ parent_pose^-1 @ pose.
    basis = np.empty_like(pose)
    for i in range(len(bones)):
        p = parent[i]
        rest_local = rest[i] if p is None else np.linalg.inv(rest[p]) @ rest[i]
        pose_local = pose[:, i] if p is None else np.linalg.inv(pose[:, p]) @ pose[:, i]
        basis[:, i] = np.linalg.inv(rest_local) @ pose_local
    quats = Rotation.from_matrix(basis[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T, len(bones), 4)
    quats = quats[..., [3, 0, 1, 2]]
    quats *= np.where(quats[..., :1] < 0, -1, 1)  # keep w >= 0 per key...
    for t in range(1, T):  # ...but avoid sign flips between frames so interpolation stays short
        flip = np.einsum('bi,bi->b', quats[t], quats[t - 1]) < 0
        quats[t, flip] *= -1

    lengths = np.array([np.linalg.norm(np.array(b['tail']) - np.array(b['head'])) for b in bones])
    lengths /= np.linalg.norm(obj[:3, :3], axis=0).mean()  # world length -> armature units
    heads_arm = pose[..., :3, 3]
    tails_arm = heads_arm + pose[..., :3, 1] * lengths[None, :, None]
    to_world = lambda x: x @ obj[:3, :3].T + obj[:3, 3]
    return Result(names, quats, basis[..., :3, 3], to_world(heads_arm), to_world(tails_arm), float(scale),
                  motion.fps, {bone: sj for bone, sj, *_ in pairs})
