"""SOMA motion -> any mapped humanoid rig, keeping the rig's own bone lengths.

For every mapped bone b following SOMA joint j:

    world_rotation_b(t) = delta_j(t) @ align_b @ rest_rotation_b

delta_j is SOMA's world-space rotation away from its rest pose, expressed in the
rig's frame. align_b is a swing that turns the rig's rest bone onto SOMA's rest
segment, so a T-pose rig follows an A-pose source (and the reverse). That is for
limbs only: hips, spine, neck and head use align_b = I, moving by SOMA's change
from its own rest, because each rig's rest trunk and neck are its neutral. Unmapped
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

from motionbridge.core import contacts

SOMA_UP = np.array([0., 1., 0.])
# SOMA's visual face frame in the Head joint's own frame: columns = left (ear to ear), forward
# (ears to eyes, ~Frankfort plane, level gaze), up. Measured on GEM body meshes (ear vertices
# 658/2862, eye joints) over three clips/body shapes; they agree within 3.4 degrees. Columns are
# (left, forward, up) like the rig frames it is paired with, so both are mirrored (det -1) and
# their product is a proper rotation.
SOMA_FACE_IN_HEAD = np.array([[-0.00144, -0.0077, 0.99997],
                              [-0.00499, 0.99996, 0.00769],
                              [0.99999, 0.00498, 0.00147]])
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
    locks: dict = None       # per side, per frame: 'heel', 'toe' or None (from contacts.detect)
    notes: list = None

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


def _pairs(roles, rig_head, soma_head):
    """[(bone, soma joint, soma child joint, next rig bone or None, align?)]

    rig_head / soma_head give rest positions by bone / joint name. Spine and neck bones are
    matched by where their midpoint falls along the chain's length, not by count: SOMA's chest
    segment is long and carries most of the bend, so a rig's long middle spine bone must follow it.
    """
    out = []

    def chain(bones, joints, end, after):
        ends = [rig_head[b] for b in bones] + [rig_head[after]]
        rig_cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ends, axis=0), axis=1))]
        marks = [soma_head[x] for x in joints + [end]]
        soma_cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(marks, axis=0), axis=1))]
        for i, b in enumerate(bones):
            middle = (rig_cum[i] + rig_cum[i + 1]) / 2 / max(rig_cum[-1], 1e-9) * soma_cum[-1]
            k = min(int(np.searchsorted(soma_cum, middle, side='right')) - 1, len(joints) - 1)
            j = joints[max(k, 0)]
            out.append((b, j, (joints + [end])[joints.index(j) + 1], bones[i + 1] if i + 1 < len(bones) else after,
                        False))

    spine, neck = roles.get('Spine', []), roles.get('Neck', [])
    out.append((roles['Hips'], 'Hips', 'Spine1', (spine or neck or [roles['Head']])[0], False))
    if spine:
        chain(spine, SPINE, 'Neck1', (neck or [roles['Head']])[0])
    if neck:
        chain(neck, NECK, 'Head', roles['Head'])
    # Trunk, neck and head transfer SOMA's change from rest, not its absolute direction: each rig's
    # rest posture there is its own neutral (SOMA's anatomical neck already slants 17 degrees forward;
    # copying that onto a rig with an upright rest neck juts the head out).
    out.append((roles['Head'], 'Head', 'HeadEnd', None, False))
    for side in ('Left', 'Right'):
        for role, j, k in LIMB:
            if side + role not in roles:
                continue
            following = {'Shoulder': 'Arm', 'Arm': 'ForeArm', 'ForeArm': 'Hand', 'Leg': 'Shin', 'Shin': 'Foot',
                         'Foot': 'ToeBase'}.get(role)
            # Only the long limb bones are aligned to SOMA's rest (that is what reconciles T-pose vs
            # A-pose and keeps knees/elbows tracking the source). Clavicles, feet and toes move
            # relative to their own rest: forcing them onto SOMA's anatomy swung shoulders 20-32
            # degrees off their skinning and tipped flat-built feet 11-16 degrees into the floor.
            out.append((roles[side + role], side + j, side + k, roles.get(side + following) if following else None,
                        role not in RELATIVE_LIMB))
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


def retarget(motion, dump, rigmap, face_start=True, feet_ik=True, head='face', match_start=True, travel='keep'):
    """motion: SomaMotion; dump: dump_rig.py JSON; rigmap: RigMap for that dump.
    feet_ik pins planted feet (contacts detected on SOMA) with leg IK.
    head='face': the rig's face points where the performer's face points (world space).
    head='carriage': the head keeps the character's built-in rest carriage and adds the
    performer's change on top (for characters modelled with a deliberate head pose).
    match_start: the motion's first frame is the character's own start pose (the dump's 'start', the frame the
    starting views drew), so the video's first frame is known exactly; each mapped bone's difference from it there
    is taken off every frame, and the hips start where the character stands. The extraction's guess of the start
    stance (feet too close, a foot forward) otherwise stays for the whole take. Off for footage that doesn't start
    from the character's pose.
    travel: 'keep' (the hips carry the travel), 'in place' (the travel taken off) or 'root bone' (the travel and
    turn moved onto the bone above the hips, for engine root motion; see _move_travel)."""
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
    # A posed rest (an idle with contrapposto) can turn and tilt the pelvis against the shoulders: one rig's rest
    # pelvis is 7.6 degrees turned and 5.8 tilted. Facing square to that pelvis walked her 8 degrees off her
    # shoulders, one arm swinging forward and the other back. Face square to the shoulders instead, and take
    # the rest's pelvis turn and tilt off the hips (below), so only the performer's pelvis motion remains.
    roles_ = rigmap.roles
    pelvis_fix = np.eye(3)
    if SQUARE_TO_SHOULDERS and all(s + k in roles_ for s in ('Left', 'Right') for k in ('Arm', 'Leg')):
        head_of = lambda k: rest[index[roles_[k]], :3, 3]
        across = head_of('LeftArm') - head_of('RightArm')
        across -= up * across.dot(up)
        if np.linalg.norm(across) > 1e-6:
            left = across / np.linalg.norm(across)
            fwd = np.cross(left, up)
            hip_line = head_of('LeftLeg') - head_of('RightLeg')
            pelvis_fix = _swing(hip_line, left)

    # SOMA -> rig frame. Facing comes from the rest pose (toes point forward).
    r = motion.rest[:, :3, 3]
    j = motion.index
    world, soma_left, soma_fwd = soma_facing(motion, face_start)
    convert = np.column_stack([left, fwd, up]) @ np.column_stack([soma_left, soma_fwd, SOMA_UP]).T

    T = motion.frames
    rotations = np.einsum('ab,tjbc,jdc,ed->tjae', convert, world[..., :3, :3], motion.rest[:, :3, :3], convert)
    soma_rest = r @ convert.T
    soma_pos = world[..., :3, 3] @ convert.T

    pairs = _pairs(rigmap.roles, {n: rest[index[n], :3, 3] for n in names},
                   {n: soma_rest[j(n)] for n in SPINE + NECK + ['Head']})
    # A posed rest can turn the chest away from the pelvis (one rig's idle rest has 7.6 degrees of
    # contrapposto). The trunk moves relative to its own rest, so arms and face are aimed relative to the
    # rig's shoulder line, not the pelvis front: otherwise every reach is skewed by that turn.
    roles_ = rigmap.roles
    shoulders = rest[index[roles_['LeftArm']], :3, 3] - rest[index[roles_['RightArm']], :3, 3]
    shoulders -= up * shoulders.dot(up)
    chest_turn = _swing(left, shoulders) if USE_CHEST_TURN and np.linalg.norm(shoulders) > 1e-6 else np.eye(3)
    upper = {roles_[s + r] for s in ('Left', 'Right') for r in ('Arm', 'ForeArm', 'Hand') if s + r in roles_}
    upper |= {b for k, v in roles_.items() if 'Hand' in k and isinstance(v, list) for b in v}  # fingers
    target_rot = {}
    for bone, sj, sk, nxt, align in pairs:
        b = index[bone]
        bone_dir = (rest[index[nxt], :3, 3] - rest[b, :3, 3]) if nxt else rest[b, :3, 1]
        seg = soma_rest[j(sk)] - soma_rest[j(sj)]
        if bone in upper:
            seg = chest_turn @ seg
        a = _swing(bone_dir, seg) if align and np.linalg.norm(bone_dir) > 1e-6 else np.eye(3)
        if bone == roles_['Hips']:
            a = pelvis_fix @ a
        target_rot[b] = np.einsum('tab,bc->tac', rotations[:, j(sj)], a @ rest[b, :3, :3])
    if head == 'face':
        # SOMA's rest pose looks ~9 degrees up, so copying its change from rest bowed every rig's
        # head. Map SOMA's measured face frame onto the rig's rest face frame (assumed to look
        # straight ahead along the rig's front) instead.
        hb = index[rigmap.roles['Head']]
        soma_face = convert @ motion.rest[j('Head'), :3, :3] @ SOMA_FACE_IN_HEAD
        rig_face = chest_turn @ np.column_stack([left, fwd, up])
        target_rot[hb] = np.einsum('tab,bc->tac', rotations[:, j('Head')],
                                   soma_face @ rig_face.T @ rest[hb, :3, :3])
    elif head != 'carriage':
        raise ValueError(f"head must be 'face' or 'carriage', not {head!r}")
    start = None
    if match_start and all('start' in b for b in bones):
        start = np.array([obj_inv @ np.array(b['start']) for b in bones])
        start[:, :3, :3] = [_orthonormal(m) for m in start[:, :3, :3]]
        for b in target_rot:  # the first frame becomes the start pose; the same turn off every frame
            target_rot[b] = target_rot[b] @ (target_rot[b][0].T @ start[b, :3, :3])

    # Root: hip-to-ankle ratio scales travel and height above the floor.
    roles = rigmap.roles
    def rig_leg(side):
        return rest[index[roles[side + 'Leg']], :3, 3].dot(up) - rest[index[roles[side + 'Foot']], :3, 3].dot(up)
    soma_leg = (soma_rest[j('LeftLeg')] - soma_rest[j('LeftFoot')]).dot(up)
    scale = (rig_leg('Left') + rig_leg('Right')) / 2 / soma_leg
    feet = [j(f'{s}{x}') for s in ('Left', 'Right') for x in ('Foot', 'ToeBase')]
    soma_ground = contacts.ground_track(motion)  # per frame: reconstructed floors drift
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
    if start is not None:
        root = root + (start[hips_b, :3, 3] - root[0])

    def fk(rotations_, root_):
        """Forward kinematics in armature space."""
        pose_ = np.repeat(rest[None], T, axis=0)
        for i in order:
            p = parent[i]
            if p is not None:
                pose_[:, i] = pose_[:, p] @ (np.linalg.inv(rest[p]) @ rest[i])
            if i in rotations_:
                pose_[:, i, :3, :3] = rotations_[i]
            if i == hips_b:
                pose_[:, i, :3, 3] = root_
        return pose_

    lengths = np.array([np.linalg.norm(np.array(b['tail']) - np.array(b['head'])) for b in bones])
    lengths /= np.linalg.norm(obj[:3, :3], axis=0).mean()  # world length -> armature units
    locks = contacts.detect(motion) if feet_ik else None
    if feet_ik:
        root, notes = _plant_feet(fk, target_rot, root, rest, parent, index, roles, locks, up, fwd, lengths)
    pose = fk(target_rot, root)
    notes = notes if feet_ik else []
    if travel != 'keep':
        notes.append(_move_travel(pose, travel, hips_b, parent, index, roles, up, motion.fps,
                                  np.linalg.norm(obj[:3, :3], axis=0).mean()))

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

    heads_arm = pose[..., :3, 3]
    tails_arm = heads_arm + pose[..., :3, 1] * lengths[None, :, None]
    to_world = lambda x: x @ obj[:3, :3].T + obj[:3, 3]
    return Result(names, quats, basis[..., :3, 3], to_world(heads_arm), to_world(tails_arm), float(scale),
                  motion.fps, {bone: sj for bone, sj, *_ in pairs}, locks, notes)



TRAVEL_CUTOFF = 0.4  # Hz: the travel taken off the body is its path low-passed, well under a walk's step rate
                     # (~1 Hz): at 1 Hz half of each step's sway and pelvis turn went onto the root
TRAVEL_PAD = 2.0     # s: the path is carried on in a straight line past both ends while filtering, so the filter's
                     # lag doesn't slow the root at the clip's ends (it ended 22 cm short; a looped clip popped)


def _ground_path(pose, hips_b, index, roles, up, fps):
    """The body's smoothed path along the floor (T, 3, from frame 0) and its smoothed turn about up (T, radians,
    from frame 0; the hip line's direction)."""
    from motionbridge.core.smooth import one_euro
    flat = lambda v: v - np.outer(v @ up, up)
    path = flat(pose[:, hips_b, :3, 3] - pose[0, hips_b, :3, 3])
    across = flat(pose[:, index[roles['LeftLeg']], :3, 3] - pose[:, index[roles['RightLeg']], :3, 3])
    ref = across[0] / np.linalg.norm(across[0])
    turn = np.unwrap(np.arctan2(np.cross(ref, across) @ up, across @ ref))
    x = np.c_[path, turn]
    T, pad, k = len(x), int(TRAVEL_PAD * fps), max(min(int(fps), len(x) - 1), 1)  # end speeds over the last second
    v0, v1 = (x[k] - x[0]) / k, (x[-1] - x[-1 - k]) / k
    x = np.r_[x[0] - np.outer(np.arange(pad, 0, -1), v0), x, x[-1] + np.outer(np.arange(1, pad + 1), v1)]
    x = one_euro(x, np.zeros(len(x)), fps, TRAVEL_CUTOFF, 0)[pad:pad + T]
    x -= x[0]
    return x[:, :3], x[:, 3]


def _move_travel(pose, travel, hips_b, parent, index, roles, up, fps, unit=1.0):
    """Takes the travel off the posed body (pose, armature space, changed in place); returns a note (unit: metres
    per armature unit).
    'in place': the hips and everything below them are shifted back by the smoothed path, so the body walks on the
    spot and keeps its turn; bones above and beside the hips (an engine root, IK targets) stay still.
    'root bone': the bone above the hips (the engine's root) carries the smoothed path and turn, turning about the
    hips; the hips and everything below them keep their world pose, so the clip looks the same in Blender, and the
    root's other children (UE's ik_foot_root, ik_hand_root) move with it. A rig without a bone above the hips keeps
    the travel on the hips (a new bone would no longer match the character already in the engine)."""
    def below(i):
        out = {i}
        for _ in range(len(parent)):  # the chain is at most that deep
            more = {j for j, q in enumerate(parent) if q in out} - out
            if not more:
                return sorted(out)
            out |= more
        return sorted(out)
    body = below(hips_b)
    path, turn = _ground_path(pose, hips_b, index, roles, up, fps)
    if travel == 'in place':
        moved = pose[:, body]
        moved[..., :3, 3] -= path[:, None]
        pose[:, body] = moved
        return f'in place: {unit * np.linalg.norm(path[-1]):.2f} m of travel taken off'
    if travel != 'root bone':
        raise ValueError(f'travel: keep, in place or root bone, not {travel!r}')
    r = parent[hips_b]
    if r is None:
        return 'root bone: the rig has no bone above the hips; travel kept on the hips'
    spin = Rotation.from_rotvec(np.outer(turn, up)).as_matrix()
    old = pose[:, r].copy()
    arm = pose[0, r, :3, 3] - pose[0, hips_b, :3, 3]
    arm -= up * (arm @ up)  # the root's floor offset from the hips at the start, turned with the body
    pose[:, r, :3, :3] = spin @ pose[0, r, :3, :3]
    pose[:, r, :3, 3] = pose[0, r, :3, 3] + path + spin @ arm - arm
    carried = [j for j in below(r) if j != r and j not in body]
    if carried:
        follow = pose[:, r] @ np.linalg.inv(old)
        pose[:, carried] = follow[:, None] @ pose[:, carried]
    return f'root bone: {unit * np.linalg.norm(path[-1]):.2f} m of travel and {np.degrees(turn[-1]):.0f} deg of turn on the root'


BLEND = 5        # frames to ease into and out of a plant
RELATIVE_LIMB = {'Shoulder', 'Foot', 'ToeBase'}
USE_HIP_DROP, USE_PELVIS_ROLL = False, True  # hip drop off: lowering the hips for a foot that can't reach bent
# the other planted knee (up to 23 degrees past the performer's in a walk); a foot short of its pin peels instead
USE_CHEST_TURN = True
SQUARE_TO_SHOULDERS = True  # facing square to the rest shoulders; the rest pelvis's own turn and tilt taken off
ROOT_FULL_PLANTS_ONLY = False
ROOT_STILL_FEET = True  # only still planted feet steer the root (see _plant_feet)
STILL_REST, STILL_MOVE = 0.005, 0.02  # leg lengths per frame (~0.1 and ~0.4 m/s at 24 fps): full say / none
ROOT_HEIGHT_HIGHEST = True  # root height from the planted foot that needs it highest (see _plant_feet)
STRAIGHT_KNEE = 25.0        # degrees of knee bend: below it a stance leg counts as straight (the highest-foot rule)
REACH_FLAT_ONLY = True  # a stance leg straightens to reach its pin only on a flat foot (see _plant_feet)
REACH_FADE = 4          # frames: that reach fades out before the heel lifts
KNEE_RAMP = 10.0            # degrees past STRAIGHT_KNEE over which a stance leg goes from keeping its length to reaching its pin
ROOT_SMOOTH = 2         # frames; halves body wobble at foot hand-offs. Was 0 when it caused 15-degree knee
                        # snaps, before KEEP_LEG_LENGTH; re-tested with it: no new snaps
KEEP_LEG_LENGTH = True  # stance legs keep their transferred length (see the IK below)
MAX_ROLL = np.radians(12)  # pelvis roll limit for planting both feet
EASE = True
KNEE_PRIOR = 0.03  # of leg length: a knee bent more than ~3 cm off the hip-ankle line keeps its own direction
REACH = 0.9999   # essentially full length; 0.995 forced ~11 degrees of knee bend on straight legs


def _ease(x):
    """Smoothstep, or linear with EASE off. Easing a foot in or out of a plant hands the body's root
    correction from one foot to the other; a linear ramp's corners showed as a jolt at every step."""
    return x * x * (3 - 2 * x) if EASE else x


def _swing_batch(a, b):
    """Per-row smallest rotation taking a[t] onto b[t]; (T, 3) -> (T, 3, 3)."""
    return np.array([_swing(x, y) for x, y in zip(a, b)])


def _runs(labels, value):
    t, out = 0, []
    while t < len(labels):
        if labels[t] == value:
            end = t
            while end < len(labels) and labels[end] == value:
                end += 1
            out.append((t, end))
            t = end
        else:
            t += 1
    return out


def _plant_feet(fk, target_rot, root, rest, parent, index, roles, locks, up, fwd, lengths, drive_root=True):
    """Pin planted feet to the floor: per-span targets, hips lowered only when a leg can't reach,
    then analytic two-bone IK on thigh and shin. Foot rotations stay as transferred.

    First pass (drive_root): the planted feet drive the body. The root is shifted (in all three axes)
    so each planted foot lands on its pin with the leg exactly as transferred. Otherwise the legs do
    the adjusting: near full extension a 2 cm hip drop bends a straight knee by ~20 degrees.
    Modifies target_rot in place; returns the adjusted root and notes."""
    notes = []
    pose = fk(target_rot, root)
    T = len(root)
    flat = lambda v: v - np.outer(v @ up, up)
    goals, legs, legs_weight, target_full, flat_foot = {}, {}, {}, {}, {}
    for side in ('Left', 'Right'):
        if not all(side + r in roles for r in ('Leg', 'Shin', 'Foot')):
            continue
        l, s, f = (index[roles[side + r]] for r in ('Leg', 'Shin', 'Foot'))
        if parent[s] != l or parent[f] != s:
            notes.append(f'{side} leg has extra bones between thigh, shin and foot; feet left unplanted')
            continue
        toe = index[roles[side + 'ToeBase']] if side + 'ToeBase' in roles else None
        ankle, toe_p = pose[:, f, :3, 3], (pose[:, toe, :3, 3] if toe is not None else None)
        ankle_h = rest[f, :3, 3] @ up
        labels = locks[side] if toe is not None else ['heel' if x else None for x in locks[side]]
        # Where a stance leg may reach for its pin (see the IK): a flat foot, faded out over the plant's last frames,
        # where the heel is already peeling. Reaching there (or rolling onto the toe) straightened the push-off leg
        # for a frame or two, and it popped back bent when the foot let go: a knee snap at every toe-off.
        on_heel = np.array([x == 'heel' for x in labels], float)
        for t in range(T):
            if on_heel[t]:
                left = next((k for k in range(1, REACH_FADE + 1) if t + k >= T or not on_heel[t + k]), None)
                if left is not None and t + left < T:
                    on_heel[t] = (left - 1) / REACH_FADE
        for t in range(1, T):  # ...and eased in over the frames before a landing, as the goal is (no snap on contact)
            if labels[t] == 'heel' and labels[t - 1] != 'heel':
                for k in range(1, BLEND + 1):
                    if t - k < 0 or labels[t - k] == 'heel':
                        break
                    on_heel[t - k] = max(on_heel[t - k], _ease(1 - k / (BLEND + 1)))
        flat_foot[side] = on_heel
        target = np.full((T, 3), np.nan)
        # Planted joints are pinned horizontally. Their height follows the source's own rise above its
        # lowest point in that plant, so heel peels and toe rolls come through instead of being held
        # flat and then released in a pop.
        def lift(points, a, b):
            z = points[a:b] @ up
            return np.outer(z - z.min(), up)

        def slow(points, a, b):
            """Planted position: the source position averaged over the plant."""
            xy = flat(points[a:b])
            out = np.empty_like(xy)
            out[:] = xy.mean(axis=0)  # PIN_WINDOW > 0 would follow slow drift; tried: it lets slide back in
            return out

        for a, b in _runs(labels, 'heel'):
            target[a:b] = slow(ankle, a, b) + up * ankle_h + lift(ankle, a, b)
        for a, b in _runs(labels, 'toe'):
            if a > 0 and labels[a - 1] == 'heel':  # heel lifting: keep the toe where the flat foot left it
                spot = target[a - 1] + (toe_p[a - 1] - ankle[a - 1])
            else:
                spot = slow(toe_p, a, b) + up * (rest[toe, :3, 3] @ up)
            spot = flat(spot) + up * (rest[toe, :3, 3] @ up)
            # ankle goal that puts the toe on its spot (plus the source's own toe lift)
            target[a:b] = spot + lift(toe_p, a, b) + (ankle[a:b] - toe_p[a:b])
        # Switching between heel and toe pins: the source heel starts moving before its label flips, so
        # the two pins disagree. Ease the jump out over BLEND + 1 frames so the foot rolls, never pops.
        for a in range(1, T):
            if labels[a] and labels[a - 1] and labels[a] != labels[a - 1]:
                gap = target[a - 1] + (ankle[a] - ankle[a - 1]) - target[a]
                for k in range(BLEND + 1):
                    if a + k < T and labels[a + k] == labels[a]:
                        target[a + k] += gap * _ease(1 - (k + 1) / (BLEND + 2))
        weight = np.zeros(T)
        planted = ~np.isnan(target[:, 0])
        for t in range(T):  # ease in/out around each plant, aiming at the nearest planted spot
            if planted[t]:
                weight[t] = 1
                continue
            near = [u for u in range(max(0, t - BLEND), min(T, t + BLEND + 1)) if planted[u]]
            if near:
                u = min(near, key=lambda u: abs(u - t))
                weight[t] = _ease(1 - abs(u - t) / (BLEND + 1))
                target[t] = ankle[t] + (target[u] - ankle[u])  # carry the nearest plant's correction
        goal = ankle.copy()
        mask = weight > 0
        goal[mask] = ankle[mask] + weight[mask, None] * (target[mask] - ankle[mask])
        thigh_len = np.linalg.norm(rest[s, :3, 3] - rest[l, :3, 3])
        shin_len = np.linalg.norm(rest[f, :3, 3] - rest[s, :3, 3])
        goals[side], legs[side] = (goal, mask), (l, s, f, thigh_len, shin_len)
        target_full[side] = np.where(np.isnan(target), ankle, target)
        legs_weight[side] = weight

    if drive_root and goals:
        pose_ = pose
        flat_ = lambda v: v - np.outer(v @ up, up)
        num, den = np.zeros((T, 3)), np.zeros(T)
        for side, (goal, mask) in goals.items():
            l, s_, f, *_ = legs[side]
            w = legs_weight[side]
            if ROOT_STILL_FEET:
                # A planted foot that the performer's leg is already moving (pushing off, or the read sliding it)
                # stops steering the body: holding the body to its pin dragged the hips back through each stance
                # and let them go at the next step, a surge every step. The leg alone holds such a foot, or it peels.
                step_ = np.linalg.norm(np.gradient(flat_(pose_[:, f, :3, 3]), axis=0), axis=1) / (legs[side][3] + legs[side][4])
                w = w * np.clip((STILL_MOVE - step_) / (STILL_MOVE - STILL_REST), 0, 1)
            if ROOT_FULL_PLANTS_ONLY:  # an easing (swinging) foot does not drag the body
                w = (w >= 0.999).astype(float)
                err = target_full[side] - pose_[:, f, :3, 3]
            else:
                err = goal - pose_[:, f, :3, 3]
            num += w[:, None] * err  # all three axes: height too
            den += w
        known = den > 1e-6
        if known.any():
            shift = np.zeros((T, 3))
            shift[known] = num[known] / den[known, None]
            if ROOT_HEIGHT_HIGHEST:
                # Height: the highest any planted foot asks for, not their average. Averaged, the two feet of a
                # double support split their disagreement: one straight leg is pushed into the floor (the floor
                # guard bends its knee) and the other can't reach (the hips drop and bend both). From the
                # highest, no knee bends past the performer's; the foot short of the floor peels onto its toe,
                # as a trailing foot does. Only while the leg asking highest is near straight: with bent legs
                # (crouch, jump landing) nothing buckles, and lifting to the highest left the other foot
                # hanging up to 19 cm above its plant.
                high = np.full(T, -np.inf)
                straight = np.zeros(T, bool)
                for side, (goal, mask) in goals.items():
                    l, s_, f = legs[side][:3]
                    w = legs_weight[side]
                    dz = (target_full[side] - pose_[:, f, :3, 3]) @ up
                    a, b = pose_[:, l, :3, 3] - pose_[:, s_, :3, 3], pose_[:, f, :3, 3] - pose_[:, s_, :3, 3]
                    bend = 180 - np.degrees(np.arccos(np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)))
                    top = (w >= 0.999) & (dz > high)
                    straight = np.where(top, bend < STRAIGHT_KNEE, straight)
                    high = np.where(top, dz, high)
                # ...and never higher than every planted leg can still reach its spot from: a straight leg
                # can't extend, so a lift past its reach left a standing foot hanging 18 cm up.
                cap = np.full(T, np.inf)
                for side, (goal, mask) in goals.items():
                    l, s_, f, a_, b_ = legs[side]
                    v = target_full[side] - (pose_[:, l, :3, 3] + shift)
                    h = np.linalg.norm(v - np.outer(v @ up, up), axis=1)
                    lift = np.sqrt(np.maximum((REACH * (a_ + b_)) ** 2 - h ** 2, 0)) + v @ up  # room to rise
                    cap = np.where(legs_weight[side] >= 0.999, np.minimum(cap, shift @ up + lift), cap)
                high = np.minimum(high, cap)
                full = np.isfinite(high) & straight & (high > shift @ up)
                shift[full] += np.outer(high[full] - shift[full] @ up, up)
            for axis in range(3):  # carry the correction smoothly through airborne frames
                shift[~known, axis] = np.interp(np.flatnonzero(~known), np.flatnonzero(known), shift[known, axis])
            k = 2 * ROOT_SMOOTH + 1
            shift = np.stack([np.convolve(np.pad(shift[:, i], ROOT_SMOOTH, mode='edge'), np.ones(k) / k, mode='valid')
                              for i in range(3)], axis=1)
            root2, more = _plant_feet(fk, target_rot, root + shift, rest, parent, index, roles, locks, up, fwd,
                                      lengths, drive_root=False)
            return root2, [f'root follows planted feet (up to {np.linalg.norm(shift, axis=1).max() * 100:.1f} cm)'] + more

    def needs():
        """Per side: how far (along up) the hip joint must come down for the planted foot to reach."""
        out = {}
        for side, (goal, mask) in goals.items():
            l, s, f, a, b = legs[side]
            v = goal - pose[:, l, :3, 3]
            z = v @ up
            h = np.linalg.norm(flat(v), axis=1)
            reach = REACH * (a + b)
            out[side] = np.where(mask & (h < reach), -np.sqrt(np.maximum(reach ** 2 - h ** 2, 0)) - z, 0)
        return out

    # Both feet planted but one can't reach (e.g. auto-rigs with unequal legs, or a source weight
    # shift): roll the pelvis about the body's forward axis to lower that hip, instead of dropping the
    # whole body and buckling the other knee. The spine keeps its own world rotation, so the torso
    # does not tilt with it.
    hips = index[roles['Hips']]
    if USE_PELVIS_ROLL and len(goals) == 2 and hips in target_rot:
        n = needs()
        both = np.minimum(legs_weight['Left'], legs_weight['Right'])
        hip_w = np.linalg.norm(pose[:, legs['Left'][0], :3, 3] - pose[:, legs['Right'][0], :3, 3], axis=1)
        # +angle about forward RAISES the left hip: forward x left = forward x (up x forward) = +up
        angle = np.clip((n['Right'] - n['Left']) / np.maximum(hip_w, 1e-6), -MAX_ROLL, MAX_ROLL) * both
        k = 2 * BLEND + 1
        angle = np.convolve(np.pad(angle, BLEND, mode='edge'), np.ones(k) / k, mode='valid')
        if np.abs(angle).max() > 1e-4:
            body_fwd = target_rot[hips] @ np.linalg.inv(rest[hips, :3, :3]) @ fwd
            body_fwd = body_fwd - np.outer(body_fwd @ up, up)
            body_fwd /= np.linalg.norm(body_fwd, axis=1, keepdims=True)
            roll = Rotation.from_rotvec(body_fwd * angle[:, None]).as_matrix()
            target_rot[hips] = np.einsum('tab,tbc->tac', roll, target_rot[hips])
            pose = fk(target_rot, root)
            notes.append(f'pelvis rolled up to {np.degrees(np.abs(angle).max()):.1f} deg to plant both feet')

    # Lower the hips just enough for every planted leg to reach its spot.
    drop = np.zeros(T)
    for side, (goal, mask) in goals.items():
        l, s, f, a, b = legs[side]
        v = goal - pose[:, l, :3, 3]
        z = v @ up
        h = np.linalg.norm(flat(v), axis=1)
        reach = REACH * (a + b)
        need = np.where(h < reach, -np.sqrt(np.maximum(reach ** 2 - h ** 2, 0)) - z, 0)
        drop = np.maximum(drop, np.where(mask, need, 0))
    if not USE_HIP_DROP:
        # Off for straight stance legs only: lowering the hips there bent the other planted knee (a walk). With
        # every planted leg already bent (a landing, a crouch) the drop costs nothing and keeps feet down.
        for side in goals:
            l, s, f = legs[side][:3]
            u, w_ = pose[:, l, :3, 3] - pose[:, s, :3, 3], pose[:, f, :3, 3] - pose[:, s, :3, 3]
            bend = 180 - np.degrees(np.arccos(np.clip((u * w_).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(w_, axis=1), -1, 1)))
            drop = np.where((legs_weight[side] > 0) & (bend < STRAIGHT_KNEE), 0, drop)
    if drop.max() > 0:
        k = 2 * BLEND + 1
        drop = np.array([drop[max(0, t - BLEND):t + BLEND + 1].max() for t in range(T)])  # cover neighbours
        drop = np.convolve(np.pad(drop, BLEND, mode='edge'), np.ones(k) / k, mode='valid')
        # Only where a foot is (easing into being) planted: smoothing must not pull a jump's airtime down.
        planted = np.max([legs_weight[side] for side in legs_weight], axis=0)
        drop *= planted
        root = root - np.outer(drop, up)
        pose = fk(target_rot, root)
        notes.append(f'hips lowered up to {drop.max() * 100:.1f} cm so planted feet reach the floor')

    # Two-bone IK, knee kept in the plane it already bends in.
    for side, (goal, mask) in goals.items():
        l, s, f, a, b = legs[side]
        frames = np.flatnonzero(mask)
        if not len(frames):
            continue
        H, K, A = (pose[frames, i, :3, 3] for i in (l, s, f))
        G = goal[frames]
        v = G - H
        dist = np.clip(np.linalg.norm(v, axis=1), abs(a - b) + 1e-4, REACH * (a + b))
        if KEEP_LEG_LENGTH:
            # Near full extension a few mm of hip-ankle distance is several degrees of knee bend, so the
            # leg keeps its transferred length and is only re-aimed at the pin: the knee angle stays the
            # performer's and the foot lands within a few mm of its pin.
            keep = np.linalg.norm(A - H, axis=1)
            # A clearly bent leg (landing, crouching) is not near full extension: it reaches its pin, or a foot
            # that GEM lands lower than the rig's legs put it hangs above the floor (up to 19 cm on a jump).
            u, w_ = H - K, A - K
            bend = 180 - np.degrees(np.arccos(np.clip((u * w_).sum(1) / np.linalg.norm(u, axis=1) / np.linalg.norm(w_, axis=1), -1, 1)))
            reach = np.clip((bend - STRAIGHT_KNEE) / KNEE_RAMP, 0, 1) if KNEE_RAMP > 0 else (bend >= STRAIGHT_KNEE)
            if REACH_FLAT_ONLY:  # bending more to reach is always fine; straightening only on a flat foot
                reach = np.where(np.linalg.norm(v, axis=1) < keep, reach, reach * flat_foot[side][frames])
            keep = keep + reach * (np.linalg.norm(v, axis=1) - keep)
            # ...except where that would put the planted ankle below the floor: bend just enough.
            floor_h = rest[f, :3, 3] @ up
            dz = (v / np.linalg.norm(v, axis=1, keepdims=True)) @ up
            limit = np.where(dz < -1e-6, (floor_h - H @ up) / np.minimum(dz, -1e-6), np.inf)
            dist = np.clip(np.minimum(keep, limit), abs(a - b) + 1e-4, REACH * (a + b))
        n = v / np.linalg.norm(v, axis=1, keepdims=True)
        # Knee direction: the performer's, leaning toward the body's forward by KNEE_PRIOR of the leg's length.
        # On a near-straight leg the performer's sideways knee offset is noise, and the floor guard above can
        # still bend it, so without the lean a straight leg's knee flicks sideways or back for a frame.
        ahead = (target_rot[hips] @ np.linalg.inv(rest[hips, :3, :3]) @ fwd)[frames] if hips in target_rot             else np.tile(fwd, (len(frames), 1))
        bend = (K - H) + KNEE_PRIOR * (a + b) * (ahead - np.outer(ahead @ up, up))
        bend = bend - np.einsum('ti,ti->t', bend, n)[:, None] * n
        weak = np.linalg.norm(bend, axis=1) < 1e-4 * (a + b)
        bend[weak] = fwd - np.outer(n[weak] @ fwd, np.ones(3)) * n[weak]  # straight leg: knees forward
        bend /= np.linalg.norm(bend, axis=1, keepdims=True)
        along = (a * a - b * b + dist * dist) / (2 * dist)
        knee = H + n * along[:, None] + bend * np.sqrt(np.maximum(a * a - along * along, 0))[:, None]
        ankle_new = H + n * dist[:, None]
        thigh = target_rot[l].copy()
        shin = target_rot[s].copy()
        thigh[frames] = np.einsum('tab,tbc->tac', _swing_batch(K - H, knee - H), thigh[frames])
        shin[frames] = np.einsum('tab,tbc->tac', _swing_batch(A - K, ankle_new - knee), shin[frames])
        target_rot[l], target_rot[s] = thigh, shin

    # Soles never go through the floor while a foot is (easing into being) planted. Source foot pitch
    # is noisy, and a few degrees toes-down drives the front of a long foot under the floor. Only lift:
    # heel strikes with toes up and toe-offs keep the performance.
    pose = fk(target_rot, root)
    lifted = 0.0
    for side in goals:
        l, s, f, a, b = legs[side]
        key = side + 'ToeBase'
        if key not in roles:
            continue
        toe = index[roles[key]]
        if toe not in target_rot or f not in target_rot:
            continue
        weight = legs_weight[side]
        for pivot, bone, moved in ((f, f, [f, toe]), (toe, toe, [toe])):
            # joint that must stay above its rest height: the toe joint for the foot, the toe tip for the toe
            pose = fk(target_rot, root)
            if bone == f:
                point, floor_h = pose[:, toe, :3, 3], rest[toe, :3, 3] @ up
            else:
                tip_local = np.linalg.inv(rest[toe]) @ np.r_[rest[toe, :3, 3] + rest[toe, :3, 1] * lengths[toe], 1]
                point = (pose[:, toe] @ tip_local)[:, :3]
                floor_h = (rest[toe] @ tip_local)[:3] @ up
            for t in np.flatnonzero(weight > 0):
                deficit = floor_h - point[t] @ up
                if deficit <= 1e-4:
                    continue
                arm = point[t] - pose[t, pivot, :3, 3]
                axis = np.cross(arm, up)
                if np.linalg.norm(axis) < 1e-9:
                    continue
                axis /= np.linalg.norm(axis)
                horizontal = np.linalg.norm(arm - up * (arm @ up))
                angle = weight[t] * np.arctan2(deficit, max(horizontal, 1e-6))
                r = Rotation.from_rotvec(axis * angle).as_matrix()  # rotates the point upward
                for k in moved:
                    target_rot[k] = target_rot[k].copy()
                    target_rot[k][t] = r @ target_rot[k][t]
                lifted = max(lifted, deficit)
    if lifted > 0:
        notes.append(f'soles lifted up to {lifted * 100:.1f} cm out of the floor')
    return root, notes
