"""A full-body character holding a prop (rifle at the hip or shoulder, sword, bat, a box in both hands).

The retargeted motion (retarget.Result, or its saved npz) moves the arms the way the performer did, but its
hands drift a few centimetres off the prop: the reconstruction does not know about it. Here:

  1. the prop is placed per frame, either from the video (2D-3D point matches solved with PnP, e.g. the matches
     fp_points.track_points follows) or, with no tracking, riding a socket on the main hand;
  2. each hand's grip on the prop (the hand's transform in the prop's frame) is found from the frames where the
     hand is near it, and how strongly the hand holds it per frame (hold weight, eased);
  3. a two-bone IK per arm brings the wrist to its grip and turns the hand to the grip's orientation, blended by
     the hold weight. The elbow stays on the side the performer's elbow was (pole from the original elbow), the
     upper arm and forearm turn the shortest way, and the hand's twist about the forearm moves into the forearm
     (a real forearm rolls; the wrist only bends). Every other bone keeps its local transform, so fingers,
     twist helpers and anything parented below follow;
  4. the prop's socket on the main hand (hand-local transform) is exported for the engine or Blender.

Everything is in armature space (Blender bone convention, Y along the bone), as in retarget.py. Nothing here
is specific to a rig, a clip or a prop.
"""
import json

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

from motionbridge.core.smooth import one_euro

NEAR, FAR = 0.06, 0.15  # m (scaled by the rig's forearm length / 0.26): a wrist closer than NEAR to its grip takes
                        # hold; it lets go only beyond FAR (hysteresis: a reconstructed arm wobbles a few cm)
PALM = 0.03             # m (scaled the same way): the wrist's distance from the prop's surface when holding it
MIN_HOLD = 6            # frames: shorter holds are dropped (a hand passing by the prop)
EASE = 5                # frames to blend into and out of a hold
REACH = 0.999           # fraction of the arm's full length the IK may stretch to
ROLL_SHARE = 1.0        # how much of the hand's twist about the forearm moves into the forearm
ROLL_LIMIT = np.radians(90)


def _orthonormal(m):
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


class Skeleton:
    """Rest pose and hierarchy from a dump_rig.py JSON, with forward kinematics to and from per-bone local pose
    (Blender basis: local pose quaternion w x y z + location)."""

    def __init__(self, dump):
        bones = dump['bones']
        self.names = [b['name'] for b in bones]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parent = [self.index.get(b['parent']) for b in bones]
        self.obj = np.array(dump['object_matrix'])
        obj_inv = np.linalg.inv(self.obj)
        rest = np.array([obj_inv @ np.array(b['rest']) for b in bones])
        rest[:, :3, :3] = [_orthonormal(r) for r in rest[:, :3, :3]]
        self.rest = rest
        self.rest_local = np.array([rest[i] if p is None else np.linalg.inv(rest[p]) @ rest[i]
                                    for i, p in enumerate(self.parent)])
        order, seen = [], set()
        while len(order) < len(bones):  # parents before children
            for i, p in enumerate(self.parent):
                if i not in seen and (p is None or p in seen):
                    order.append(i)
                    seen.add(i)
        self.order = order
        self.children = [[j for j, p in enumerate(self.parent) if p == i] for i in range(len(bones))]
        scale = np.linalg.norm(self.obj[:3, :3], axis=0).mean()
        self.lengths = np.array([np.linalg.norm(np.array(b['tail']) - np.array(b['head'])) for b in bones]) / scale

    def descendants(self, i):
        out, stack = [], list(self.children[i])
        while stack:
            j = stack.pop()
            out.append(j)
            stack += self.children[j]
        return out

    def pose(self, quaternions, locations):
        """(T, B, 4) w x y z + (T, B, 3) local pose -> (T, B, 4, 4) armature-space bone matrices."""
        T, B = quaternions.shape[:2]
        basis = np.repeat(np.eye(4)[None, None], T, 0).repeat(B, 1)
        q = np.asarray(quaternions, float)
        basis[..., :3, :3] = Rotation.from_quat(q[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(T, B, 3, 3)
        basis[..., :3, 3] = locations
        pose = np.empty_like(basis)
        for i in self.order:
            p = self.parent[i]
            local = self.rest_local[i] @ basis[:, i]
            pose[:, i] = local if p is None else pose[:, p] @ local
        return pose

    def basis(self, pose):
        """Inverse of pose(): (T, B, 4, 4) -> quaternions (T, B, 4) w x y z (no sign flips between frames),
        locations (T, B, 3)."""
        T, B = pose.shape[:2]
        basis = np.empty_like(pose)
        for i, p in enumerate(self.parent):
            local = pose[:, i] if p is None else np.linalg.inv(pose[:, p]) @ pose[:, i]
            basis[:, i] = np.linalg.inv(self.rest_local[i]) @ local
        q = Rotation.from_matrix(basis[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T, B, 4)[..., [3, 0, 1, 2]]
        q *= np.where(q[..., :1] < 0, -1, 1)
        for t in range(1, T):
            flip = np.einsum('bi,bi->b', q[t], q[t - 1]) < 0
            q[t, flip] *= -1
        return q, basis[..., :3, 3]

    def heads_tails(self, pose):
        """World-space heads and tails, as retarget.Result stores them."""
        h = pose[..., :3, 3]
        t = h + pose[..., :3, 1] * self.lengths[None, :, None]
        w = lambda x: x @ self.obj[:3, :3].T + self.obj[:3, 3]
        return w(h), w(t)


def arm_chain(skeleton, roles, side):
    """(upper arm, forearm, hand) bone indices for 'Left' / 'Right' from a rigmap's roles."""
    return tuple(skeleton.index[roles[side + r]] for r in ('Arm', 'ForeArm', 'Hand'))


# ---------------------------------------------------------------------------------------------- prop placement

def prop_from_socket(pose, hand, socket):
    """The prop riding the main hand: (T, 4, 4) = hand matrix @ socket (prop in the hand's frame)."""
    return pose[:, hand] @ socket


def prop_from_matches(X, uv, K, world_from_camera, placement=None, valid=None, ransac_px=3.0):
    """The prop from 2D-3D matches per frame (e.g. what fp_points.track_points follows in a video).
    X: (N, 3) points on the prop in its own frame (or at `placement`, if given, as fp_points keeps them);
    uv: (T, N, 2) where each is seen (NaN when lost); K: pinhole matrix (fp_points.intrinsics);
    world_from_camera: (4, 4) Blender camera matrix (looks down -Z, Y up), in the same space the result should
    be in (armature space: inv(object_matrix) @ camera world).
    Returns (T, 4, 4) prop matrices (frames that cannot be solved hold the previous one) and per-frame median
    reprojection error in px (NaN when unsolved)."""
    import cv2
    T = len(uv)
    X = np.asarray(X, float)
    if placement is not None:  # matches given on the placed prop: back to the prop's own frame
        X = (X - placement[:3, 3]) @ placement[:3, :3]
    flip = np.diag([1., -1., -1., 1.])  # Blender camera (-Z forward, Y up) <-> OpenCV (Z forward, Y down)
    cam_from_world = flip @ np.linalg.inv(world_from_camera)
    out, err = np.zeros((T, 4, 4)), np.full(T, np.nan)
    last = placement if placement is not None else np.eye(4)
    for t in range(T):
        ok = np.isfinite(uv[t]).all(1) if valid is None else valid[t]
        if ok.sum() >= 6:
            guess = cam_from_world @ last
            rv0, _ = cv2.Rodrigues(guess[:3, :3])
            good, rv, tv, inl = cv2.solvePnPRansac(X[ok], uv[t][ok], K, None, rv0.copy(), guess[:3, 3].copy(),
                                                   useExtrinsicGuess=True, reprojectionError=ransac_px,
                                                   flags=cv2.SOLVEPNP_ITERATIVE)
            if good and inl is not None and len(inl) >= 6:
                m = np.eye(4)
                m[:3, :3] = cv2.Rodrigues(rv)[0]
                m[:3, 3] = tv.ravel()
                last = np.linalg.inv(cam_from_world) @ m
                proj, _ = cv2.projectPoints(X[ok], rv, tv, K, None)
                err[t] = np.median(np.linalg.norm(proj[:, 0] - uv[t][ok], axis=1))
        out[t] = last
    return out, err


def smooth_prop(motion, fps, cutoff=4.0):
    """Zero-phase low-pass of a prop path (T, 4, 4): positions, and rotations about their mean (no wrap-around)."""
    T = len(motion)
    out = motion.copy()
    out[:, :3, 3] = one_euro(motion[:, :3, 3], np.zeros(T), fps, cutoff, 0)
    R = Rotation.from_matrix(motion[:, :3, :3])
    mean = R.mean()
    v = one_euro((mean.inv() * R).as_rotvec(), np.zeros(T), fps, cutoff, 0)
    out[:, :3, :3] = (mean * Rotation.from_rotvec(v)).as_matrix()
    return out


# ------------------------------------------------------------------------------------------------- grips, holds

def mean_transform(ms):
    """Average of rigid transforms (N, 4, 4): mean position, chordal mean rotation."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(ms[:, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(ms[:, :3, 3], axis=0)
    return out


def grip_of(pose, hand, prop, frames=None):
    """The hand's grip on the prop: its transform in the prop's frame, averaged over `frames` (default: all)."""
    rel = np.linalg.inv(prop) @ pose[:, hand]
    return mean_transform(rel if frames is None else rel[frames])


def held_cluster(pose, hand, prop, reach, radius=None):
    """Frames where the hand sits at its most common place in the prop's frame (bool (T,)). A hold is a dense
    cluster there (the hand stays put on the prop); a hand moving freely is spread out. Being near the prop's
    centre says nothing: a box carried by its ends is held far from it."""
    radius = (NEAR + FAR) / 2 * reach if radius is None else radius
    rel = np.einsum('tij,tj->ti', np.linalg.inv(prop), np.c_[pose[:, hand, :3, 3], np.ones(len(pose))])[:, :3]
    d = np.linalg.norm(rel[:, None] - rel[None], axis=2)
    count = (d < radius).sum(1)
    return d[np.argmax(count)] < radius


def hold_weight(pose, hand, prop, grip, reach, near=NEAR, far=FAR, min_hold=MIN_HOLD, ease=EASE):
    """Per frame 0..1: how much the hand holds the prop, from how close its wrist is to the grip. Distances are
    scaled by `reach` (the arm's forearm length / 0.26 m) so small and large characters behave alike; short
    holds are dropped and the rest eased in and out."""
    want = (prop @ grip)[:, :3, 3]
    d = np.linalg.norm(pose[:, hand, :3, 3] - want, axis=1) / reach
    on = np.zeros(len(d), bool)
    for t in range(len(d)):  # take hold when near, let go only when far
        on[t] = d[t] < near or (t > 0 and on[t - 1] and d[t] < far)
    for t in range(len(d) - 2, -1, -1):  # and backwards, so a hold already under way at the start counts
        on[t] |= on[t + 1] and d[t] < far
    # drop short runs (a hand passing the prop) and fill short gaps (a hand briefly read off it)
    on = _runs_fill(on, min_hold, True)
    on = ~_runs_fill(~on, min_hold, True)
    w = uniform_filter1d(on.astype(float), 2 * ease + 1, mode='nearest')
    w = np.where(on, np.maximum(w, 0.5), w)
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(w, 0, 1))  # smoothstep-like ease


def snap_grip(grip, distance, reach, palm=PALM, steps=20):
    """Move a grip (in the prop's frame) along the prop's surface normal so the wrist sits `palm` (scaled by
    reach) off the surface. distance: signed distance in the prop's frame, (N, 3) -> (N,), e.g.
    lambda p: item.distance(p, np.eye(4)) for an fp_track.Item. The grip found from a reconstructed motion is
    off by its low-frequency wobble; the hand touching the prop pins one direction of that error."""
    g = grip.copy()
    for _ in range(steps):
        p = g[:3, 3]
        d = float(distance(p[None])[0])
        e = 1e-3
        n = np.array([float(distance((p + e * a)[None])[0]) - d for a in np.eye(3)]) / e
        if np.linalg.norm(n) < 1e-9:
            break
        n /= np.linalg.norm(n)
        step = d - palm * reach
        g[:3, 3] = p - step * n
        if abs(step) < 1e-4:
            break
    return g


def _runs_fill(on, n, value):
    """Runs of `value` shorter than n frames are flipped."""
    out = on.copy()
    t = 0
    while t < len(on):
        u = t
        while u < len(on) and on[u] == on[t]:
            u += 1
        if on[t] == value and u - t < n:
            out[t:u] = not value
        t = u
    return out


# ---------------------------------------------------------------------------------------------------- arm IK

def _swing(a, b):
    """Shortest rotation (3x3) turning direction a onto b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:  # opposite: any perpendicular axis
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _about(R, p):
    """4x4 turning by R about point p."""
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = p - R @ p
    return m


def two_bone(s, e, w, target, l1=None, l2=None, reach=REACH):
    """Elbow position for a shoulder s reaching target, with upper/forearm lengths l1, l2 (default: from s, e,
    w) and the elbow kept on the side of the line s -> target where the original elbow e was. Returns
    (elbow, wrist actually reached)."""
    l1 = np.linalg.norm(e - s) if l1 is None else l1
    l2 = np.linalg.norm(w - e) if l2 is None else l2
    d = target - s
    D = np.linalg.norm(d)
    u = d / D if D > 1e-9 else (w - s) / np.linalg.norm(w - s)
    D = np.clip(D, abs(l1 - l2) + 1e-6, (l1 + l2) * reach)
    pole = (e - s) - u * ((e - s) @ u)
    if np.linalg.norm(pole) < 1e-6 * l1:  # straight original arm: bend the way it bent last (caller's e) or down
        pole = np.cross(u, [1., 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1., 0])
    v = pole / np.linalg.norm(pole)
    x = (l1 * l1 - l2 * l2 + D * D) / (2 * D)
    h = np.sqrt(max(l1 * l1 - x * x, 0))
    return s + x * u + h * v, s + D * u


def solve_arm(skeleton, pose, chain, wrist, hand_rot, weight, roll_share=ROLL_SHARE):
    """One frame, one arm: bring the hand to `wrist` (armature space) turned to `hand_rot` (3x3), blended with
    its own pose by `weight`. pose: (B, 4, 4), changed in place for the arm and every bone below it."""
    if weight <= 0:
        return
    ua, fa, ha = chain
    old = pose.copy()
    s, e, w = old[ua, :3, 3], old[fa, :3, 3], old[ha, :3, 3]
    target = w + weight * (wrist - w)
    r0 = Rotation.from_matrix(old[ha, :3, :3])
    rot = (r0 * Rotation.from_rotvec(weight * (r0.inv() * Rotation.from_matrix(hand_rot)).as_rotvec())).as_matrix()
    e2, w2 = two_bone(s, e, w, target)
    R1 = _swing(e - s, e2 - s)
    A1 = _about(R1, s)                                   # upper arm (and everything on it) about the shoulder
    w1 = (A1 @ np.r_[w, 1])[:3]
    R2 = _swing(w1 - e2, w2 - e2)
    A2 = _about(R2, e2)                                  # then the forearm about the new elbow
    new = {ua: A1 @ old[ua], fa: A2 @ A1 @ old[fa]}
    # the hand's twist about the forearm moves into the forearm: the wrist only bends
    axis = new[fa][:3, 1] / np.linalg.norm(new[fa][:3, 1])
    carried = (A2 @ A1 @ old[ha])[:3, :3]                # where the hand would point riding the forearm
    delta = Rotation.from_matrix(rot @ carried.T).as_rotvec()
    twist = np.clip(roll_share * (delta @ axis), -ROLL_LIMIT, ROLL_LIMIT)
    A3 = _about(Rotation.from_rotvec(axis * twist).as_matrix(), new[fa][:3, 3])
    new[fa] = A3 @ new[fa]
    h = np.eye(4)
    h[:3, :3] = rot
    h[:3, 3] = (A3 @ A2 @ A1 @ np.r_[w, 1])[:3]
    new[ha] = h
    # everything below keeps its local transform (fingers, twist helpers, props parented to the hand)
    moved = {ua: None}
    pose[ua], pose[fa], pose[ha] = new[ua], new[fa], new[ha]
    for i in skeleton.order:
        p = skeleton.parent[i]
        if i in new or p is None:
            continue
        if p in new or p in moved:
            pose[i] = pose[p] @ np.linalg.inv(old[p]) @ old[i]
            moved[i] = None


def hold_prop(skeleton, pose, chains, prop, grips, weights):
    """Both (or any) hands onto the prop. pose: (T, B, 4, 4) armature space (a copy is returned);
    chains/grips/weights: per hand, from arm_chain, grip_of, hold_weight. Returns the new pose."""
    out = pose.copy()
    for t in range(len(pose)):
        for chain, grip, w in zip(chains, grips, weights):
            g = prop[t] @ grip
            solve_arm(skeleton, out[t], chain, g[:3, 3], g[:3, :3], float(w[t]))
    return out


def grip_error(pose, hand, prop, grip):
    """Per frame: wrist distance to its grip (m, armature units) and hand turn off the grip (degrees)."""
    g = prop @ grip
    d = np.linalg.norm(pose[:, hand, :3, 3] - g[:, :3, 3], axis=1)
    a = Rotation.from_matrix(np.einsum('tji,tjk->tik', pose[:, hand, :3, :3], g[:, :3, :3])).magnitude()
    return d, np.degrees(a)


# --------------------------------------------------------------------------------------------------- socket

def socket_of(pose, hand, prop, weights=None):
    """The prop in the main hand's frame: per frame (T, 4, 4) and one constant socket (averaged over the frames
    the hand holds it, weight > 0.5, or all)."""
    rel = np.linalg.inv(pose[:, hand]) @ prop
    use = slice(None) if weights is None or not (weights > 0.5).any() else weights > 0.5
    return rel, mean_transform(rel[use])


def export_socket(path, skeleton, hand, socket, prop_world=None):
    """JSON for the engine / Blender: the bone the prop is parented to and the prop's matrix in that bone's
    frame (armature units; Blender's bone frame is the pose bone matrix: Y along the bone)."""
    d = {'bone': skeleton.names[hand], 'matrix': np.asarray(socket).tolist()}
    if prop_world is not None:
        d['frames'] = len(prop_world)
    with open(path, 'w') as f:
        json.dump(d, f, indent=1)
    return d


# ------------------------------------------------------------------------------------------------- the step

def attach(skeleton, quaternions, locations, roles, prop=None, main='Right', socket=None, grips=None,
           hands=('Right', 'Left'), fps=30.0, weights=None, distance=None, log=print):
    """One call for a retargeted motion: returns dict(quaternions, locations, heads, tails, prop (T, 4, 4),
    socket (4, 4) on the main hand, weights {side: (T,)}, error {side: (before, after) mean wrist mm while held}).
    prop: (T, 4, 4) from the video (prop_from_matches / fp_points), or None to ride `socket` on the main hand
    (socket required then). grips: {side: 4x4 in the prop's frame}, default found from the motion itself.
    weights: {side: (T,)} hold weights to use instead of the ones found from the motion.
    distance: the prop's signed distance in its own frame ((N, 3) -> (N,)); found grips are then snapped to
    rest on its surface (snap_grip)."""
    pose = skeleton.pose(quaternions, locations)
    T = len(pose)
    chains = {s: arm_chain(skeleton, roles, s) for s in hands}
    riding = prop is None
    if riding:
        if socket is None:
            raise ValueError('no prop path and no socket: give one')
        prop = prop_from_socket(pose, chains[main][2], socket)
    else:
        prop = smooth_prop(prop, fps)
    reach = {s: np.linalg.norm(pose[0, chains[s][2], :3, 3] - pose[0, chains[s][1], :3, 3]) / 0.26 for s in hands}
    grips = dict(grips or {})
    given, weights = dict(weights or {}), {}
    for s in hands:
        hand = chains[s][2]
        if s not in grips:  # first guess: where the hand stays put in the prop's frame, then refine on the hold
            grips[s] = grip_of(pose, hand, prop, held_cluster(pose, hand, prop, reach[s]))
            w = hold_weight(pose, hand, prop, grips[s], reach[s])
            if (w > 0.5).sum() >= MIN_HOLD:
                grips[s] = grip_of(pose, hand, prop, w > 0.5)
            if distance is not None:
                grips[s] = snap_grip(grips[s], distance, reach[s])
        weights[s] = given[s] if s in given else hold_weight(pose, hand, prop, grips[s], reach[s])
        if riding and s == main:
            weights[s] = np.ones(T)  # the prop rides this hand: it holds it throughout
    out = hold_prop(skeleton, pose, [chains[s] for s in hands], prop, [grips[s] for s in hands],
                    [weights[s] for s in hands])
    error = {}
    for s in hands:
        held = weights[s] > 0.5
        if held.any():
            b = grip_error(pose, chains[s][2], prop, grips[s])[0][held].mean() * 1000
            a = grip_error(out, chains[s][2], prop, grips[s])[0][held].mean() * 1000
            error[s] = (b, a)
            if log:
                log(f'{s.lower()} hand holds the prop on {int(held.sum())} of {T} frames; wrist off its grip '
                    f'{b:.1f} -> {a:.1f} mm')
    q, l = skeleton.basis(out)
    heads, tails = skeleton.heads_tails(out)
    _, sock = socket_of(out, chains[main][2], prop, weights[main])
    return dict(quaternions=q, locations=l, heads=heads, tails=tails, prop=prop, socket=sock, weights=weights,
                grips=grips, error=error, pose=out)
