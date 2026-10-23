"""First-person hands: pose any FPS arm rig to a generated video, frame by frame, with its main prop.

Inputs are the prepared scene (blender/fp_scene.py: rig, camera, prop geometry) and the learned part
(gem/fp_extract.py: SAM 2 arm and prop masks, SAM 3D Body's hands per frame). Frame 0 is known: the
video starts from a render of this scene. From there:

  1. each hand is tracked from the previous frame, matching the extracted hand's bone directions (each
     rig keeps its own proportions) and screen points, without passing through the prop
  2. the main prop: while the hand that holds it in the start frame holds it, that hand decides where it
     is (its start-frame grip is the socket); a prop nobody holds is tracked from its outline
  3. hands and prop are smoothed (One Euro, zero-phase), the holding hand is put back exactly on the prop,
     fingers are kept out of the prop and the other hand, and remaining pops are eased
Everything else in the scene (loose parts, spare magazines) is left as the user placed it.

Where the hand sits: rigs whose hand hangs off an IK control (the usual FPS layout) are placed by
moving that control; rigs whose hand is attached to the forearm are placed by turning the arm.
"""
import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, map_coordinates, uniform_filter1d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from motionbridge.core import hands
from motionbridge.core.hand_fit import project
from motionbridge.core.smooth import _speed, one_euro, settings

FPS = 24
# Hand points as MediaPipe orders them (wrist, then base to tip per finger) from SAM 3D Body's 70
MHR_RIGHT = [41] + [21 + 4 * f + k for f in range(5) for k in (3, 2, 1, 0)]
SEG = [(0, 1), (1, 2), (2, 3), (3, 4)] + [(a, a + 1) for f in range(1, 5) for a in (1 + 4 * f, 2 + 4 * f, 3 + 4 * f)] + \
      [(0, 5), (0, 9), (0, 13), (0, 17)]
FINGER_SEG = [s for s in SEG if not (s[0] == 0 and s[1] in (5, 9, 13, 17))]  # fingers and the whole thumb
PALM = [(0, 9), (9, 10), (10, 11), (11, 12), (0, 5), (5, 6), (6, 7), (7, 8), (0, 17)]

# Hand tracking: expected size of each term (residuals are divided by these)
DIR_SIGMA = 0.12      # bone direction (unit vector difference)
KP2D_SIGMA = 0.03     # screen points, fraction of the hand's size on screen
DEPTH_SIGMA = 0.04    # m, wrist depth from the extraction (scaled to the rig's hand)
CONTACT_SIGMA = 0.002 # m, finger/palm surface inside the item
FINGER_RADIUS, PALM_RADIUS = 0.008, 0.014
FRONT_MARGIN = 0.005  # m: the hand is this far in front of the item where the video shows it so
STEP_FINGER = 0.3     # rad per frame
PLACEMENT = {'control': {'prior': 0.15, 'step': 0.015},  # m: hand control offset from the start / per frame
             'arm': {'prior': 1.5, 'step': 0.15}}       # rad: arm turn from the start / per frame
DEPTH_MAP_SCALE = 4   # item depth map at 1/4 resolution
FINGER_SMOOTH = 0.35  # fingers get this share of the hand's smoothing strength
WRIST_TWIST = 25.0    # deg: the hand may turn this much about the forearm (the forearm rolls for more)
WRIST_BEND = 75.0     # deg: and bend this much against it


def to_hand21(kp):
    return kp[..., MHR_RIGHT, :]


def directions(p):
    v = np.array([p[b] - p[a] for a, b in SEG])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def hand_size(p):
    return sum(np.linalg.norm(p[b] - p[a]) for a, b in PALM)


class Rig:
    """The scene's arm rig as a function of a few numbers: placement, then wrist and fingers."""

    def __init__(self, dump, side='Right'):
        self.dump = dump
        self.world = np.array(dump['rig_world'])
        self.scale = np.linalg.norm(self.world[:3, 0])
        bones = []
        for b in dump['bones']:
            rest = self.world @ np.array(b['rest'])
            head = rest[:3, 3]
            tail = head + rest[:3, 1] / np.linalg.norm(rest[:3, 1]) * b['length'] * self.scale
            bones.append({'name': b['name'], 'parent': b['parent'], 'rest': rest.tolist(), 'head': head.tolist(),
                          'tail': tail.tolist(), 'deform': True})
        flat = {'bones': bones, 'object_matrix': self.world.tolist(), 'armature': dump['armature']}
        self.sk = hands.Skeleton(flat)
        found = [h for h in hands.find_hands(flat) if h.side == side]
        if not found:
            raise ValueError(f'no {side.lower()} hand with four or five finger chains found on {dump["armature"]}')
        self.hand = found[0]
        self.basis0 = np.array([b['basis'] for b in dump['bones']])
        self.model = hands.HandModel(self.sk, self.hand, self.basis0)
        self.pose0 = {b['name']: self.world @ np.array(b['pose']) for b in dump['bones']}
        # placement: a control bone at the wrist (moved), or the forearm and upper arm (turned)
        h = self.sk.index[self.hand.hand]
        rest = self.sk.rest
        ctrl = h
        while self.sk.parent[ctrl] is not None and np.linalg.norm(rest[self.sk.parent[ctrl], :3, 3] - rest[h, :3, 3]) < 1e-3 * max(self.scale, 1e-3):
            ctrl = self.sk.parent[ctrl]
        ancestors = []
        p = self.sk.parent[ctrl]
        while p is not None:
            ancestors.append(p)
            p = self.sk.parent[p]
        if len(ancestors) >= 2 and ctrl == h:
            self.mode, self.arm = 'arm', ancestors[:2]   # forearm, upper arm
            self.na = 6
        else:
            self.mode, self.ctrl = 'control', ctrl
            self.na = 3
            self.to_local = np.linalg.inv(rest[ctrl, :3, :3]) if self.sk.parent[ctrl] is None else None
            if self.to_local is None:  # a control under a fixed parent: world offsets through the parent's rest
                self.to_local = np.linalg.inv(self.sk.pose(self.basis0)[ctrl, :3, :3])
        self.size = self.na + self.hand.size
        self.names = self.sk.names
        lo, hi = self.hand.limits()
        lim = 1.0 if self.mode == 'control' else 1.5  # m / rad
        self.lo, self.hi = np.r_[np.full(self.na, -lim), lo], np.r_[np.full(self.na, lim), hi]
        self.span0 = None
        # the forearm the wrist bends against: the bone outside the hand whose tail ends at the wrist
        in_hand = {self.hand.hand} | {n for j in list(self.hand.fingers.values()) + list(self.hand.palm.values()) for n in j}
        wrist0 = self.pose0[self.hand.hand][:3, 3]
        tails = {b['name']: self.pose0[b['name']][:3, 3] + self.pose0[b['name']][:3, 1] / np.linalg.norm(self.pose0[b['name']][:3, 1])
                 * b['length'] * self.scale for b in dump['bones'] if b['name'] not in in_hand and b['length'] * self.scale > 0.1}
        fore = min(tails, key=lambda n: np.linalg.norm(tails[n] - wrist0), default=None)
        self.forearm = None
        if fore is not None and np.linalg.norm(tails[fore] - wrist0) < 0.05 and dump['bones'][self.sk.index[fore]]['parent']:
            upper = dump['bones'][self.sk.index[fore]]['parent']
            frame = lambda n: self.pose0[n][:3, :3] / np.linalg.norm(self.pose0[n][:3, :3], axis=0)
            self.forearm = {'name': fore, 'shoulder': self.pose0[upper][:3, 3], 'elbow0': self.pose0[fore][:3, 3],
                            'L1': np.linalg.norm(self.pose0[fore][:3, 3] - self.pose0[upper][:3, 3]),
                            'L2': np.linalg.norm(tails[fore] - self.pose0[fore][:3, 3]), 'F0': frame(fore)}
            self.forearm['rel0'] = self.forearm['F0'].T @ self.hand_frame(np.zeros(self.size))[:3, :3]

    def forearm_frame(self, x):
        """The forearm's frame (columns; y along it) for parameters x. With the hand on an IK control, the
        arm is a two-bone chain from the shoulder to the wrist, bending in its start plane, as Blender's IK
        solves it; with an attached hand, it is the posed forearm bone."""
        fa = self.forearm
        if self.mode == 'arm':
            m = self.posed(x)[self.sk.index[fa['name']], :3, :3]
            return m / np.linalg.norm(m, axis=0)
        s, w = fa['shoulder'], self.hand_frame(x)[:3, 3]
        d = np.clip(np.linalg.norm(w - s), 1e-6, fa['L1'] + fa['L2'] - 1e-6)
        u = (w - s) / np.linalg.norm(w - s)
        a = (fa['L1'] ** 2 - fa['L2'] ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(max(fa['L1'] ** 2 - a ** 2, 0))
        v = (fa['elbow0'] - s) - ((fa['elbow0'] - s) @ u) * u
        v = v / (np.linalg.norm(v) + 1e-9)
        f = w - (s + a * u + h * v)
        f = f / np.linalg.norm(f)
        f0 = fa['F0'][:, 1]
        turn = Rotation.align_vectors([f], [f0])[0]  # the shortest turn from its start direction
        return turn.as_matrix() @ fa['F0']

    def wrist_strain(self, x):
        """(twist, bend) of the hand against the forearm, degrees, measured from the start pose."""
        if self.forearm is None:
            return 0.0, 0.0
        rel = self.forearm_frame(x).T @ self.hand_frame(x)[:3, :3]
        q = Rotation.from_matrix(rel @ self.forearm['rel0'].T).as_quat()  # x, y, z, w in the forearm's frame
        twist = np.degrees(2 * np.arctan2(q[1], q[3]))
        twist = (twist + 180) % 360 - 180
        total = np.degrees(Rotation.from_quat(q).magnitude())
        return twist, max(total - abs(twist), 0.0)

    def wrist_penalty(self, x):
        twist, bend = self.wrist_strain(x)
        return np.array([max(abs(twist) - WRIST_TWIST, 0) / 3.0, max(bend - WRIST_BEND, 0) / 3.0])

    def camera(self, W, H):
        """The scene camera for frames of W x H; frames resized unevenly from the render keep its view."""
        d = self.dump
        return {'world': d['cam_world'], 'lens': d['lens'], 'sensor': d['sensor'], 'W': W, 'H': H,
                'aspect': (H / d['H']) / (W / d['W'])}

    def basis(self, x):
        b = self.model.basis(x[self.na:])
        if self.mode == 'control':
            b[self.ctrl, :3, 3] = self.basis0[self.ctrl, :3, 3] + self.to_local @ x[:3]
        else:
            for k, i in enumerate(self.arm):
                b[i, :3, :3] = self.basis0[i, :3, :3] @ Rotation.from_rotvec(x[3 * k:3 * k + 3]).as_matrix()
        return b

    def posed(self, x):
        """The armature posed by parameters x; the last one is kept (keypoints, hand frame and wrist strain of
        one x share it: the solvers ask for all three per evaluation)."""
        key = np.asarray(x, float).tobytes()
        if getattr(self, '_posed_key', None) != key:
            self._posed_key, self._posed = key, self.sk.pose(self.basis(x))
        return self._posed

    def keypoints(self, x):
        pose = self.posed(x)
        return np.array([self.sk.point(pose, *kb) for kb in self.hand.keypoint_bones()])

    def hand_frame(self, x):
        H = self.posed(x)[self.sk.index[self.hand.hand]].copy()
        H[:3, :3] /= np.linalg.norm(H[:3, :3], axis=0)
        return H

    def arm_prompts(self, cam):
        """Screen points on the arm in the start frame (wrist, knuckles, forearm) for SAM 2."""
        kp = project(self.keypoints(np.zeros(self.size)), cam)
        wrist = self.pose0[self.hand.hand][:3, 3]
        # the forearm: outside the hand, its tail ends at the wrist (attached or IK-driven alike)
        in_hand = {self.hand.hand} | {n for j in list(self.hand.fingers.values()) + list(self.hand.palm.values()) for n in j}
        tails = {b['name']: self.pose0[b['name']][:3, 3] + self.pose0[b['name']][:3, 1] / np.linalg.norm(self.pose0[b['name']][:3, 1])
                 * b['length'] * self.scale for b in self.dump['bones'] if b['name'] not in in_hand and b['length'] * self.scale > 0.1}
        fore = min(tails, key=lambda n: np.linalg.norm(tails[n] - wrist), default=None)
        pts = list(kp[[0, 5, 9, 17]])
        if fore and np.linalg.norm(tails[fore] - wrist) < 0.05:
            e = self.pose0[fore][:3, 3]
            pts += list(project(np.array([e + a * (wrist - e) for a in (0.6, 0.85)]), cam))
        pts = np.array(pts)
        on = lambda p: p[(p[:, 0] > 0) & (p[:, 0] < cam['W']) & (p[:, 1] > 0) & (p[:, 1] < cam['H'])]
        if len(on(pts)) < 2:  # wrist and knuckles off screen (a hand below the frame edge): whatever of it shows
            pts = np.r_[pts, kp]
        return on(pts)


class Item:
    """The held object: surface points and a signed distance (sealed against holes in the mesh)."""

    def __init__(self, npz):
        d = np.load(npz)
        self.parts = [d[f'v_{i}'] for i in range(len(d['names']))]
        self.points = np.concatenate(self.parts)
        self.centre = self.points.mean(0)
        step = float(d['x'][1] - d['x'][0])
        surface = np.abs(d['sdf']) < 0.75 * step
        closed = ndimage.binary_dilation(surface, iterations=2)  # gaps up to ~2 cells are sealed
        solid = ndimage.binary_erosion(ndimage.binary_fill_holes(closed), iterations=2) | surface
        grid = np.where(solid, -distance_transform_edt(solid), distance_transform_edt(~solid)) * step
        self._sdf = RegularGridInterpolator((d['x'], d['y'], d['z']), grid, bounds_error=False, fill_value=0.05)

    def distance(self, points, M):
        """Signed distance (m) of world points to the item placed by 4x4 M (relative to the scene)."""
        return self._sdf((points - M[:3, 3]) @ M[:3, :3])

    def prompts(self, cam):
        """A few screen points on the item in the start frame for SAM 2."""
        uv = project(self.points, cam)
        c = uv.mean(0)
        axis = np.linalg.svd(uv - c, full_matrices=False)[2][0]
        spread = (uv - c) @ axis
        targets = [c + axis * np.percentile(spread, q) * 0.6 for q in (10, 90)] + [c]
        return np.array([uv[np.argmin(np.linalg.norm(uv - t, axis=1))] for t in targets])

    def depth_map(self, M, cam, cam_pos, forward):
        q = DEPTH_MAP_SCALE
        v = self.points @ M[:3, :3].T + M[:3, 3]
        uv = project(v, cam) / q
        z = (v - cam_pos) @ forward
        H, W = cam['H'] // q, cam['W'] // q
        D = np.full((H, W), np.inf)
        u, w = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < W) & (w >= 0) & (w < H)
        np.minimum.at(D, (w[ok], u[ok]), z[ok])
        return ndimage.grey_erosion(np.where(np.isinf(D), 1e3, D), size=(3, 3))


def rigid(x, centre):
    """6 numbers (turn about the item's centre, move) -> 4x4."""
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    M[:3, 3] = centre + x[3:] - M[:3, :3] @ centre
    return M


def track_item(item, masks, cam, log=print, progress=None):
    """Item pose per frame (T, 4, 4) from its masks, starting at its scene placement."""
    T, H, W = masks.shape
    rng = np.random.default_rng(0)
    pts = item.points[rng.choice(len(item.points), min(3000, len(item.points)), replace=False)]
    cap = 15.0
    soft = lambda d: cap * np.log1p(d / cap)
    out, x = np.zeros((T, 4, 4)), np.zeros(6)
    for t in range(T):
        m = masks[t]
        if m.sum() >= 50:
            outside = distance_transform_edt(~m)
            ys, xs = np.nonzero(m)
            pick = np.random.default_rng(t).choice(len(xs), min(1500, len(xs)), replace=False)
            mask_pts = np.c_[xs[pick], ys[pick]].astype(float)
            prev = x.copy()

            def res(x):
                M = rigid(x, item.centre)
                uv = project(pts @ M[:3, :3].T + M[:3, 3], cam)
                d_out = soft(map_coordinates(outside, [uv[:, 1], uv[:, 0]], order=1, mode='nearest'))
                d_in = soft(cKDTree(uv).query(mask_pts)[0])
                return np.r_[d_out / np.sqrt(len(d_out)), d_in / np.sqrt(len(d_in)),
                             (x[:3] - prev[:3]) / 0.3, (x[3:] - prev[3:]) / 0.05]
            x = least_squares(res, x, max_nfev=200, diff_step=1e-4).x
        out[t] = rigid(x, item.centre)
        if progress:
            progress(t + 1, T)
        if log and (t % 24 == 0 or t == T - 1):
            log(f'item frame {t}: moved {np.linalg.norm(x[3:]) * 100:.1f} cm, turned {np.degrees(np.linalg.norm(x[:3])):.1f} deg')
    return out


def _distance(items, motions, t, points):
    """Distance of points to the nearest item in frame t."""
    return np.min([it.distance(points, M[t]) for it, M in zip(items, motions)], axis=0)


def track_hand(rig, items, motions, kp2d, kp3d, arm_masks, cam, log=print, progress=None, see_through=False):
    """Rig parameters per frame (T, P), from the known start pose through the extracted hands.
    items/motions: every item and its pose per frame (T, 4, 4); the hand enters none of them."""
    T = len(kp2d)
    C = np.array(cam['world'])
    R = C[:3, :3] / np.linalg.norm(C[:3, :3], axis=0)
    forward = -R[:, 2]
    rig_size = hand_size(rig.keypoints(np.zeros(rig.size)))
    place = PLACEMENT[rig.mode]
    step = np.r_[np.full(rig.na, place['step']), np.full(3, 0.15), np.full(rig.hand.size - 3, STEP_FINGER)]
    scale = np.r_[np.full(rig.na, 0.1 if rig.mode == 'control' else 1.0), np.ones(rig.hand.size)]  # natural size of each number
    q = DEPTH_MAP_SCALE
    params, prev, errs = np.zeros((T, rig.size)), np.zeros(rig.size), np.full(T, np.nan)
    for t in range(T):
        if progress:
            progress(t + 1, T)
        if np.isnan(kp2d[t]).all():
            params[t] = prev
            continue
        maps = [it.depth_map(M[t], cam, C[:3, 3], forward) for it, M in zip(items, motions)]
        depth = np.min(maps, axis=0) if maps else np.full(arm_masks[t][::q, ::q].shape, np.inf)  # no prop: nothing in front
        seen_arm = arm_masks[t][::q, ::q][:depth.shape[0], :depth.shape[1]]
        uv = np.nanmedian(to_hand21(kp2d[t]), axis=0)
        span = np.linalg.norm(uv[0] - uv[12])
        ex = to_hand21(kp3d[t])
        ext = (np.nanmedian(ex, axis=0) * rig_size / np.nanmedian([hand_size(p) for p in ex]) * [1, -1, -1]) @ R.T + C[:3, 3]
        target = directions(ext)

        def res(x):
            p = rig.keypoints(x)
            f = np.array([p[a] + s * (p[b] - p[a]) for a, b in FINGER_SEG for s in (0.25, 0.5, 0.75, 1.0)])
            palm = np.array([p[0] + s * (p[k] - p[0]) for k in (5, 9, 13, 17) for s in (0.33, 0.66)])
            contact = np.r_[np.maximum(FINGER_RADIUS - _distance(items, motions, t, f), 0),
                            np.maximum(PALM_RADIUS - _distance(items, motions, t, palm), 0)]
            pts = np.r_[p, f, palm]
            uq = project(pts, cam) / q
            ui = np.clip(np.round(uq[:, 0]).astype(int), 0, depth.shape[1] - 1)
            vi = np.clip(np.round(uq[:, 1]).astype(int), 0, depth.shape[0] - 1)
            seen = seen_arm[vi, ui] & (depth[vi, ui] < 100) & (not see_through)  # a see-through prop shows the hand behind it too
            front = np.where(seen, np.maximum((pts - C[:3, 3]) @ forward - depth[vi, ui] + FRONT_MARGIN, 0), 0)
            return np.r_[front / FRONT_MARGIN, (directions(p) - target).ravel() / DIR_SIGMA,
                         ((project(p, cam) - uv) / (KP2D_SIGMA * span)).ravel(), [(p[0] - ext[0]) @ forward / DEPTH_SIGMA],
                         contact / CONTACT_SIGMA, x[:rig.na] / place['prior'], (x - prev) / step, rig.wrist_penalty(x)]

        # two starts: the previous frame, and the previous frame re-aligned to this frame's hand
        # (placement and wrist only, items ignored); the better explanation of the frame wins
        k = rig.na + 3

        def align(v):
            y = prev.copy()
            y[:k] = v
            p = rig.keypoints(y)
            return np.r_[(directions(p) - target).ravel() / DIR_SIGMA, ((project(p, cam) - uv) / (KP2D_SIGMA * span)).ravel(),
                         [(p[0] - ext[0]) @ forward / DEPTH_SIGMA]]
        realigned = prev.copy()
        realigned[:k] = least_squares(align, np.clip(prev[:k], rig.lo[:k] + 1e-6, rig.hi[:k] - 1e-6),
                                      bounds=(rig.lo[:k], rig.hi[:k]), x_scale=scale[:k], max_nfev=100).x
        best = None
        for x0 in (prev, realigned):
            sol = least_squares(res, np.clip(x0, rig.lo + 1e-6, rig.hi - 1e-6), bounds=(rig.lo, rig.hi), max_nfev=200, x_scale=scale)
            if best is None or sol.cost < best.cost:
                best = sol
        params[t] = prev = best.x
        errs[t] = np.median(np.linalg.norm(project(rig.keypoints(best.x), cam) - uv, axis=1)) / span
        if log and (t % 24 == 0 or t == T - 1):
            log(f'{rig.hand.side.lower()} hand frame {t}: {errs[t] * 100:.1f}% of the hand off its video position')
    return params, errs


def _filter():
    return lambda v, cut, beta: one_euro(v, _speed(v, FPS), FPS, cut, beta)


def _turns(mats):
    """Rotations as small vectors around their average (no wrap-around near 180 degrees), for filtering.
    Returns (average, vectors); _mats() turns them back."""
    ref = Rotation.from_matrix(mats).mean()
    return ref, (ref.inv() * Rotation.from_matrix(mats)).as_rotvec()


def _mats(ref, vectors):
    return (ref * Rotation.from_rotvec(vectors)).as_matrix()


def smooth_items(items, motions, strength=0.7):
    """Each item as a rigid object: its centre and its turn filtered."""
    filt = _filter()
    _, beta_rot, beta_pos = settings(strength)
    out = []
    for item, M in zip(items, motions):
        centre = M[:, :3, :3] @ item.centre + M[:, :3, 3]
        ref, rv = _turns(M[:, :3, :3])
        Ms = np.repeat(np.eye(4)[None], len(M), 0)
        Ms[:, :3, :3] = _mats(ref, filt(rv, 1.0, beta_rot))
        Ms[:, :3, 3] = filt(centre, 1.0, beta_pos) - Ms[:, :3, :3] @ item.centre
        out.append(Ms)
    return out


def smooth_hand(rig, items, motions, smoothed, params, strength=0.7, log=print):
    """Smooth a hand; while it holds an item it rides on that item's smoothed path, its pose on the
    item held steady. A hand can hold different items in turn (a magazine, then the slide)."""
    filt = _filter()
    T, k = len(params), rig.na + 3
    min_cut, beta_rot, beta_pos = settings(strength)
    finger_cut = settings(strength * FINGER_SMOOTH)[0]  # fingers smoothed less: taps and clicks are quick, small moves
    xs = params.copy()
    groups = [(slice(0, rig.na), beta_pos if rig.mode == 'control' else beta_rot, min_cut), (slice(rig.na, k), beta_rot, min_cut)]
    groups += [(slice(k + 4 * i, k + 4 * i + 4), beta_rot, finger_cut) for i in range(len(rig.hand.fingers))]
    for sl, beta, cut in groups:
        xs[:, sl] = filt(params[:, sl], cut, beta)
    out = np.clip(xs, rig.lo, rig.hi)
    if not items:
        return out
    touch = np.zeros((len(items), T))
    for t in range(T):
        p = rig.keypoints(params[t])
        f = np.array([p[a] + s * (p[b] - p[a]) for a, b in FINGER_SEG for s in (0.5, 1.0)])
        for i, (it, M) in enumerate(zip(items, motions)):
            touch[i, t] = (it.distance(f, M[t]) < 0.012).sum()
    holding = np.where(touch.max(0) >= 10, touch.argmax(0), -1)
    held_frames = {}
    for i, (M, Ms) in enumerate(zip(motions, smoothed)):
        held = uniform_filter1d((holding == i).astype(float), 9)
        on = held > 0.05
        if on.sum() <= 4:
            continue
        held_frames[i] = int((holding == i).sum())
        rel = np.array([np.linalg.inv(M[t]) @ rig.hand_frame(params[t]) for t in range(T)])
        ref, rrv = _turns(rel[:, :3, :3])
        rc = rel[:, :3, 3].copy()
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for a0, a1 in zip(edges[::2], edges[1::2]):  # steady within each stretch of holding it
            if a1 - a0 > 4:
                rrv[a0:a1], rc[a0:a1] = filt(rrv[a0:a1], 0.4, 0.2), filt(rc[a0:a1], 0.4, 2.0)
        for t in np.flatnonzero(on):
            rel_t = np.eye(4)
            rel_t[:3, :3], rel_t[:3, 3] = _mats(ref, rrv[t]), rc[t]
            target = Ms[t] @ rel_t
            cur = rig.hand_frame(out[t])
            a = held[t]
            pos = cur[:3, 3] * (1 - a) + target[:3, 3] * a
            r0 = Rotation.from_matrix(cur[:3, :3])
            turn = r0 * Rotation.from_rotvec(a * (r0.inv() * Rotation.from_matrix(target[:3, :3])).as_rotvec())

            def res(v, t=t, pos=pos, turn=turn):
                y = out[t].copy()
                y[:k] = v
                Hf = rig.hand_frame(y)
                return np.r_[(Hf[:3, 3] - pos) / 0.002, (Rotation.from_matrix(Hf[:3, :3]) * turn.inv()).as_rotvec() / 0.02]
            out[t, :k] = least_squares(res, out[t, :k]).x
    if log:
        log(f'{rig.hand.side.lower()} hand holds: ' + (', '.join(f'item {i} on {n} frames' for i, n in held_frames.items()) or 'nothing'))
    return out


def combined_basis(rigs, params):
    """Bone basis per frame for the whole armature from each hand's parameters: (T, B, 4, 4)."""
    T = len(params[0])
    out = np.repeat(rigs[0].basis0[None], T, 0)
    for rig, P in zip(rigs, params):
        for t in range(T):
            b = rig.basis(P[t])
            changed = np.abs(b - rig.basis0).reshape(len(b), -1).max(1) > 1e-9
            out[t, changed] = b[changed]
    return out


def jitter(rig, params):
    """Mean frame-to-frame acceleration of the hand's 21 points, mm per frame squared."""
    p = np.array([rig.keypoints(x) for x in params])
    return float(np.linalg.norm(np.diff(p, 2, axis=0), axis=2).mean() * 1000)


# The main prop ---------------------------------------------------------------------------------------
# One prop is tracked: the one a hand holds in the start frame (or the one named). While that hand holds it,
# the hand decides where it is: its outline is ambiguous exactly when it is held (hidden, edge-on, seen
# from behind), and the hand is well tracked. Everything else in the scene is left as the user placed it.

SURFACE = 0.006        # m: a finger's or the palm's half-thickness; centre lines stay this far outside surfaces
SOCKET_TOUCH = 8       # finger points within 15 mm of the prop in the start frame: that hand holds it
SOCKET_FINGERS = 0.8   # while holding, the fingers lean this much toward their start-frame grip
EASE = 6               # frames to ease the hand onto / off the prop
POP = (4.0, 0.025, 6)  # a step over 4x the usual and 25 mm is a pop; it is eased over 6 frames each side


def observation(rig, kp2d, kp3d, cam):
    """One frame's extracted hand: screen points, their size, and bone directions in the world."""
    C = np.array(cam['world'])
    R = C[:3, :3] / np.linalg.norm(C[:3, :3], axis=0)
    uv = np.nanmedian(to_hand21(kp2d), axis=0)
    span = np.linalg.norm(uv[0] - uv[12])
    ex = to_hand21(kp3d)
    rig_size = hand_size(rig.keypoints(np.zeros(rig.size)))
    ext = (np.nanmedian(ex, axis=0) * rig_size / np.nanmedian([hand_size(p) for p in ex]) * [1, -1, -1]) @ R.T + C[:3, 3]
    return uv, span, directions(ext)


def _surface_points(p):
    f = np.array([p[a] + s * (p[b] - p[a]) for a, b in FINGER_SEG for s in (0.25, 0.5, 0.75, 1.0)])
    palm = np.array([p[0] + s * (p[k] - p[0]) for k in (5, 9, 13, 17) for s in (0.33, 0.66)])
    return np.r_[f, palm]


def _touching(rig, item, x=None):
    p = rig.keypoints(np.zeros(rig.size) if x is None else x)
    f = np.array([p[a] + s * (p[b] - p[a]) for a, b in FINGER_SEG for s in (0.5, 1.0)])
    return int((item.distance(f, np.eye(4)) < 0.015).sum())


def main_prop(rigs, items, names, wanted=''):
    """Index of the main prop: the one named, else the one a hand holds most firmly in the start frame."""
    if wanted:
        if wanted not in names:
            raise ValueError(f'no prop named "{wanted}"; the scene has: {", ".join(names)}')
        return names.index(wanted)
    touch = [max(_touching(r, it) for r in rigs) for it in items]
    if not touch or max(touch) < SOCKET_TOUCH:
        return None
    return int(np.argmax(touch))


def socket(rigs, prop):
    """(hand, grip, fingers): the hand holding the prop in the start frame, its frame in the prop's frame,
    and its finger values; None when no hand holds it there."""
    touch = [_touching(r, prop) for r in rigs]
    h = int(np.argmax(touch))
    if touch[h] < SOCKET_TOUCH:
        return None
    x0 = np.zeros(rigs[h].size)
    return h, rigs[h].hand_frame(x0), x0[rigs[h].na + 3:].copy()


def prop_from_hand(rig, params, grip):
    """The prop's pose per frame while the hand holds it: where the hand's grip puts it."""
    return np.array([rig.hand_frame(x) @ np.linalg.inv(grip) for x in params])


def _ease(a, b, T):
    """Weight over frames a..b: eased in and out, except at the clip's ends."""
    w = np.ones(b - a)
    for k in range(min(EASE, b - a)):
        f = (k + 1) / (EASE + 1)
        f = f * f * (3 - 2 * f)
        if a > 0:
            w[k] = min(w[k], f)
        if b < T:
            w[b - a - 1 - k] = min(w[b - a - 1 - k], f)
    return w


def _blend(A, B, w):
    """Pose between A (w = 0) and B (w = 1)."""
    out = np.eye(4)
    ra = Rotation.from_matrix(A[:3, :3])
    out[:3, :3] = (ra * Rotation.from_rotvec(w * (ra.inv() * Rotation.from_matrix(B[:3, :3])).as_rotvec())).as_matrix()
    out[:3, 3] = A[:3, 3] * (1 - w) + B[:3, 3] * w
    return out


def _place_hand(rig, x, target):
    """Hand placement and wrist that put the hand's frame at target; the fingers stay as they are."""
    k = rig.na + 3

    def res(v):
        y = x.copy()
        y[:k] = v
        Hf = rig.hand_frame(y)
        return np.r_[(Hf[:3, 3] - target[:3, 3]) / 0.002,
                     (Rotation.from_matrix(Hf[:3, :3]) * Rotation.from_matrix(target[:3, :3]).inv()).as_rotvec() / 0.02]
    y = x.copy()
    y[:k] = least_squares(res, x[:k]).x
    return y


def hold_prop(rig, params, prop_motion, grip, fingers):
    """The holding hand placed exactly on the (smoothed) prop, its fingers leaning to their start grip."""
    P = params.copy()
    k = rig.na + 3
    for t in range(len(P)):
        P[t, k:] = P[t, k:] + SOCKET_FINGERS * (fingers - P[t, k:])
        P[t] = _place_hand(rig, P[t], prop_motion[t] @ grip)
    return P


def settle_contacts(rigs, prop, motion, params, arm_masks, cam, log=print):
    """Last pass, per frame and hand, with the smallest changes (fingers mostly): no finger or palm centre
    line inside the prop or inside the other hand."""
    T = len(params[0])
    params = [P.copy() for P in params]
    changed = [0] * len(rigs)
    for h, rig in enumerate(rigs):
        P = params[h]
        give = np.r_[np.full(rig.na, 0.004 if rig.mode == 'control' else 0.04), np.full(3, 0.04), np.full(rig.hand.size - 3, 0.2)]
        for t in range(T):
            others = [_surface_points(r.keypoints(Q[t])) for o, (r, Q) in enumerate(zip(rigs, params)) if o != h]
            tree = cKDTree(np.concatenate(others)) if others else None
            x0, prev = P[t].copy(), (P[t - 1] if t > 0 else None)

            def res(x):
                surf = _surface_points(rig.keypoints(x))
                r = [(x - x0) / give]
                if prev is not None:  # no flicker: close to the previous settled frame too
                    r.append((x - prev) / (give * 0.5))
                if prop is not None:
                    r.append(np.maximum(SURFACE - prop.distance(surf, motion[t]), 0) / 0.001)
                if tree is not None:
                    r.append(np.maximum(2 * SURFACE - tree.query(surf)[0], 0) / 0.001)
                return np.concatenate(r)
            r0 = res(x0)
            if np.abs(r0[len(x0) * (1 if prev is None else 2):]).max() < 0.5:
                continue
            P[t] = least_squares(res, np.clip(x0, rig.lo + 1e-6, rig.hi - 1e-6), bounds=(rig.lo, rig.hi), max_nfev=100).x
            changed[h] += 1
    if log:
        log('contacts settled on ' + ', '.join(f'{n} frames ({r.hand.side.lower()} hand)' for r, n in zip(rigs, changed)))
    return params


def _ease_mats(mats, a, b):
    key = Rotation.from_matrix(mats[[a, b], :3, :3])
    for t in range(a + 1, b):
        f = (t - a) / (b - a)
        f = f * f * (3 - 2 * f)
        mats[t, :3, :3] = Slerp([0, 1], key)([f]).as_matrix()[0]
        mats[t, :3, 3] = mats[a, :3, 3] * (1 - f) + mats[b, :3, 3] * f


def _pops(path):
    factor, floor, pad = POP
    step = np.linalg.norm(np.diff(path, axis=0), axis=1)
    out = []
    for t in np.flatnonzero(step > max(factor * np.median(step), floor)):
        a, b = max(t + 1 - pad, 0), min(t + 1 + pad, len(path) - 1)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def ease_pops(rigs, basis, prop=None, motion=None):
    """Where a hand (or the prop) moves far more in one frame than it usually does, replace the motion
    around it with an eased transition between the frames either side. basis: (T, B, 4, 4)."""
    basis = basis.copy()
    motion = None if motion is None else motion.copy()
    T = len(basis)
    eased = []
    for rig in rigs:
        wrist = np.array([rig.sk.pose(basis[t])[rig.sk.index[rig.hand.hand], :3, 3] for t in range(T)])
        bones = [rig.sk.index[n] for n in [rig.hand.hand] + [n for j in rig.hand.fingers.values() for n in j]]
        if rig.mode == 'control':
            bones.append(rig.ctrl)
        for a, b in _pops(wrist):
            for i in bones:
                _ease_mats(basis[:, i], a, b)
            eased.append(f'{rig.hand.side.lower()} hand {a}-{b}')
    if motion is not None:
        for a, b in _pops(motion[:, :3, :3] @ prop.centre + motion[:, :3, 3]):
            _ease_mats(motion, a, b)
            eased.append(f'prop {a}-{b}')
    return basis, motion, eased


def contact_report(rigs, prop, motion, params):
    """Per hand: frames with a finger or palm centre line more than 2 mm inside the prop."""
    out = []
    for rig, P in zip(rigs, params):
        out.append([t for t in range(len(P)) if prop is not None and
                    (prop.distance(np.r_[rig.keypoints(P[t]), _surface_points(rig.keypoints(P[t]))], motion[t]) < -0.002).any()])
    return out


# Grip takes ----------------------------------------------------------------------------------------------
# A short take where the hands reach for a floating prop and hold it: its settled end becomes the start pose
# (grip) for later animations. The video says where each hand is; the fingers that are near the prop then
# close onto its surface, since the wrap mostly happens on the side the camera cannot see.

WRAP_REACH = 0.07      # m: fingers whose base is this close to the prop wrap onto it


def wrap_fingers(rig, x, prop, M):
    """Close the fingers near the prop until their middle and tip segments rest on its surface: nothing
    inside it, middle and tip joints bending together, no sideways spread. Placement and wrist unchanged."""
    k = rig.na + 3
    segs = {f: [(1 + 4 * j + a, 2 + 4 * j + a) for a in range(3)] for j, f in enumerate(hands.FINGERS) if f in rig.hand.fingers}
    p0 = rig.keypoints(x)
    wrap = [f for f in rig.hand.fingers if prop.distance(p0[[segs[f][0][0]]], M)[0] < WRAP_REACH]

    def res(v):
        y = x.copy()
        y[k:] = v
        p = rig.keypoints(y)
        r = [np.maximum(SURFACE - prop.distance(_surface_points(p), M), 0) / 0.001]
        for i, f in enumerate(rig.hand.fingers):
            o = k + 4 * i
            if f in wrap:
                ends = np.array([p[a] + s * (p[b] - p[a]) for a, b in segs[f][1:] for s in (0.5, 1.0)])
                r.append((prop.distance(ends, M) - SURFACE) / 0.003)
            else:
                r.append((y[o:o + 4] - x[o:o + 4]) / 0.1)
            r.append([(y[o + 3] - 0.7 * y[o + 2]) / 0.2, y[o + 1] / 0.3])
        return np.concatenate([np.atleast_1d(q) for q in r])
    v = least_squares(res, np.clip(x[k:], rig.lo[k:] + 1e-6, rig.hi[k:] - 1e-6), bounds=(rig.lo[k:], rig.hi[k:])).x
    y = x.copy()
    y[k:] = v
    return y, wrap


PALM_REST = 0.012      # m: the palm's centre line sits this far off the prop's surface when it holds it


def rest_palm(rig, x, prop, M, cam):
    """Slide the hand (placement and a little wrist turn) so its palm rests on the prop while it stays where it is
    on screen: depth is what a single video measures worst."""
    k = rig.na + 3
    p0 = rig.keypoints(x)
    uv0 = project(p0, cam)
    span = np.linalg.norm(uv0[0] - uv0[12])
    palm_idx = [0, 5, 9, 13, 17]

    def res(v):
        y = x.copy()
        y[:k] = v
        p = rig.keypoints(y)
        palm = np.array([p[0] + s * (p[j] - p[0]) for j in (5, 9, 13, 17) for s in (0.5, 1.0)])
        d = prop.distance(palm, M)
        return np.r_[((project(p, cam)[palm_idx] - uv0[palm_idx]) / (0.02 * span)).ravel(),
                     (np.min(d) - PALM_REST) / 0.003, np.maximum(SURFACE - prop.distance(_surface_points(p), M), 0) / 0.001,
                     (v[rig.na:] - x[rig.na:k]) / 0.3]
    y = x.copy()
    y[:k] = least_squares(res, np.clip(x[:k], rig.lo[:k] + 1e-6, rig.hi[:k] - 1e-6), bounds=(rig.lo[:k], rig.hi[:k])).x
    return y


def settled_grip(rigs, params, prop, motion, frames=8, cam=None):
    """The grip at the end of a grip take: each hand's and the prop's pose averaged over the last frames,
    then the fingers near the prop wrapped onto it. Returns (basis (B, 4, 4), prop pose (4, 4), notes)."""
    M = np.eye(4)
    ref, rv = _turns(motion[-frames:, :3, :3])
    M[:3, :3] = _mats(ref, rv.mean(0))
    M[:3, 3] = motion[-frames:, :3, 3].mean(0)
    xs, notes = [], []
    for rig, P in zip(rigs, params):
        x = np.clip(P[-frames:].mean(0), rig.lo, rig.hi)
        if cam is not None:
            before = rig.hand_frame(x)[:3, 3]
            x = rest_palm(rig, x, prop, M, cam)
            notes.append(f'{rig.hand.side.lower()} hand moved {np.linalg.norm(rig.hand_frame(x)[:3, 3] - before) * 100:.1f} cm to rest on the prop')
        x, wrapped = wrap_fingers(rig, x, prop, M)
        xs.append(x)
        notes.append(f'{rig.hand.side.lower()} hand: {", ".join(wrapped) or "no fingers"} wrapped onto the prop')
    return combined_basis(rigs, [x[None] for x in xs])[0], M, notes
