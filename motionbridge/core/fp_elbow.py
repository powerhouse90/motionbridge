"""Elbow swing from the video. With the hand tracked, the arm chain (shoulder -> elbow -> wrist) still has one
free turn: the elbow can swing about the shoulder-wrist line. Blender's IK keeps it in its start plane; here each
frame's swing is chosen so the upper arm and forearm, drawn as tapered capsules, cover the video's arm mask (SAM 2)
outside the hand. Capsule thickness is one number per arm, calibrated on the frames showing the most arm (the
start frame is no use for it: there the arms are mostly hand and off-screen). Frames showing little arm outside
the hand count for less, so the smoothness and the start-plane prior carry them.
A Viterbi pass over a grid of swings gives the best path with a smoothness cost and joint limits; the result
composes with the forearm roll (same output as fp_twist.forearm_roll)."""
import numpy as np
from scipy.spatial.transform import Rotation
from motionbridge.core import fp_twist
from motionbridge.core.hand_fit import project
from motionbridge.core.smooth import one_euro

LIMIT = 80.0       # deg: the elbow swings at most this far from its start plane either way
STEP = 4.0         # deg: grid of swings tried per frame
SMOOTH = 0.02      # cost per (deg/frame)^2 / 100 of swing change between frames (1 - IoU units)
PRIOR = 0.05       # cost at the limit, quadratic: an arm the mask can't decide stays near its start plane
SCALE = 4          # masks are compared at 1/SCALE resolution
NEAR = 0.04        # m: the camera's near plane; a shoulder behind the camera is clipped there
HAND_PAD = 0.35    # the hand's region (excluded) is its keypoint box grown by this fraction of its size
THICK = np.linspace(0.06, 0.40, 18)  # capsule radius / forearm length, calibrated per arm
CALIB = 10         # frames used for the thickness calibration (the ones showing the most arm)
CUTOFF = 2.0       # Hz: the grid path is low-passed (zero phase) so the swing doesn't step by STEP
SEEN = 200         # mask pixels (at 1/SCALE) outside the hand for a frame to count fully
TAPER = (1.25, 1.1, 0.75)  # radius at shoulder, elbow, wrist relative to the calibrated one


def chain(rig, pose, phi, off=None):
    """(shoulder, elbow, wrist) for one frame's posed armature and a swing phi (deg) about the shoulder-wrist
    line, measured from the start plane (phi = 0 is Blender's IK result)."""
    fa = rig.forearm
    wr = pose[rig.sk.index[rig.hand.hand]][:3, 3]
    _, s, e0 = fp_twist.reach(rig, wr, off)  # the shoulder comes forward when the hand is out of reach
    dd = np.clip(np.linalg.norm(wr - s), 1e-6, fa['L1'] + fa['L2'] - 1e-4)
    u = (wr - s) / np.linalg.norm(wr - s)
    a = (fa['L1'] ** 2 - fa['L2'] ** 2 + dd ** 2) / (2 * dd)
    h = np.sqrt(max(fa['L1'] ** 2 - a ** 2, 0))
    v = (e0 - s) - ((e0 - s) @ u) * u
    v /= np.linalg.norm(v) + 1e-9
    v = Rotation.from_rotvec(u * np.radians(phi)).apply(v)
    return s, s + a * u + h * v, wr


def _cam_space(points, cam):
    m = np.linalg.inv(np.array(cam['world']))
    return points @ m[:3, :3].T + m[:3, 3]


def _segment(a, b, ra, rb, cam):
    """A 3D segment with end radii, clipped to the near plane -> 2D ends and pixel radii (or None)."""
    ca, cb = _cam_space(np.array([a, b]), cam)
    za, zb = -ca[2], -cb[2]
    if za < NEAR and zb < NEAR:
        return None
    if za < NEAR or zb < NEAR:  # move the end behind the near plane onto it
        t = (NEAR - za) / (zb - za)
        p, r = a + t * (b - a), ra + t * (rb - ra)
        a, ra = (p, r) if za < NEAR else (a, ra)
        b, rb = (p, r) if zb < NEAR else (b, rb)
        za, zb = max(za, NEAR), max(zb, NEAR)
    f = cam['lens'] / cam['sensor'] * cam['W']
    q = project(np.array([a, b]), cam)
    return q, np.array([ra * f / za, rb * f / zb])


def capsules(segs, cam, shape):
    """Boolean raster (H/SCALE, W/SCALE) of tapered capsules [(a, b, ra, rb)] seen through cam."""
    H, W = shape
    ys, xs = np.mgrid[0:H, 0:W]
    px = np.stack([xs, ys], -1).reshape(-1, 2) * SCALE + SCALE / 2
    out = np.zeros(len(px), bool)
    for a, b, ra, rb in segs:
        s = _segment(a, b, ra, rb, cam)
        if s is None:
            continue
        (qa, qb), (pa, pb) = s
        d = qb - qa
        t = np.clip(((px - qa) @ d) / max(d @ d, 1e-9), 0, 1)
        out |= np.linalg.norm(px - (qa + t[:, None] * d), axis=1) <= pa + t * (pb - pa)
    return out.reshape(H, W)


def _down(mask):
    H, W = mask.shape
    return mask[:H - H % SCALE, :W - W % SCALE].reshape(H // SCALE, SCALE, W // SCALE, SCALE).mean((1, 3)) > 0.5


def hand_zone(rig, pose, cam, shape):
    """The hand's screen region (its keypoint box, grown), at mask resolution: left out of the comparison,
    the hand is already tracked and its shape isn't a capsule."""
    p = project(np.array([rig.sk.point(pose, *kb) for kb in rig.hand.keypoint_bones()]), cam) / SCALE
    lo, hi = p.min(0), p.max(0)
    pad = (hi - lo) * HAND_PAD
    lo, hi = np.floor(lo - pad).astype(int), np.ceil(hi + pad).astype(int)
    z = np.zeros(shape, bool)
    z[max(lo[1], 0):max(hi[1], 0), max(lo[0], 0):max(hi[0], 0)] = True
    return z


def _arm_segs(rig, pose, phi, k, off=None):
    s, e, w = chain(rig, pose, phi, off)
    r = k * rig.forearm['L2']
    return [(s, e, r * TAPER[0], r * TAPER[1]), (e, w, r * TAPER[1], r * TAPER[2])]


STRAIGHT = 0.25        # cost per radian the wrist bends away from how it sat on the forearm at the start: a real
                       # wrist stays fairly straight, so where the mask shows little arm the arm lines up with the hand


def bend(rig, pose, phi, off=None):
    """How far (rad) the wrist bends, the forearm from the shoulder solve against the hand, relative to the start pose."""
    fa = rig.forearm
    s, e, w = chain(rig, pose, phi, off)
    f = (w - e) / np.linalg.norm(w - e)
    h = pose[rig.sk.index[rig.hand.hand]][:3, 1]
    h = h / np.linalg.norm(h)
    if 'bend0' not in fa:
        h0 = rig.pose0[rig.hand.hand][:3, 1] / np.linalg.norm(rig.pose0[rig.hand.hand][:3, 1])
        f0 = rig.pose0[rig.hand.hand][:3, 3] - fa['elbow0']
        fa['bend0'] = float(np.arccos(np.clip(h0 @ (f0 / np.linalg.norm(f0)), -1, 1)))
    return abs(float(np.arccos(np.clip(f @ h, -1, 1))) - fa['bend0'])


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def fit_elbows(rigs, basis, masks, cam, fps=24, log=print, offset=None):
    """rigs: fp_track.Rig per hand; basis: (T, B, 4, 4); masks: {side: (T, H, W) bool arm masks}.
    offset: {side: (T, 3)} shoulder offsets (fit_shoulders), applied before the swing.
    Returns ({side: swing per frame, deg}, report {side: dict(thick, iou_plane, iou_fit)})."""
    T = len(basis)
    grid = np.arange(-LIMIT, LIMIT + 1e-6, STEP)
    poses = [rigs[0].sk.pose(b) for b in basis]
    angles, report = {}, {}
    for rig in rigs:
        side = rig.hand.side
        if rig.forearm is None or side not in masks:
            continue
        m = [_down(masks[side][t]) for t in range(T)]
        shape = m[0].shape
        keep = [~hand_zone(rig, poses[t], cam, shape) for t in range(T)]
        off = offset[side] if offset and side in offset else np.zeros((T, 3))
        score = lambda t, phi, k: iou(capsules(_arm_segs(rig, poses[t], phi, k, off[t]), cam, shape) & keep[t], m[t] & keep[t])
        seen = np.array([(m[t] & keep[t]).sum() for t in range(T)])
        calib = [t for t in np.argsort(-seen)[:CALIB] if seen[t] > SEEN / 4]
        coarse = grid[::2]
        k = max(THICK, key=lambda k: np.mean([max(score(t, phi, k) for phi in coarse) for t in calib])) if calib else THICK[len(THICK) // 2]
        w = np.minimum(seen / SEEN, 1.0)[:, None]
        unary = w * np.array([[1 - score(t, phi, k) for phi in grid] for t in range(T)]) + PRIOR * (grid / LIMIT) ** 2
        unary += STRAIGHT * np.array([[bend(rig, poses[t], phi, off[t]) for phi in grid] for t in range(T)])
        # Viterbi: best swing path, change between frames costs SMOOTH per (deg/frame)^2 / 100
        jump = SMOOTH * ((grid[:, None] - grid[None, :]) / 10) ** 2
        cost, back = unary[0].copy(), np.zeros((T, len(grid)), int)
        for t in range(1, T):
            tot = cost[None, :] + jump  # [to, from]
            back[t] = tot.argmin(1)
            cost = tot.min(1) + unary[t]
        path = np.zeros(T, int)
        path[-1] = cost.argmin()
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        angles[side] = one_euro(grid[path][:, None].astype(float), np.zeros(T), fps, CUTOFF, 0)[:, 0] if T > 2 else grid[path].astype(float)
        zero = int(np.argmin(np.abs(grid)))
        report[side] = dict(thick=k, seen=seen, iou_plane=np.array([score(t, 0.0, k) for t in range(T)]),
                            iou_fit=np.array([score(t, angles[side][t], k) for t in range(T)]))
        if log:
            r = report[side]
            vis = seen > SEEN
            log(f'{side.lower()} elbow: thickness {k:.2f} x forearm, IoU outside the hand on the {vis.sum()} frames showing '
                f'arm {np.mean(r["iou_plane"][vis]):.2f} -> {np.mean(r["iou_fit"][vis]):.2f} (mean), swing median {np.median(np.abs(angles[side])):.0f} deg, '
                f'max {np.abs(angles[side]).max():.0f} deg')
    return angles, report


def arm_world(rigs, basis, angles, log=print, offset=None):
    """fp_twist.forearm_roll with each frame's elbow swing and shoulder offset: (arm_names, arm_world (T, n, 4, 4), rolls)."""
    return fp_twist.forearm_roll(rigs, basis, log=log, swing=angles, offset=offset)


SHOULDER_RANGE = 0.15   # m: the shoulder may move this far (each camera axis) to fit the arm to its mask
SHOULDER_STEP = 0.075   # m: coarse grid of shoulder moves tried, then refined around the pick at a third of it
SHOULDER_EVERY = 8      # the shoulder is searched on every Nth frame, interpolated between (its path is low-passed
                        # at SHOULDER_CUTOFF anyway)
SHOULDER_SLACK = 0.03   # IoU: among fits this close to the frame's best, the one moving least is kept
SHOULDER_THICK = 0.20   # capsule radius / forearm length while searching (calibrated later, at the found shoulder)
SHOULDER_CUTOFF = 1.0   # Hz: the shoulder path is low-passed (zero phase): a shoulder moves slowly


def fit_shoulders(rigs, basis, masks, cam, fps=24, log=print):
    """The shoulder's place per frame, from the arm masks. A rig's shoulder is fixed to the camera, but a video's
    arm comes from wherever the model drew it (an FPS viewmodel pushed forward and down when aiming): with the
    shoulder fixed the arm can't bend to match, and a hand beyond reach stretches it. The shoulder is off-screen,
    so the mask fixes only the direction the arm comes from: of the moves that fit about as well as the best,
    the one closest to the last frame's and to none is kept. Frames showing little arm keep the last move.
    Returns {side: (T, 3) world offsets}."""
    T = len(basis)
    C = np.array(cam['world'])
    Rc = C[:3, :3] / np.linalg.norm(C[:3, :3], axis=0)
    grid = lambda r, st: np.array([Rc @ np.array([x, y, z]) for x in np.arange(-r, r + 1e-9, st)
                                   for y in np.arange(-r, r + 1e-9, st) for z in np.arange(-r, r + 1e-9, st)])
    coarse, fine = grid(SHOULDER_RANGE, SHOULDER_STEP), grid(SHOULDER_STEP / 3, SHOULDER_STEP / 3)
    phis_coarse, phis = np.arange(-LIMIT, LIMIT + 1e-6, 40.0), np.arange(-LIMIT, LIMIT + 1e-6, 20.0)
    poses = [rigs[0].sk.pose(b) for b in basis]
    keys = sorted(set(range(0, T, SHOULDER_EVERY)) | {T - 1})
    out = {}
    for rig in rigs:
        side, fa = rig.hand.side, rig.forearm
        if fa is None or side not in masks:
            continue
        L = (fa['L1'] + fa['L2']) * fp_twist.REACH
        prev, at = np.zeros(3), []
        for t in keys:
            m = _down(masks[side][t])
            keep = ~hand_zone(rig, poses[t], cam, m.shape)
            target = m & keep
            seen = min(target.sum() / SEEN, 1.0)  # how much the mask can say
            wr = poses[t][rig.sk.index[rig.hand.hand]][:3, 3]

            def pick(moves, phis, best=None):
                scores = np.full(len(moves), -1.0)
                for i, d in enumerate(moves):
                    if np.linalg.norm(wr - fa['shoulder'] - d) > L:
                        continue  # the hand must stay in reach
                    scores[i] = max(seen * iou(capsules(_arm_segs(rig, poses[t], phi, SHOULDER_THICK, d), cam, m.shape) & keep, target)
                                    - STRAIGHT * bend(rig, poses[t], phi, d) for phi in phis)
                top = max(scores.max(), best or -1.0)
                good = scores >= top - SHOULDER_SLACK
                cost = np.linalg.norm(moves - prev, axis=1) + 0.5 * np.linalg.norm(moves, axis=1)
                return moves[np.where(good, cost, np.inf).argmin()] if good.any() else None, scores.max()

            near, top = pick(coarse, phis_coarse)  # coarse: where the arm comes from
            if near is not None:
                better, _ = pick(near + fine, phis, top)  # fine, around it
                prev = better if better is not None else near
            at.append(prev)
        at = np.array(at)
        path = np.stack([np.interp(np.arange(T), keys, at[:, j]) for j in range(3)], 1)
        out[side] = one_euro(path, np.zeros(T), fps, SHOULDER_CUTOFF, 0) if T > 2 else path
        if log:
            n = np.linalg.norm(out[side], axis=1)
            log(f'{side.lower()} shoulder: moved median {np.median(n) * 100:.0f} cm, max {n.max() * 100:.0f} cm (fitted to its mask)')
    return out