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
from motionbridge.core.fp_twist import _aim
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


def chain(rig, pose, phi):
    """(shoulder, elbow, wrist) for one frame's posed armature and a swing phi (deg) about the shoulder-wrist
    line, measured from the start plane (phi = 0 is Blender's IK result)."""
    fa = rig.forearm
    s = fa['shoulder']
    wr = pose[rig.sk.index[rig.hand.hand]][:3, 3]
    dd = np.clip(np.linalg.norm(wr - s), 1e-6, fa['L1'] + fa['L2'] - 1e-4)
    u = (wr - s) / np.linalg.norm(wr - s)
    a = (fa['L1'] ** 2 - fa['L2'] ** 2 + dd ** 2) / (2 * dd)
    h = np.sqrt(max(fa['L1'] ** 2 - a ** 2, 0))
    v = (fa['elbow0'] - s) - ((fa['elbow0'] - s) @ u) * u
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


def _arm_segs(rig, pose, phi, k):
    s, e, w = chain(rig, pose, phi)
    r = k * rig.forearm['L2']
    return [(s, e, r * TAPER[0], r * TAPER[1]), (e, w, r * TAPER[1], r * TAPER[2])]


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def fit_elbows(rigs, basis, masks, cam, fps=24, log=print):
    """rigs: fp_track.Rig per hand; basis: (T, B, 4, 4); masks: {side: (T, H, W) bool arm masks}.
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
        score = lambda t, phi, k: iou(capsules(_arm_segs(rig, poses[t], phi, k), cam, shape) & keep[t], m[t] & keep[t])
        seen = np.array([(m[t] & keep[t]).sum() for t in range(T)])
        calib = [t for t in np.argsort(-seen)[:CALIB] if seen[t] > SEEN / 4]
        coarse = grid[::2]
        k = max(THICK, key=lambda k: np.mean([max(score(t, phi, k) for phi in coarse) for t in calib])) if calib else THICK[len(THICK) // 2]
        w = np.minimum(seen / SEEN, 1.0)[:, None]
        unary = w * np.array([[1 - score(t, phi, k) for phi in grid] for t in range(T)]) + PRIOR * (grid / LIMIT) ** 2
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


def arm_world(rigs, basis, angles, log=print):
    """fp_twist.forearm_roll with each frame's elbow swing: (arm_names, arm_world (T, 2 per arm, 4, 4), rolls)."""
    T = len(basis)
    names, worlds, rolls = [], [], {}
    for rig in rigs:
        fa = rig.forearm
        if fa is None:
            continue
        sk = rig.sk
        upper = rig.dump['bones'][sk.index[fa['name']]]['parent']
        P0u, P0f = rig.pose0[upper], rig.pose0[fa['name']]
        hand = sk.index[rig.hand.hand]
        phis = angles.get(rig.hand.side, np.zeros(T))
        Wu, Wf, roll = np.zeros((T, 4, 4)), np.zeros((T, 4, 4)), np.zeros(T)
        for t in range(T):
            pose = sk.pose(basis[t])
            Hm = pose[hand]
            Hr = Hm[:3, :3] / np.linalg.norm(Hm[:3, :3], axis=0)
            s, e, wr = chain(rig, pose, phis[t])
            Wu[t], Wf[t] = _aim(P0u, s, e), _aim(P0f, e, wr)
            Ff = Wf[t][:3, :3] / np.linalg.norm(Wf[t][:3, :3], axis=0)
            q = Rotation.from_matrix(Ff.T @ Hr @ fa['rel0'].T).as_quat()
            twist = (2 * np.arctan2(q[1], q[3]) + np.pi) % (2 * np.pi) - np.pi
            Wf[t][:3, :3] = Rotation.from_rotvec(Ff[:, 1] * twist).as_matrix() @ Wf[t][:3, :3]
            roll[t] = np.degrees(twist)
        names += [upper, fa['name']]
        worlds += [Wu, Wf]
        rolls[rig.hand.side] = roll
        if log:
            log(f'{rig.hand.side.lower()} forearm roll (with elbow swing): median {np.median(np.abs(roll)):.0f} deg, '
                f'max {np.abs(roll).max():.0f} deg')
    return np.array(names), (np.stack(worlds, 1) if worlds else np.zeros((T, 0, 4, 4))), rolls
