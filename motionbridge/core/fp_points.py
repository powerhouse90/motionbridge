"""The main prop placed from point matches: distinctive points on the prop in frame 0, whose 3D position on the
model is known (frame 0 is our render of the scene), are followed through the video with optical flow, and each
frame the prop's pose is solved from those 2D-3D matches (PnP with outlier rejection). Its outline alone is
ambiguous exactly when it is held (hidden, edge-on); its texture is not.

Any rigid prop: give its triangles at the scene placement (world), or none and its Item's signed distance is
ray-marched instead. Poses are relative to the scene placement, as everywhere in fp_track: a point X on the
placed prop is at motion[t] @ X in frame t (motion[0] is the identity).
"""
import cv2
import numpy as np
from scipy import ndimage

FEATURES = 600         # points picked on the prop in frame 0
REFILL = 80            # fewer points than this still followed: top up with fresh ones
REFILL_FEATURES = 300
MIN_GAP = 6            # px between picked points
FB_ERROR = 1.5         # px: a point that does not come back to where it was (forward-backward flow) is lost
ARM_MARGIN = 8         # px: points this close to an arm are not picked (the arm may slide over them)
RANSAC_PX = 3.0        # px: a match further than this off the solved pose is an outlier and is dropped
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def intrinsics(cam):
    """Pinhole matrix for the camera dict (as hand_fit.project: horizontal sensor fit, aspect for uneven resizes)."""
    W, H = cam['W'], cam['H']
    f = cam['lens'] / cam['sensor'] * W
    return np.array([[f, 0, W / 2], [0, f * cam.get('aspect', 1.0), H / 2], [0, 0, 1]])


def _rays(px, cam):
    """World-space unit directions through pixels (Blender camera: looks down -z, y up)."""
    K = intrinsics(cam)
    C = np.array(cam['world'])
    R = C[:3, :3] / np.linalg.norm(C[:3, :3], axis=0)
    d = np.c_[(px[:, 0] - K[0, 2]) / K[0, 0], -(px[:, 1] - K[1, 2]) / K[1, 1], -np.ones(len(px))] @ R.T
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def raycast_triangles(tris, origin, dirs, chunk=4_000_000):
    """First hit per ray from one origin on triangles (N, 3, 3); NaN where none (Moller-Trumbore, vectorised)."""
    A, E1, E2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    tv = origin - A
    qv = np.cross(tv, E1)
    out = np.full((len(dirs), 3), np.nan)
    step = max(1, chunk // max(len(tris), 1))
    for a in range(0, len(dirs), step):
        d = dirs[a:a + step]
        pv = np.cross(d[:, None], E2[None])                     # (r, N, 3)
        det = (E1[None] * pv).sum(2)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
        u = (tv[None] * pv).sum(2) * inv
        v = (d[:, None] * qv[None]).sum(2) * inv
        t = (qv * E2).sum(1)[None] * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        t = np.where(hit, t, np.inf).min(1)
        k = np.isfinite(t)
        out[a:a + step][k] = origin + d[k] * t[k, None]
    return out


def raycast_item(item, origin, dirs, far=5.0, eps=5e-4, steps=300):
    """First hit per ray on the item's (sealed) signed distance at its scene placement, by sphere tracing."""
    t = np.zeros(len(dirs))
    hit = np.zeros(len(dirs), bool)
    live = np.ones(len(dirs), bool)
    for _ in range(steps):
        if not live.any():
            break
        d = item.distance(origin + dirs[live] * t[live, None], np.eye(4))
        idx = np.flatnonzero(live)
        hit[idx[d < eps]] = True
        t[idx] += np.maximum(d, eps)
        live &= ~hit & (t < far)
    out = np.full((len(dirs), 3), np.nan)
    out[hit] = origin + dirs[hit] * t[hit, None]
    return out


def _gray(f):
    return f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def track_points(frames, prop_masks, arm_masks, cam, tris=None, item=None, log=print, progress=None, keep=()):
    """Prop pose per frame (T, 4, 4) relative to its scene placement, from points followed on its texture.
    frames: T images (H, W) or (H, W, 3) BGR; prop_masks, arm_masks: (T, H, W) bool (arm_masks: both arms).
    tris: the prop's triangles (N, 3, 3) at the scene placement, or None to ray-march item's signed distance.
    keep: frames whose surviving matches are returned (for checks). Returns (motion, err, held, matches):
    err = median reprojection of the matches used (px, NaN where too few), held = points still followed."""
    T = len(frames)
    H, W = prop_masks.shape[1:]
    if (H, W) != (cam['H'], cam['W']):
        raise ValueError(f'masks are {W}x{H}, the camera is {cam["W"]}x{cam["H"]}')
    if tris is None and item is None:
        raise ValueError('give the prop triangles or its Item')
    gray = [_gray(np.asarray(f)) for f in frames]
    K = intrinsics(cam)
    C = np.array(cam['world'])
    C[:3, :3] /= np.linalg.norm(C[:3, :3], axis=0)
    Cinv = np.linalg.inv(C)
    D = np.diag([1.0, -1, -1])  # Blender camera axes -> OpenCV's
    flip = np.eye(4)
    flip[:3, :3] = D

    def to_world(rvec, tvec):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
        return C @ flip @ m

    def cast(origin, dirs):
        return raycast_triangles(tris, origin, dirs) if tris is not None else raycast_item(item, origin, dirs)

    def acquire(t, rvec, tvec, have, n):
        """Fresh points on the prop in frame t, their 3D from the current pose (on the placed prop)."""
        region = prop_masks[t] & ~ndimage.binary_dilation(arm_masks[t], iterations=ARM_MARGIN)
        new = cv2.goodFeaturesToTrack(gray[t], n, 0.01, MIN_GAP, mask=region.astype(np.uint8) * 255)
        if new is None:
            return np.zeros((0, 3)), np.zeros((0, 2), np.float32)
        new = new.reshape(-1, 2)
        if len(have):
            new = new[np.min(np.linalg.norm(new[:, None] - have[None], axis=-1), axis=1) > MIN_GAP]
        to_model = np.linalg.inv(to_world(rvec, tvec))  # world -> the prop at its scene placement
        o = to_model[:3, :3] @ C[:3, 3] + to_model[:3, 3]
        hit = cast(o, _rays(new, cam) @ to_model[:3, :3].T)
        k = ~np.isnan(hit[:, 0])
        return hit[k], new[k].astype(np.float32)

    rvec, tvec = cv2.Rodrigues(D @ Cinv[:3, :3])[0], (D @ Cinv[:3, 3]).reshape(3, 1)  # frame 0: at its placement
    X, p = acquire(0, rvec, tvec, np.zeros((0, 2)), FEATURES)
    if log:
        log(f'prop points: {len(X)} picked in frame 0')
    motion = np.repeat(np.eye(4)[None], T, 0)
    motion[0] = to_world(rvec, tvec)
    err, held, matches = np.full(T, np.nan), np.zeros(T, int), {}
    err[0], held[0] = 0.0, len(X)
    p = p.reshape(-1, 1, 2)
    alive = np.ones(len(X), bool)
    for t in range(1, T):
        if len(X):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], p, None, **LK)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t - 1], nxt, None, **LK)
            good = alive & (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - p, axis=-1).ravel() < FB_ERROR)
            ij = np.round(nxt.reshape(-1, 2)).astype(int)
            inside = (ij[:, 0] >= 0) & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H)
            good &= inside
            good[inside] &= ~arm_masks[t][ij[inside, 1], ij[inside, 0]]  # a hand moved over it: that point is lost
            alive, p = good, nxt
        if alive.sum() >= 8:
            ok, rv, tv, inl = cv2.solvePnPRansac(X[alive], p.reshape(-1, 2)[alive].astype(np.float64), K, None, rvec.copy(), tvec.copy(),
                                                 useExtrinsicGuess=True, reprojectionError=RANSAC_PX, iterationsCount=200)
            if ok and inl is not None:
                rvec, tvec = rv, tv
                out = np.ones(alive.sum(), bool)
                out[inl.ravel()] = False
                alive[np.flatnonzero(alive)[out]] = False
                q = cv2.projectPoints(X[alive], rvec, tvec, K, None)[0].reshape(-1, 2)
                err[t] = float(np.median(np.linalg.norm(q - p.reshape(-1, 2)[alive], axis=1)))
        motion[t] = to_world(rvec, tvec)  # too few points: the prop holds its last pose
        held[t] = int(alive.sum())
        if t in keep:
            matches[t] = (X[alive].copy(), p.reshape(-1, 2)[alive].copy())
        if alive.sum() < REFILL:  # running low: top up with fresh points
            X, p = X[alive], p.reshape(-1, 2)[alive]
            Xn, pn = acquire(t, rvec, tvec, p, REFILL_FEATURES)
            X, p = np.r_[X, Xn], np.r_[p, pn].astype(np.float32).reshape(-1, 1, 2)
            alive = np.ones(len(X), bool)
        if progress:
            progress(t + 1, T)
        if log and (t % 24 == 0 or t == T - 1):
            log(f'prop frame {t}: {held[t]} points held, {err[t]:.1f} px off')
    return motion, err, held, matches


# The outline check. Points followed on the texture pin the prop where they sit, but when they bunch on one part
# of it (the rest hidden, untextured or redrawn), a prop that comes closer can be read as one that turned. The
# prop's outline (SAM 2's mask) sees all of it: where the posed prop's silhouette disagrees with the mask, the
# pose is refit to the mask from the points' pose, on every OUTLINE_EVERY-th frame, and the correction is
# carried smoothly between them.
OUTLINE_SCALE = 4         # masks compared at 1/OUTLINE_SCALE resolution
OUTLINE_OK = 0.85         # IoU: the points' pose agrees with the outline; left alone
OUTLINE_GAIN = 0.05       # IoU: a refit must beat the points' pose by this much to be taken
OUTLINE_EVERY = 8
OUTLINE_STARTS = (0.0, -0.05, -0.10, -0.15)  # m along the view: start the refit closer, too (the usual mix-up)
OUTLINE_CUTOFF = 1.5      # Hz: the correction is low-passed between its frames


def silhouette(tris, M, cam, shape, scale=OUTLINE_SCALE):
    """Boolean mask (shape) of the prop's triangles posed by M (relative to the scene) seen through cam;
    triangles at or behind the camera are left out."""
    from motionbridge.core.hand_fit import project
    C = np.array(cam['world'])
    fwd = -C[:3, 2] / np.linalg.norm(C[:3, 2])
    w = tris @ M[:3, :3].T + M[:3, 3]
    front = ((w - C[:3, 3]) @ fwd > 0.02).all(1)
    q = (project(w[front].reshape(-1, 3), cam) / scale).reshape(-1, 3, 2)
    im = np.zeros(shape, np.uint8)
    cv2.fillPoly(im, list(np.round(q).astype(np.int32)), 1)
    return im > 0


GLOVE_SAT = 90            # capture mode's gloves (orange right, blue left) are saturated and bright, the prop dull:
GLOVE_VAL = 110           # pixels this saturated and bright in the gloves' hues are hand, whatever SAM's prop mask says
GLOVE_HUES = ((8, 35), (85, 120))  # OpenCV hue (0-180): orange, blue


def glove_mask(frame):
    """Capture mode: where a glove shows in a BGR frame (closed over small gaps)."""
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
    hue = np.zeros(h.shape, bool)
    for lo, hi in GLOVE_HUES:
        hue |= (h >= lo) & (h <= hi)
    g = (s > GLOVE_SAT) & (v > GLOVE_VAL) & hue
    return cv2.morphologyEx(g.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def _outline_targets(mask, arms, scale=OUTLINE_SCALE):
    """The prop's mask at comparison size, and where it counts: under an arm the prop may be hidden, so it can't
    say either way."""
    H, W = mask.shape
    sz = (W // scale, H // scale)
    m = cv2.resize(mask.astype(np.uint8), sz, interpolation=cv2.INTER_AREA) > 0
    a = cv2.resize(arms.astype(np.uint8), sz, interpolation=cv2.INTER_AREA) > 0
    return m & ~a, ~a


def _iou(a, b, care):
    return (a & b & care).sum() / max(((a | b) & care).sum(), 1)


def outline_fit(M0, tris, mask, arms, cam):
    """The pose near M0 whose silhouette best matches the mask (6 DoF about the prop's centre, Powell from M0
    and from M0 moved closer). Returns (pose, IoU of M0, IoU of the pose)."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation
    m, care = _outline_targets(mask, arms)
    C = np.array(cam['world'])
    fwd = -C[:3, 2] / np.linalg.norm(C[:3, 2])
    c0 = tris.reshape(-1, 3).mean(0)
    c = M0[:3, :3] @ c0 + M0[:3, 3]

    def pose(v):
        M = np.eye(4)
        M[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix() @ M0[:3, :3]
        M[:3, 3] = c + v[3:] - M[:3, :3] @ c0
        return M

    cost = lambda v: 1 - _iou(silhouette(tris, pose(v), cam, m.shape), m, care)
    base = 1 - cost(np.zeros(6))
    best = None
    for dz in OUTLINE_STARTS:
        r = minimize(cost, np.r_[0, 0, 0, fwd * dz], method='Powell', options={'maxiter': 2000, 'xtol': 1e-3, 'ftol': 1e-4})
        if best is None or r.fun < best.fun:
            best = r
    return pose(best.x), base, 1 - best.fun


def outline_correct(motion, tris, prop_masks, arm_masks, cam, fps=24, log=print, progress=None, frames=None):
    """motion with the outline check applied: on every OUTLINE_EVERY-th frame (and the last) whose pose
    disagrees with the prop's mask, the pose refit to it; the correction (refit relative to the points' pose)
    carried between those frames and low-passed. frames (capture mode): the gloves' colours are cut from the
    prop's mask too, where SAM's arm masks missed some hand. Returns (motion, report rows (frame, IoU before,
    after))."""
    from scipy.spatial.transform import Rotation
    T = len(motion)
    keys = sorted(set(range(0, T, OUTLINE_EVERY)) | {T - 1})
    delta, rows = [], []
    for n, t in enumerate(keys):
        if progress:
            progress(n + 1, len(keys))
        shape = (prop_masks[t].shape[0] // OUTLINE_SCALE, prop_masks[t].shape[1] // OUTLINE_SCALE)
        arms = arm_masks[t] | glove_mask(frames[t]) if frames is not None else arm_masks[t]
        m, care = _outline_targets(prop_masks[t], arms)
        base = _iou(silhouette(tris, motion[t], cam, shape), m, care)
        D = np.eye(4)
        if m.sum() > 50 and base < OUTLINE_OK:
            M, base, fit = outline_fit(motion[t], tris, prop_masks[t], arms, cam)
            if fit > base + OUTLINE_GAIN:
                D = M @ np.linalg.inv(motion[t])
            rows.append((t, base, fit))
        delta.append(D)
    if all(np.allclose(D, np.eye(4)) for D in delta):
        return motion, rows
    rv = np.array([Rotation.from_matrix(D[:3, :3]).as_rotvec() for D in delta])
    tr = np.array([D[:3, 3] for D in delta])
    f = np.arange(T)
    rv = np.stack([np.interp(f, keys, rv[:, j]) for j in range(3)], 1)
    tr = np.stack([np.interp(f, keys, tr[:, j]) for j in range(3)], 1)
    if T > 2:
        from motionbridge.core.smooth import one_euro
        rv = one_euro(rv, np.zeros(T), fps, OUTLINE_CUTOFF, 0)
        tr = one_euro(tr, np.zeros(T), fps, OUTLINE_CUTOFF, 0)
    out = motion.copy()
    for t in range(T):
        D = np.eye(4)
        D[:3, :3] = Rotation.from_rotvec(rv[t]).as_matrix()
        D[:3, 3] = tr[t]
        out[t] = D @ motion[t]
    if log:
        fixed = [r for r in rows if r[2] > r[1] + OUTLINE_GAIN]
        log(f'prop outline check: {len(rows)} of {len(keys)} checked frames disagreed with the mask, {len(fixed)} refit '
            f'(IoU {np.mean([r[1] for r in fixed]):.2f} -> {np.mean([r[2] for r in fixed]):.2f})' if fixed else
            f'prop outline check: the points agree with the outline')
    return out, rows
