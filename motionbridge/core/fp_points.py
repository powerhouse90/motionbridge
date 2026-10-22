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
