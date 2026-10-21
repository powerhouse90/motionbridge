"""Fit a HandModel to 2D hand points (MediaPipe's 21) through a known camera.

anatomical=True (the default) is what production uses:
  - joint limits per finger value (hands.LIMITS)
  - the middle and tip joints of a finger bend together (tip ~ 0.7 x middle), as real fingers do
  - smoothness: each frame stays near the previous one
  - finger gating: a finger whose points disagree with the rest of the hand is dropped for that
    frame and holds its previous pose, instead of twisting the joints to chase a bad point
anatomical=False fits every value independently with weak priors (the earlier probe), for comparison.
"""
import numpy as np
from scipy.optimize import least_squares

from motionbridge.core.hands import FINGERS, PER_FINGER

COUPLE = 0.7       # tip bend / middle bend
# Expected sizes of each term (residuals are divided by these, so the terms are balanced, not guessed):
KP_SIGMA = 0.05    # keypoint noise, fraction of the hand's span on screen
STEP_SIGMA = 0.35  # radians a finger value may change between frames
COUPLE_SIGMA = 0.25
HOLD_SIGMA = 0.03  # a gated finger stays where it was
GATE = 2.5         # a finger is dropped when its error exceeds GATE x the hand's median finger error...
GATE_MIN = 0.08    # ...and GATE_MIN of the hand's size on screen


def project(points, camera):
    """World points (N, 3) -> pixels (N, 2) through camera {world, lens, sensor, W, H}."""
    m = np.linalg.inv(np.array(camera['world']))
    c = points @ m[:3, :3].T + m[:3, 3]
    f = camera['lens'] / camera['sensor'] * camera['W']
    return np.c_[camera['W'] / 2 + f * c[:, 0] / -c[:, 2], camera['H'] / 2 - f * c[:, 1] / -c[:, 2]]


def finger_slices(model):
    out, k = {}, 3
    for f in FINGERS:
        if f in model.hand.fingers:
            out[f] = (k, slice(1 + 4 * FINGERS.index(f), 5 + 4 * FINGERS.index(f)))  # param start, keypoint rows
            k += PER_FINGER
    return out


def fit(model, observed, camera, anatomical=True, start=None):
    """observed: (T, 21, 2) pixels, NaN where missing. Returns params (T, P) and per-frame used-finger masks."""
    T, P = len(observed), model.hand.size
    lo, hi = model.hand.limits() if anatomical else (np.full(P, -np.pi), np.full(P, np.pi))
    prev = np.zeros(P) if start is None else start.copy()
    span = np.linalg.norm(np.subtract(*project(model.keypoints(prev)[[0, 12]], camera)))
    fingers = finger_slices(model)
    params, used = np.zeros((T, P)), np.zeros((T, len(fingers)), bool)
    for t in range(T):
        obs = observed[t]
        if np.isnan(obs[0, 0]):
            params[t] = prev
            continue
        active = {f: True for f in fingers}

        def residual(x):
            uv = project(model.keypoints(x), camera)
            rows = [0] + [r for f, (_, sl) in fingers.items() if active[f] for r in range(sl.start, sl.stop)]
            r = [(uv[rows] - obs[rows]).ravel() / (KP_SIGMA * span)]
            if anatomical:
                r.append((x - prev) / STEP_SIGMA)
                for f, (k, _) in fingers.items():
                    if f != 'Thumb':
                        r.append([(x[k + 3] - COUPLE * x[k + 2]) / COUPLE_SIGMA])
                    if not active[f]:
                        r.append((x[k:k + PER_FINGER] - prev[k:k + PER_FINGER]) / HOLD_SIGMA)
            else:
                r.append((x - prev) / (3 * STEP_SIGMA))
            return np.concatenate([np.atleast_1d(v) for v in r])

        x0 = np.clip(prev, lo + 1e-6, hi - 1e-6)
        x = least_squares(residual, x0, bounds=(lo, hi), max_nfev=100).x
        if anatomical:
            uv = project(model.keypoints(x), camera)
            err = {f: np.linalg.norm(uv[sl] - obs[sl], axis=1).mean() / span for f, (_, sl) in fingers.items()}
            med = np.median(list(err.values()))
            bad = [f for f, e in err.items() if e > GATE * med and e > GATE_MIN]
            if bad:
                for f in bad:
                    active[f] = False
                x = least_squares(residual, np.clip(x, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), max_nfev=100).x
        params[t], prev = x, x
        used[t] = [active[f] for f in fingers]
    return params, used
