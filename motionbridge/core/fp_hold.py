"""Holds that last through what the video can't show. A hand holding the prop keeps its grip until there is
evidence it let go: when it leaves the frame, goes behind the prop or its reading falls apart, the hold carries
on and the hand stays at its pose on the prop. While it can be seen, the tracking leads and the hold only keeps
it from slipping.

reliability() scores each hand on each frame from what the video gives (how much of the hand is on screen,
whether SAM read it, how far the fit is from the read); hold_lock() finds the holds (only reliable frames can
start or end one) and pulls the hand, fingers included, toward its settled pose on the prop by how unreliable
the frame is. Presses (fingertips on a surface, no wrap) never start a hold."""
import numpy as np

from motionbridge.core import fp_events as ev
from motionbridge.core import fp_track as fp
from motionbridge.core.hand_fit import project
from motionbridge.core.smooth import one_euro

MARGIN = 0.02      # share of the frame: keypoints this close to the edge count as off screen
FIT_GOOD = 0.06    # fit error (share of the hand's size) up to which a frame counts fully
FIT_BAD = 0.15     # ... and from which not at all
TRUST = 0.5        # reliability needed for a frame to start or end a hold
SETTLED = 0.7      # reliability of the frames a hold's settled pose is taken from
CUTOFF = 2.0       # Hz: reliability is low-passed so the lock doesn't flicker
FLOOR = 0.15       # the lock's pull on a fully visible hand: enough to stop slip, not to override the video
APART = 0.3        # hand spans: the read palm this far from the prop on screen has let go (depth can't say so:
                   # one camera reads it badly, so a hand drifting off the prop only in depth still holds)


def reliability(kp2d, errs, W, H, fps=fp.FPS):
    """(T,) in [0, 1] per hand: share of SAM's hand keypoints on screen (squared: a half-visible hand is read
    badly), times how close the fit came to the read. No read: 0."""
    T = len(errs)
    r = np.zeros(T)
    for t in range(T):
        uv = np.nanmedian(fp.to_hand21(kp2d[t]), axis=0) if not np.isnan(kp2d[t]).all() else None
        if uv is None or np.isnan(errs[t]):
            continue
        on = ((uv[:, 0] > MARGIN * W) & (uv[:, 0] < (1 - MARGIN) * W) & (uv[:, 1] > MARGIN * H) & (uv[:, 1] < (1 - MARGIN) * H)).mean()
        fit = np.clip((FIT_BAD - errs[t]) / (FIT_BAD - FIT_GOOD), 0, 1)
        r[t] = on ** 2 * fit
    return np.clip(one_euro(r[:, None], np.zeros(T), fps, CUTOFF, 0)[:, 0], 0, 1) if T > 2 else r


def apart(kp2d, prop, motion, cam, T):
    """(T,) bool: the read palm is clearly away from the prop on screen (no read: False)."""
    out = np.zeros(T, bool)
    for t in range(T):
        if np.isnan(kp2d[t]).all():
            continue
        uv = np.nanmedian(fp.to_hand21(kp2d[t]), axis=0)
        span = np.linalg.norm(uv[0] - uv[12])
        pts = project(prop.points @ motion[t][:3, :3].T + motion[t][:3, 3], cam)
        palm = uv[[0, 5, 9, 13, 17]]
        d = np.min(np.linalg.norm(palm[:, None] - pts[None], axis=2))
        out[t] = d > APART * span
    return out


def holds(rig, params, prop, motion, rel, kp2d=None, cam=None, fps=fp.FPS):
    """[(start, end)] of the hand's holds: started and ended on reliable frames only (hysteresis as fp_events),
    carried through unreliable ones. With the read (kp2d, cam), a hold ends only when the hand also leaves the
    prop on screen."""
    c = ev.contacts(ev.hand_points(rig, params), prop, motion)
    score, wrapped = ev.hold_score(c), c['wrapped']
    T = len(params)
    away = apart(kp2d, prop, motion, cam, T) if kp2d is not None else np.ones(T, bool)
    on, spans, start = False, [], 0
    for t in range(T):
        if rel[t] >= TRUST:
            if not on and score[t] >= ev.HOLD_ON and wrapped[t] >= ev.MIN_WRAP:
                on, start = True, t
            elif on and score[t] < ev.HOLD_OFF and away[t]:
                on = False
                spans.append((start, t))
    if on:
        spans.append((start, len(params)))
    n = ev._frames(ev.MIN_HOLD, fps)
    return [(a, b) for a, b in spans if b - a >= n]


def hold_lock(rig, params, prop, motion, rel, kp2d=None, cam=None, fps=fp.FPS, log=print):
    """params with each hold kept: the hand pulled to its settled pose on the prop (taken from the hold's
    reliable frames) by 1 - reliability (at least FLOOR), eased in and out at the hold's ends; the fingers
    likewise toward their settled grip. Returns (params, spans)."""
    P = params.copy()
    T, k = len(P), rig.na + 3
    spans = holds(rig, params, prop, motion, rel, kp2d, cam, fps)
    Hf = np.array([rig.hand_frame(x) for x in params])
    n = ev._frames(ev.EASE, fps)
    for a, b in spans:
        good = [t for t in range(a, b) if rel[t] >= SETTLED] or [a]
        ref = ev.mean_pose(np.linalg.inv(motion[good]) @ Hf[good])
        fingers = np.median(params[good, k:], axis=0)
        edge = np.ones(b - a)
        if a > 0:
            edge[:n] = np.minimum(edge[:n], ev._ramp(n)[:b - a])
        if b < T:
            edge[-n:] = np.minimum(edge[-n:], ev._ramp(n)[::-1][-(b - a):])
        for i, t in enumerate(range(a, b)):
            w = edge[i] * max(1 - rel[t], FLOOR)
            P[t] = fp._place_hand(rig, params[t], fp._blend(Hf[t], motion[t] @ ref, w))
            P[t, k:] = (1 - w) * params[t, k:] + w * fingers
        if log:
            unseen = sum(rel[t] < TRUST for t in range(a, b))
            log(f'{rig.hand.side.lower()} hand holds the prop over frames {a}-{b - 1} ({unseen} of them unreliable: '
                f'kept on its grip there)')
    return P, spans
