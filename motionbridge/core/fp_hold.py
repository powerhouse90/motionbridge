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
UNSEEN_ON = 0.5    # a locked hand this much on screen ...
UNSEEN_MASK = 0.2  # ... with this little of it on the video's arm mask is contradicted: the hand isn't there
EDGE = 0.15        # share of the frame: a hand lost this close to an edge left through it
PAST = 0.3         # hand spans: a bridged hand goes this far past the edge it left through
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


def holds(rig, params, prop, motion, rel, kp2d=None, cam=None, fps=fp.FPS, block=None):
    """[(start, end)] of the hand's holds: started and ended on reliable frames only (hysteresis as fp_events),
    carried through unreliable ones. With the read (kp2d, cam), a hold ends only when the hand also leaves the
    prop on screen. block: (T,) frames the video contradicts a hold on: it ends there, and only a reliable frame
    can start one again."""
    c = ev.contacts(ev.hand_points(rig, params), prop, motion)
    score, wrapped = ev.hold_score(c), c['wrapped']
    T = len(params)
    away = apart(kp2d, prop, motion, cam, T) if kp2d is not None else np.ones(T, bool)
    on, spans, start = False, [], 0
    for t in range(T):
        if on and block is not None and block[t]:
            on = False
            spans.append((start, t))
        elif rel[t] >= TRUST:
            if not on and score[t] >= ev.HOLD_ON and wrapped[t] >= ev.MIN_WRAP:
                on, start = True, t
            elif on and score[t] < ev.HOLD_OFF and away[t]:
                on = False
                spans.append((start, t))
    if on:
        spans.append((start, len(params)))
    n = ev._frames(ev.MIN_HOLD, fps)
    return [(a, b) for a, b in spans if b - a >= n]


def _settled(rig, params, motion, rel, Hf, a, b):
    """A hold's settled pose on the prop and finger grip, from its reliable frames."""
    k = rig.na + 3
    good = [t for t in range(a, b) if rel[t] >= SETTLED] or [a]
    return ev.mean_pose(np.linalg.inv(motion[good]) @ Hf[good]), np.median(params[good, k:], axis=0)


def contradicted(rig, params, motion, rel, spans, masks, cam):
    """(T,) bool: unreliable frames inside a hold where the hand, put at its grip, would be on screen but the
    video's arm mask shows no hand there (a hand that went off with a part of the prop, e.g. a magazine)."""
    T = len(params)
    out = np.zeros(T, bool)
    Hf = np.array([rig.hand_frame(x) for x in params])
    H, W = masks.shape[1:]
    for a, b in spans:
        ref, fingers = _settled(rig, params, motion, rel, Hf, a, b)
        for t in range(a, b):
            if rel[t] >= TRUST:
                continue
            x = fp._place_hand(rig, params[t], motion[t] @ ref)
            x[rig.na + 3:] = fingers
            uv = project(rig.keypoints(x), cam) * [W / cam['W'], H / cam['H']]
            on = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if on.mean() >= UNSEEN_ON:
                q = uv[on].astype(int)
                out[t] = masks[t][q[:, 1], q[:, 0]].mean() < UNSEEN_MASK
    return out


def hold_lock(rig, params, prop, motion, rel, kp2d=None, cam=None, fps=fp.FPS, log=print, masks=None):
    """params with each hold kept: the hand pulled to its settled pose on the prop (taken from the hold's
    reliable frames) by 1 - reliability (at least FLOOR), eased in and out at the hold's ends; the fingers
    likewise toward their settled grip. With the arm masks, a hold the video contradicts ends there
    (contradicted()). Returns (params, spans)."""
    P = params.copy()
    T, k = len(P), rig.na + 3
    spans = holds(rig, params, prop, motion, rel, kp2d, cam, fps)
    if masks is not None and cam is not None:
        block = contradicted(rig, params, motion, rel, spans, masks, cam)
        if block.any():
            spans = holds(rig, params, prop, motion, rel, kp2d, cam, fps, block)
            if log:
                log(f"{rig.hand.side.lower()} hand: the video contradicts its hold on {int(block.sum())} frames "
                    f"(no hand where its grip would put it): let go there")
    Hf = np.array([rig.hand_frame(x) for x in params])
    n = ev._frames(ev.EASE, fps)
    for a, b in spans:
        ref, fingers = _settled(rig, params, motion, rel, Hf, a, b)
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


def _past_edge(rig, x, cam):
    """The hand frame of x moved across the view (at its depth) to just past the frame edge it is nearest, if it
    is within EDGE of one; else None."""
    p = rig.keypoints(x)
    uv = project(p, cam)
    pc = uv[[0, 5, 9, 13, 17]].mean(0)
    W, H = cam['W'], cam['H']
    gaps = {'left': pc[0], 'right': W - pc[0], 'top': pc[1], 'bottom': H - pc[1]}
    side = min(gaps, key=gaps.get)
    if gaps[side] > EDGE * (W if side in ('left', 'right') else H):
        return None
    span = np.linalg.norm(uv[0] - uv[12])
    C = np.array(cam['world'])
    R = C[:3, :3] / np.linalg.norm(C[:3, :3], axis=0)
    depth = (p[[0, 5, 9, 13, 17]].mean(0) - C[:3, 3]) @ -R[:, 2]
    focal = cam['lens'] / cam['sensor'] * W
    metres = (max(gaps[side], 0) + PAST * span) * depth / focal
    move = {'left': -R[:, 0], 'right': R[:, 0], 'top': R[:, 1], 'bottom': -R[:, 1]}[side] * metres
    Hf = rig.hand_frame(x).copy()
    Hf[:3, 3] += move
    return Hf


def bridge(rig, params, rel, spans, log=print, cam=None):
    """A free hand the video can't follow (off screen, misread) outside any hold: across each run of unreliable
    frames the hand goes smoothly from its last reliable pose to its next one (hand frame and fingers
    interpolated), replacing the reading by how unreliable the frame is. No reliable frame after: it keeps its
    last one; none before: it eases from the next one. With the camera, a hand lost near a frame edge goes out
    past it on the way (and comes back in from it), as a hand that left the frame did. Returns params."""
    P = params.copy()
    T, k = len(P), rig.na + 3
    held = np.zeros(T, bool)
    for a, b in spans:
        held[a:b] = True
    bad = (rel < TRUST) & ~held
    Hf = np.array([rig.hand_frame(x) for x in params])
    for a, b in ev._runs(bad):
        before, after = a - 1, b  # the reliable (or held) frames around the gap
        if before < 0 and after >= T:
            continue  # never seen well: nothing better to go on
        A = Hf[before] if before >= 0 else Hf[after]
        B = Hf[after] if after < T else Hf[before]
        fa = params[before, k:] if before >= 0 else params[after, k:]
        fb = params[after, k:] if after < T else params[before, k:]
        E = None
        if cam is not None:
            E = _past_edge(rig, params[before], cam) if before >= 0 else None
            if E is None and after < T:
                E = _past_edge(rig, params[after], cam)
        n = b - a + 1
        for t in range(a, b):
            s = (t - a + 1) / n
            if E is None:
                path = fp._blend(A, B, s * s * (3 - 2 * s))
            elif after >= T:  # out through the edge, not back
                u = min(2 * s, 1.0)
                path = fp._blend(A, E, u * u * (3 - 2 * u))
            elif before < 0:
                u = max(2 * s - 1, 0.0)
                path = fp._blend(E, B, u * u * (3 - 2 * u))
            else:
                u = 2 * s if s < 0.5 else 2 * s - 1
                u = u * u * (3 - 2 * u)
                path = fp._blend(A, E, u) if s < 0.5 else fp._blend(E, B, u)
            w = 1 - rel[t]
            P[t] = fp._place_hand(rig, params[t], fp._blend(Hf[t], path, w))
            P[t, k:] = (1 - w) * params[t, k:] + w * ((1 - s) * fa + s * fb)
        if log:
            log(f'{rig.hand.side.lower()} hand not held and not readable over frames {a}-{b - 1}: bridged between its '
                f'poses at frames {before if before >= 0 else after} and {after if after < T else before}'
                + (' (out past the frame edge it left through)' if E is not None else ''))
    return P


GRIP_REGION = 0.6      # palm lengths: a known grip counts for a hold whose palm is this near it on the prop
GRIP_BETTER = 1.5      # hold score: a known grip replaces the video's reading only if it holds this much better
GRIP_SETTLED = 0.9     # reliability of frames whose grip is learned from the video (with the hand holding well)


def _grip_score(rig, x, prop, M):
    c = ev.contacts(rig.keypoints(x)[None], prop, M[None])
    return float(ev.hold_score(c)[0]), float(c['palm'][0])


def known_grips(rig, params, prop, motion, rel, spans, start=True):
    """Grips this hand is known to make on the prop, relative to it: the scene's start pose (if it holds the
    prop there) and, per hold, its best-read frame where the video saw the hand well and it held well.
    Returns [(hand frame on the prop, fingers, palm centre on the prop, score)]."""
    k = rig.na + 3
    out = []

    def add(x, M):
        score, palm = _grip_score(rig, x, prop, M)
        if score >= ev.HOLD_ON and palm < PALM_NEAR_GRIP:
            pc = rig.keypoints(x)[[0, 5, 9, 13, 17]].mean(0)
            out.append((np.linalg.inv(M) @ rig.hand_frame(x), x[k:].copy(), (np.linalg.inv(M) @ np.r_[pc, 1])[:3], score))

    if start:
        add(np.zeros(rig.size), motion[0])
    for a, b in spans:
        good = [t for t in range(a, b) if rel[t] >= GRIP_SETTLED]
        if good:
            t = max(good, key=lambda t: _grip_score(rig, params[t], prop, motion[t])[0])
            add(params[t], motion[t])
    return out


PALM_NEAR_GRIP = 0.12  # palm lengths: a known grip's palm rests this close to the prop


def reuse_grips(rig, params, prop, motion, spans, grips, fps=fp.FPS, log=print):
    """Holds where the video's reading of the hand holds the prop worse than a grip the hand is known to make on
    the same part of it (a regrip, half hidden, read the wrong way round): the hand eases into that grip over
    the hold's first EASE (and out over its last, if it lets go before the end); fingers likewise. Returns params."""
    P = params.copy()
    T, k = len(P), rig.na + 3
    if not grips:
        return P
    L = ev.palm_length(np.array([rig.keypoints(x) for x in params[:1]]))
    n = ev._frames(ev.EASE, fps)
    Hf = np.array([rig.hand_frame(x) for x in params])
    for a, b in spans:
        mid = (a + b) // 2
        pcs = np.array([(np.linalg.inv(motion[t]) @ np.r_[rig.keypoints(params[t])[[0, 5, 9, 13, 17]].mean(0), 1])[:3]
                        for t in range(a, b)])
        here = np.median(pcs, axis=0)
        near = [g for g in grips if np.linalg.norm(g[2] - here) < GRIP_REGION * L]
        if not near:
            continue
        g = max(near, key=lambda g: g[3])
        video = np.median([_grip_score(rig, params[t], prop, motion[t])[0] for t in range(a, b, max(1, (b - a) // 6))])
        if g[3] < video + GRIP_BETTER:
            continue
        w = np.ones(b - a)
        if a > 0:
            w[:n] = np.minimum(w[:n], ev._ramp(n)[:b - a])
        if b < T:
            w[-n:] = np.minimum(w[-n:], ev._ramp(n)[::-1][-(b - a):])
        for i, t in enumerate(range(a, b)):
            P[t] = fp._place_hand(rig, params[t], fp._blend(Hf[t], motion[t] @ g[0], w[i]))
            P[t, k:] = (1 - w[i]) * params[t, k:] + w[i] * g[1]
        if log:
            log(f'{rig.hand.side.lower()} hand, frames {a}-{b - 1}: the video\'s grip (hold score {video:.1f}) replaced by the '
                f'grip it is known to make there (score {g[3]:.1f})')
    return P
