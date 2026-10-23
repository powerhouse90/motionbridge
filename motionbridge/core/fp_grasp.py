"""The grasp step: where a hand holds the main prop, its fingers are made to hold it.

The tracked hand is right on screen (SAM 3D Body's joints) but its distance is what a single video measures
worst, so a hand can sit a few cm behind the handle it holds, and the "no penetration" rule then keeps it
there. Per frame, every joint stays where the skeleton puts it on screen, the hand's depth is free, and each
finger gets a contact rule picked from the tracked pose:

  wrap   the finger's base is within reach, the palm is on the prop and the prop there is thin enough to close
         a hand around (a handle, a grip, a bar): the prop lies between the palm's centre and the fingertip,
         the tip rests on its surface, the palm rests on it
  press  the fingertip is at the surface but the hand cannot close around it (a keyboard, a button, a trigger
         face, a wall): the tip rests on the surface
  free   neither: screen points and bone directions only

A finger the video shows clearly less curled than the rest of a curled hand, its tip apart from theirs on screen,
is not part of the wrap (a trigger finger, a finger along the slide): it presses where it meets the prop, or is free.

The fit has several minima (the hand behind the handle scores well: fewer terms disagree), so it starts from
the tracked pose and from poses slid toward the camera, and of the fits nearly as good as the best, the one
whose fingers hold the most wins.

A whole take: pass 1 solves each held frame on its own (in parallel, with the search above), the hands are
smoothed, then pass 2 settles each held frame again from its smoothed pose, held near it, against the video's
hand low-passed over time (no search: it is already next to the answer), and a final light smoothing. Smoothing
alone undoes wraps (it averages between different ones), grasping alone is jittery (each frame's read noise);
together they keep both.

Lengths are in hand lengths (wrist to middle fingertip, the rig's own; [metres] for a 19 cm hand)."""
import os
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from motionbridge.core import fp_track as fp
from motionbridge.core import hands
from motionbridge.core.hand_fit import project

HOLD_RAY = 0.155       # [3 cm] a palm whose line of sight passes this close to the prop holds it
SIGHT = 0.78           # [15 cm] ... looked for this far either side of the palm along its line of sight
WRAP_REACH = 0.36      # [7 cm] a finger whose base is this close can wrap the prop
PALM_NEAR = 0.3        # [6 cm] and only if the palm is this close (after the depth slide)
ENCLOSE = 0.45         # [8.7 cm] a hand closes around a prop no wider than this in the plane its fingers curl in
PRESS_REACH = 0.08     # [1.5 cm] a fingertip this close to the surface (after the depth slide) presses on it
DEPTH_SLACK = 0.6      # [12 cm] the hand's tracked depth may be off by this much either way
PALM_REST = 0.062      # [12 mm] the palm's centre line sits this far off the surface when it holds the prop
LINE_MARGIN = 0.02     # [4 mm] the palm-to-tip line goes this deep into the prop
LINE_TOL, TIP_TOL, PALM_TOL = 0.0155, 0.031, 0.0207  # [3, 6, 4 mm] expected size of those terms
ACROSS = 0.052         # [1 cm] the hand moves across the view (not along it) by about this much
STARTS = (0.155, 0.31, 0.465, 0.62)  # [3-12 cm] extra starts, the hand slid toward the camera by these
SCREEN = 0.02          # screen points, fraction of the hand's size on screen: joints stay where the video shows them
DIRS = 0.5             # bone directions count, but less than the screen
STEP = {'control': 0.103, 'arm': 0.1}  # [2 cm] / rad: placement change against a reference pose (pass 2, neighbours)
STEP_WRIST, STEP_FINGER = 0.2, 0.3     # rad
CHOICE = 2.0           # fits whose cost is within this factor of the best compete on how many fingers hold
LINE = np.linspace(0.15, 0.95, 9)      # where on the palm-to-tip line the prop is looked for
MAX_NFEV = 150
OBS_CUT = 3.0          # Hz: the video's hand is low-passed this much for pass 2, so it doesn't take each frame's read noise back in
APART_CURL = 0.8       # a finger curled less than this fraction of the other fingers' (median) curl ...
APART_TIP = 0.5        # ... with its tip this far from theirs on screen (fraction of the hand's size) is apart from the wrap
CURLED = 120.0         # degrees: ... and only when the other fingers really are curled
MODE_WINDOW = 2        # frames either side: a finger's mode is the majority over this window (one-frame flips are noise)


def hand_length(rig):
    p = rig.keypoints(np.zeros(rig.size))
    return float(np.linalg.norm(p[12] - p[0]))


def _view(cam):
    C = np.array(cam['world'])
    f = -C[:3, 2]
    return C[:3, 3], f / np.linalg.norm(f)


def _segs(rig):
    """Keypoint index pairs per finger segment, base to tip (MediaPipe 21 layout)."""
    return {f: [(1 + 4 * j + a, 2 + 4 * j + a) for a in range(3)] for j, f in enumerate(hands.FINGERS) if f in rig.hand.fingers}


def _palm(p):
    return np.array([p[0] + s * (p[j] - p[0]) for j in (5, 9, 13, 17) for s in (0.5, 1.0)])


def _palm_centre(p):
    return p[[0, 5, 9, 13, 17]].mean(0)


def holds(rig, x, prop, M, cam):
    """The hand holds (or touches) the prop: its palm's line of sight passes close to it. The line of sight, not
    the palm itself: the hand's tracked depth is the unreliable part."""
    L = hand_length(rig)
    cpos, _ = _view(cam)
    palm = _palm(rig.keypoints(x))
    ray = palm - cpos
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    samples = (palm[:, None] + ray[:, None] * np.linspace(-SIGHT * L, SIGHT * L, 13)[None, :, None]).reshape(-1, 3)
    return bool(prop.distance(samples, M).min() < HOLD_RAY * L)


def sight(points, prop, M, cam, L):
    """Per point, the prop's distance from its line of sight within DEPTH_SLACK either side: how close it could be
    with its depth fixed (the depth is what a single video measures worst). Also the nearest place on it."""
    cpos, _ = _view(cam)
    ray = points - cpos
    ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    along = np.linspace(-DEPTH_SLACK * L, DEPTH_SLACK * L, 25)
    samples = points[:, None] + ray[:, None] * along[None, :, None]
    d = prop.distance(samples.reshape(-1, 3), M).reshape(len(points), len(along))
    k = d.argmin(1)
    return d[np.arange(len(points)), k], samples[np.arange(len(points)), k]


def enclosable(prop, M, p, L):
    """Can the fingers close around the prop where the palm is? Through the prop's surface nearest the palm,
    along the fingers' direction, the prop must be narrower than a curled hand (a handle across the fingers,
    not a keyboard or a wall under them). p: the hand's 21 points, palm placed at its nearest approach."""
    pc = _palm_centre(p)
    n = np.cross(p[5] - p[0], p[17] - p[0])
    n /= np.linalg.norm(n)
    e = p[9] - p[0]
    e -= (e @ n) * n
    e /= np.linalg.norm(e)
    R, k = 0.6 * L, 41
    g = np.linspace(-R, R, k)
    grid = pc + g[:, None, None] * e + g[None, :, None] * n  # rows along the fingers, columns along the palm normal
    solid = prop.distance(grid.reshape(-1, 3), M).reshape(k, k) < 0
    if not solid.any():
        return False
    ij = np.argwhere(solid)
    i, j = ij[np.argmin(np.linalg.norm(ij - k // 2, axis=1))]
    col = solid[:, j]
    a, b = i, i
    while a > 0 and col[a - 1]:
        a -= 1
    while b < k - 1 and col[b + 1]:
        b += 1
    if a == 0 or b == k - 1:  # runs off the window: wider than a hand
        return False
    return bool((b - a + 1) * (g[1] - g[0]) < ENCLOSE * L)


def depth_slide(rig, x, prop, M, cam):
    """How far along the view the hand would slide to just meet the prop without entering it (m, + = away from
    the camera); 0 if it already touches or cannot within DEPTH_SLACK."""
    L = hand_length(rig)
    _, fwd = _view(cam)
    surf = fp._surface_points(rig.keypoints(x))
    dzs = np.linspace(-DEPTH_SLACK * L, DEPTH_SLACK * L, 49)
    score = []
    for dz in dzs:
        d = prop.distance(surf + fwd * dz, M)
        score.append(max(d.min() - fp.SURFACE, 0) + 10 * np.maximum(fp.SURFACE - d, 0).sum())
    score = np.array(score)
    ok = score < max(score.min(), 0) + 1e-3
    return float(dzs[ok][np.argmin(np.abs(dzs[ok]))]) if score.min() < 0.25 * PRESS_REACH * L else 0.0


def finger_modes(rig, x, prop, M, cam, obs=None):
    """{finger: 'wrap' | 'press' | 'free'} from a tracked pose (see the module notes). Wrapping is judged with the
    depth left open (lines of sight); pressing after sliding the whole hand along the view to meet the prop, since
    a fingertip lifted a little off a key must not count. With obs (fp_track.observation), a finger the video shows
    apart from the grip doesn't wrap: it presses if its tip's line of sight meets the prop, else it is free."""
    L = hand_length(rig)
    _, fwd = _view(cam)
    p = rig.keypoints(x)
    segs = _segs(rig)
    dp, near = sight(_palm(p), prop, M, cam, L)
    closes = dp.min() < PALM_NEAR * L and enclosable(prop, M, p + (near[dp.argmin()] - _palm(p)[dp.argmin()]), L)
    bases = sight(p[[s[0][0] for s in segs.values()]], prop, M, cam, L)[0]
    tips = prop.distance(p[[s[2][1] for s in segs.values()]] + fwd * depth_slide(rig, x, prop, M, cam), M)
    modes = {f: 'wrap' if closes and b < WRAP_REACH * L else 'press' if t < PRESS_REACH * L else 'free'
             for f, b, t in zip(segs, bases, tips)}
    for f in apart(rig, obs) if obs is not None else ():
        if modes.get(f) == 'wrap':
            modes[f] = 'press' if sight(p[[segs[f][2][1]]], prop, M, cam, L)[0][0] < PRESS_REACH * L else 'free'
    return modes


def _curl(d, f):
    """Degrees a finger (1..4: index..pinky) bends, metacarpal through tip, from bone directions (fp_track.SEG order)."""
    chain = [d[15 + f]] + [d[4 + 3 * (f - 1) + k] for k in range(3)]
    return sum(np.degrees(np.arccos(np.clip(a @ b, -1, 1))) for a, b in zip(chain, chain[1:]))


def apart(rig, obs):
    """Fingers the video shows apart from the hand's grip: clearly less curled than the others, which are curled,
    and their tip away from the others' on screen (a trigger finger). Thumb excluded: it always curls differently."""
    uv, span, d = obs
    names = [f for f in hands.FINGERS[1:] if f in rig.hand.fingers]
    k = [hands.FINGERS.index(f) for f in names]
    curl = np.array([_curl(d, f) for f in k])
    tips = uv[[4 * f + 4 for f in k]]
    out = set()
    for j, f in enumerate(names):
        others = np.delete(np.arange(len(names)), j)
        if len(others) < 2:
            continue
        ref = np.median(curl[others])
        far = np.linalg.norm(tips[j] - tips[others].mean(0)) / span
        if ref >= CURLED and curl[j] < APART_CURL * ref and far > APART_TIP:
            out.add(f)
    return out


def holding(rig, x, prop, M, modes):
    """How many fingers hold as their mode asks: wrap -> the prop between palm centre and tip, press -> the
    tip on the surface."""
    L = hand_length(rig)
    p = rig.keypoints(x)
    pc = _palm_centre(p)
    n = 0
    for f, s in _segs(rig).items():
        tip = p[s[2][1]]
        if modes.get(f) == 'wrap':
            n += prop.distance(pc + LINE[:, None] * (tip - pc), M).min() < 0
        elif modes.get(f) == 'press':
            n += abs(prop.distance(tip[None], M)[0] - fp.SURFACE) < TIP_TOL * L
    return int(n)


def _slid(rig, x, offset):
    """x with the hand moved by a world offset (wrist and fingers as they are)."""
    if rig.mode == 'control':  # the hand follows its control linearly
        J = np.array([rig.hand_frame(np.r_[e, np.zeros(rig.size - 3)])[:3, 3] - rig.hand_frame(np.zeros(rig.size))[:3, 3] for e in np.eye(3)]).T
        return np.r_[x[:3] + np.linalg.pinv(J) @ offset, x[3:]]
    H = rig.hand_frame(x)
    H[:3, 3] += offset
    return fp._place_hand(rig, x, H)


def grasp(rig, prop, M, obs, cam, x0, prev=None, modes=None, search=True):
    """One frame: the hand's parameters with its fingers holding the prop (placed by M). obs: fp_track.observation
    for this frame; x0: the tracked pose (screen anchor); prev: a pose to stay near (pass 2), or None; search:
    False when x0 is already next to the answer (a smoothed pose): one fit from it, contact rules on.
    Returns (x, modes, fingers holding)."""
    uv, span, target = obs
    L = hand_length(rig)
    _, fwd = _view(cam)
    modes = modes or finger_modes(rig, x0, prop, M, cam, obs)
    segs = _segs(rig)
    wrap = [segs[f][2][1] for f, m in modes.items() if m == 'wrap']
    press = [segs[f][2][1] for f, m in modes.items() if m == 'press']
    step = np.r_[np.full(rig.na, STEP[rig.mode] * (L if rig.mode == 'control' else 1)), np.full(3, STEP_WRIST),
                 np.full(rig.hand.size - 3, STEP_FINGER)]
    H0 = rig.hand_frame(x0)[:3, 3]

    def res(x, contact=True):
        p = rig.keypoints(x)
        pc = _palm_centre(p)
        r = [((project(p, cam) - uv) / (SCREEN * span)).ravel(), (fp.directions(p) - target).ravel() / DIRS,
             np.maximum(fp.SURFACE - prop.distance(fp._surface_points(p), M), 0) / 0.001]
        if wrap:
            r.append([(prop.distance(_palm(p), M).min() - PALM_REST * L) / (PALM_TOL * L)])
        if contact:
            for k in wrap:
                r.append([max(prop.distance(pc + LINE[:, None] * (p[k] - pc), M).min() + LINE_MARGIN * L, 0) / (LINE_TOL * L)])
                r.append([(prop.distance(p[[k]], M)[0] - fp.SURFACE) / (TIP_TOL * L)])
            for k in press:
                r.append([(prop.distance(p[[k]], M)[0] - fp.SURFACE) / (TIP_TOL * L)])
        shift = rig.hand_frame(x)[:3, 3] - H0
        r.append((shift - (shift @ fwd) * fwd) / (ACROSS * L))  # depth is free; across the view it stays put
        if prev is not None:
            r.append((x - prev) / step)
        r.append(rig.wrist_penalty(x))
        return np.concatenate([np.atleast_1d(q) for q in r])

    if not search:
        x = least_squares(lambda x: res(x, bool(wrap or press)), np.clip(x0, rig.lo + 1e-6, rig.hi - 1e-6),
                          bounds=(rig.lo, rig.hi), max_nfev=MAX_NFEV).x
        return x, modes, holding(rig, x, prop, M, modes)
    starts = [x0] if prev is None or np.allclose(prev, x0) else [prev, x0]
    if wrap:  # the hand behind the handle is a minimum the fit cannot climb out of: start it in front too
        starts += [_slid(rig, x0, -fwd * s * L) for s in STARTS]
    elif press:
        dz = depth_slide(rig, x0, prop, M, cam)
        if abs(dz) > 1e-4:
            starts.append(_slid(rig, x0, fwd * dz))
    # candidates from the fit with the contact rules and from the fit without (the rules can pull a hand off the
    # video, and a hand slid in front often closes on its own); judged on the video alone
    fits = [(lambda x, c=c: res(x, c)) for c in ((True, False) if wrap or press else (False,))]
    xs = [least_squares(f, np.clip(st, rig.lo + 1e-6, rig.hi - 1e-6), bounds=(rig.lo, rig.hi), max_nfev=MAX_NFEV).x
          for f in fits for st in starts]
    cost = [0.5 * np.sum(res(x, False) ** 2) for x in xs]
    held = [holding(rig, x, prop, M, modes) for x in xs]
    low = min(cost)
    best = max((k for k in range(len(xs)) if cost[k] <= CHOICE * low), key=lambda k: (held[k], -cost[k]))
    return xs[best], modes, held[best]


# A whole take, in parallel ----------------------------------------------------------------------------

_STATE = {}


def _init(state):
    _STATE.update(state)
    try:  # one BLAS thread per worker: the workers are the parallelism
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _job(a):
    i, t, x0, prev, modes, search = a
    s = _STATE
    obs = s['obs' if search else 'obs_smooth'][i][t]
    x, modes, n = grasp(s['rigs'][i], s['prop'], s['motion'][t], obs, s['cam'], x0, prev, modes, search)
    return i, t, x, modes, n


def _solve(jobs, state, workers, log, progress, label):
    """Run grasp jobs, in worker processes when workers > 0 (spawned: the caller's main module needs the usual
    `if __name__ == '__main__'` guard)."""
    t0, out, n = time.time(), [], len(jobs)
    every = max(1, n // 10)
    if workers <= 0:
        _init(state)
        it = map(_job, jobs)
        pool = None
    else:
        import multiprocessing as mp
        keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
        saved = {k: os.environ.get(k) for k in keys}
        os.environ.update({k: '1' for k in keys})  # spawned workers read these when numpy loads
        try:
            pool = mp.get_context('spawn').Pool(workers, _init, (state,))
        finally:
            for k, v in saved.items():
                os.environ.pop(k) if v is None else os.environ.__setitem__(k, v)
        it = pool.imap_unordered(_job, jobs)
    try:
        for k, r in enumerate(it, 1):
            out.append(r)
            if progress:
                progress(k, n)
            if log and (k % every == 0 or k == n):
                log(f'{label}: {k}/{n} frames ({time.time() - t0:.0f}s)')
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


def steady_modes(modes, window=MODE_WINDOW):
    """{t: {finger: mode}} -> the same with each finger's mode the majority over held frames within +-window
    (ties keep the frame's own): the per-frame tests flip on single frames where the geometry is borderline."""
    ts = sorted(modes)
    out = {}
    for t in ts:
        near = [u for u in ts if abs(u - t) <= window]
        out[t] = {}
        for f, own in modes[t].items():
            votes = [modes[u][f] for u in near if f in modes[u]]
            best = max(set(votes), key=lambda m: (votes.count(m), m == own))
            out[t][f] = best
    return out


def smoothed_observations(obs, cut=OBS_CUT):
    """A hand's per-frame observations low-passed over time (zero-phase, fixed cutoff): screen points, size and
    bone directions (re-normalised)."""
    T = len(obs)
    if T < 3:
        return list(obs)

    def lp(v):  # frames without a reading (the hand off screen) are filled from their neighbours first
        f = v.reshape(T, -1).astype(float).copy()
        ok = np.isfinite(f).all(1)
        if ok.sum() < 2:
            return v
        for j in range(f.shape[1]):
            f[~ok, j] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), f[ok, j])
        return fp.one_euro(f, np.zeros(T), fp.FPS, cut, 0).reshape(v.shape)
    uv = lp(np.array([o[0] for o in obs]))
    span = lp(np.array([[o[1]] for o in obs]))[:, 0]
    d = lp(np.array([o[2] for o in obs]))
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return list(zip(uv, span, d))


def grasp_take(rigs, prop, motion, kp2d, kp3d, params, cam, workers=8, strength=0.7, settle=False, log=print,
               progress=None):
    """The grasp step over a take. rigs: fp_track.Rig per hand; motion: the prop's pose per frame (T, 4, 4),
    smoothed; kp2d/kp3d: the extracted hands per rig (as fp_track.track_hand takes them); params: the tracked
    parameters per rig (T, P); workers: processes (0 = in this process, e.g. inside ComfyUI); settle: run
    fp_track.settle_contacts between the passes (off: the v6 recipe does without). Returns (params per rig,
    report): per hand the frames held, the finger modes per held frame (pass 1 'modes', pass 2 'modes2'), the
    fingers holding per frame after 'pass1', 'smoothed', 'pass2' and 'final', and 'jitter' (tracked, smoothed,
    pass 2, final; mm/frame^2)."""
    T = len(motion)
    obs = [[fp.observation(r, kp2d[i][t], kp3d[i][t], cam) for t in range(T)] for i, r in enumerate(rigs)]
    obs_smooth = [smoothed_observations(o) for o in obs]
    state = {'rigs': rigs, 'prop': prop, 'motion': motion, 'obs': obs, 'obs_smooth': obs_smooth, 'cam': cam}
    read = lambda o: all(np.isfinite(np.asarray(q, float)).all() for q in o)
    # frames without a reading of the hand have nothing to match: fp_hold keeps them on the grip or bridges them
    held = [[t for t in range(T) if read(obs[i][t]) and holds(r, params[i][t], prop, motion[t], cam)] for i, r in enumerate(rigs)]
    if log:
        log('holding the prop: ' + ', '.join(f'{r.hand.side.lower()} hand {len(h)} of {T} frames' for r, h in zip(rigs, held)))
    modes1 = [steady_modes({t: finger_modes(r, params[i][t], prop, motion[t], cam, obs[i][t]) for t in held[i]})
              for i, r in enumerate(rigs)]
    jobs = [(i, t, params[i][t], None, modes1[i][t], True) for i in range(len(rigs)) for t in held[i]]
    stage = lambda k, n: progress(k, 2 * len(jobs)) if progress else None
    first = _solve(jobs, state, workers, log, stage, 'grasp pass 1')
    p1 = [P.copy() for P in params]
    modes = [dict() for _ in rigs]
    for i, t, x, m, _ in first:
        p1[i][t], modes[i][t] = x, m
    count = lambda Ps, ms: [[holding(r, P[t], prop, motion[t], ms[i][t]) if t in ms[i] else 0 for t in range(T)]
                            for i, (r, P) in enumerate(zip(rigs, Ps))]
    report = {'held': held, 'modes': modes, 'pass1': count(p1, modes)}
    smooth = [fp.smooth_hand(r, [], [], [], P, strength=strength, log=None) for r, P in zip(rigs, p1)]
    if settle:
        smooth = fp.settle_contacts(rigs, prop, motion, smooth, None, cam, log=log)
    report['smoothed'] = count(smooth, modes)
    # pass 2: modes picked again from the smoothed pose (where the hand now is), against the smoothed video
    modes2 = [steady_modes({t: finger_modes(r, smooth[i][t], prop, motion[t], cam, obs_smooth[i][t]) for t in held[i]})
              for i, r in enumerate(rigs)]
    jobs = [(i, t, smooth[i][t], smooth[i][t], modes2[i][t], False) for i in range(len(rigs)) for t in held[i]]
    stage = lambda k, n: progress(len(jobs) + k, 2 * len(jobs)) if progress else None
    p2 = [P.copy() for P in smooth]
    for i, t, x, _, _ in _solve(jobs, state, workers, log, stage, 'grasp pass 2'):
        p2[i][t] = x
    report['modes2'] = modes2
    report['pass2'] = count(p2, modes2)
    out = [fp.smooth_hand(r, [], [], [], P, strength=strength, log=None) for r, P in zip(rigs, p2)]
    # frames pass 2 never touched (the hand not on the prop) keep their one smoothing: smoothed twice, a typing
    # hand's taps are gone; held frames take the second one (it settles pass 2), blended in over a few frames
    for i in range(len(rigs)):
        w = np.zeros(T)
        w[list(held[i])] = 1.0
        w = np.clip(np.convolve(w, np.ones(7) / 7, mode='same') * 7 / 4, 0, 1)
        out[i] = w[:, None] * out[i] + (1 - w[:, None]) * smooth[i]
    report['final'] = count(out, modes2)
    report['jitter'] = [(fp.jitter(r, params[i]), fp.jitter(r, smooth[i]), fp.jitter(r, p2[i]), fp.jitter(r, out[i]))
                        for i, r in enumerate(rigs)]
    if log:
        for i, r in enumerate(rigs):
            log(f'{r.hand.side.lower()} hand: fingers holding per frame {report["final"][i]}; jitter (mm/frame^2) '
                'tracked {:.1f}, smoothed {:.1f}, pass 2 {:.1f}, final {:.1f}'.format(*report['jitter'][i]))
    return out, report
