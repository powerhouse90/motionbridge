"""Hold, grab and release events for first-person hands and their main prop.

Per hand and frame a hold score from the hand's contact with the prop (fingers wrapped round it, fingertips on
it, the palm on it), turned into held spans with hysteresis and minimum durations; then the event list (grab,
release, handoff, and the prop's own state: held, still, moving, falling), a parent schedule (which hand the
prop is attached to, with a constant hand-to-prop offset per span, so Blender can key a Child Of switch), and a
lock that holds each hand fixed on the prop over its spans (no slip), eased in and out at the boundaries.

Everything is in hand lengths (wrist to middle knuckle) or seconds, so it carries across rigs and frame rates.
Poses: prop motion (T, 4, 4) relative to its scene placement, as fp_track produces it; hands as fp_track.Rig
parameters (or their 21 keypoints and hand frames for the parts that need no rig)."""
import numpy as np
from scipy.spatial.transform import Rotation

from motionbridge.core import fp_track as fp

FPS = fp.FPS
TOUCH = 0.12        # palm lengths: a fingertip this close to the surface touches it
PALM_NEAR = 0.35    # palm lengths: the palm this close to the surface rests on it
HOLD_ON = 2.5       # hold score to start a hold (2 wrapped fingers and the palm, or 3 tips and the palm)
HOLD_OFF = 1.5      # ... and to end it: a hold survives a finger lifting for a moment
MIN_WRAP = 1        # a hold starts only round the prop: a hand hovering beside it (tips and palm near) doesn't
SETTLE = 0.15       # palm lengths: the hand within this of its settled pose on the prop has closed its grip ...
SETTLE_TURN = 0.2   # rad: ... and within this turn
MIN_HOLD = 0.25     # s: shorter holds are a brush past the prop
MIN_GAP = 0.2       # s: shorter gaps inside a hold are read noise, not a release
HANDOFF = 1.5       # s: the other hand grabs at most this long before this one lets go
STILL = 0.05        # m/s: the prop moving slower than this (and not turning) is still
STILL_TURN = 0.3    # rad/s
FALL = 0.5          # g: unheld and accelerating down at least this much is falling
PROP_MIN = 0.1      # s: shorter prop states are noise (a drop of a few cm lasts barely longer)
EASE = 0.25         # s: the lock eases on and off over this long at each end of a hold
UP = np.array([0.0, 0.0, 1.0])  # Blender's up
G = 9.81
S_ = np.linspace(0.15, 0.95, 9)  # samples along the palm-centre-to-fingertip line
TIPS = (4, 8, 12, 16, 20)


def _frames(seconds, fps):
    return max(1, int(round(seconds * fps)))


def palm_length(points):
    """Wrist to middle knuckle, median over frames: the hand's yardstick (curl doesn't change it)."""
    return float(np.median(np.linalg.norm(points[:, 9] - points[:, 0], axis=1)))


def hand_points(rig, params):
    return np.array([rig.keypoints(x) for x in params])


def hand_frames(rig, params, scaled=True):
    """Hand bone's world matrix per frame. scaled: as Blender's Child Of sees it (armature scale included)."""
    i = rig.sk.index[rig.hand.hand]
    return np.array([rig.sk.pose(rig.basis(x))[i] if scaled else rig.hand_frame(x) for x in params])


def contacts(points, prop, motion):
    """Per frame: fingers wrapped round the prop (the palm-centre-to-tip line passes through it), fingertips on
    its surface (not counting wrapped ones), and the palm's distance to it in palm lengths."""
    T = len(points)
    L = palm_length(points)
    wrapped, touching, palm = np.zeros(T, int), np.zeros(T, int), np.zeros(T)
    for t in range(T):
        p, M = points[t], motion[t]
        pc = p[[0, 5, 9, 13, 17]].mean(0)
        lines = np.array([pc + S_[:, None] * (p[k] - pc) for k in TIPS])
        through = prop.distance(lines.reshape(-1, 3), M).reshape(len(TIPS), -1).min(1) < 0
        tips = prop.distance(p[list(TIPS)], M) < TOUCH * L
        wrapped[t], touching[t] = through.sum(), (tips & ~through).sum()
        pts = np.array([p[0] + s * (p[j] - p[0]) for j in (5, 9, 13, 17) for s in (0.5, 1.0)])
        palm[t] = prop.distance(pts, M).min() / L
    return {'wrapped': wrapped, 'touching': touching, 'palm': palm, 'palm_length': L}


def hold_score(c):
    """A wrapped finger counts 1, a fingertip resting on the prop 0.5, the palm on it 1."""
    return c['wrapped'] + 0.5 * c['touching'] + (c['palm'] < PALM_NEAR)


def _runs(mask):
    """(start, end) of each run of True, end exclusive."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def held_spans(score, wrapped, fps=FPS, on=HOLD_ON, off=HOLD_OFF, min_hold=MIN_HOLD, min_gap=MIN_GAP):
    """Hysteresis on the hold score (starting needs a wrapped finger), then short gaps closed and short holds
    dropped: [(start, end)]."""
    held = np.zeros(len(score), bool)
    state = False
    for t, s in enumerate(score):
        state = (s >= on and wrapped[t] >= MIN_WRAP) if not state else s >= off
        held[t] = state
    gap = _frames(min_gap, fps)
    for a, b in _runs(~held):
        if a > 0 and b < len(held) and b - a < gap:
            held[a:b] = True
    return [(int(a), int(b)) for a, b in _runs(held) if b - a >= _frames(min_hold, fps)]


def prop_state(motion, centre, held, fps=FPS, up=UP):
    """Per frame: 'held', or, unheld, 'still' / 'falling' / 'moving' from the prop's centre path and turn."""
    c = motion[:, :3, :3] @ centre + motion[:, :3, 3]
    v = np.gradient(c, axis=0) * fps
    a = np.gradient(v, axis=0) * fps
    turn = np.r_[0, np.linalg.norm((Rotation.from_matrix(motion[1:, :3, :3]) *
                                    Rotation.from_matrix(motion[:-1, :3, :3]).inv()).as_rotvec(), axis=1)] * fps
    state = np.where((np.linalg.norm(v, axis=1) < STILL) & (turn < STILL_TURN), 'still',
                     np.where(a @ up < -FALL * G, 'falling', 'moving'))
    state = np.where(held, 'held', state)
    # a state shorter than this is noise: it takes its neighbour's
    n = _frames(PROP_MIN, fps)
    out = state.copy()
    starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
    ends = np.r_[starts[1:], len(out)]
    for a, b in zip(starts, ends):
        if b - a < n and a > 0:
            out[a:b] = out[a - 1]
    return out


def events(spans, state, fps=FPS):
    """spans: {hand: [(start, end)]}. Grab / release per hand, a handoff where one hand takes the prop from the
    other (it grabs while the other holds, and the other lets go soon after), and the prop's state changes."""
    ev = []
    for hand, ss in spans.items():
        for a, b in ss:
            ev.append({'frame': a, 'type': 'grab', 'hand': hand})
            if b < len(state):
                ev.append({'frame': b, 'type': 'release', 'hand': hand})
    window = _frames(HANDOFF, fps)
    for giver, gs in spans.items():
        for taker, ts in spans.items():
            if giver == taker:
                continue
            for _, gb in gs:
                for ta, tb in ts:
                    if ta < gb <= ta + window and tb > gb and gb < len(state):
                        ev.append({'frame': gb, 'type': 'handoff', 'from': giver, 'to': taker})
    for t in range(1, len(state)):
        if state[t] != state[t - 1]:
            ev.append({'frame': t, 'type': 'prop', 'state': str(state[t])})
    order = {'release': 0, 'handoff': 1, 'prop': 2, 'grab': 3}
    return sorted(ev, key=lambda e: (e['frame'], order[e['type']]))


def parent_schedule(spans, T):
    """Which hand the prop is attached to per frame (None: the world). The current parent keeps it while it
    holds (no flicker when both hands hold); then the hand that has held longest takes it."""
    holding = {h: np.zeros(T, bool) for h in spans}
    since = {h: np.full(T, T) for h in spans}
    for h, ss in spans.items():
        for a, b in ss:
            holding[h][a:b] = True
            since[h][a:b] = a
    parent, cur = [], None
    for t in range(T):
        if cur is None or not holding[cur][t]:
            live = [h for h in spans if holding[h][t]]
            cur = min(live, key=lambda h: since[h][t]) if live else None
        parent.append(cur)
    segs = []
    for t, p in enumerate(parent):
        if segs and segs[-1]['parent'] == p:
            segs[-1]['end'] = t + 1
        else:
            segs.append({'parent': p, 'start': t, 'end': t + 1})
    return parent, segs


def mean_pose(mats):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_matrix(mats[:, :3, :3]).mean().as_matrix()
    out[:3, 3] = np.median(mats[:, :3, 3], axis=0)
    return out


def blender_keys(segs, rel, names):
    """Keyframes for a prop with one Child Of per hand (target: the armature, subtarget: the hand bone, inverse
    identity): per frame each constraint's influence (constant interpolation) and the prop's matrix factor.
    names: {hand: bone name}."""
    T = len(rel)
    infl = {h: np.zeros(T) for h in names}
    for s in segs:
        if s['parent'] is not None:
            infl[s['parent']][s['start']:s['end']] = 1
    return {'bones': names, 'influence': {h: v.tolist() for h, v in infl.items()}, 'matrix': rel.tolist(),
            'segments': [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()} for s in segs]}


def _ramp(n):
    f = np.arange(1, n + 1) / (n + 1)
    return f * f * (3 - 2 * f)


def settle(Hf, motion, span, L):
    """A hold's settled part: where the hand's pose on the prop stays near its typical one (the reach-in and
    the let-go fall outside it). Hf: unscaled hand frames (T, 4, 4); L: palm length. -> (s0, e0, pose on prop)."""
    a, b = span
    rel = np.linalg.inv(motion[a:b]) @ Hf[a:b]
    n = b - a
    ref = mean_pose(rel[n // 4: max(n // 4 + 1, 3 * n // 4)])  # the middle of the hold
    for _ in range(3):
        off = np.linalg.norm(rel[:, :3, 3] - ref[:3, 3], axis=1) / L
        turn = (Rotation.from_matrix(rel[:, :3, :3]) * Rotation.from_matrix(ref[:3, :3]).inv()).magnitude()
        ok = np.flatnonzero((off < SETTLE) & (turn < SETTLE_TURN))
        if not len(ok):
            return a, a, ref
        ref = mean_pose(rel[ok[0]:ok[-1] + 1])
    return a + int(ok[0]), a + int(ok[-1]) + 1, ref


def weights(a, s0, e0, b, T, fps=FPS):
    """How much the lock applies over a hold: 0 -> 1 across the reach-in (at least EASE long, not at the clip's
    start), 1 while settled, 1 -> 0 across the let-go."""
    w = np.zeros(T)
    n = _frames(EASE, fps)
    w[s0:e0] = 1
    up = max(s0 - a, n if s0 > 0 else 0)
    if up:
        lo = max(0, s0 - up)
        w[lo:s0] = _ramp(s0 - lo)
    down = max(b - e0, n if e0 < T else 0)
    if down:
        hi = min(T, e0 + down)
        w[e0:hi] = _ramp(hi - e0)[::-1]
    return w


def lock(rig, params, motion, holds, fps=FPS):
    """Each hold's settled grip: the hand placed exactly at its pose on the prop, eased in across the reach and
    out across the let-go; the fingers stay as tracked. holds: [(a, s0, e0, b, pose on prop)]. -> new params."""
    P = params.copy()
    T = len(P)
    Hf = np.array([rig.hand_frame(x) for x in params])
    for a, s0, e0, b, ref in holds:
        w = weights(a, s0, e0, b, T, fps)
        for t in np.flatnonzero(w > 0):
            P[t] = fp._place_hand(rig, params[t], fp._blend(Hf[t], motion[t] @ ref, w[t]))
    return P


def slip(rig, params, motion, spans):
    """Over spans [(start, end)]: the hand's palm centre on the prop, frame to frame (mm/frame) and its spread
    from its mean over the span (mm, worst)."""
    step, spread = [], []
    for a, b in spans:
        if b - a < 2:
            continue
        pc = np.array([rig.keypoints(x)[[0, 5, 9, 13, 17]].mean(0) for x in params[a:b]])
        local = np.einsum('tji,tj->ti', motion[a:b, :3, :3], pc - motion[a:b, :3, 3])  # in the prop's frame
        step += list(np.linalg.norm(np.diff(local, axis=0), axis=1))
        spread.append(np.linalg.norm(local - local.mean(0), axis=1).max())
    return (float(np.mean(step) * 1000) if step else 0.0, float(max(spread) * 1000) if spread else 0.0)


def analyse(rigs, params, prop, motion, fps=FPS):
    """The whole pass for a take: contacts, holds (grab, settled, let-go), prop state, events, the parent
    schedule over the settled parts, and the prop's keys relative to its parent."""
    T = len(motion)
    pts = {r.hand.side: hand_points(r, P) for r, P in zip(rigs, params)}
    con = {h: contacts(p, prop, motion) for h, p in pts.items()}
    spans = {h: held_spans(hold_score(c), c['wrapped'], fps) for h, c in con.items()}
    holds, settled, scale = {}, {}, {}
    for r, P in zip(rigs, params):
        h = r.hand.side
        Hf = np.array([r.hand_frame(x) for x in P])
        holds[h] = []
        for a, b in spans[h]:
            s0, e0, ref = settle(Hf, motion, (a, b), con[h]['palm_length'])
            holds[h].append((a, s0, e0, b, ref))
        settled[h] = [(s0, e0) for _, s0, e0, _, _ in holds[h] if e0 > s0]
        Hs = hand_frames(r, P[:1])[0]
        scale[h] = np.linalg.inv(Hf[0]) @ Hs  # the bone's own scale (the armature's), as Child Of sees it
    held = np.zeros(T, bool)  # the prop is held while a hand has it in its settled grip (not while reaching / letting go)
    for ss in settled.values():
        for a, b in ss:
            held[a:b] = True
    state = prop_state(motion, prop.centre, held, fps)
    ev = events(spans, state, fps)
    for h, hs in holds.items():
        for a, s0, e0, b, _ in hs:
            if e0 > s0:
                ev += [{'frame': s0, 'type': 'attach', 'hand': h}] + ([{'frame': e0, 'type': 'detach', 'hand': h}] if e0 < T else [])
    order = {'release': 0, 'detach': 1, 'handoff': 2, 'prop': 3, 'grab': 4, 'attach': 5}
    ev.sort(key=lambda e: (e['frame'], order[e['type']]))
    parent, segs = parent_schedule(settled, T)
    rel = np.array(motion, float)
    for sg in segs:
        if sg['parent'] is None:
            continue
        ref = next(r_ for _, s0, e0, _, r_ in holds[sg['parent']] if s0 <= sg['start'] < e0)
        # the prop relative to the hand bone (scaled, as Blender sees it) once the hand is locked on it
        sg['offset'] = np.linalg.inv(ref @ scale[sg['parent']])
        sg['offset_from'] = 'settled grip'
        rel[sg['start']:sg['end']] = sg['offset']
        i = [r.hand.side for r in rigs].index(sg['parent'])
        tracked = hand_frames(rigs[i], params[i][sg['start']:sg['end']])
        at = lambda M: M[:, :3, :3] @ prop.centre + M[:, :3, 3]  # the prop's centre, not its far-off origin
        stray = np.linalg.norm(at(tracked @ sg['offset']) - at(motion[sg['start']:sg['end']]), axis=1)
        sg['stray_mm'] = (float(np.median(stray) * 1000), float(stray.max() * 1000))  # before the lock
    return {'contacts': con, 'spans': spans, 'holds': holds, 'state': state, 'events': ev, 'parent': parent,
            'segments': segs, 'rel': rel}
