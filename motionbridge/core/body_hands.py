"""Fingers for the full-body pipeline: SAM 3D Body's hand reads (gem/body_hands_extract.py) onto any rig's
finger bones, on top of a retargeted motion.

The body and wrist stay exactly as retargeted from GEM. Only the finger joints move, and only by the hand's
shape: each read hand and the rig's hand are expressed in their own palm frame (wrist -> middle base, across the
knuckles), so the fit needs neither the camera nor the hand's world turn, and a hand read in camera space
fits a rig standing anywhere. The hand's size and proportions don't matter either: only bone directions are
compared.

Rig side: hands.find_hands (finger chains by shape, not names) and hands.HandModel (bend axes from the palm's
geometry), so any rig with four or five finger chains works. Leading palm bones (metacarpals) keep the
retarget's pose. Per finger 4 numbers (base bend, spread, middle bend, tip bend) relative to the rig's rest.

    obs = body_hands.load_reads('hands.npz')              # {side: (hand21 (T, 21, 3), spread (T,))}
    fits = body_hands.fit(dump, result, obs)              # {side: (Hand, params (T, P), used (T, fingers))}
    body_hands.apply(result, dump, fits)                  # the retarget Result, fingers replaced
"""
import numpy as np
from scipy.optimize import least_squares

from motionbridge.core import hands
from motionbridge.core.smooth import _speed, one_euro, settings

# The 21 points (MediaPipe order) as segments along each finger, the whole thumb included
SEG = [(0, 1), (1, 2), (2, 3), (3, 4)] + [(a, a + 1) for f in range(1, 5) for a in (1 + 4 * f, 2 + 4 * f, 3 + 4 * f)]
DIR_SIGMA = 0.15      # unit-vector difference per segment: a body-video hand is a small crop, its read is coarse
STEP_SIGMA = 0.35     # rad a finger value may change between frames
COUPLE = 0.7          # tip bend / middle bend, as real fingers move (hand_fit.py)
COUPLE_SIGMA = 0.25
HOLD_SIGMA = 0.03     # a finger that isn't read this frame stays where it was
SPREAD_BAD = 0.12     # box disagreement (fraction of the hand on screen) above which a read is not used
GATE, GATE_MIN = 2.5, 0.35  # a finger is dropped when its error is GATE x the hand's median and above GATE_MIN
STRENGTH = 0.6        # smoothing (smooth.settings)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def palm_frame(p):
    """(3, 3) columns: across the knuckles, wrist -> middle base, out of the plane. Built the same way for
    the read hand and the rig's hand, so a left hand compares with a left hand."""
    y = _unit(p[9] - p[0])
    z = _unit(np.cross(p[5] - p[17], y))
    return np.column_stack([np.cross(y, z), y, z])


def shape(p):
    """Finger segment directions in the palm's own frame: (len(SEG), 3); NaN rows where points are missing."""
    d = _unit(np.array([p[b] - p[a] for a, b in SEG]))
    return d @ palm_frame(p)


def load_reads(path):
    """body_hands_extract.py output -> {side: (hand21 (T, 21, 3), spread (T,))}."""
    d = np.load(path)
    return {s: (d[f'hand3d_{s}'].astype(np.float64), d[f'spread_{s}'].astype(np.float64))
            for s in ('Right', 'Left') if f'hand3d_{s}' in d.files}


class HandSolver:
    """One rig hand as a function of its finger numbers. Forward kinematics runs on the hand's own subtree
    (the hand bone as root), which is all the palm-frame shape depends on."""

    def __init__(self, dump, hand):
        names = [b['name'] for b in dump['bones']]
        keep, frontier = {hand.hand}, [hand.hand]
        children = {}
        for b in dump['bones']:
            children.setdefault(b['parent'], []).append(b['name'])
        while frontier:
            n = frontier.pop()
            for c in children.get(n, []):
                keep.add(c); frontier.append(c)
        bones = [dict(b, parent=None if b['name'] == hand.hand else b['parent']) for b in dump['bones'] if b['name'] in keep]
        self.hand = hand
        self.rows = [names.index(b['name']) for b in bones]  # subtree bone -> row in the full rig
        self.sk = hands.Skeleton({'bones': bones})
        self.model = hands.HandModel(self.sk, hand)
        self.joints = [self.sk.index[n] for f in hands.FINGERS if f in hand.fingers for n in hand.fingers[f]]
        lo, hi = hand.limits()
        self.lo, self.hi = lo[3:], hi[3:]
        self.size = hand.size - 3
        self.slices = {}
        k = 0
        for f in hands.FINGERS:
            if f in hand.fingers:
                self.slices[f] = (k, [SEG.index(s) for s in SEG if 1 + 4 * hands.FINGERS.index(f) <= s[0] <= 4 + 4 * hands.FINGERS.index(f) or (f == 'Thumb' and s[0] == 0)])
                k += hands.PER_FINGER

    def set_pose(self, full_basis=None):
        """Palm bones (and anything else in the hand that isn't a finger joint) as the retarget posed them;
        finger joints at rest, so the numbers are absolute (relative to the rig's rest)."""
        b0 = np.repeat(np.eye(4)[None], len(self.rows), 0) if full_basis is None else full_basis[self.rows].copy()
        b0[self.joints] = np.eye(4)
        b0[self.sk.index[self.hand.hand]] = np.eye(4)  # the hand's own turn doesn't change its shape
        self.model.basis0 = b0

    def keypoints(self, xf):
        return self.model.keypoints(np.r_[0., 0., 0., xf])

    def basis(self, xf):
        """Finger joints' local basis (n, 4, 4) for these numbers, in self.joints order."""
        return self.model.basis(np.r_[0., 0., 0., xf])[self.joints]


def fit_hand(solver, reads, spread=None, basis=None, log=None):
    """reads: (T, 21, 3) read hand points (any frame, any scale); spread: (T,) read confidence; basis: (T, B,
    4, 4) the retargeted local basis of the whole rig (palm bones) or None. -> params (T, P), used (T, fingers)."""
    T = len(reads)
    fingers = list(solver.slices)
    params, used = np.zeros((T, solver.size)), np.zeros((T, len(fingers)), bool)
    prev = np.zeros(solver.size)
    for t in range(T):
        solver.set_pose(None if basis is None else basis[t])
        target = shape(reads[t]) if np.isfinite(reads[t]).all() else None
        if target is None or (spread is not None and not spread[t] <= SPREAD_BAD):
            params[t] = prev  # nothing usable: hold
            continue
        active = {f: True for f in fingers}

        def residual(x):
            d = shape(solver.keypoints(x)) - target
            r = []
            for f, (k, rows) in solver.slices.items():
                if active[f]:
                    r.append(d[rows].ravel() / DIR_SIGMA)
                else:
                    r.append((x[k:k + 4] - prev[k:k + 4]) / HOLD_SIGMA)
                if f != 'Thumb':
                    r.append([(x[k + 3] - COUPLE * x[k + 2]) / COUPLE_SIGMA])
            r.append((x - prev) / STEP_SIGMA)
            return np.concatenate(r)

        x0 = np.clip(prev, solver.lo + 1e-6, solver.hi - 1e-6)
        x = least_squares(residual, x0, bounds=(solver.lo, solver.hi), max_nfev=60, loss='soft_l1').x
        d = np.linalg.norm(shape(solver.keypoints(x)) - target, axis=1)
        err = {f: d[rows].mean() for f, (_, rows) in solver.slices.items()}
        med = np.median(list(err.values()))
        bad = [f for f, e in err.items() if e > GATE * med and e > GATE_MIN]
        if bad:  # a finger the read got wrong (hidden, swapped) holds instead of twisting to chase it
            for f in bad:
                active[f] = False
            x = least_squares(residual, np.clip(x, solver.lo + 1e-6, solver.hi - 1e-6), bounds=(solver.lo, solver.hi),
                              max_nfev=60, loss='soft_l1').x
        params[t], prev = x, x
        used[t] = [active[f] for f in fingers]
        if log and t % 20 == 0:
            log(f'{solver.hand.side} hand: frame {t}/{T}')
    return params, used


def smooth(params, fps, strength=STRENGTH):
    """Zero-phase adaptive low-pass per finger (smooth.one_euro)."""
    if len(params) < 5 or strength <= 0:
        return params
    min_cut, beta_rot, _ = settings(strength)
    out = params.copy()
    for k in range(0, params.shape[1], hands.PER_FINGER):
        v = params[:, k:k + hands.PER_FINGER]
        out[:, k:k + hands.PER_FINGER] = one_euro(v, _speed(v, fps), fps, min_cut, beta_rot)
    return out


def result_basis(result):
    """retarget.Result -> (T, B, 4, 4) local basis matrices."""
    from scipy.spatial.transform import Rotation
    T, B = result.quaternions.shape[:2]
    q = result.quaternions.reshape(-1, 4)
    m = np.repeat(np.eye(4)[None], T * B, 0)
    m[:, :3, :3] = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    m = m.reshape(T, B, 4, 4)
    m[..., :3, 3] = result.locations
    return m


def fit(dump, result, reads, fps=None, strength=STRENGTH, log=None):
    """dump: dump_rig JSON of the character; result: its retarget.Result (or None to fit on the rest pose);
    reads: load_reads(...). -> {side: (Hand, params (T, P), used)} for each side found on both."""
    basis = None if result is None else result_basis(result)
    fps = fps or (result.fps if result is not None else 24.0)
    out = {}
    for hand in hands.find_hands(dump):
        if hand.side not in reads:
            continue
        pts, spread = reads[hand.side]
        if basis is not None and len(basis) != len(pts):
            raise ValueError(f'{len(pts)} hand reads for a {len(basis)}-frame motion')
        solver = HandSolver(dump, hand)
        P, used = fit_hand(solver, pts, spread, basis, log)
        out[hand.side] = (hand, smooth(P, fps, strength), used)
        if log:
            log(f'{hand.side} hand: fitted {len(P)} frames, fingers read on {used.mean() * 100:.0f}% of finger-frames')
    return out


def apply(result, dump, fits):
    """Write the fitted finger joints into result (in place): quaternions of those bones, then heads/tails of
    every bone below each hand by forward kinematics. Everything else is untouched."""
    from scipy.spatial.transform import Rotation
    names = result.names
    basis = result_basis(result)
    sk = hands.Skeleton(dump)
    changed, below = set(), set()  # finger joints; they and every bone under them (their points move)
    for side, (hand, P, _) in fits.items():
        solver = HandSolver(dump, hand)
        rows = [names.index(solver.sk.names[j]) for j in solver.joints]
        for t in range(len(P)):
            solver.set_pose(basis[t])
            basis[t, rows] = solver.basis(P[t])
        changed |= set(rows)
        for j in range(len(solver.rows)):  # subtree bones under a finger joint
            k = j
            while k is not None and k not in solver.joints:
                k = solver.sk.parent[k]
            if k is not None:
                below.add(solver.rows[j])
    rows = sorted(changed)
    q = Rotation.from_matrix(basis[:, rows, :3, :3].reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]]
    q = q.reshape(len(basis), len(rows), 4)
    q *= np.where(q[..., :1] < 0, -1, 1)
    for t in range(1, len(q)):  # no sign flips between frames (as retarget.py)
        flip = np.einsum('bi,bi->b', q[t], q[t - 1]) < 0
        q[t, flip] *= -1
    result.quaternions[:, rows] = q
    for t in range(len(basis)):
        pose = sk.pose(basis[t])
        for i in sorted(below):
            result.heads[t, i] = sk.point(pose, names[i], 'head')
            result.tails[t, i] = sk.point(pose, names[i], 'tail')
    return result
