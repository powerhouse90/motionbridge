"""Articulated main props: parts that move on the prop's body (a pistol's slide and trigger, a magazine that
comes out, a revolver's cylinder, a steering wheel, a laptop's lid).

The body's own motion comes from fp_points (M_t: scene placement -> frame t). Each part hangs off the body,
or off another part, by one joint:
  slide   translation along an axis, metres                       (slide, bolt, charging handle)
  hinge   rotation about an axis through a point, radians         (trigger, hammer, lid, cylinder, wheel)
  detach  a slide that lets go past its travel; from then on the part is free, 6 DoF in the world (magazine)
  fixed   moves with its parent
Axes, pivots and limits are in scene-placement world coordinates (frame 0), as the parts sit when the scene
is built, so a part's placement at frame t is M_t @ parent chain @ joint(q_t).

A part is seen per frame as image points (its rest-position 3D points and where they are on screen, e.g.
LK tracks as fp_points makes them) and/or a mask. Each joint value is found per frame from a grid over its
range (no local minimum across the travel; a step prior keeps symmetric parts such as a three-spoke wheel
from jumping between look-alike angles), then the whole curve is refined at once with its limits and an
acceleration penalty. A detaching part switches to free when it reaches the end of its travel or a free
fit explains it much better than the seated one; it stays free for the rest of the take.

Where joints come from (first that applies): an explicit description (JSON), Blender custom properties on
the part object (mb_joint, mb_axis, mb_origin, mb_limits), then name rules (KEYWORDS) with the axis and
pivot guessed from the geometry of the part and its parent.
"""
import json
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from motionbridge.core.hand_fit import project

JOINTS = ('slide', 'hinge', 'detach', 'fixed')
# name token -> joint, axis rule, pivot rule. Axis rules: 'long' the parent's longest extent, 'own' the part's
# longest extent (a slide runs along its own length, a magazine leaves along its length), 'thin' the part's
# thinnest extent (a wheel turns about its face normal), 'edge' the line the part touches its parent along
# (a lid's hinge), 'side' the parent's thinnest extent (a trigger pivots about the gun's sideways axis).
# Pivot rules: 'contact' where part and parent touch, 'centre' the part's centre.
KEYWORDS = [
    ('magazine', 'detach', 'own', 'contact'), ('mag', 'detach', 'own', 'contact'), ('clip', 'detach', 'own', 'contact'),
    ('battery', 'detach', 'own', 'contact'),
    ('slide', 'slide', 'own', 'contact'), ('bolt', 'slide', 'own', 'contact'), ('charging', 'slide', 'own', 'contact'),
    ('pump', 'slide', 'own', 'contact'), ('forend', 'slide', 'own', 'contact'), ('plunger', 'slide', 'own', 'contact'),
    ('trigger', 'hinge', 'side', 'contact'), ('hammer', 'hinge', 'side', 'contact'), ('lever', 'hinge', 'side', 'contact'),
    ('safety', 'hinge', 'side', 'contact'), ('lock', 'hinge', 'side', 'contact'),
    ('lid', 'hinge', 'edge', 'contact'), ('screen', 'hinge', 'edge', 'contact'), ('door', 'hinge', 'edge', 'contact'),
    ('cover', 'hinge', 'edge', 'contact'), ('stock', 'hinge', 'edge', 'contact'), ('flap', 'hinge', 'edge', 'contact'),
    ('cylinder', 'hinge', 'own', 'centre'), ('drum', 'hinge', 'own', 'centre'), ('wheel', 'hinge', 'thin', 'centre'),
    ('knob', 'hinge', 'thin', 'centre'), ('dial', 'hinge', 'thin', 'centre'), ('crank', 'hinge', 'thin', 'contact'),
]
# Default travel when nothing says otherwise: slides as a fraction of the part's length, hinges in radians
SLIDE_TRAVEL = 0.5
DETACH_TRAVEL = 1.0   # a seated part lets go once it has moved its own length
HINGE_RANGE = {'trigger': (-0.6, 0.6), 'hammer': (-1.2, 1.2), 'lever': (-1.6, 1.6), 'safety': (-1.6, 1.6),
               'lock': (-1.6, 1.6), 'lid': (-np.pi, np.pi), 'screen': (-np.pi, np.pi), 'door': (-2.2, 2.2),
               'cover': (-np.pi, np.pi), 'stock': (-np.pi, np.pi), 'flap': (-np.pi, np.pi)}
SPIN = 4 * np.pi      # wheels, cylinders, knobs: two turns either way
CONTACT = 0.004       # m: part points this close to the parent touch it

# Tracking: expected size of each term (residuals are divided by these)
PX_SIGMA = 1.5        # px, a tracked image point
MASK_SIGMA = 2.0      # px, the part's outline against its mask
PER_FRAME = 20.0      # a frame's observation counts as this many measurements, however many points it has
STEP_SIGMA = {'slide': 0.02, 'hinge': 0.35, 'detach': 0.03}   # per frame, the per-frame search's step prior
ACC_SIGMA = {'slide': 0.006, 'hinge': 0.08, 'detach': 0.006}  # per frame^2, the whole-curve smoothness
FREE_ACC = (0.08, 0.006)   # rad, m per frame^2 for a free part
BODY_SIGMA = (np.radians(1.5), 0.008)  # rad, m: how far the body's own track may be off (fp_points: ~1 cm, a few deg)
FREE_STEP = (0.3, 0.03)    # rad, m per frame: how far a free part moves between frames (per-frame search prior)
GRID = 61             # per-frame search: values tried across the range
DETACH_RATIO = 0.35   # free cost below this fraction of the seated cost (at the end of travel) -> it let go
MASK_SAMPLES = 1500   # part points used against a mask (dense enough that its outline is covered)
OUTLINE_SAMPLES = 200 # mask outline pixels used for coverage


class Part:
    """One moving part: rest points (world, scene placement), its joint and its parent ('' = the body)."""

    def __init__(self, name, points, joint, axis=None, origin=None, limits=None, parent='', release=None):
        assert joint in JOINTS, joint
        self.name, self.points, self.joint, self.parent = name, np.asarray(points, float), joint, parent
        self.axis = None if axis is None else np.asarray(axis, float) / np.linalg.norm(axis)
        self.origin = self.points.mean(0) if origin is None else np.asarray(origin, float)
        self.limits = (-np.inf, np.inf) if limits is None else tuple(float(v) for v in limits)
        self.release = release  # detach: the joint value at which it lets go

    def move(self, q):
        """4x4 of this part's joint at value q, in scene-placement coordinates."""
        D = np.eye(4)
        if self.joint in ('slide', 'detach'):
            D[:3, 3] = self.axis * q
        elif self.joint == 'hinge':
            R = Rotation.from_rotvec(self.axis * q).as_matrix()
            D[:3, :3], D[:3, 3] = R, self.origin - R @ self.origin
        return D

    def describe(self):
        return {'name': self.name, 'joint': self.joint, 'parent': self.parent,
                'axis': None if self.axis is None else self.axis.tolist(), 'origin': self.origin.tolist(),
                'limits': list(self.limits), 'release': self.release}


def _extent_axes(points):
    """Principal axes of a point cloud, longest first, and the extent along each."""
    c = points - points.mean(0)
    axes = np.linalg.svd(c, full_matrices=False)[2]
    return axes, np.ptp(c @ axes.T, axis=0)


def _contact(part, parent):
    """Part points touching the parent (or the closest few when none quite touch)."""
    d, _ = cKDTree(parent).query(part)
    near = d < max(CONTACT, np.sort(d)[min(len(d) - 1, 20)])
    return part[near]


def guess(name, points, parent_points, parent=''):
    """A Part from its name and geometry alone, or None when the name says nothing (it is then fixed)."""
    low = name.lower()
    rule = next((k for k in KEYWORDS if k[0] in low), None)
    if rule is None:
        return None
    word, joint, axis_rule, pivot_rule = rule
    own, own_ext = _extent_axes(points)
    par, _ = _extent_axes(parent_points)
    touch = _contact(points, parent_points)
    if axis_rule == 'long':
        axis = par[0]
    elif axis_rule == 'own':
        axis = own[0]
    elif axis_rule == 'thin':
        axis = own[2]
    elif axis_rule == 'side':
        axis = par[2]
    else:  # 'edge': the touching strip's long direction
        axis = _extent_axes(touch)[0][0] if len(touch) > 3 else par[0]
    if pivot_rule == 'contact' and len(touch):
        origin = touch.mean(0)
    else:  # a spinning part turns about the middle of its extent (an odd spoke pulls the mean off it)
        c = points - points.mean(0)
        x = c @ own.T
        origin = points.mean(0) + own.T @ ((x.max(0) + x.min(0)) / 2)
    length = float(np.ptp((points - points.mean(0)) @ axis))
    if joint == 'slide':
        limits = (-SLIDE_TRAVEL * length, SLIDE_TRAVEL * length)
    elif joint == 'detach':
        # leaving the parent: the direction that takes the part's centre away from the parent's centre
        if (points.mean(0) - parent_points.mean(0)) @ axis < 0:
            axis = -axis
        limits = (-0.1 * length, DETACH_TRAVEL * length)
    elif word in HINGE_RANGE:
        limits = HINGE_RANGE[word]
    else:
        limits = (-SPIN, SPIN)
    p = Part(name, points, joint, axis, origin, limits, parent)
    if joint == 'detach':
        p.release = limits[1]
    return p


class PropParts:
    """The body and its moving parts, parents before children."""

    def __init__(self, body, parts):
        self.body = np.asarray(body, float)
        self.parts = _ordered(parts)

    def __getitem__(self, name):
        return next(p for p in self.parts if p.name == name)

    @classmethod
    def from_item(cls, npz, spec=None):
        """From a scene item (blender/fp_scene.py item_k.npz: names, v_i). The largest piece is the body
        unless spec names it; spec (dict or JSON path) overrides any part's joint, axis, origin, limits."""
        d = np.load(npz)
        pieces = {str(n): d[f'v_{i}'] for i, n in enumerate(d['names'])}
        return cls.from_pieces(pieces, spec)

    @classmethod
    def from_pieces(cls, pieces, spec=None):
        if isinstance(spec, str):
            spec = json.load(open(spec))
        spec = spec or {}
        given = {p['name']: p for p in spec.get('parts', [])}
        # the body: the largest piece whose name doesn't say it moves (a lid can be bigger than its base)
        still = [n for n in pieces if not any(k[0] in n.lower() for k in KEYWORDS)] or list(pieces)
        body = spec.get('body') or max(still, key=lambda n: np.ptp(pieces[n], axis=0).prod())
        parts, fixed = [], [pieces[body]]
        for name, v in pieces.items():
            if name == body:
                continue
            if name in given:
                g = given[name]
                parent = g.get('parent', '')
                parts.append(Part(name, v, g['joint'], g.get('axis'), g.get('origin'), g.get('limits'), parent,
                                  g.get('release')))
                if g['joint'] == 'fixed':
                    fixed.append(v)
                continue
            p = guess(name, v, pieces[body])
            if p is None:
                fixed.append(v)  # unnamed pieces ride on the body
            else:
                parts.append(p)
        return cls(np.concatenate(fixed), [p for p in parts if p.joint != 'fixed'])

    def describe(self):
        return {'parts': [p.describe() for p in self.parts]}


def _ordered(parts):
    out, names = [], set()
    while len(out) < len(parts):
        ready = [p for p in parts if p not in out and (p.parent in names or p.parent == '')]
        assert ready, 'part parents form a loop or name a missing part'
        for p in ready:
            out.append(p)
            names.add(p.name)
    return out


# --- observations ------------------------------------------------------------------------------------------

def mask_data(mask, rng=None):
    """What the tracker needs from one frame's part mask: distance outside it and outline pixels."""
    if mask is None or not mask.any():
        return None
    rng = rng or np.random.default_rng(0)
    out = ndimage.distance_transform_edt(~mask)
    edge = np.argwhere(mask & ~ndimage.binary_erosion(mask))[:, ::-1].astype(float)
    if len(edge) > OUTLINE_SAMPLES:
        edge = edge[rng.choice(len(edge), OUTLINE_SAMPLES, replace=False)]
    return {'out': out, 'edge': edge}


def _residual(sample, D, cam, points=None, mask=None):
    """One frame's misfit of a part placed by D: its tracked points, then its outline against the mask."""
    r = []
    if points is not None and len(points[0]):
        X, uv = points
        e = (project(X @ D[:3, :3].T + D[:3, 3], cam) - uv) / PX_SIGMA
        r.append(e.ravel() * np.sqrt(PER_FRAME / len(X)))
    if mask is not None:
        q = project(sample @ D[:3, :3].T + D[:3, 3], cam)
        H, W = mask['out'].shape
        inside = ndimage.map_coordinates(mask['out'], [np.clip(q[:, 1], 0, H - 1), np.clip(q[:, 0], 0, W - 1)], order=1)
        framed = (q[:, 0] >= 0) & (q[:, 0] <= W - 1) & (q[:, 1] >= 0) & (q[:, 1] <= H - 1)
        a = np.where(framed, inside, 0) / MASK_SIGMA * np.sqrt(PER_FRAME / len(q))  # off screen: unseen, not wrong
        b = cKDTree(q).query(mask['edge'])[0] / MASK_SIGMA * np.sqrt(PER_FRAME / len(mask['edge']))
        r += [a, b]
    return np.concatenate(r) if r else np.zeros(0)


# --- tracking ----------------------------------------------------------------------------------------------

def track_parts(parts, motion, cam, points=None, masks=None, log=print):
    """Every part's joint curve over the take.
    parts: PropParts. motion: (T, 4, 4) body motion (fp_points). points: {name: [None | (X (K,3) rest
    points, uv (K,2) pixels)] * T}. masks: {name: (T, H, W) bool}. Returns {name: {'value': (T,) joint
    values (nan while free), 'free': (T, 4, 4) placement while free (identity rows elsewhere), 'is_free':
    (T,) bool, 'world': (T, 4, 4) the part's placement relative to the scene}}."""
    points, masks = points or {}, masks or {}
    T = len(motion)
    rng = np.random.default_rng(0)
    out = {}
    for p in parts.parts:
        res = _observer(p.points, points.get(p.name), masks.get(p.name), cam, T, rng)
        base = np.array([motion[t] @ (out[p.parent]['world'][t] if p.parent else np.eye(4)) for t in range(T)])
        seen = np.array([res(t, base[t]).size > 0 for t in range(T)])
        q, is_free, free = _per_frame(p, base, res, seen, T)
        q = _refine(p, base, res, seen, q, ~is_free)
        if is_free.any():
            free = _refine_free(free, res, is_free, p.points.mean(0))
        world = np.array([free[t] if is_free[t] else base[t] @ p.move(q[t]) for t in range(T)])
        q = np.where(is_free, np.nan, q)
        if log:
            fr = np.flatnonzero(is_free)
            log(f'  part {p.name} ({p.joint}): {np.nanmin(q):.3f} .. {np.nanmax(q):.3f}'
                + (f', free from frame {fr[0]}' if len(fr) else ''))
        out[p.name] = {'value': q, 'free': free, 'is_free': is_free, 'world': world}
    return out


def _observer(rest, points, masks, cam, T, rng):
    """res(t, D): one frame's misfit of a piece with rest points `rest` placed by D."""
    pts = points or [None] * T
    md = [mask_data(masks[t], rng) if masks is not None else None for t in range(T)]
    sample = rest if len(rest) <= MASK_SAMPLES else rest[rng.choice(len(rest), MASK_SAMPLES, replace=False)]
    return lambda t, D: _residual(sample, D, cam, pts[t], md[t])


def refine_body(parts, motion, tracked, cam, points=None, masks=None, body_points=None, body_mask=None):
    """The body's track corrected frame by frame together with its seated parts' joints: the articulated prop
    fitted as one, so the parts' own points pin the body too and body error stops leaking into the joints.
    Run track_parts again on the returned motion. body_points / body_mask: the body's own observations, in
    the track_parts format. Returns the corrected (T, 4, 4) motion."""
    points, masks = points or {}, masks or {}
    T = len(motion)
    rng = np.random.default_rng(0)
    body = _observer(parts.body, body_points, body_mask, cam, T, rng)
    obs = {p.name: _observer(p.points, points.get(p.name), masks.get(p.name), cam, T, rng) for p in parts.parts}
    c0 = parts.body.mean(0)
    sig = np.r_[np.full(3, BODY_SIGMA[0]), np.full(3, BODY_SIGMA[1])]
    out = motion.copy()
    for t in range(T):
        seated = [p for p in parts.parts if not tracked[p.name]['is_free'][t]]
        q0 = np.array([tracked[p.name]['value'][t] for p in seated])
        c = motion[t][:3, :3] @ c0 + motion[t][:3, 3]

        def place(v):
            M = _pose(v[:6], motion[t], c)
            world = {}
            for p, q in zip(seated, v[6:]):
                parent = world.get(p.parent, np.eye(4)) if p.parent else np.eye(4)
                world[p.name] = parent @ p.move(q)
            return M, world

        def f(v):
            M, world = place(v)
            r = [body(t, M), v[:6] / sig, (v[6:] - q0) / np.array([STEP_SIGMA[p.joint] for p in seated])]
            r += [obs[p.name](t, M @ world[p.name]) for p in seated]
            r += [obs[p.name](t, tracked[p.name]['free'][t]) for p in parts.parts if p not in seated]
            return np.concatenate(r)
        lo = np.r_[np.full(6, -np.inf), [p.limits[0] for p in seated]]
        hi = np.r_[np.full(6, np.inf), [p.limits[1] for p in seated]]
        v0 = np.r_[np.zeros(6), np.clip(q0, lo[6:] + 1e-9, hi[6:] - 1e-9)]
        s = least_squares(f, v0, bounds=(lo, hi), diff_step=1e-4, loss='soft_l1', f_scale=3)
        out[t] = place(s.x)[0]
    return out


def _cost(r):
    return float(r @ r) if r.size else 0.0


def _per_frame(p, base, res, seen, T):
    lo, hi = p.limits
    grid = np.linspace(lo, hi, GRID)
    q, is_free, free = np.zeros(T), np.zeros(T, bool), np.repeat(np.eye(4)[None], T, 0)
    centre = p.points.mean(0)
    prev = float(np.clip(0, lo, hi))  # frame 0 is the scene as built: every joint at rest
    for t in range(T):
        if is_free[t - 1] if t else False:  # once let go, it stays free
            is_free[t] = True
            free[t] = _fit_free(res, t, free[t - 1], centre)
            continue
        if not seen[t]:
            q[t] = prev
            continue
        step = lambda v: ((v - prev) / STEP_SIGMA[p.joint]) ** 2
        costs = [_cost(res(t, base[t] @ p.move(v))) + step(v) for v in grid]
        v0 = grid[int(np.argmin(costs))]
        f = lambda v: np.r_[res(t, base[t] @ p.move(v[0])), [(v[0] - prev) / STEP_SIGMA[p.joint]]]
        s = least_squares(f, [np.clip(v0, lo + 1e-9, hi - 1e-9)], bounds=([lo], [hi]), diff_step=1e-4)
        q[t] = prev = s.x[0]
        if p.joint == 'detach':
            seated = _cost(res(t, base[t] @ p.move(q[t])))
            at_end = q[t] >= p.release - 0.05 * (p.release - lo)
            if at_end or t and q[t] > 0.5 * p.release:
                F = _fit_free(res, t, base[t] @ p.move(q[t]), centre)
                if at_end or _cost(res(t, F)) < DETACH_RATIO * seated:
                    is_free[t], free[t] = True, F
    return q, is_free, free


def _pose(v, ref, c):
    """A free part's placement: ref turned by rotvec v[:3] about world point c, then moved by v[3:]."""
    D = np.eye(4)
    R = Rotation.from_rotvec(v[:3]).as_matrix()
    D[:3, :3], D[:3, 3] = R, c - R @ c + v[3:]
    return D @ ref


def _fit_free(res, t, ref, centre):
    c = ref[:3, :3] @ centre + ref[:3, 3]  # turn about the part's own centre: turn and shift stay apart
    step = np.r_[np.full(3, FREE_STEP[0]), np.full(3, FREE_STEP[1])]
    s = least_squares(lambda v: np.r_[res(t, _pose(v, ref, c)), v / step], np.zeros(6), diff_step=1e-4, loss='soft_l1', f_scale=3)
    return _pose(s.x, ref, c)


def _refine(p, base, res, seen, q, on):
    """The whole seated curve at once: per-frame fits plus an acceleration penalty, within the limits."""
    idx = np.flatnonzero(on)
    if len(idx) < 3:
        return q
    lo, hi = p.limits
    n = [res(t, base[t] @ p.move(q[t])).size for t in idx]
    acc = ACC_SIGMA[p.joint]

    def f(v):
        x = q.copy(); x[idx] = v
        r = [res(t, base[t] @ p.move(x[t])) for t in idx]
        return np.concatenate(r + [np.diff(v, 2) / acc])
    J = lil_matrix((sum(n) + len(idx) - 2, len(idx)))
    row = 0
    for k, m in enumerate(n):
        J[row:row + m, k] = 1; row += m
    for k in range(len(idx) - 2):
        J[row + k, k:k + 3] = 1
    s = least_squares(f, np.clip(q[idx], lo + 1e-9, hi - 1e-9), bounds=(lo, hi), jac_sparsity=J, diff_step=1e-4,
                      loss='soft_l1', f_scale=3, max_nfev=60)
    out = q.copy(); out[idx] = s.x
    return out


def _refine_free(free, res, is_free, centre):
    """The free stretch at once: per-frame fits plus an acceleration penalty on turn and position."""
    idx = np.flatnonzero(is_free)
    if len(idx) < 3:
        return free
    ref = free[idx[0]].copy()
    c = ref[:3, :3] @ centre + ref[:3, 3]

    def start(D):  # the v with _pose(v, ref, c) == D
        R = D[:3, :3] @ ref[:3, :3].T
        return np.r_[Rotation.from_matrix(R).as_rotvec(), D[:3, 3] - (R @ ref[:3, 3] + c - R @ c)]
    v0 = np.array([start(free[t]) for t in idx])
    n = [res(t, free[t]).size for t in idx]
    sig = np.r_[np.full(3, FREE_ACC[0]), np.full(3, FREE_ACC[1])]

    def f(v):
        v = v.reshape(-1, 6)
        r = [res(t, _pose(v[k], ref, c)) for k, t in enumerate(idx)]
        return np.concatenate(r + [(np.diff(v, 2, axis=0) / sig).ravel()])
    J = lil_matrix((sum(n) + 6 * (len(idx) - 2), 6 * len(idx)))
    row = 0
    for k, m in enumerate(n):
        J[row:row + m, 6 * k:6 * k + 6] = 1; row += m
    for k in range(len(idx) - 2):
        for j in range(6):
            J[row + 6 * k + j, [6 * k + j, 6 * (k + 1) + j, 6 * (k + 2) + j]] = 1
    s = least_squares(f, v0.ravel(), jac_sparsity=J, diff_step=1e-4, loss='soft_l1', f_scale=3, max_nfev=40)
    out = free.copy()
    for k, t in enumerate(idx):
        out[t] = _pose(s.x.reshape(-1, 6)[k], ref, c)
    return out


# --- keying ------------------------------------------------------------------------------------------------

def keying(parts, tracked, rest_world=None):
    """What fp_apply keys for each part object: its world matrix per frame (part placement relative to the
    scene, times the object's rest matrix_world), and the joint value per frame for rigs whose part is a
    bone. For an object parented in Blender, local = inverse(parent's keyed world) @ world.
    rest_world: {name: 4x4} the objects' matrix_world in the scene (identity when omitted).
    Returns [{'object', 'parent', 'joint', 'axis', 'origin', 'value' (T,), 'world' (T,4,4), 'free' (T,)}]."""
    rest_world = rest_world or {}
    out = []
    for p in parts.parts:
        tr = tracked[p.name]
        W = tr['world'] @ np.asarray(rest_world.get(p.name, np.eye(4)))
        out.append({'object': p.name, 'parent': p.parent, 'joint': p.joint,
                    'axis': None if p.axis is None else p.axis.tolist(), 'origin': p.origin.tolist(),
                    'value': tr['value'], 'world': W, 'free': tr['is_free']})
    return out


def save_keys(path, keys):
    """keying() output as one npz fp_apply can read: per part world_<name>, value_<name>, free_<name>."""
    np.savez(path, names=np.array([k['object'] for k in keys]),
             meta=json.dumps([{x: k[x] for x in ('object', 'parent', 'joint', 'axis', 'origin')} for k in keys]),
             **{f'world_{i}': k['world'] for i, k in enumerate(keys)},
             **{f'value_{i}': k['value'] for i, k in enumerate(keys)},
             **{f'free_{i}': k['free'] for i, k in enumerate(keys)})


# Blender side, for fp_apply (runs inside Blender; kept here so the format lives in one place):
BLENDER_KEYS = '''
import bpy, json, numpy as np
from mathutils import Matrix
def key_parts(npz_path, frame_start=1):
    d = np.load(npz_path)
    for i, name in enumerate(d['names']):
        o = bpy.data.objects.get(str(name))
        if o is None: continue
        W = d[f'world_{i}']
        for t in range(len(W)):
            o.matrix_world = Matrix(W[t].tolist())
            o.keyframe_insert('location', frame=frame_start + t)
            o.keyframe_insert('rotation_euler' if o.rotation_mode not in ('QUATERNION', 'AXIS_ANGLE') else 'rotation_quaternion',
                              frame=frame_start + t)
'''


def blender_dump(root_name):
    """Inside Blender: the root's mesh pieces (world vertices) and any mb_* custom properties, in the
    from_pieces(spec=...) format. Returns (pieces, spec)."""
    import bpy  # only inside Blender
    root = bpy.data.objects[root_name]
    pieces, spec = {}, {'parts': []}
    for o in [root] + list(root.children_recursive):
        if o.type != 'MESH':
            continue
        M = np.array(o.matrix_world)
        v = np.array([tuple(x.co) for x in o.data.vertices])
        pieces[o.name] = v @ M[:3, :3].T + M[:3, 3]
        if 'mb_joint' in o:
            spec['parts'].append({'name': o.name, 'joint': str(o['mb_joint']),
                                  'axis': list(o.get('mb_axis', [])) or None, 'origin': list(o.get('mb_origin', [])) or None,
                                  'limits': list(o.get('mb_limits', [])) or None,
                                  'parent': o.parent.name if o.parent and o.parent != root else ''})
    return pieces, spec
