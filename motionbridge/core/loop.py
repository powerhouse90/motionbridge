"""Loops: one cycle of a take, made seamless, for walk / run / idle clips a game plays on repeat.

A cycle is frames S..E-1 of the take; frame E is where it would start again. Its first `blend` frames fade from
the take's own continuation past the cut (frames E.., moved back by the cycle's travel) into frames S.., so the cut
runs on into real motion and eases into the start: every frame is the take's, the fade only mixes two of its own
poses. S and E are searched (or given): the pair whose `blend`-frame windows match best.

The cycle is laid down several times after the take's frames before S (its lead-in, so the take's first frame
stays first: the retarget's start-pose match needs it). The retarget cuts out a middle cycle again after foot
planting and travel (see retarget.retarget's `loop`), so the seam is clean on the character too: the cycles
around it give the planting and smoothing the same context on both sides of the seam.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from motionbridge.core.soma import SomaMotion

BLEND = 8              # frames of crossfade at the seam
CYCLE = (0.6, 4.0)     # s: shortest and longest cycle searched
REPEATS = 5            # cycles laid down; the retarget keeps the middle one
TRAVELS = 0.2          # m/s: a cycle that moves faster than this travels (walk, run): each repeat carries on from
                       # the last. Slower (an idle's sway) is drift, faded out at the seam instead of piling up.
SKIP = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky', 'Jaw', 'Eye', 'End')  # joints the match ignores


def _locals(motion):
    W, par = motion.world, motion.parents
    out = np.empty_like(W)
    for j in range(len(par)):
        out[:, j] = W[:, j] if par[j] < 0 else np.linalg.inv(W[:, par[j]]) @ W[:, j]
    return out


def find(motion, blend=BLEND, cycle=CYCLE, start=None, end=None):
    """(S, E, cost): the cycle whose seam matches best; cost is the mean bone rotation difference (degrees) between
    the `blend` frames from E and those from S. start / end (frames) pin either one."""
    T, fps = motion.frames, motion.fps
    body = [j for j in range(1, len(motion.names)) if not any(k in motion.names[j] for k in SKIP)]
    q = Rotation.from_matrix(_locals(motion)[:, body, :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T, len(body), 4)

    def cost(s, e):
        d = np.abs((q[s:s + blend] * q[e:e + blend]).sum(-1))
        return float(np.degrees(2 * np.arccos(np.clip(d, 0, 1))).mean())
    ends = [end] if end is not None else range(1, T - blend + 1)
    best = None
    for e in ends:
        if e + blend > T:
            raise ValueError(f'the cycle must end at least {blend} frames before the take does (frame {T - blend})')
        starts = [start] if start is not None else range(0, e)
        for s in starts:
            if e - s <= blend or s + blend > T:  # the fade must finish inside the cycle
                continue
            if (start is None or end is None) and not cycle[0] * fps <= e - s <= cycle[1] * fps:
                continue
            c = cost(s, e)
            if best is None or c < best[2]:
                best = (s, e, c)
    if best is None:
        raise ValueError(f'no cycle of {cycle[0]:g}-{cycle[1]:g} s, longer than the {blend}-frame crossfade, fits in '
                         f'this {T / fps:.1f} s take' + (' with the pinned frames' if start is not None or end is not None else ''))
    return best


def make(motion, S, E, blend=BLEND, repeats=REPEATS):
    """The take's frames before S, then the seamless cycle S..E-1 `repeats` times. Returns (motion, info) with
    info['start'] the first frame of the middle cycle and info['frames'] the cycle's length (the retarget also keeps
    the next cycle's first frame, so the clip ends on its own start: what engines and Blender expect of a loop)."""
    n, fps = E - S, motion.fps
    if n <= blend:
        raise ValueError(f'a {n}-frame cycle is too short for a {blend}-frame crossfade')
    names, par = list(motion.names), motion.parents
    hip = names.index('Hips')
    local = _locals(motion)
    W = motion.world
    step = W[E, hip, :3, 3] - W[S, hip, :3, 3]
    step[1] = 0  # Y-up: the cycle's travel along the floor
    travels = np.linalg.norm(step) / (n / fps) > TRAVELS
    if not travels:
        step = np.zeros(3)
    cyc = local[S:E].copy()
    for t in range(min(blend, n)):
        w = (t + 1) / (blend + 1)
        w = w * w * (3 - 2 * w)  # 0 -> 1, eased
        for j in range(len(par)):
            if par[j] < 0 and j != hip:
                continue
            r = Slerp([0, 1], Rotation.from_matrix(np.stack([local[E + t, j, :3, :3], local[S + t, j, :3, :3]])))(w)
            cyc[t, j, :3, :3] = r.as_matrix()
        cyc[t, hip, :3, 3] = (1 - w) * (local[E + t, hip, :3, 3] - step) + w * local[S + t, hip, :3, 3]
    reps = [local[:S]]
    for k in range(repeats):
        c = cyc.copy()
        c[:, hip, :3, 3] += step * k
        reps.append(c)
    loc = np.concatenate(reps)
    world = np.empty_like(loc)
    for j in range(len(par)):
        world[:, j] = loc[:, j] if par[j] < 0 else world[:, par[j]] @ loc[:, j]
    out = SomaMotion(motion.names, par, world, motion.rest, fps, motion.source + f' loop [{S}:{E}] x{repeats}')
    return out, dict(start=S + (repeats // 2) * n, frames=n, source=(S, E), travels=bool(travels),
                     travel_m=float(np.linalg.norm(step)))
