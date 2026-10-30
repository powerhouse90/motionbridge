"""Animation spacing for captured moves: a long ease in, fast through the middle, a shorter ease out, a small settle
(Williams' spacing; Thomas & Johnston's slow in and slow out, timing and follow through).

Captured strikes arrive with the even, rounded spacing of a filtered estimate: a punch eases into its extreme as
gently as it left. This finds each move of each limb (the hand's or foot's speed relative to the body rising to a
peak and falling back) and keeps its poses and its length but redistributes them in time: slow out of the start,
fastest past the middle, then a shorter ease onto the extreme (a dead stop reads as a mistake: SNAP well under 1
makes one), then a small overshoot that eases back onto it. Only that limb's joints are retimed, so a jab doesn't
drag the stance with it; a planted foot doesn't move, so it has no moves to retime.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from motionbridge.core.soma import SomaMotion

# limb (its top joint), the end whose speed finds the moves, and the joint that speed is measured against
LIMBS = (('LeftArm', 'LeftHand', 'Chest'), ('RightArm', 'RightHand', 'Chest'),
         ('LeftLeg', 'LeftFoot', 'Hips'), ('RightLeg', 'RightFoot', 'Hips'))
PEAK = 1.5         # m/s: a move peaks at least this fast (a relaxed arm swing stays under it)
LOW = 0.25         # a move starts and ends where the speed falls below this fraction of its peak
MIN_FRAMES = 5     # shorter bursts are left alone
EASE_IN = 3.0      # how slowly a move leaves its start (1 = no ease)
SNAP = 1.2         # ease onto the extreme: less than EASE_IN keeps it shorter than the ease in (under ~0.5 is a dead stop)
SETTLE = 0.15      # s after the extreme that the overshoot takes to ease back
OVERSHOOT = 0.25   # of the move's last frame step, carried past the extreme before settling back
SKIP = 0.5         # share of a move's fast frames cut (0 = none, 0.5 = every other one)
SKIP_ABOVE = 5.0   # m/s: only frames where the hand or foot moves faster than this (relative to the body) are cut


def _locals(world, parents):
    L = world.copy()
    for j, p in enumerate(parents):
        if p >= 0:
            L[:, j] = np.linalg.inv(world[:, p]) @ world[:, j]
    return L


def _fk(L, parents):
    W = L.copy()
    for j, p in enumerate(parents):
        if p >= 0:
            W[:, j] = W[:, p] @ L[:, j]
    return W


def _subtree(parents, top):
    out = {top}
    for j, p in enumerate(parents):   # parents come before children
        if p in out:
            out.add(j)
    return sorted(out)


def moves(speed, fps):
    """(start, end) frames of each burst of speed (T,) that peaks above PEAK."""
    out, t, T = [], 0, len(speed)
    peaks = [i for i in range(1, T - 1) if speed[i] >= PEAK and speed[i] >= speed[i - 1] and speed[i] >= speed[i + 1]]
    for i in sorted(peaks, key=lambda i: -speed[i]):
        if any(s <= i <= e for s, e in out):
            continue
        low = LOW * speed[i]
        s = i
        while s > 0 and speed[s] > low:
            s -= 1
        e = i
        while e < T - 1 and speed[e] > low:
            e += 1
        if e - s >= MIN_FRAMES and not any(s < e2 and s2 < e for s2, e2 in out):
            out.append((s, e))
    return sorted(out)


def progress(n, ease_in=EASE_IN, snap=SNAP):
    """Share of the move covered at each of n + 1 frames: speed ~ u^ease_in (1 - u)^snap."""
    u = np.linspace(0, 1, 2001)
    v = u ** ease_in * (1 - u) ** snap
    g = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2)])
    return np.interp(np.linspace(0, 1, n + 1), u, g / g[-1])


def respace(motion, ease_in=EASE_IN, snap=SNAP, settle=SETTLE, overshoot=OVERSHOOT, skip=SKIP, skip_above=SKIP_ABOVE,
            log=None):
    """SomaMotion with each limb's moves respaced. Returns (motion, [(limb, start, end), ...]).
    skip: share of the frames where the limb moves faster than skip_above (m/s) that are cut (0.5: every other one);
    the limb jumps there, reaches the extreme that much sooner and holds it, so the move keeps its length and the
    other limbs their timing. Slower moves are only respaced."""
    names, parents, fps = list(motion.names), motion.parents, motion.fps
    L = _locals(motion.world, parents)
    out = L.copy()
    P = motion.positions
    found = []
    settle_frames = max(int(round(settle * fps)), 2)
    for top, end_joint, ref in LIMBS:
        joints = _subtree(parents, names.index(top))
        rel = P[:, names.index(end_joint)] - P[:, names.index(ref)]
        speed = np.linalg.norm(np.gradient(rel, axis=0), axis=1) * fps
        rots = Rotation.from_matrix(L[:, joints, :3, :3].reshape(-1, 3, 3)).as_quat().reshape(len(L), len(joints), 4)
        for s, e in moves(speed, fps):
            # where the move's frames are along its path, and where the new spacing wants each frame to be
            path = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(rel[s:e + 1], axis=0), axis=1))])
            if path[-1] < 1e-6:
                continue
            path = path / path[-1] + np.linspace(0, 1e-6, len(path))   # strictly increasing
            when = s + np.interp(progress(e - s, ease_in, snap), path, np.arange(e - s + 1))
            # skipped frames: only where the respaced limb moves faster than skip_above, every so often there
            at = np.stack([np.interp(when, np.arange(len(rel)), rel[:, i]) for i in range(3)], 1)
            fast = [k for k in range(1, len(when) - 1) if np.linalg.norm(at[k + 1] - at[k]) * fps > skip_above]
            drop = fast[::max(int(round(1 / skip)), 1)][:int(round(skip * len(fast)))] if skip > 0 and fast else []
            cut = len(drop)
            if cut:
                keep = [k for k in range(len(when)) if k not in drop]
                when = np.concatenate([when[keep], np.full(cut, when[-1])])   # jump through the fast part, then hold
            arrive = s + len(when) - 1 - cut                        # the frame it reaches the extreme
            for k, j in enumerate(joints):
                slerp = Slerp(np.arange(len(L)), Rotation.from_quat(rots[:, k]))
                out[s:e + 1, j, :3, :3] = slerp(when).as_matrix()
            # follow through: carry on past the extreme by part of the last step, then ease back onto it
            stop = min(arrive + settle_frames, len(L) - 1)
            if stop > arrive:
                tau = np.arange(1, stop - arrive + 1) / (stop - arrive + 1)
                amount = overshoot * np.sin(np.pi * tau)
                for j in joints:
                    step = Rotation.from_matrix(out[arrive, j, :3, :3]) * Rotation.from_matrix(out[arrive - 1, j, :3, :3]).inv()
                    for i, a in zip(range(arrive + 1, stop + 1), amount):
                        extra = Rotation.from_rotvec(step.as_rotvec() * a)
                        out[i, j, :3, :3] = (extra * Rotation.from_matrix(out[i, j, :3, :3])).as_matrix()
            found.append((top, s, e))
    if log:
        log(f'respaced {len(found)} moves: ' + ', '.join(f'{t} {s}-{e}' for t, s, e in found))
    return SomaMotion(motion.names, parents, _fk(out, parents), motion.rest, fps, motion.source + ' +spacing'), found
