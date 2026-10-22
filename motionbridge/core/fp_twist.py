"""Forearm roll (pronation / supination). A tracked hand keeps its exact pose, but its turn about the forearm's
axis belongs in the forearm: a real forearm rolls along its length and the wrist only bends. Per frame the arm
chain (shoulder -> elbow -> wrist) is solved as Blender's IK would (elbow in its start plane), then the forearm
is rolled by the hand's twist. The result is the upper arm's and forearm's world matrices per frame, to be set
directly with the IK muted (blender side)."""
import numpy as np
from scipy.spatial.transform import Rotation


def _aim(P0, head, tail):
    """The start-pose bone matrix P0 turned (shortest turn) so it runs from head to tail."""
    d0 = P0[:3, 1] / np.linalg.norm(P0[:3, 1])
    dn = (tail - head) / np.linalg.norm(tail - head)
    m = np.eye(4)
    m[:3, :3] = Rotation.align_vectors([dn], [d0])[0].as_matrix() @ P0[:3, :3]
    m[:3, 3] = head
    return m


REACH = 0.97      # the arm reaches at most this share of its length; beyond, the shoulder comes forward
SHOULDER_MAX = 35.0  # deg: how far the shoulder bone may turn forward (protraction) to reach


def clavicle(rig):
    """The bone the upper arm hangs from (the shoulder / clavicle), or None."""
    upper = rig.dump['bones'][rig.sk.index[rig.forearm['name']]]['parent']
    return rig.dump['bones'][rig.sk.index[upper]]['parent']


def reach(rig, wr, off=None):
    """A hand beyond the arm's reach: the shoulder bone turned forward about its head just enough (at most
    SHOULDER_MAX) to bring it back, as a shoulder comes forward to shoulder a rifle. The bones keep their length.
    -> (rotation (3, 3) about the clavicle's head, shoulder, elbow reference point), identity if in reach."""
    fa = rig.forearm
    off = np.zeros(3) if off is None else off
    s0, e0, c = fa['shoulder'] + off, fa['elbow0'] + off, clavicle(rig)
    L = (fa['L1'] + fa['L2']) * REACH
    if c is None or np.linalg.norm(wr - s0) <= L:
        return np.eye(3), s0, e0
    ch = rig.pose0[c][:3, 3] + off
    a, b = s0 - ch, wr - ch
    axis = np.cross(a, b)
    if np.linalg.norm(axis) < 1e-9:
        return np.eye(3), s0, e0
    axis /= np.linalg.norm(axis)
    turn = lambda deg: Rotation.from_rotvec(axis * np.radians(deg)).as_matrix()
    lo, hi = 0.0, SHOULDER_MAX
    if np.linalg.norm(wr - ch - turn(hi) @ a) > L:
        lo = hi  # can't reach even fully forward: go as far as allowed
    for _ in range(30):
        if lo == hi:
            break
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if np.linalg.norm(wr - ch - turn(mid) @ a) > L else (lo, mid)
    R = turn(hi)
    return R, ch + R @ a, ch + R @ (e0 - ch)


def forearm_roll(rigs, basis, log=print, swing=None, offset=None):
    """rigs: fp_track.Rig per hand (sharing one armature); basis: (T, B, 4, 4) bone basis per frame.
    offset: {side: (T, 3)} the shoulder bone moved (world, m), e.g. fp_elbow's fit of the arm to its mask.
    swing: {side: (T,) deg} elbow swing about the shoulder-wrist line (fp_elbow), 0 = the start plane.
    Returns (arm_names, arm_world (T, 2-3 per arm, 4, 4), rolls {side: degrees per frame}): per arm the shoulder
    bone (only when some frame needs it to reach), then the upper arm and forearm. Arms without a forearm the
    wrist hangs off (Rig.forearm is None) are left out."""
    T = len(basis)
    names, worlds, rolls = [], [], {}
    for rig in rigs:
        fa = rig.forearm
        if fa is None:
            if log:
                log(f'{rig.hand.side.lower()} arm: no forearm found at the wrist, no roll')
            continue
        sk = rig.sk
        upper = rig.dump['bones'][sk.index[fa['name']]]['parent']
        P0u, P0f = rig.pose0[upper], rig.pose0[fa['name']]
        hand = sk.index[rig.hand.hand]
        c = clavicle(rig)
        Wc, Wu, Wf, roll = np.zeros((T, 4, 4)), np.zeros((T, 4, 4)), np.zeros((T, 4, 4)), np.zeros(T)
        moved = False
        for t in range(T):
            Hm = sk.pose(basis[t])[hand]
            wr = Hm[:3, 3]
            Hr = Hm[:3, :3] / np.linalg.norm(Hm[:3, :3], axis=0)
            off = offset[rig.hand.side][t] if offset and rig.hand.side in offset else np.zeros(3)
            R, s, e0 = reach(rig, wr, off)
            moved |= not np.allclose(R, np.eye(3)) or np.abs(off).max() > 1e-6
            if c is not None:
                Wc[t] = rig.pose0[c].copy()
                Wc[t][:3, :3] = R @ Wc[t][:3, :3]
                Wc[t][:3, 3] += off
            dd = np.clip(np.linalg.norm(wr - s), 1e-6, fa['L1'] + fa['L2'] - 1e-4)
            u = (wr - s) / np.linalg.norm(wr - s)
            a = (fa['L1'] ** 2 - fa['L2'] ** 2 + dd ** 2) / (2 * dd)
            h = np.sqrt(max(fa['L1'] ** 2 - a ** 2, 0))
            v = (e0 - s) - ((e0 - s) @ u) * u
            v /= np.linalg.norm(v) + 1e-9
            if swing and rig.hand.side in swing:
                v = Rotation.from_rotvec(u * np.radians(swing[rig.hand.side][t])).apply(v)
            elbow = s + a * u + h * v
            Wu[t], Wf[t] = _aim(P0u, s, elbow), _aim(P0f, elbow, wr)
            # the hand's roll about the forearm axis, relative to the start pose: moved into the forearm
            Ff = Wf[t][:3, :3] / np.linalg.norm(Wf[t][:3, :3], axis=0)
            q = Rotation.from_matrix(Ff.T @ Hr @ fa['rel0'].T).as_quat()
            twist = (2 * np.arctan2(q[1], q[3]) + np.pi) % (2 * np.pi) - np.pi
            Wf[t][:3, :3] = Rotation.from_rotvec(Ff[:, 1] * twist).as_matrix() @ Wf[t][:3, :3]
            roll[t] = np.degrees(twist)
        if moved:
            names.append(c)
            worlds.append(Wc)
            if log:
                ang = [np.degrees(Rotation.from_matrix(reach(rig, sk.pose(basis[t])[hand][:3, 3], offset[rig.hand.side][t] if offset and rig.hand.side in offset else None)[0]).magnitude()) for t in range(T)]
                log(f'{rig.hand.side.lower()} shoulder forward to reach: on {sum(a > 0.1 for a in ang)} frames, max {max(ang):.0f} deg')
        names += [upper, fa['name']]
        worlds += [Wu, Wf]
        rolls[rig.hand.side] = roll
        if log:
            log(f'{rig.hand.side.lower()} forearm roll: median {np.median(np.abs(roll)):.0f} deg, max {np.abs(roll).max():.0f} deg')
    return np.array(names), (np.stack(worlds, 1) if worlds else np.zeros((T, 0, 4, 4))), rolls
