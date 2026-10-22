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


def forearm_roll(rigs, basis, log=print):
    """rigs: fp_track.Rig per hand (sharing one armature); basis: (T, B, 4, 4) bone basis per frame.
    Returns (arm_names, arm_world (T, 2 per arm, 4, 4), rolls {side: degrees per frame}). Arms without a
    forearm the wrist hangs off (Rig.forearm is None) are left out."""
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
        s = fa['shoulder']
        Wu, Wf, roll = np.zeros((T, 4, 4)), np.zeros((T, 4, 4)), np.zeros(T)
        for t in range(T):
            Hm = sk.pose(basis[t])[hand]
            wr = Hm[:3, 3]
            Hr = Hm[:3, :3] / np.linalg.norm(Hm[:3, :3], axis=0)
            dd = np.clip(np.linalg.norm(wr - s), 1e-6, fa['L1'] + fa['L2'] - 1e-4)
            u = (wr - s) / np.linalg.norm(wr - s)
            a = (fa['L1'] ** 2 - fa['L2'] ** 2 + dd ** 2) / (2 * dd)
            h = np.sqrt(max(fa['L1'] ** 2 - a ** 2, 0))
            v = (fa['elbow0'] - s) - ((fa['elbow0'] - s) @ u) * u
            v /= np.linalg.norm(v) + 1e-9
            elbow = s + a * u + h * v
            Wu[t], Wf[t] = _aim(P0u, s, elbow), _aim(P0f, elbow, wr)
            # the hand's roll about the forearm axis, relative to the start pose: moved into the forearm
            Ff = Wf[t][:3, :3] / np.linalg.norm(Wf[t][:3, :3], axis=0)
            q = Rotation.from_matrix(Ff.T @ Hr @ fa['rel0'].T).as_quat()
            twist = (2 * np.arctan2(q[1], q[3]) + np.pi) % (2 * np.pi) - np.pi
            Wf[t][:3, :3] = Rotation.from_rotvec(Ff[:, 1] * twist).as_matrix() @ Wf[t][:3, :3]
            roll[t] = np.degrees(twist)
        names += [upper, fa['name']]
        worlds += [Wu, Wf]
        rolls[rig.hand.side] = roll
        if log:
            log(f'{rig.hand.side.lower()} forearm roll: median {np.median(np.abs(roll)):.0f} deg, max {np.abs(roll).max():.0f} deg')
    return np.array(names), (np.stack(worlds, 1) if worlds else np.zeros((T, 0, 4, 4))), rolls
