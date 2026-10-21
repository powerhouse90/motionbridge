"""Hands on any rig: find them, and pose their fingers with a few anatomical numbers.

Nothing here depends on bone names or roll axes. A hand is a bone with at least four finger
chains of similar length (rigmap._fingers sorts them into thumb..pinky by shape). Leading palm bones
(metacarpals, as on UE5 or MakeHuman rigs) stay fixed; the last three bones of each chain are the
finger's joints. Bend axes come from the palm's geometry: the palm faces along
cross(wrist->index, wrist->pinky) on a right hand (the opposite on a left), and a finger bends
toward its palm around cross(bone direction, palm normal).

Per finger 4 numbers, radians: base bend, base spread (sideways, around the palm normal), middle
bend, tip bend. Plus the wrist as a rotation vector on the hand bone. Positive bend curls toward
the palm, so a fist is all positive.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from motionbridge.core import rigmap

FINGERS = rigmap.FINGERS  # Thumb, Index, Middle, Ring, Pinky
PER_FINGER = 4
# Joint limits per finger value (bend base, spread, bend middle, bend tip), radians
LIMITS = {'Thumb': ((-0.4, 1.2), (-0.6, 0.8), (-0.3, 1.3), (-0.4, 1.5)),
          'other': ((-0.35, 1.7), (-0.35, 0.35), (-0.05, 1.9), (-0.1, 1.5))}


@dataclass
class Hand:
    side: str                      # 'Left' or 'Right'
    hand: str                      # hand bone name
    fingers: dict                  # finger -> list of joint bone names (3, after any palm bones)
    palm: dict = field(default_factory=dict)  # finger -> leading palm bones (fixed)

    @property
    def size(self):
        return 3 + PER_FINGER * len(self.fingers)

    def keypoint_bones(self):
        """MediaPipe's 21 points as (bone, 'head'|'tail'): wrist, then per finger 4 points base to tip."""
        out = [(self.hand, 'head')]
        for f in FINGERS:
            j = self.fingers.get(f)
            out += [(j[0], 'head'), (j[1], 'head'), (j[2], 'head'), (j[2], 'tail')] if j else [None] * 4
        return out

    def limits(self):
        lo, hi = [-np.pi] * 3, [np.pi] * 3
        for f in FINGERS:
            if f in self.fingers:
                for a, b in LIMITS['Thumb' if f == 'Thumb' else 'other']:
                    lo.append(a); hi.append(b)
        return np.array(lo), np.array(hi)


def find_hands(dump):
    """Hands of any rig (full body or arms only), with finger chains sorted thumb..pinky."""
    rig = rigmap._Rig(dump)
    hands = []
    for n in rig.names:
        chains = [rig.trim(rig.descend(c)) for c in rig.children[n] if rig.weighted(c)]
        chains = [c for c in chains if len(c) >= 2]
        if len(chains) < 4:
            continue
        reach = [max(np.linalg.norm(rig.tail[x] - rig.head[n]) for x in c) for c in chains]
        if max(reach) > 2.5 * min(reach):
            continue  # fingers are alike in length; a pelvis or chest branches into long and short chains
        side = rig.parsed[n][0] or next((rig.parsed[x][0] for c in chains for x in c if rig.parsed[x][0]), None)
        if side is None:
            continue
        m = rigmap.RigMap()
        rigmap._fingers(rig, m, side, n)
        fingers, palm = {}, {}
        for f in FINGERS:
            chain = m.roles.get(f'{side}Hand{f}')
            if chain and len(chain) >= 3:
                fingers[f], palm[f] = chain[-3:], chain[:-3]
        if len(fingers) >= 4:
            hands.append(Hand(side, n, fingers, palm))
    return hands


class Skeleton:
    """Forward kinematics of a dumped rig, in Blender's convention (pose = parent pose @ rest offset @ basis)."""

    def __init__(self, dump):
        bones = dump['bones']
        self.names = [b['name'] for b in bones]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parent = [self.index.get(b['parent']) for b in bones]
        self.rest = np.array([b['rest'] for b in bones])          # world space at rest
        self.length = np.array([np.linalg.norm(np.subtract(b['tail'], b['head'])) for b in bones])  # world units
        self.order, seen = [], set()
        while len(self.order) < len(bones):
            for i, p in enumerate(self.parent):
                if i not in seen and (p is None or p in seen):
                    self.order.append(i); seen.add(i)
        self.offset = np.array([self.rest[i] if p is None else np.linalg.inv(self.rest[p]) @ self.rest[i]
                                for i, p in enumerate(self.parent)])

    def pose(self, basis):
        """basis: (B, 4, 4) local pose matrices -> (B, 4, 4) world pose."""
        out = np.empty_like(self.rest)
        for i in self.order:
            p = self.parent[i]
            out[i] = (self.offset[i] if p is None else out[p] @ self.offset[i]) @ basis[i]
        return out

    def point(self, pose, bone, end):
        m = pose[self.index[bone]]
        if end == 'head':
            return m[:3, 3].copy()
        return m[:3, 3] + m[:3, 1] / np.linalg.norm(m[:3, 1]) * self.length[self.index[bone]]


class HandModel:
    """Maps a hand's 3 + 4 x fingers numbers to bone basis matrices on top of a start pose."""

    def __init__(self, skeleton, hand, basis0=None):
        self.sk, self.hand = skeleton, hand
        self.basis0 = basis0 if basis0 is not None else np.repeat(np.eye(4)[None], len(skeleton.names), axis=0)
        rest = skeleton.rest
        p = lambda n: rest[skeleton.index[n], :3, 3]
        wrist = p(hand.hand)
        normal = np.cross(p(hand.fingers['Index'][0]) - wrist, p(hand.fingers['Pinky' if 'Pinky' in hand.fingers else 'Ring'][0]) - wrist)
        normal /= np.linalg.norm(normal)
        if hand.side == 'Left':
            normal = -normal
        self.palm_normal = normal                         # world space at rest, points out of the palm
        self.axes = {}                                    # bone -> (bend axis, spread axis) in bone-local space
        for f, joints in hand.fingers.items():
            for k, n in enumerate(joints):
                r = rest[skeleton.index[n], :3, :3]
                r = r / np.linalg.norm(r, axis=0)
                d = r[:, 1]
                bend = np.cross(d, normal); bend /= np.linalg.norm(bend) + 1e-9
                spread = normal if f != 'Thumb' else np.cross(bend, d)
                self.axes[n] = (r.T @ bend, r.T @ spread)

    def basis(self, x):
        b = self.basis0.copy()
        h = self.sk.index[self.hand.hand]
        b[h, :3, :3] = self.basis0[h, :3, :3] @ Rotation.from_rotvec(x[:3]).as_matrix()
        k = 3
        for f in FINGERS:
            if f not in self.hand.fingers:
                continue
            bend0, spread, bend1, bend2 = x[k:k + 4]; k += 4
            for n, angle, sp in zip(self.hand.fingers[f], (bend0, bend1, bend2), (spread, 0, 0)):
                ax_b, ax_s = self.axes[n]
                rot = Rotation.from_rotvec(ax_s * sp) * Rotation.from_rotvec(ax_b * angle)
                i = self.sk.index[n]
                b[i, :3, :3] = self.basis0[i, :3, :3] @ rot.as_matrix()
        return b

    def keypoints(self, x):
        """21 world points (NaN rows for missing fingers)."""
        pose = self.sk.pose(self.basis(x))
        return np.array([self.sk.point(pose, *kb) if kb else np.full(3, np.nan) for kb in self.hand.keypoint_bones()])
