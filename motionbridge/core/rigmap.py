"""Find the SOMA body parts on an arbitrary humanoid rig.

Topology and rest geometry decide every assignment; bone names only confirm or
contradict it, which sets the confidence. The input is dump_rig.py's JSON
(Blender world space, Z-up, meters).

Roles use SOMA joint names. Single bones: Hips, Head and, per side (Left/Right),
Shoulder, Arm, ForeArm, Hand, Leg, Shin, Foot, ToeBase. Chains (ordered root to
tip): Spine, Neck and, per side, HandThumb/HandIndex/HandMiddle/HandRing/HandPinky.
"""
from dataclasses import dataclass, field
import re

import numpy as np

UP = np.array([0., 0., 1.])
SIDES = ('Left', 'Right')
FINGERS = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
LIMB_ROLES = ('Shoulder', 'Arm', 'ForeArm', 'Hand', 'Leg', 'Shin', 'Foot', 'ToeBase')
REQUIRED = ['Hips', 'Head'] + [s + r for s in SIDES for r in ('Arm', 'ForeArm', 'Hand', 'Leg', 'Shin', 'Foot')]

# Name evidence. First match wins, so fingers precede "hand" and "neck" precedes "twist".
KEYWORDS = [
    (r'thumb', 'Thumb'), (r'index|pointer', 'Index'), (r'middle', 'Middle'), (r'ring', 'Ring'),
    (r'pinky|little', 'Pinky'), (r'neck', 'Neck'), (r'twist|roll|helper|corrective|share', 'twist'),
    (r'clav|collar|shoulder', 'Shoulder'), (r'forearm|lowerarm|loarm|elbow', 'ForeArm'),
    (r'upperarm|uparm|humerus|^arm\d*$', 'Arm'), (r'hand|wrist|palm', 'Hand'),
    (r'thigh|upleg|upperleg', 'Leg'), (r'calf|shin|lowerleg|knee|^leg\d*$', 'Shin'),
    (r'foot|ankle', 'Foot'), (r'toe|ball', 'ToeBase'), (r'head', 'Head'),
    (r'spine|waist|chest|torso|abdomen|belly|back|rib', 'Spine'), (r'hip|pelvis', 'Hips'),
]
PREFIX = re.compile(r'^(mixamorig\d*[:_]|.*:|def[-_]|org[-_]|cc_base_|j_bip_(c_)?|bip\d*[ _])', re.I)


def parse_name(name):
    """(side 'Left'/'Right'/None, kind from KEYWORDS or None)."""
    n = re.sub(r'\.\d{3}$', '', PREFIX.sub('', name))
    side = None
    for s, letter in (('Left', 'lL'), ('Right', 'rR')):
        word = s.lower()
        patterns = [rf'^{word}', rf'{word}$', rf'^[{letter}][_.\-\s]', rf'[_.\-\s][{letter}]$',
                    rf'[_.\-\s][{letter}][_.\-\s]', rf'^{letter[1]}(?=[A-Z])']
        if any(re.search(p, n, 0 if p.endswith('[A-Z])') else re.I) for p in patterns):
            side = s
            n = re.sub(rf'(?i){word}|(^|[_.\-\s])[{letter}](?=$|[_.\-\s])|^{letter[1]}(?=[A-Z])', ' ', n)
            break
    base = re.sub(r'[^a-z0-9]', '', n.lower())
    kind = next((k for pattern, k in KEYWORDS if re.search(pattern, base)), None)
    return side, kind


@dataclass
class RigMap:
    roles: dict = field(default_factory=dict)       # role -> bone name, or list of bone names for chains
    confidence: dict = field(default_factory=dict)  # role -> 0..1
    up: list = None
    forward: list = None
    left: list = None
    height: float = 0.0
    ground: float = 0.0
    hips_aux: list = field(default_factory=list)    # centre bones between hips and the legs (e.g. a Pelvis)
    extras: dict = field(default_factory=dict)      # unassigned weighted bones -> guessed kind
    notes: list = field(default_factory=list)

    def missing(self):
        return [r for r in REQUIRED if r not in self.roles]

    def to_json(self):
        return {k: getattr(self, k) for k in ('roles', 'confidence', 'up', 'forward', 'left', 'height',
                                              'ground', 'hips_aux', 'extras', 'notes')}


class _Rig:
    def __init__(self, dump):
        bones = dump['bones']
        use = [b for b in bones if b['deform']] or bones
        keep = {b['name'] for b in use}
        by_name = {b['name']: b for b in bones}

        def deform_parent(b):  # nearest kept ancestor, so control bones in between are skipped
            p = b['parent']
            while p is not None and p not in keep:
                p = by_name[p]['parent']
            return p

        self.names = [b['name'] for b in use]
        self.parent = {b['name']: deform_parent(b) for b in use}
        self.children = {n: [] for n in self.names}
        for n, p in self.parent.items():
            if p is not None:
                self.children[p].append(n)
        self.head = {b['name']: np.array(b['head']) for b in use}
        self.tail = {b['name']: np.array(b['tail']) for b in use}
        self.weight = {b['name']: b.get('weighted_vertices', 1) for b in use}
        self.girth = {b['name']: b.get('girth', 0.0) for b in use}
        self.parsed = {n: parse_name(n) for n in self.names}

    def ancestors(self, n):
        out = []
        while n is not None:
            out.append(n)
            n = self.parent[n]
        return out

    def lca(self, *names):
        common = set(self.ancestors(names[0]))
        for n in names[1:]:
            common &= set(self.ancestors(n))
        return next(a for a in self.ancestors(names[0]) if a in common)

    def path(self, a, b):
        """Bones from a (exclusive) down to b (inclusive); a must be an ancestor of b."""
        chain = self.ancestors(b)
        return list(reversed(chain[:chain.index(a)]))

    def subtree(self, n):
        out, stack = [], [n]
        while stack:
            x = stack.pop()
            out.append(x)
            stack += self.children[x]
        return out

    def weighted(self, n):
        return sum(self.weight[x] for x in self.subtree(n)) > 0

    def descend(self, start):
        """Follow the child whose subtree reaches farthest from start, skipping twist helpers."""
        chain, origin = [start], self.head[start]
        while True:
            options = [c for c in self.children[chain[-1]] if self.weighted(c)]
            if not options:
                return chain
            def reach(c):
                far = max(np.linalg.norm(self.tail[x] - origin) for x in self.subtree(c))
                return far - (10 if self.parsed[c][1] == 'twist' else 0)
            chain.append(max(options, key=reach))

    def trim(self, chain):
        """Drop trailing end markers (e.g. HeadTop_End): leaves moving far fewer vertices than their parent."""
        while len(chain) > 1 and self.weight[chain[-1]] <= 0.15 * self.weight[chain[-2]]:
            chain = chain[:-1]
        return chain


def build(dump):
    rig = _Rig(dump)
    m = RigMap(up=UP.tolist())
    tails = np.array([rig.tail[n] for n in rig.names])
    heads = np.array([rig.head[n] for n in rig.names])
    points = np.vstack([heads, tails])
    m.ground = float(points[:, 2].min())
    m.height = float(points[:, 2].max() - m.ground)
    h, ground = m.height, m.ground

    # Lateral axis: arms and legs spread sideways in a T or A rest pose, so it has the larger extent.
    lateral = int(np.argmax(np.ptp(points[:, :2], axis=0)))
    top = max(rig.names, key=lambda n: rig.tail[n][2])
    centre = rig.tail[top][lateral]

    low = [n for n in rig.names if rig.tail[n][2] < ground + 0.12 * h]
    sides = [[n for n in low if rig.tail[n][lateral] > centre + 0.02 * h],
             [n for n in low if rig.tail[n][lateral] < centre - 0.02 * h]]
    if not all(sides):
        m.notes.append('could not find two separate legs reaching the floor')
        return m
    feet = [min(s, key=lambda n: rig.tail[n][2]) for s in sides]
    hips = rig.lca(feet[0], feet[1], top)

    upper = [n for n in rig.names if rig.tail[n][2] > ground + 0.35 * h]
    hands = [max(upper, key=lambda n: rig.tail[n][lateral]), min(upper, key=lambda n: rig.tail[n][lateral])]
    chest = rig.lca(*hands)
    if hips not in rig.ancestors(chest):
        m.notes.append('arms do not attach above the hips; spine guessed from the head path')
        chest = rig.lca(chest, top)

    # Spine, neck, head.
    spine = rig.path(hips, chest)
    post = rig.path(chest, top) if chest in rig.ancestors(top) else []
    named_head = [n for n in post if rig.parsed[n][1] == 'Head']
    if named_head:
        head = named_head[0]
    elif len(post) == 1:
        head = post[0]
    elif post and all(rig.girth[n] > 0 for n in post):
        # The skull is wider than the neck: the head is the first bone after the narrowest
        # neck bone that is clearly wider (some rigs split the skull into several bones).
        narrow = min(range(len(post) - 1), key=lambda i: rig.girth[post[i]])
        head = next((n for n in post[narrow + 1:] if rig.girth[n] > 1.15 * rig.girth[post[narrow]]),
                    post[narrow + 1])
    elif post:
        head = max(rig.trim(post), key=lambda n: rig.weight[n])
        m.notes.append('no girth data; head chosen by weighted vertex count')
    else:
        head = chest
        m.notes.append('no bones above the chest; head shares the chest bone')
    m.roles['Hips'], m.roles['Head'] = hips, head
    m.roles['Spine'] = spine
    m.roles['Neck'] = post[:post.index(head)] if head in post else []

    # Facing: toes point forward. Left = up x forward.
    leg_chains, toe_dirs = [], []
    first = [rig.path(hips, f) for f in feet]
    shared = [n for n in first[0] if n in first[1]]
    m.hips_aux = shared
    for f_path in first:
        rest = [n for n in f_path if n not in shared]
        chain = rig.trim(rig.descend(rest[0]))
        leg_chains.append(chain)
        end = chain[-1]
        toe_dirs.append(rig.tail[end] - rig.head[chain[-2] if len(chain) > 1 else end])
    fwd = np.mean(toe_dirs, axis=0)
    fwd[2] = 0
    if np.linalg.norm(fwd) < 1e-4:
        fwd = np.array([0., -1., 0.])
        m.notes.append('feet give no facing direction; assuming Blender front (-Y)')
    fwd /= np.linalg.norm(fwd)
    left = np.cross(UP, fwd)
    m.forward, m.left = fwd.round(4).tolist(), left.round(4).tolist()

    def side_of(points_):
        return 'Left' if np.dot(np.mean(points_, axis=0) - rig.head[hips], left) > 0 else 'Right'

    # Legs.
    for chain in leg_chains:
        side = side_of([rig.tail[n] for n in chain])
        toe = None
        if len(chain) >= 3:
            d = rig.tail[chain[-1]] - rig.head[chain[-1]]
            if abs(d[2]) < 0.5 * np.linalg.norm(d):
                toe = chain[-1]
        foot = chain[-2] if toe else chain[-1]
        body = chain[:chain.index(foot)]
        if body:
            m.roles[side + 'Leg'] = body[0]
        if len(body) > 1:
            m.roles[side + 'Shin'] = body[1]
            for extra in body[2:]:
                m.extras[extra] = 'leg segment'
        m.roles[side + 'Foot'] = foot
        if toe:
            m.roles[side + 'ToeBase'] = toe

    # Arms and fingers.
    for tip in hands:
        arm = rig.path(chest, tip)
        # The hand is the last branch point toward the fingertips; helper bones can branch earlier.
        hand = next((n for n in reversed(arm) if sum(rig.weighted(c) for c in rig.children[n]) >= 3), None)
        if hand is None:
            arm = rig.trim(arm)
            hand = arm[-1]
        side = side_of([rig.tail[hand]])
        segment = [n for n in arm[:arm.index(hand)] if rig.parsed[n][1] != 'twist']
        m.roles[side + 'Hand'] = hand
        if len(segment) >= 3:
            m.roles[side + 'Shoulder'] = segment[0]
            segment = segment[1:]
        if segment:
            length = [np.linalg.norm(rig.tail[n] - rig.head[n]) for n in segment]
            half = sum(length) / 2
            start = np.cumsum([0] + length[:-1])
            upper_seg = [n for n, s in zip(segment, start) if s < half]
            lower_seg = [n for n, s in zip(segment, start) if s >= half] or upper_seg[-1:]
            m.roles[side + 'Arm'] = upper_seg[0]
            if lower_seg[0] != upper_seg[0]:
                m.roles[side + 'ForeArm'] = lower_seg[0]
            for extra in segment:
                if extra not in (upper_seg[0], lower_seg[0]):
                    m.extras[extra] = 'arm segment'
        _fingers(rig, m, side, hand)

    _score(rig, m)
    if not named_head and len(post) > 2:
        m.confidence['Head'] = min(m.confidence['Head'], 0.6)
        m.confidence['Neck'] = min(m.confidence.get('Neck', 0.6), 0.6)  # the neck ends where the head starts
        m.notes.append(f'head picked by girth among {len(post)} neck/head bones; check it')
    assigned = set(_flat(m.roles.values())) | set(m.hips_aux) | set(m.extras)
    for n in rig.names:
        if n not in assigned and rig.weight[n] > 0:
            m.extras[n] = rig.parsed[n][1] or 'unknown'
    if m.missing():
        m.notes.append('missing required roles: ' + ', '.join(m.missing()))
    return m


def _fingers(rig, m, side, hand):
    chains = [rig.trim(rig.descend(c)) for c in rig.children[hand] if rig.weighted(c)]
    chains = [c for c in chains if rig.parsed[c[0]][1] != 'twist']
    if not chains:
        return
    named = {}
    for c in chains:
        kinds = {rig.parsed[n][1] for n in c} & set(FINGERS)
        if len(kinds) == 1:
            named[kinds.pop()] = c
    if len(named) == len(chains):
        for finger, c in named.items():
            m.roles[f'{side}Hand{finger}'] = c
        return
    # Geometry: the thumb diverges most from the hand's direction; the rest are ordered
    # across the palm starting next to the thumb (index) and ending at the pinky.
    hand_dir = rig.tail[hand] - rig.head[hand]
    hand_dir /= np.linalg.norm(hand_dir) + 1e-9
    def direction(c):
        d = rig.tail[c[-1]] - rig.head[c[0]]
        return d / (np.linalg.norm(d) + 1e-9)
    order = list(chains)
    thumb = min(order, key=lambda c: np.dot(direction(c), hand_dir)) if len(order) >= 4 else None
    rest = [c for c in order if c is not thumb]
    bases = np.array([rig.head[c[0]] for c in rest])
    if len(rest) > 1:
        axis = np.linalg.svd(bases - bases.mean(0))[2][0]
        rest.sort(key=lambda c: np.dot(rig.head[c[0]], axis))
        if thumb is not None and np.linalg.norm(rig.head[rest[-1][0]] - rig.head[thumb[0]]) < \
                np.linalg.norm(rig.head[rest[0][0]] - rig.head[thumb[0]]):
            rest.reverse()
    if thumb is not None:
        m.roles[f'{side}HandThumb'] = thumb
    for finger, c in zip(FINGERS[1:], rest):
        m.roles[f'{side}Hand{finger}'] = c
    if len(rest) > 4:
        m.notes.append(f'{side} hand has {len(rest) + (thumb is not None)} finger chains; extras unassigned')


def _flat(values):
    for v in values:
        if isinstance(v, list):
            yield from v
        elif v is not None:
            yield v


def _expected(role):
    for r in LIMB_ROLES + tuple('Hand' + f for f in FINGERS):
        if role.endswith(r) and role[:-len(r)] in SIDES:
            return role[:-len(r)], r.replace('Hand', '') if r.startswith('Hand') and r != 'Hand' else r
    return None, role


def _score(rig, m):
    """1.0 names agree, 0.8 names silent, 0.5 names contradict (noted)."""
    for role, bones in m.roles.items():
        side, kind = _expected(role)
        scores = []
        for b in (bones if isinstance(bones, list) else [bones]):
            if b is None:
                continue
            name_side, name_kind = rig.parsed[b]
            s = 0.8 if name_kind is None else 1.0 if name_kind == kind or (kind == 'Neck' and name_kind == 'Neck') \
                else 0.5
            if kind == 'Spine' and name_kind in ('Spine', 'Hips'):
                s = 1.0
            if side and name_side and name_side != side:
                s = 0.3
                m.notes.append(f'{b}: named {name_side} but sits on the {side} side')
            elif s == 0.5:
                m.notes.append(f'{b}: mapped to {role} but its name suggests {name_kind}')
            scores.append(s)
        m.confidence[role] = min(scores) if scores else 0.0
