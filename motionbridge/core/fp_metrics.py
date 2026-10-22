"""How well a first-person hand matches its video and holds its prop: per frame and over a track.

Everything is measured from the hand's 21 points (MediaPipe layout), so a pose from any solver or any saved
track (parameters or a whole-armature basis) is scored the same way. Lengths in mm, angles in degrees.

Per frame
  screen_pct   median distance of the 21 points from the video's (SAM 3D Body) on screen, % of the hand's size
  dir_deg      median angle between the hand's bones and the video's bone directions
  contacts     per finger what it does to the prop: 'wrap' (the prop lies between the palm's centre and the
               fingertip), 'press' (the tip on the surface), 'free'
  holding      fingers wrapping or pressing
  inside       hand surface points inside the prop, and the deepest (mm)
  palm_mm      the palm's nearest approach to the prop (negative: inside)
  wrist        (twist, bend) against the forearm, when the pose is given as parameters
  parts        per finger, the prop part nearest its tip (when the prop's part names are given)
  expected     with an expected contact per finger: how many match ('wrap|press' accepts either, '@Trigger'
               also asks for that part)

Over a track
  jitter       mean frame-to-frame acceleration of the 21 points, mm/frame^2
  holding      fingers holding per frame
  slip         while the palm rests on the prop on consecutive frames, how far it moves over the prop's surface
               per frame (mm): a held prop should ride with the hand
  inside       hand points inside the prop per frame"""
import numpy as np

from motionbridge.core import fp_track as fp
from motionbridge.core import hands
from motionbridge.core.hand_fit import project

LINE = np.linspace(0.15, 0.95, 9)  # where on the palm-to-tip line the prop is looked for (as fp_grasp)
PRESS_MM = 6.0                     # a fingertip this close to the surface presses on it
REST_MM = 25.0                     # a palm this close to the prop rests on it (for slip)


def segs(fingers):
    """Keypoint index pairs per finger segment, base to tip."""
    return {f: [(1 + 4 * j + a, 2 + 4 * j + a) for a in range(3)] for j, f in enumerate(hands.FINGERS) if f in fingers}


def keypoints_from_basis(rig, B):
    """The hand's 21 points from a whole-armature bone basis (as saved in a track's 'basis')."""
    pose = rig.sk.pose(B)
    return np.array([rig.sk.point(pose, *kb) for kb in rig.hand.keypoint_bones()])


def palm_centre(p):
    return p[[0, 5, 9, 13, 17]].mean(0)


def palm_points(p):
    return np.array([p[0] + s * (p[j] - p[0]) for j in (5, 9, 13, 17) for s in (0.5, 1.0)])


def contacts(p, fingers, prop, M):
    """{finger: 'wrap' | 'press' | 'free'} as the hand at points p actually touches the prop."""
    pc = palm_centre(p)
    out = {}
    for f, s in segs(fingers).items():
        tip = p[s[2][1]]
        if prop.distance(pc + LINE[:, None] * (tip - pc), M).min() < 0:
            out[f] = 'wrap'
        elif abs(prop.distance(tip[None], M)[0] - fp.SURFACE) * 1000 < PRESS_MM:
            out[f] = 'press'
        else:
            out[f] = 'free'
    return out


def parts(p, fingers, prop, M, names):
    """{finger: name of the prop part nearest its tip} (prop.parts are the parts' surface points, in order)."""
    out = {}
    for f, s in segs(fingers).items():
        tip = np.linalg.inv(M)[:3] @ np.r_[p[s[2][1]], 1]
        out[f] = str(names[int(np.argmin([np.linalg.norm(q - tip, axis=1).min() for q in prop.parts]))])
    return out


def matches(observed, expected, touched=None):
    """Fingers whose contact is one the expectation allows: 'wrap|press' allows either, 'wrap|press@Trigger' also
    asks for the tip to be on that part. None without an expectation."""
    if not expected:
        return None
    n = 0
    for f, e in expected.items():
        modes, _, part = str(e).partition('@')
        n += observed.get(f) in modes.split('|') and (not part or touched is None or touched.get(f) == part)
    return int(n)


def frame(rig, p, prop, M, cam, obs=None, x=None, expected=None, names=None):
    """Metrics of one frame's hand points p (21, 3) against the prop placed by M; obs = fp_track.observation."""
    out = {}
    if obs is not None:
        uv, span, target = obs
        out['screen_pct'] = float(np.median(np.linalg.norm(project(p, cam) - uv, axis=1)) / span * 100)
        d = fp.directions(p)
        cos = np.clip(np.sum(d * target, axis=-1) / (np.linalg.norm(d, axis=-1) * np.linalg.norm(target, axis=-1) + 1e-12), -1, 1)
        out['dir_deg'] = float(np.degrees(np.median(np.arccos(cos))))
    c = contacts(p, rig.hand.fingers, prop, M)
    out['contacts'] = c
    touched = parts(p, rig.hand.fingers, prop, M, names) if names is not None else None
    if touched is not None:
        out['parts'] = touched
    out['holding'] = int(sum(v != 'free' for v in c.values()))
    d = prop.distance(fp._surface_points(p), M)
    out['inside'] = int((d < 0).sum())
    out['inside_mm'] = float(max(-d.min(), 0) * 1000)
    out['palm_mm'] = float(prop.distance(palm_points(p), M).min() * 1000)
    if x is not None:
        out['wrist'] = [float(v) for v in rig.wrist_strain(x)]
    m = matches(c, expected, touched)
    if m is not None:
        out['expected'] = expected
        out['matched'] = m
        out['of'] = len(expected)
    return out


def track(rig, P, prop, motion):
    """Metrics of a whole track: P (T, 21, 3) hand points, motion (T, 4, 4) the prop's placement."""
    T = len(P)
    acc = np.linalg.norm(np.diff(P, 2, axis=0), axis=2).mean() * 1000 if T > 2 else 0.0
    held = [int(sum(v != 'free' for v in contacts(P[t], rig.hand.fingers, prop, motion[t]).values())) for t in range(T)]
    inside = [int((prop.distance(fp._surface_points(P[t]), motion[t]) < 0).sum()) for t in range(T)]
    # the palm's centre in the prop's own frame: still while the prop is held
    local = np.array([np.linalg.inv(motion[t]) @ np.r_[palm_centre(P[t]), 1] for t in range(T)])[:, :3]
    rest = np.array([prop.distance(palm_points(P[t]), motion[t]).min() * 1000 < REST_MM for t in range(T)])
    pairs = rest[:-1] & rest[1:]
    slip = np.linalg.norm(np.diff(local, axis=0), axis=1)[pairs] * 1000
    return {'jitter': float(acc), 'holding': held, 'inside': inside,
            'slip_mm': float(slip.mean()) if len(slip) else None, 'rest_frames': int(rest.sum())}
