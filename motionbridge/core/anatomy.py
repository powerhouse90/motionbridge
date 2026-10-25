"""How a human arm and hand move: joint limits a pose may not pass, and comfortable ranges a pose is pulled
toward where the video doesn't say otherwise (an arm off screen, a hand half hidden). Angles in degrees.

The wrist is measured from the hand's 21 points against the forearm's direction: flexion (palm toward the
inner forearm) / extension (back of the hand up), and radial (toward the thumb) / ulnar (toward the little
finger) deviation. The wrist itself doesn't twist: turning the hand is the forearm's pronation / supination
(fp_twist). The elbow bends 0-150 degrees, never backward, and sits low (below the shoulder-to-wrist line)
unless the video shows otherwise."""
import numpy as np

# wrist: (hard limit, comfortable) per direction
WRIST = {'flexion': (80.0, 15.0), 'extension': (70.0, 25.0), 'radial': (20.0, 10.0), 'ulnar': (35.0, 15.0)}
ELBOW = (0.0, 150.0)       # flexion range
COMFORT_COST = 1.0 / 30.0  # cost per degree past the comfortable range (squared-hinge, per 30 degrees)
LIMIT_COST = 1.0           # cost per degree past the hard limit
ELBOW_UP_COST = 2.0        # cost per metre the elbow sits above the shoulder-to-wrist line (elbows hang low)
UP = np.array([0.0, 0.0, 1.0])


def hand_axes(p, side):
    """The hand's axes from its 21 points: (along: wrist to middle knuckle, palm: out of the palm, thumb: toward
    the thumb side), unit vectors."""
    along = p[9] - p[0]
    along = along / np.linalg.norm(along)
    across = p[5] - p[17]  # index knuckle minus little-finger knuckle: toward the thumb side
    across = across - (across @ along) * along
    across = across / np.linalg.norm(across)
    palm = np.cross(along, across) if side == 'Left' else np.cross(across, along)
    return along, palm / np.linalg.norm(palm), across


def wrist_angles(p, forearm, side):
    """(flexion +/extension -, radial +/ulnar -) of the wrist in degrees: the forearm's direction (elbow to wrist,
    unit) seen in the hand's frame. p: the hand's 21 points."""
    along, palm, thumb = hand_axes(p, side)
    f = forearm / np.linalg.norm(forearm)
    # the hand's direction relative to the forearm's: tilted toward the palm is flexion, toward the thumb radial
    fe = np.degrees(np.arctan2(-(f @ palm), f @ along))
    ru = np.degrees(np.arctan2(-(f @ thumb), f @ along))
    return fe, ru


def _hinge(x, comfort, limit):
    over = max(x - comfort, 0.0)
    hard = max(x - limit, 0.0)
    return (over * COMFORT_COST) ** 2 + hard * LIMIT_COST


def wrist_cost(p, forearm, side):
    """Cost of a wrist pose: nothing within the comfortable ranges, growing past them, steep past the limits."""
    fe, ru = wrist_angles(p, forearm, side)
    c = _hinge(fe, WRIST['flexion'][1], WRIST['flexion'][0]) if fe > 0 else _hinge(-fe, WRIST['extension'][1], WRIST['extension'][0])
    c += _hinge(ru, WRIST['radial'][1], WRIST['radial'][0]) if ru > 0 else _hinge(-ru, WRIST['ulnar'][1], WRIST['ulnar'][0])
    return c


def elbow_cost(shoulder, elbow, wrist):
    """Cost of an elbow: past its range, or raised above the line from the shoulder to the wrist."""
    u, f = elbow - shoulder, wrist - elbow
    flex = np.degrees(np.arccos(np.clip(u @ f / (np.linalg.norm(u) * np.linalg.norm(f)), -1, 1)))
    c = max(flex - ELBOW[1], 0.0) * LIMIT_COST
    mid = shoulder + (wrist - shoulder) * ((elbow - shoulder) @ (wrist - shoulder)) / max((wrist - shoulder) @ (wrist - shoulder), 1e-9)
    c += max((elbow - mid) @ UP, 0.0) * ELBOW_UP_COST
    return c


# share of the hand's size on screen each joint may move when straightening: the fingers and knuckles (what the
# video shows sharpest) stay put; the wrist (at the fuzzy edge of the sleeve, under a glove's cuff) may shift, up to
# STRAIGHT_WRIST more at strength 1: from the camera a hand's tilt is plain to see, so a straighter wrist than the
# video's moves the wrist on screen
STRAIGHT_SCREEN = np.array([0.10] + [0.015] * 4 + [0.03, 0.015, 0.015, 0.015] * 4)
STRAIGHT_WRIST = 1.5
STRAIGHT_THUMB = 0.45
STRAIGHT_CURL = 3.0       # rad: how far the fingers may re-curl at strength 1 to keep their tips where the video has them
STRAIGHT_SCALE = 5.0      # deg off the wrist's rest angle that costs as much as the screen tolerance, at strength 1
STRAIGHT_SCENERY = 0.002  # m: nothing of the hand inside the scenery (soft wall)


def rest_wrist(rig):
    """(flexion, radial) of the wrist in the scene's start pose: how the hand sits on the forearm as built."""
    p = rig.keypoints(np.zeros(rig.size))
    return wrist_angles(p, rig.pose0[rig.hand.hand][:3, 3] - rig.forearm['elbow0'], rig.hand.side)


def straighten(rig, x, elbow, cam, strength, rest, scenery=()):
    """The hand tilted toward its rest angle on the forearm (elbow, fixed, to the wrist) where the video can't
    tell: every joint stays where it is on screen (within STRAIGHT_SCREEN of the hand's size; the wrist within
    STRAIGHT_WRIST more at strength 1), so the fingers and knuckles keep to the video and the wrist gives.
    strength: 0 (as tracked) to 1.
    rest: rest_wrist(rig). Returns params."""
    from scipy.optimize import least_squares
    from motionbridge.core import fp_track as fp
    side = rig.hand.side
    p0 = rig.keypoints(x)
    uv0 = fp.project(p0, cam)
    span = np.linalg.norm(uv0[0] - uv0[12])
    w = strength / STRAIGHT_SCALE
    tol = STRAIGHT_SCREEN.copy()
    tol[0] += strength * STRAIGHT_WRIST
    tol[1:3] += strength * STRAIGHT_THUMB  # the thumb's base turns with the wrist

    def res(y):
        p = rig.keypoints(y)
        fe, ru = wrist_angles(p, p[0] - elbow, side)
        r = [((fp.project(p, cam) - uv0) / (tol[:, None] * span)).ravel(), [w * (fe - rest[0]), w * (ru - rest[1])]]
        for it in scenery:
            r.append(np.maximum(STRAIGHT_SCENERY - it.distance(np.r_[p, fp._surface_points(p)], np.eye(4)), 0) / 0.002)
        r.append((y[rig.na + 3:] - x[rig.na + 3:]) / (0.3 + strength * STRAIGHT_CURL))  # the fingers' shape as tracked
        return np.concatenate([np.atleast_1d(q) for q in r])

    return least_squares(res, np.clip(x, rig.lo + 1e-6, rig.hi - 1e-6), bounds=(rig.lo, rig.hi), max_nfev=3000).x
