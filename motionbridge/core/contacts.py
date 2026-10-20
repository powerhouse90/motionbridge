"""Foot contacts, detected once on the canonical SOMA motion.

A heel (ankle joint) or toe joint is planted when it is close to its own lowest height
in the clip and moving slowly. Hysteresis, minimum span length and gap filling keep the
labels from chattering. Each foot then gets a per-frame lock: 'heel' when the heel is
planted (the ankle is pinned), 'toe' when only the toe is (heel lifted, toe pinned),
or None in the air.
"""
import numpy as np

SOMA_UP = 1
HEIGHT_ON, HEIGHT_OFF = 0.03, 0.05   # m above the joint's own low point: enter / leave contact
SPEED_ON, SPEED_OFF = 0.45, 0.8      # m/s horizontal: enter / leave contact
MIN_SPAN, MAX_GAP = 3, 2             # frames


def _hysteresis(height, speed):
    on = (height < HEIGHT_ON) & (speed < SPEED_ON)
    keep = (height < HEIGHT_OFF) & (speed < SPEED_OFF)
    state, out = False, np.zeros(len(height), bool)
    for t in range(len(height)):
        state = on[t] or (state and keep[t])
        out[t] = state
    return out


def _clean(mask):
    """Fill gaps of at most MAX_GAP frames, then drop spans shorter than MIN_SPAN."""
    m = mask.copy()
    t = 0
    while t < len(m):
        if not m[t]:
            end = t
            while end < len(m) and not m[end]:
                end += 1
            if 0 < t and end < len(m) and end - t <= MAX_GAP:
                m[t:end] = True
            t = end
        else:
            t += 1
    t = 0
    while t < len(m):
        if m[t]:
            end = t
            while end < len(m) and m[end]:
                end += 1
            if end - t < MIN_SPAN:
                m[t:end] = False
            t = end
        else:
            t += 1
    return m


def ground_track(motion, window_s=1.5):
    """Floor height per frame. Video reconstruction lets the floor drift (several cm over a few
    seconds), so use a smoothed rolling minimum of the lowest foot joint instead of one constant.
    The window is long enough to span a jump's airtime."""
    feet = [motion.index(f'{s}{x}') for s in ('Left', 'Right') for x in ('Foot', 'ToeBase')]
    low = motion.positions[:, feet, SOMA_UP].min(axis=1)
    half = max(1, int(window_s * motion.fps / 2))
    padded = np.pad(low, half, mode='edge')
    rolling = np.array([padded[t:t + 2 * half + 1].min() for t in range(len(low))])
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(np.pad(rolling, half, mode='edge'), kernel, mode='valid')


def joint_contact(positions, joint, fps, ground):
    p = positions[:, joint]
    above = p[:, SOMA_UP] - ground
    height = above - np.percentile(above, 10)  # relative to the joint's own planted height
    flat = np.delete(p, SOMA_UP, axis=1)
    speed = np.r_[0.0, np.linalg.norm(np.diff(flat, axis=0), axis=1) * fps]
    speed = np.convolve(speed, np.ones(3) / 3, mode='same')  # one noisy frame should not break a plant
    return _clean(_hysteresis(height, speed))


def detect(motion):
    """{'Left': [lock per frame], 'Right': [...]} with lock in {'heel', 'toe', None}."""
    p = motion.positions
    ground = ground_track(motion)
    out = {}
    for side in ('Left', 'Right'):
        heel = joint_contact(p, motion.index(side + 'Foot'), motion.fps, ground)
        toe = joint_contact(p, motion.index(side + 'ToeBase'), motion.fps, ground)
        out[side] = ['heel' if h else 'toe' if o else None for h, o in zip(heel, toe)]
    return out


def summary(locks):
    """Compact text for reports, e.g. 'HHHHttt....HHH' (H heel, t toe, . air)."""
    return ''.join({'heel': 'H', 'toe': 't', None: '.'}[x] for x in locks)
