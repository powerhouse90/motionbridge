"""Foot contacts, detected once on the canonical SOMA motion.

A heel (ankle joint) or toe joint is planted when it is close to its own lowest height
in the clip and moving slowly. Hysteresis, minimum span length and gap filling keep the
labels from chattering. A plant also holds only while the joint stays within DRIFT of where
it was planted: video reconstructions slide planted feet, and the instantaneous speed test
alone glued a shuffling step into one long "plant" whose pin the rig then slid towards. Each foot then gets a per-frame lock: 'heel' when the heel is
planted (the ankle is pinned), 'toe' when only the toe is (heel lifted, toe pinned),
or None in the air.
"""
import numpy as np

SOMA_UP = 1
HEIGHT_ON, HEIGHT_OFF = 0.03, 0.05   # m above the joint's own low point: enter / leave contact
SPEED_ON, SPEED_OFF = 0.45, 0.8      # m/s horizontal: enter / leave contact
MIN_SPAN, MAX_GAP = 3, 2             # frames
DRIFT = 0.035                        # m: a plant ends when the joint leaves this radius around its anchor
USE_DRIFT, CENTRAL_SPEED = False, False  # off: mixed on the golden set (runs/body_quality/STATUS.md)
ANCHOR_TAU = 0.4                     # s: the anchor follows slow drift (GEM floats standing feet a few cm
                                     # over seconds) but not a shuffle; release above ~DRIFT / TAU = 9 cm/s


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


def _plants(on, keep, flat, fps):
    """Spans [a, b) planted. Each strict core (on) is cut where it drifts more than DRIFT from its
    running anchor, then grown both ways over lenient frames (keep) that stay within DRIFT of the
    anchor, so a plant's onset is as lenient as its release. Neighbouring plants touch only if their
    anchors agree; otherwise one free frame stays between them so they are pinned separately."""
    T, spans = len(on), []
    rate = min(1.0, 1.0 / (ANCHOR_TAU * fps))
    t = 0
    while t < T:
        if not on[t]:
            t += 1
            continue
        a, anchor = t, flat[t].copy()
        first = anchor.copy()
        while t + 1 < T and on[t + 1] and np.linalg.norm(flat[t + 1] - anchor) < DRIFT:
            t += 1
            anchor += rate * (flat[t] - anchor)
        spans.append([a, t + 1, first, anchor.copy()])  # anchor where it starts and where it ends
        t += 1
    for i, (a, b, first, last) in enumerate(spans):  # grow over lenient frames near each end's anchor
        lo = spans[i - 1][1] if i else 0
        anchor = first.copy()
        while a - 1 >= lo and keep[a - 1] and np.linalg.norm(flat[a - 1] - anchor) < DRIFT:
            a -= 1
            anchor += rate * (flat[a] - anchor)
        hi = spans[i + 1][0] if i + 1 < len(spans) else T
        anchor = last.copy()
        while b < hi and keep[b] and np.linalg.norm(flat[b] - anchor) < DRIFT:
            anchor += rate * (flat[b] - anchor)
            b += 1
        spans[i] = [a, b, first, last]
    merged = []
    for a, b, first, last in spans:
        if merged and a - merged[-1][1] <= MAX_GAP and np.linalg.norm(first - merged[-1][3]) < DRIFT:
            merged[-1][1], merged[-1][3] = b, last  # same spot: one plant across a short gap
        elif merged and a <= merged[-1][1]:
            merged[-1][1] = a - 1  # a different spot: keep a free frame between the two pins
            merged.append([a, b, first, last])
        else:
            merged.append([a, b, first, last])
    out = np.zeros(T, bool)
    for a, b, *_ in merged:
        if b - a >= MIN_SPAN:
            out[a:b] = True
    return out


def joint_contact(positions, joint, fps, ground):
    p = positions[:, joint]
    above = p[:, SOMA_UP] - ground
    height = above - np.percentile(above, 10)  # relative to the joint's own planted height
    flat = np.delete(p, SOMA_UP, axis=1)
    # central differences: a one-sided difference gave frame 0 zero speed, planting any foot moving at the cut
    if CENTRAL_SPEED and len(flat) > 1:
        speed = np.linalg.norm(np.gradient(flat, axis=0), axis=1) * fps
    else:
        speed = np.r_[0.0, np.linalg.norm(np.diff(flat, axis=0), axis=1) * fps]
    speed = np.convolve(np.pad(speed, 1, mode='edge'), np.ones(3) / 3, mode='valid')  # one noisy frame should not break a plant
    on = (height < HEIGHT_ON) & (speed < SPEED_ON)
    keep = (height < HEIGHT_OFF) & (speed < SPEED_OFF)
    if not USE_DRIFT:
        return _clean(_hysteresis(height, speed))
    return _plants(on, keep, flat, fps)


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
