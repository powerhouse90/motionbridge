"""EXPERIMENTAL, NOT USED: correct GEM's head pitch against the 2D face keypoints of the same video.

Result so far: on the H3 walk this tilts the head further down, while the frame shows her
looking ahead. SOMA's eye/jaw joints and ViTPose's points likely do not mean the same place.

GEM's head orientation can sit 15-30 degrees off what the face in the frame shows. Its
companion 2D detector (ViTPose, SOMA keypoints) places head top, jaw and eyes on the image
independently. Per frame, find the tilt of the head about its own left-right axis that best
reprojects those points with GEM's camera, smooth it over time, and apply it to the head's
world rotation (head joint fixed; jaw, eyes and head end follow).
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

from motionbridge.core.soma import SomaMotion

FACE = ('HeadEnd', 'Jaw', 'LeftEye', 'RightEye')
SEARCH = np.arange(-45, 45.25, 0.25)  # degrees
MIN_CONFIDENCE = 0.5


def _project(points, k):
    q = points @ k.T
    return q[..., :2] / q[..., 2:3]


def head_tilt(gem_export, keypoints):
    """Per-frame tilt (degrees, about the head's left axis) and 2D error before/after (px)."""
    d = np.load(gem_export)
    names = d['names'].tolist()
    joints, cams, rot = d['incam_joints'], d['K'], d['incam_transforms']
    rest = d['reference_transforms'][0]
    j = lambda n: names.index(n) - 1  # joints and keypoints omit the static Root
    head, face = j('Head'), [j(n) for n in FACE]
    hi = names.index('Head')
    left_rest = np.array([1., 0, 0])  # SOMA rest: +X is the character's left
    tilt, before, after = np.zeros(len(joints)), [], []
    for t in range(len(joints)):
        weight = keypoints[t, face, 2]
        if weight.min() < MIN_CONFIDENCE:
            tilt[t] = np.nan
            continue
        axis = rot[t, hi, :3, :3] @ rest[hi, :3, :3].T @ left_rest
        axis /= np.linalg.norm(axis)
        offsets = joints[t, face] - joints[t, head]
        rots = Rotation.from_rotvec(np.outer(np.radians(SEARCH), axis)).as_matrix()
        moved = joints[t, head] + np.einsum('sab,fb->sfa', rots, offsets)
        err = (np.linalg.norm(_project(moved, cams[t]) - keypoints[t, face, :2], axis=2) * weight).sum(1) / weight.sum()
        best = int(np.argmin(err))
        tilt[t] = SEARCH[best]
        before.append(err[np.argmin(np.abs(SEARCH))])
        after.append(err[best])
    return tilt, float(np.mean(before)) if before else 0.0, float(np.mean(after)) if after else 0.0


def smooth(tilt, fps):
    """Fill undetected frames, remove single-frame outliers, then low-pass (~0.2 s)."""
    t = tilt.copy()
    ok = ~np.isnan(t)
    if not ok.any():
        return np.zeros_like(t)
    t[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), t[ok])
    t = median_filter(t, size=5, mode='nearest')
    half = max(1, int(round(0.1 * fps)))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(np.pad(t, half, mode='edge'), kernel, mode='valid')


def apply_tilt(motion, tilt_deg):
    """Rotate the head and everything under it about the head joint, per frame (world space)."""
    world = motion.world.copy()
    hi = motion.index('Head')
    under = [i for i in range(len(motion.names)) if hi in _ancestors(motion.parents, i)]
    for t, deg in enumerate(tilt_deg):
        axis = world[t, hi, :3, :3] @ motion.rest[hi, :3, :3].T @ np.array([1., 0, 0])
        r = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.radians(deg)).as_matrix()
        pivot = world[t, hi, :3, 3].copy()
        for i in under:
            world[t, i, :3, :3] = r @ world[t, i, :3, :3]
            world[t, i, :3, 3] = pivot + r @ (world[t, i, :3, 3] - pivot)
    return SomaMotion(motion.names, motion.parents, world, motion.rest, motion.fps,
                      motion.source + ' +head_pose')


def _ancestors(parents, i):
    out = []
    while i >= 0:
        out.append(i)
        i = parents[i]
    return out


def refine(motion, gem_export, keypoints):
    """Corrected motion plus a short report."""
    raw, before, after = head_tilt(gem_export, keypoints)
    tilt = np.clip(smooth(raw, motion.fps), SEARCH[0], SEARCH[-1])
    report = {'tilt_median_deg': round(float(np.median(tilt)), 1), 'tilt_range_deg':
              [round(float(tilt.min()), 1), round(float(tilt.max()), 1)],
              'face_error_px': [round(before, 1), round(after, 1)],
              'frames_without_face': int(np.isnan(raw).sum())}
    return apply_tilt(motion, tilt), report
