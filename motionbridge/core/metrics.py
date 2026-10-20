"""Measurements shared by source motions and retargeted results.

All functions take joint positions (T, J, 3) in meters plus the index of the up axis,
so SOMA (Y-up) and Blender (Z-up) results are measured the same way.
"""
import numpy as np


def ground_height(positions, feet, up):
    """Robust floor estimate: a low percentile of the lowest foot joint per frame."""
    return float(np.percentile(positions[:, feet, up].min(axis=1), 5))


def foot_slide(positions, foot, up, fps, ground, contact_height=0.03):
    """Horizontal foot speed (m/s) on frames where the foot is within contact_height of the floor.

    Returns (mean speed during contact, fraction of frames in contact). A planted foot should be near 0.
    """
    p = positions[:, foot]
    flat = [i for i in range(3) if i != up]
    speed = np.linalg.norm(np.diff(p[:, flat], axis=0), axis=1) * fps
    contact = (p[1:, up] - ground) < contact_height
    if not contact.any():
        return 0.0, 0.0
    return float(speed[contact].mean()), float(contact.mean())


def root_travel(positions, root, up):
    """Horizontal distance (m) the root covers from first to last frame, and its path length."""
    flat = [i for i in range(3) if i != up]
    p = positions[:, root][:, flat]
    return float(np.linalg.norm(p[-1] - p[0])), float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def jitter(positions, fps):
    """Mean joint acceleration magnitude (m/s^2); high values mean shaky motion."""
    if len(positions) < 3:
        return 0.0
    acc = np.diff(positions, n=2, axis=0) * fps * fps
    return float(np.linalg.norm(acc, axis=2).mean())
