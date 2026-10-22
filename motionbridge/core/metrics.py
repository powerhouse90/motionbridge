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


def planted_skate(positions, joint, contact, up):
    """Mean horizontal drift (m per frame) of a joint over consecutive frames both marked in contact."""
    flat = [i for i in range(3) if i != up]
    step = np.linalg.norm(np.diff(positions[:, joint][:, flat], axis=0), axis=1)
    both = np.asarray(contact[1:]) & np.asarray(contact[:-1])
    return float(step[both].mean()) if both.any() else 0.0, int(both.sum())


def floor_penetration(positions, joints, floors, up):
    """How far the lowest of the given joints goes below its own floor height, per frame (m, >= 0).
    floors: each joint's height when standing on the floor (its rest height)."""
    depth = np.asarray(floors)[None] - positions[:, joints, up]
    return np.maximum(depth.max(axis=1), 0)


def bone_length_drift(positions, parents, skip=()):
    """Largest relative change (max - min) / mean over time of any joint-to-parent distance.
    skip: joints that translate by design (the root / hips) are left out."""
    worst = 0.0
    for i, p in enumerate(parents):
        if p is None or p < 0 or i in skip:
            continue
        d = np.linalg.norm(positions[:, i] - positions[:, p], axis=1)
        if d.mean() > 1e-3:
            worst = max(worst, float((d.max() - d.min()) / d.mean()))
    return worst


def knee_violations(hip, knee, ankle, toe, max_flex=np.radians(160), reverse_min=np.radians(10)):
    """Frames where a knee bends backwards (away from the toes) by more than reverse_min, or flexes
    past max_flex. Inputs are (T, 3) joint positions; returns (reverse frames, overflex frames)."""
    thigh, shin = knee - hip, ankle - knee
    cos = np.einsum('ti,ti->t', thigh, shin) / (np.linalg.norm(thigh, axis=1) * np.linalg.norm(shin, axis=1) + 1e-12)
    flex = np.arccos(np.clip(cos, -1, 1))
    line = ankle - hip
    n = line / (np.linalg.norm(line, axis=1, keepdims=True) + 1e-12)
    out = (knee - hip) - np.einsum('ti,ti->t', knee - hip, n)[:, None] * n  # where the knee sticks out
    foot = toe - ankle
    foot = foot - np.einsum('ti,ti->t', foot, n)[:, None] * n
    reverse = (flex > reverse_min) & (np.einsum('ti,ti->t', out, foot) < 0)
    return int(reverse.sum()), int((flex > max_flex).sum())


def floor_skate(positions, joints, floors, up, touch=0.01):
    """Detector-free foot skate: mean horizontal step (m per frame) of every foot point that is within
    `touch` of its floor height on both frames. This is what reads as skating on screen."""
    flat = [i for i in range(3) if i != up]
    steps = []
    for j, f in zip(joints, floors):
        p = positions[:, j]
        on = np.abs(p[:, up] - f) < touch
        both = on[1:] & on[:-1]
        steps.append(np.linalg.norm(np.diff(p[:, flat], axis=0), axis=1)[both])
    steps = np.concatenate(steps) if steps else np.zeros(0)
    return float(steps.mean()) if len(steps) else 0.0, int(len(steps))
