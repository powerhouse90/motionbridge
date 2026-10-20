"""Remove frame-to-frame wobble from a SomaMotion without dulling fast moves.

Video pose estimates shake by a few millimetres from frame to frame. A fixed low-pass filter
would also round off punches and kicks, so this uses a One Euro filter (Casiez et al. 2012):
its cutoff rises with each joint's own speed, smoothing hard while a joint is slow (where shake
shows) and barely at all while it moves fast. It runs forwards then backwards, so nothing lags.

Each joint's rotation relative to its parent is filtered, plus the hips' position; bone offsets
stay as they are, so limb lengths cannot drift.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from motionbridge.core.soma import SomaMotion

D_CUTOFF = 1.0  # Hz, smoothing of the speed estimate itself


def _alpha(cutoff, fps):
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau * fps)


def _speed(x, fps):
    """Smoothed per-frame speed of signal x (T, D) -> (T,)."""
    window = min(len(x) - (1 - len(x) % 2), 7)
    if window < 5:
        return np.linalg.norm(np.gradient(x, axis=0), axis=1) * fps
    return np.abs(np.linalg.norm(savgol_filter(x, window, 2, deriv=1, axis=0), axis=1)) * fps


def one_euro(x, speed, fps, min_cutoff, beta):
    """Zero-phase adaptive low-pass of x (T, D); speed (T,) sets each frame's cutoff."""
    a = _alpha(min_cutoff + beta * speed, fps)[:, None]
    y = x.copy()
    for t in range(1, len(y)):  # forwards
        y[t] = y[t - 1] + a[t] * (x[t] - y[t - 1])
    for t in range(len(y) - 2, -1, -1):  # backwards over the forward result
        y[t] = y[t + 1] + a[t] * (y[t] - y[t + 1])
    return y


def settings(strength):
    """strength 0..1 -> (min cutoff Hz, rotation beta per rad/s, position beta per m/s)."""
    strength = float(np.clip(strength, 0, 1))
    return 8.0 - 6.5 * strength, 0.6, 6.0


def smooth(motion, strength=0.5):
    """SomaMotion -> SomaMotion with wobble removed. strength 0 leaves it nearly untouched."""
    if strength <= 0 or motion.frames < 5:
        return motion
    min_cutoff, beta_rot, beta_pos = settings(strength)
    fps, world, parents = motion.fps, motion.world, motion.parents
    T, J = world.shape[:2]
    local = world.copy()
    for j in range(J):
        if parents[j] >= 0:
            local[:, j] = np.linalg.inv(world[:, parents[j]]) @ world[:, j]
    out = local.copy()
    for j in range(J):
        q = Rotation.from_matrix(local[:, j, :3, :3]).as_quat()
        for t in range(1, T):  # keep each quaternion in the same hemisphere as the last
            if np.dot(q[t], q[t - 1]) < 0:
                q[t] = -q[t]
        angular = np.linalg.norm((Rotation.from_quat(q[1:]) * Rotation.from_quat(q[:-1]).inv()).as_rotvec(), axis=1)
        speed = np.r_[angular[:1], angular] * fps
        speed = one_euro(speed[:, None], np.zeros(T), fps, D_CUTOFF * 4, 0)[:, 0]
        q = one_euro(q, speed, fps, min_cutoff, beta_rot)
        out[:, j, :3, :3] = Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True)).as_matrix()
    hips = motion.index('Hips')
    p = local[:, hips, :3, 3]
    out[:, hips, :3, 3] = one_euro(p, _speed(p, fps), fps, min_cutoff, beta_pos)
    rebuilt = out.copy()
    done = set()
    while len(done) < J:  # parents before children
        for j in range(J):
            if j not in done and (parents[j] < 0 or parents[j] in done):
                if parents[j] >= 0:
                    rebuilt[:, j] = rebuilt[:, parents[j]] @ out[:, j]
                done.add(j)
    return SomaMotion(motion.names, parents, rebuilt, motion.rest, fps, motion.source + f' smooth={strength:g}')
