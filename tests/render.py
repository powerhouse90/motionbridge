"""Draw skeletons as front and side views for contact sheets. Test tooling only."""
import subprocess

import cv2
import numpy as np

PANEL = (280, 340)  # width, height
LEFT, RIGHT, CENTER = (235, 140, 40), (40, 60, 220), (170, 170, 170)  # BGR


def side_colors(names):
    """Left limbs blue, right limbs red, everything else gray. Matches common naming styles."""
    colors = []
    for n in names:
        low = n.lower()
        if low.startswith(('left', 'l_', 'l.')) or low.endswith(('_l', '.l', 'left')) or 'left' in low:
            colors.append(LEFT)
        elif low.startswith(('right', 'r_', 'r.')) or low.endswith(('_r', '.r', 'right')) or 'right' in low:
            colors.append(RIGHT)
        else:
            colors.append(CENTER)
    return colors


class Views:
    """Fixed framing for a whole clip so motion is visible, not re-centred every frame."""

    def __init__(self, positions, up, forward, ground):
        self.up, self.forward, self.ground = up, forward, ground
        self.across = 3 - up - forward
        lo, hi = positions.reshape(-1, 3).min(0), positions.reshape(-1, 3).max(0)
        span = max(hi[self.across] - lo[self.across], hi[forward] - lo[forward], hi[up] - ground) * 1.15
        self.scale = (PANEL[1] - 40) / max(span, 1e-3)
        self.mid = (lo + hi) / 2

    def project(self, p, horizontal, flip):
        x = (p[horizontal] - self.mid[horizontal]) * self.scale * (-1 if flip else 1) + PANEL[0] / 2
        y = PANEL[1] - 20 - (p[self.up] - self.ground) * self.scale
        return int(round(x)), int(round(y))

    def draw(self, pose, parents, colors, label, front):
        img = np.full((PANEL[1], PANEL[0], 3), 24, np.uint8)
        horizontal = self.across if front else self.forward
        floor_y = self.project(np.full(3, self.ground), horizontal, False)[1]
        cv2.line(img, (0, floor_y), (PANEL[0], floor_y), (70, 70, 70), 1)
        for j, parent in enumerate(parents):
            if parent < 0 or parents[parent] < 0:  # skip the static root's link to the hips
                continue
            a = self.project(pose[j], horizontal, front)
            b = self.project(pose[parent], horizontal, front)
            cv2.line(img, a, b, colors[j], 2, cv2.LINE_AA)
        cv2.putText(img, label + (' front' if front else ' side'), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        return img


def segments_panel(segments, colors, left, forward, up, ground, label, on_top=()):
    """Front view of fixed segments [(a, b), ...] as seen by someone facing the character.

    left/forward/up are the character's directions; its left appears on the image's right.
    Segments whose index is in on_top are drawn last.
    """
    basis = np.array([left, forward, up], float)
    segments = [(basis @ a, basis @ b) for a, b in segments]
    points = np.array([p for s in segments for p in s])[None]
    views = Views(points, 2, 1, ground)
    img = np.full((PANEL[1], PANEL[0], 3), 24, np.uint8)
    order = [i for i in range(len(segments)) if i not in on_top] + list(on_top)
    for i in order:
        a, b = segments[i]
        cv2.line(img, views.project(a, 0, False), views.project(b, 0, False), colors[i], 2, cv2.LINE_AA)
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def animated_segments(heads, tails, colors, left, forward, up, ground, label):
    """Front view per frame of moving segments; heads/tails (T, N, 3). Fixed framing for the clip."""
    basis = np.array([left, forward, up], float)
    h, t = heads @ basis.T, tails @ basis.T
    views = Views(np.concatenate([h, t], axis=1), 2, 1, ground)
    frames = []
    for f in range(len(h)):
        img = np.full((PANEL[1], PANEL[0], 3), 24, np.uint8)
        floor_y = views.project(np.array([0, 0, ground]), 0, False)[1]
        cv2.line(img, (0, floor_y), (PANEL[0], floor_y), (70, 70, 70), 1)
        for i in np.argsort([c == CENTER for c in colors], kind='stable')[::-1]:
            cv2.line(img, views.project(h[f, i], 0, False), views.project(t[f, i], 0, False), colors[i], 2,
                     cv2.LINE_AA)
        cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 220, 220), 1, cv2.LINE_AA)
        frames.append(img)
    return frames


def write_grid(path, columns, picks=4):
    """Still sheet: one row per picked frame, one column per clip."""
    length = min(len(c) for c in columns)
    idx = np.linspace(0, length - 1, picks).round().astype(int)
    cv2.imwrite(str(path), np.vstack([np.hstack([c[i] for c in columns]) for i in idx]))


def clip_frames(positions, parents, names, up, forward, ground, label):
    """One image per frame: front view beside side view."""
    views, colors = Views(positions, up, forward, ground), side_colors(names)
    return [np.hstack([views.draw(p, parents, colors, label, True), views.draw(p, parents, colors, label, False)])
            for p in positions]


def write_video(path, rows, fps):
    """rows: list of frame lists, stacked vertically; shorter clips hold their last frame."""
    length = max(len(r) for r in rows)
    first = np.vstack([r[0] for r in rows])
    # H.264 through ffmpeg: OpenCV's mp4v output does not play in browsers or app previews.
    h, w = first.shape[:2]
    ffmpeg = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt',
                               'bgr24', '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt',
                               'yuv420p', '-crf', '20', str(path)], stdin=subprocess.PIPE)
    for t in range(length):
        ffmpeg.stdin.write(np.ascontiguousarray(np.vstack([r[min(t, len(r) - 1)] for r in rows])).tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError(f'ffmpeg failed writing {path}')


def write_strip(path, rows, picks=5):
    """Still contact sheet: evenly spaced frames of each clip, one clip per row."""
    lines = []
    for r in rows:
        idx = np.linspace(0, len(r) - 1, picks).round().astype(int)
        lines.append(np.hstack([r[i] for i in idx]))
    width = max(l.shape[1] for l in lines)
    lines = [np.pad(l, ((0, 0), (0, width - l.shape[1]), (0, 0))) for l in lines]
    cv2.imwrite(str(path), np.vstack(lines))
