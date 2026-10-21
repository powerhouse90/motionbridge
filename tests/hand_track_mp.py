"""MediaPipe hand landmarks for frames in a folder, on the whole frame or on per-frame crops.

python hand_track_mp.py FRAMES_DIR MODEL.task OUT.npz [CROPS.npy]
CROPS: (T, 4) x0, y0, x1, y1 in pixels, one hand per crop. Output: points (T, 21, 3) in frame pixels
(NaN where no hand), world (T, 21, 3) MediaPipe's metric hand shape, score (T,). Runs in its own environment (mediapipe, opencv, numpy only).
"""
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

frames_dir, model, out = sys.argv[1:4]
crops = np.load(sys.argv[4]) if len(sys.argv) > 4 else None
files = sorted(Path(frames_dir).glob('*.png')) or sorted(Path(frames_dir).glob('*.jpg'))
det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mpt.BaseOptions(model_asset_path=model), running_mode=vision.RunningMode.IMAGE,
    num_hands=1, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2))
points = np.full((len(files), 21, 3), np.nan)
world = np.full((len(files), 21, 3), np.nan)  # MediaPipe's metric 3D hand, origin at the hand's centre
score = np.zeros(len(files))
for t, f in enumerate(files):
    img = cv2.imread(str(f))
    x0, y0, x1, y1 = (0, 0, img.shape[1], img.shape[0]) if crops is None else np.round(crops[t]).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if x1 - x0 < 16 or y1 - y0 < 16:
        continue
    crop = img[y0:y1, x0:x1]
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    if r.hand_landmarks:
        lm = r.hand_landmarks[0]
        points[t] = [[q.x * crop.shape[1] + x0, q.y * crop.shape[0] + y0, q.z] for q in lm]
        score[t] = r.handedness[0][0].score
        world[t] = [[q.x, q.y, q.z] for q in r.hand_world_landmarks[0]]
np.savez(out, points=points, world=world, score=score)
print(f'hands found in {int((~np.isnan(points[:, 0, 0])).sum())} of {len(files)} frames')
