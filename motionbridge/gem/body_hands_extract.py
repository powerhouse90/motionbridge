"""Full-body hands, the learned part, in GEM's environment: SAM 3D Body's hand decoder on a crop around each
hand on every frame, read the same way as the first-person path (fp_extract.py: several boxes, median, a left
hand read from the mirrored image).

python body_hands_extract.py --gem-root DIR --video clip.mp4 --extraction GEM_OUT --out hands.npz

The crops come from GEM itself: its motion_export.npz holds the body's joints in camera space (incam_joints)
and the lens per frame (K), so each hand's wrist, elbow and finger joints project onto the frame without
another detector. Writes, per side S in (Right, Left):
  kp2d_S, kp3d_S   (T, boxes, 70, 2|3)  SAM 3D Body's 70 keypoints per box (camera space, metres; a left hand
                                         sits in the right-hand slots, mirrored back to the real image)
  hand2d_S, hand3d_S (T, 21, 2|3)       MediaPipe-21 order (wrist, then thumb..pinky base to tip), median of boxes
  spread_S         (T,)                 how much the boxes disagree: median 2D distance from the median, as a
                                         fraction of the hand's size on screen (a confident read is ~0.02;
                                         a garbage read, e.g. a hidden hand, is several times that)
  span_S           (T,)                 the hand's size on screen, px (NaN where the hand is off frame)
Note: SAM 3D Body and MHR are under Meta's SAM License, not Apache-2.0.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BOX_SCALES = (0.85, 1.0, 1.2)
BOX_SHIFTS = (0.0, 0.12, -0.12)        # along the forearm, fractions of the box size
MHR_RIGHT = [41] + [21 + 4 * f + k for f in range(5) for k in (3, 2, 1, 0)]  # MHR70 -> MediaPipe 21 (as fp_track)
MIN_SPAN = 12                          # px: smaller hands are not read
EDGE = 0.25                            # a wrist this far (fractions of the span) outside the frame is off frame


def hand_boxes(wrist, elbow, centre, span):
    """Boxes (n, 4) xyxy around a hand: centred on its joints, nudged along the forearm."""
    along = wrist - elbow
    along = along / max(np.linalg.norm(along), 1e-6)
    h = 0.8 * span
    return np.array([np.r_[centre + along * d * h - s * h, centre + along * d * h + s * h]
                     for s in BOX_SCALES for d in BOX_SHIFTS])


def project(K, p):
    q = p @ K.T
    return q[..., :2] / q[..., 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gem-root', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--extraction', required=True, help='GEM output folder (holds motion_export.npz)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    gem = Path(a.gem_root).resolve()
    export = np.load(Path(a.extraction) / 'motion_export.npz')
    names = [str(n) for n in export['names'][1:]]   # incam_joints omits the static Root
    J, Ks = export['incam_joints'].astype(np.float64), export['K'].astype(np.float64)

    script_dir = Path(__file__).resolve().parent
    sys.path[:] = [e for e in sys.path if Path(e or os.curdir).resolve() != script_dir]
    sys.path[:0] = [str(gem), str(gem / 'third_party' / 'sam-3d-body')]
    load = torch.load
    torch.load = lambda *x, **k: load(*x, **{**k, 'weights_only': False})  # the checkpoint holds plain objects
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to
    model, cfg = load_sam_3d_body(str(gem / 'inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt'), device='cuda',
                                  mhr_path=str(gem / 'inputs/mhr_data/mhr_model.pt'))
    estimator = SAM3DBodyEstimator(model, cfg)

    capture = cv2.VideoCapture(str(a.video))
    T = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(J))
    n = len(BOX_SCALES) * len(BOX_SHIFTS)
    sides = ('Right', 'Left')
    out = {}
    for s in sides:
        out[f'kp2d_{s}'] = np.full((T, n, 70, 2), np.nan, np.float32)
        out[f'kp3d_{s}'] = np.full((T, n, 70, 3), np.nan, np.float32)
        out[f'span_{s}'] = np.full(T, np.nan, np.float32)
    for t in range(T):
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        K = Ks[t]
        for s in sides:
            j = lambda x: J[t, names.index(s + x)]
            fingers = np.array([j(f'Hand{f}End') for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')])
            wrist3, tip3 = j('Hand'), j('HandMiddleEnd')
            span = K[0, 0] * np.linalg.norm(tip3 - wrist3) / wrist3[2]  # the hand's length on screen, unforeshortened
            wrist, elbow = project(K, wrist3), project(K, j('ForeArm'))
            centre = project(K, np.r_[wrist3[None], fingers]).mean(0)
            inside = -EDGE * span < wrist[0] < W + EDGE * span and -EDGE * span < wrist[1] < H + EDGE * span
            if span < MIN_SPAN or wrist3[2] <= 0 or not inside:
                continue
            out[f'span_{s}'][t] = span
            mirror = s == 'Left'  # the hand decoder knows right hands only
            if mirror:
                flip = lambda p: np.array([W - p[0], p[1]])
                wrist, elbow, centre = flip(wrist), flip(elbow), flip(centre)
            boxes = hand_boxes(wrist, elbow, centre, span)
            image = np.ascontiguousarray(rgb[:, ::-1]) if mirror else rgb
            Km = K.copy()
            if mirror:
                Km[0, 2] = W - K[0, 2]
            with torch.no_grad():
                batch = recursive_to(prepare_batch(image, estimator.transform_hand, boxes,
                                                   cam_int=torch.from_numpy(Km).float()[None]), 'cuda')
                model._initialize_batch(batch)
                res = recursive_to(recursive_to(model.forward_step(batch, decoder_type='hand')['mhr_hand'], 'cpu'), 'numpy')
            kp2d = res['pred_keypoints_2d'][:, :70].copy()
            kp3d = (res['pred_keypoints_3d'] + res['pred_cam_t'][:, None])[:, :70].copy()
            if mirror:  # back to the real image: x mirrored about the image's centre line
                kp2d[..., 0] = W - kp2d[..., 0]
                kp3d[..., 0] = -kp3d[..., 0]
            out[f'kp2d_{s}'][t], out[f'kp3d_{s}'][t] = kp2d, kp3d
        print('hands', t, flush=True)
    for s in sides:
        k2, k3 = out[f'kp2d_{s}'][:, :, MHR_RIGHT], out[f'kp3d_{s}'][:, :, MHR_RIGHT]
        with np.errstate(all='ignore'):
            import warnings
            warnings.simplefilter('ignore', RuntimeWarning)
            out[f'hand2d_{s}'], out[f'hand3d_{s}'] = np.nanmedian(k2, axis=1), np.nanmedian(k3, axis=1)
            dev = np.linalg.norm(k2 - out[f'hand2d_{s}'][:, None], axis=-1)       # (T, boxes, 21)
            out[f'spread_{s}'] = np.nanmedian(dev, axis=(1, 2)) / out[f'span_{s}']
    np.savez_compressed(a.out, **out)
    print('BODYHANDS ok', T, 'frames', flush=True)


if __name__ == '__main__':
    main()
