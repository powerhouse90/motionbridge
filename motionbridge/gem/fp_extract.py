"""First-person hands, the learned part, in GEM's environment: SAM 2 masks of each arm and each item
through the clip, then SAM 3D Body's hand decoder on every frame for each hand.

python fp_extract.py --gem-root DIR --sam2-checkpoint FILE --frames DIR --prompts prompts.json --out out.npz

prompts.json: {"arms": {"Right": [[x, y], ...], "Left": [...]}, "items": [[[x, y], ...], ...], "focal": px,
"spans": {"Right": hand size in px, ...}}, all on frame 0, where the scene is known (the start frame was
rendered from it). Writes, per arm side S and item k: arm_S (T, H, W), item_k (T, H, W) masks, and
kp2d_S, kp3d_S (T, boxes, 70, 2|3): SAM 3D Body's 70 keypoints for several boxes around the hand
(camera space, metres; a left hand's points sit in the right-hand slots, mirrored back), so the caller
can take a robust median. The hand decoder knows right hands only, so a left hand is read from the
mirrored image, as SAM 3D Body itself does.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

BOX_SCALES = (0.85, 1.0, 1.2)
BOX_SHIFTS = ((0, 0), (0.12, 0), (-0.12, 0))


def masks(frames_dir, checkpoint, objects, T, H, W):
    """objects: {key: prompt points} -> {key: (T, H, W) bool}."""
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml', str(checkpoint), device='cuda')
    keys = list(objects)
    out = {k: np.zeros((T, H, W), bool) for k in keys}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(frames_dir), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for obj, key in enumerate(keys, 1):
            pts = np.array(objects[key], np.float32)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=obj, points=pts, labels=np.ones(len(pts), np.int32))
        for t, ids, logits in predictor.propagate_in_video(state):
            for k, obj in enumerate(ids):
                out[keys[obj - 1]][t] = (logits[k, 0] > 0).cpu().numpy()
            print('masks', t, flush=True)
    del predictor
    torch.cuda.empty_cache()
    return out


def hand_boxes(mask, span):
    """Boxes around the hand: the arm enters from the image border, the hand is its far end."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    edge = (xs < 3) | (xs > W - 4) | (ys < 3) | (ys > H - 4)
    entry = np.array([xs[edge].mean(), ys[edge].mean()]) if edge.any() else np.array([W / 2, H])
    far = np.c_[xs, ys][np.argmax(np.hypot(xs - entry[0], ys - entry[1]))].astype(float)
    centre = far - (far - entry) / max(np.linalg.norm(far - entry), 1e-6) * 0.55 * span
    h = 0.8 * span
    return np.array([np.r_[centre + np.array(d) * h - s * h, centre + np.array(d) * h + s * h]
                     for s in BOX_SCALES for d in BOX_SHIFTS])


def hands(frames, arm_masks, gem_root, focal, spans):
    """{side: (kp2d, kp3d)} for the sides in arm_masks."""
    sys.path[:0] = [str(gem_root), str(gem_root / 'third_party' / 'sam-3d-body')]
    load = torch.load
    torch.load = lambda *a, **k: load(*a, **{**k, 'weights_only': False})  # the checkpoint holds plain objects
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    from sam_3d_body.data.utils.prepare_batch import prepare_batch
    from sam_3d_body.utils import recursive_to
    model, cfg = load_sam_3d_body(str(gem_root / 'inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt'), device='cuda',
                                  mhr_path=str(gem_root / 'inputs/mhr_data/mhr_model.pt'))
    estimator = SAM3DBodyEstimator(model, cfg)
    n = len(BOX_SCALES) * len(BOX_SHIFTS)
    out = {s: (np.full((len(frames), n, 70, 2), np.nan, np.float32), np.full((len(frames), n, 70, 3), np.nan, np.float32))
           for s in arm_masks}
    for t, f in enumerate(frames):
        rgb = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        K = torch.tensor([[[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]]], dtype=torch.float32)
        for side, m in arm_masks.items():
            mirror = side == 'Left'
            boxes = hand_boxes(m[t][:, ::-1] if mirror else m[t], spans[side])
            if boxes is None:
                continue
            image = np.ascontiguousarray(rgb[:, ::-1]) if mirror else rgb
            with torch.no_grad():
                batch = recursive_to(prepare_batch(image, estimator.transform_hand, boxes, cam_int=K.clone()), 'cuda')
                model._initialize_batch(batch)
                res = recursive_to(recursive_to(model.forward_step(batch, decoder_type='hand')['mhr_hand'], 'cpu'), 'numpy')
            kp2d = res['pred_keypoints_2d'][:, :70].copy()
            kp3d = (res['pred_keypoints_3d'] + res['pred_cam_t'][:, None])[:, :70].copy()
            if mirror:  # back to the real image: x mirrored about the image centre
                kp2d[..., 0] = W - kp2d[..., 0]
                kp3d[..., 0] = -kp3d[..., 0]
            out[side][0][t], out[side][1][t] = kp2d, kp3d
        print('hands', t, flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gem-root', required=True)
    p.add_argument('--sam2-checkpoint', required=True)
    p.add_argument('--frames', required=True, help='folder of 00000.jpg ...')
    p.add_argument('--prompts', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    frames = sorted(Path(a.frames).glob('*.jpg'))
    print('frames', len(frames), flush=True)
    H, W = cv2.imread(str(frames[0])).shape[:2]
    prompts = json.loads(Path(a.prompts).read_text())
    objects = {f'arm_{s}': pts for s, pts in prompts['arms'].items()}
    objects.update({f'item_{k}': pts for k, pts in enumerate(prompts['items'])})
    m = masks(a.frames, a.sam2_checkpoint, objects, len(frames), H, W)
    print('masks done', flush=True)
    found = hands(frames, {s: m[f'arm_{s}'] for s in prompts['arms']}, Path(a.gem_root), prompts['focal'], prompts['spans'])
    result = dict(m)
    for s, (k2, k3) in found.items():
        result[f'kp2d_{s}'], result[f'kp3d_{s}'] = k2, k3
    np.savez_compressed(a.out, **result)
    print('FPEXTRACT ok', len(frames), 'frames,', ', '.join(objects))


if __name__ == '__main__':
    main()
