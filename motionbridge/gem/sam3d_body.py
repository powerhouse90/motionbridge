"""Per-frame SAM 3D Body (Meta) on a video, reusing GEM's person boxes and camera.

Run with the GEM environment's Python (SAM 3D Body ships inside the GEM-X checkout):
    python motionbridge/gem/sam3d_body.py --gem-root GEM --video clip.mp4 --extraction DIR --out sam3d.npz

GEM only uses SAM 3D Body's image features and predicts posture with its own temporal model.
This runs the full SAM 3D Body pose head per frame, to compare posture against GEM.
Note: SAM 3D Body and MHR are under Meta's SAM License, not Apache-2.0.
"""
import argparse
import os
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--gem-root', required=True)
p.add_argument('--video', required=True)
p.add_argument('--extraction', required=True, help='GEM extraction folder (boxes and camera are reused)')
p.add_argument('--out', required=True)
a = p.parse_args()
gem = Path(a.gem_root).resolve()
out = Path(a.out).resolve()
video = Path(a.video).resolve()
extraction = Path(a.extraction).resolve()
script_dir = Path(__file__).resolve().parent
sys.path[:] = [e for e in sys.path if Path(e or os.curdir).resolve() != script_dir]
sys.path.insert(0, str(gem))
sys.path.insert(0, str(gem / 'third_party' / 'sam-3d-body'))
os.chdir(gem)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

original_load = torch.load


def trusted_load(*args, **kwargs):  # installed model files are trusted local dependencies
    kwargs.setdefault('weights_only', False)
    return original_load(*args, **kwargs)


torch.load = trusted_load
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402

model, cfg = load_sam_3d_body(str(gem / 'inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt'),
                              device=torch.device('cuda'), mhr_path=str(gem / 'inputs/mhr_data/mhr_model.pt'))
estimator = SAM3DBodyEstimator(model, cfg)
boxes = original_load(extraction / 'capture' / 'preprocess' / 'bbx.pt', weights_only=True)['bbx_xyxy'].numpy()
cams = np.load(extraction / 'motion_export.npz')['K']
capture = cv2.VideoCapture(str(video))
keep = ('pred_vertices', 'pred_cam_t', 'pred_keypoints_3d', 'pred_joint_coords', 'pred_global_rots', 'focal_length')
frames = {k: [] for k in keep}
t = 0
while True:
    ok, bgr = capture.read()
    if not ok:
        break
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = estimator.process_one_image(rgb, bboxes=boxes[t].reshape(1, 4),
                                         cam_int=torch.from_numpy(cams[t]).float()[None], inference_type='full')
    for k in keep:
        frames[k].append(np.asarray(result[0][k], dtype=np.float32))
    t += 1
    print('SAM3D frame', t, flush=True)
np.savez_compressed(out, faces=estimator.faces, **{k: np.stack(v) for k, v in frames.items()})
print('SAM3D_DONE', out, flush=True)
