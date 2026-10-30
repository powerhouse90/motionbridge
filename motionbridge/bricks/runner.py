"""MotionBricks transition, run with MotionBricks' own Python (its repo's .venv). One generation: four context
poses and four target poses in, the motion between them out, at 30 Hz.

python runner.py --repo MOTIONBRICKS --inp in.npz --out out.npz

in.npz: rotations (8, 77, 4) local XYZW quaternions, SOMA joint order (context then target); hips (8, 3) the hips'
position in model space (Y up, MotionBricks' proportions); tokens: 0 lets the model choose the length, else 6-16
(4 poses each). out.npz: rotations (N, 77, 4), hips (N, 3), contacts (N, 4), from the last context pose to the
first target pose.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
p.add_argument('--inp', required=True)
p.add_argument('--out', required=True)
p.add_argument('--fp32', action='store_true', help='full precision (more GPU memory)')
a = p.parse_args()
os.chdir(a.repo)  # TensorRT plans are cached under the repo
sys.path.insert(0, a.repo)

import warp as wp  # noqa: E402

wp.init()
wp.set_device('cuda:0')
from motionbricks.io.usd import load_skeleton_from_usd  # noqa: E402
from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH, get_asset_file  # noqa: E402
from motionbricks.model.config import MotionBricksInferenceConfig  # noqa: E402
from motionbricks.model.soma.contract import SOMA_N1_FP16_ONNX_PATH, SOMA_N1_FP32_ONNX_PATH  # noqa: E402
from motionbricks.model.soma.engine import MotionBricksSOMAEngine  # noqa: E402

d = np.load(a.inp)
skeleton = load_skeleton_from_usd(get_asset_file('human', 'soma_base_skel_minimal_no_root.usd'), SOMA_SKELETON_PRIM_PATH)
reference = skeleton.reference_local_transforms.numpy()


def poses(rotations, hips):
    """Local transforms (n, 77, 7): MotionBricks' own bone offsets, the given rotations, the hips placed."""
    out = np.repeat(reference[None], len(rotations), 0).astype(np.float32)
    out[:, :, 3:] = rotations
    out[:, 0, :3] = hips
    return out


onnx = SOMA_N1_FP32_ONNX_PATH if a.fp32 else SOMA_N1_FP16_ONNX_PATH
tokens = int(d['tokens'])
config = MotionBricksInferenceConfig(allowed_token_counts=(tokens,)) if tokens else MotionBricksInferenceConfig()
with MotionBricksSOMAEngine(onnx, model_to_runtime=wp.transform_identity()) as engine, wp.ScopedStream(engine.stream):
    context, target = engine.create_input(skeleton), engine.create_input(skeleton)
    output = engine.create_output(skeleton)
    both = poses(d['rotations'], d['hips'])
    context.update_from_poses(wp.array(both[:4], dtype=wp.transform), stream=engine.stream)
    target.update_from_poses(wp.array(both[4:], dtype=wp.transform), stream=engine.stream)
    engine.generate_animation(context, target, output, config=config)
    engine.synchronize()
    n = output.num_frames
    gen = output.poses[:n].numpy().reshape(n, -1, 7)
    contacts = output.foot_contacts[:n].numpy() if hasattr(output, 'foot_contacts') else np.zeros((n, 4))
np.savez(a.out, rotations=gen[:, :, 3:], hips=gen[:, 0, :3], contacts=contacts, bone_lengths=reference[:, :3])
print(f'MOTIONBRICKS ok: {n} frames ({n / 30:.2f} s)', flush=True)
