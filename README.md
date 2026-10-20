# MotionBridge

Type an action, get it on your own rigged character, any body shape, on an 8GB GPU.

MotionBridge is a ComfyUI node pack. A motion source (MiniMax H3 video through
NVIDIA GEM, or text through NVIDIA Kimodo) writes one canonical skeleton, SOMA's
77 joints. Cleanup runs once on that skeleton. A generic retargeter then maps it
onto any humanoid rig without per-character calibration.

**Status:** early scaffold. Nothing here runs in ComfyUI yet.

## Layout

| Folder | Contents | May import |
| --- | --- | --- |
| `motionbridge/core` | SOMA motion, cleanup, bone mapping, retarget math | NumPy, SciPy |
| `motionbridge/blender` | Scripts run inside Blender: inspect rigs, apply keys, export | `bpy`, `core` |
| `motionbridge/comfy` | Thin ComfyUI nodes that call `core` and `blender` | ComfyUI |
| `tests` | Golden-set runner and metrics | everything |
| `golden` | Small committed test motions (transforms only) | |

## House rules

1. No per-clip or per-character code. A fix that needs a character name or a
   frame range does not go in.
2. Every change runs the golden set (`python -m tests.golden all`) and is judged
   on its contact sheet and metrics, across every motion and rig.
3. `core` never imports ComfyUI or `bpy`.
4. No hardcoded machine paths. They live in `config.toml` (copy
   `config.example.toml`).
5. No model weights, generated runs or licensed characters in git.

## Setup (development)

1. Copy `config.example.toml` to `config.toml` and fill in your paths.
2. Use a Python 3.11+ environment with NumPy, SciPy and OpenCV (the GEM
   environment works).
3. Run `python -m tests.golden all` from the repository root.

## License

Apache License 2.0. Models and assets used at runtime (H3, GEM, SAM 3D Body,
MHR, Kimodo) keep their own licenses and are downloaded by the user.
