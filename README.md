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
3. Run `python -m tests.golden all` from the repository root: motion metrics and
   contact sheet, rig inspection, then bone mapping (`runs/golden/maps.png`).
4. Run `python -m tests.test_names` after touching name rules.

## Bone mapping

`core/rigmap.py` finds SOMA's body parts on any humanoid rig from hierarchy and
rest geometry: the floor, leg ends, head top and arm extremes locate the hips and
chest; paths between them give spine, neck, arms and legs; the hand is the last
branch point before the fingertips; the head is the first bone wider than the
narrowest neck bone. Names only confirm or contradict, which sets each role's
confidence (1.0 agree, 0.8 silent, 0.5 contradict). The golden runner re-maps
every rig with its bones renamed `b0, b1, ...`; a role that changes without being
flagged below 0.8 confidence is a silent error and fails the run.

## Retargeting

`core/retarget.py` turns a SOMA motion into local pose keys for any mapped rig.
Each mapped bone takes SOMA's world-space rotation change from rest, after a
swing that lines its rest direction up with SOMA's rest segment, so T-pose and
A-pose rigs both work. Spine, neck and finger chains of any length map by
position along the chain (fingers from the tip). Forward kinematics keeps the
rig's own bone lengths; the hips keep the rig's standing height and receive
SOMA's travel and rise scaled by the hip-to-ankle ratio. By default the clip is
turned so frame 1 faces the rig's front.

`blender/apply_motion.py` keys the result (one action per motion), reads every
bone head back from Blender, and saves a new BLEND plus an FBX with all actions.
`python -m tests.golden retarget` and `bake` run every motion on every rig.

## License

Apache License 2.0. Models and assets used at runtime (H3, GEM, SAM 3D Body,
MHR, Kimodo) keep their own licenses and are downloaded by the user.
