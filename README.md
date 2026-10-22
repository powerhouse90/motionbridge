# MotionBridge

Type an action, get it on your own rigged character, any body shape, on an 8GB GPU.

MotionBridge is a ComfyUI node pack. A motion source (MiniMax H3 video through
NVIDIA GEM, or text through NVIDIA Kimodo) writes one canonical skeleton, SOMA's
77 joints. Cleanup runs once on that skeleton. A generic retargeter then maps it
onto any humanoid rig without per-character calibration.

## Workflows and guides

Drag a workflow from [`workflows/`](workflows) into ComfyUI. Each one has notes on every step.

| Workflow | What it does | Guide |
| --- | --- | --- |
| [`full_body.json`](workflows/full_body.json) | Your character: starting views → H3 → GEM motion → smooth → retarget → BLEND/FBX/GLB | [Full body](docs/full_body.md) |
| [`fp_hands.json`](workflows/fp_hands.json) | FPS arms: neutral start, H3 grip take, hands, fingers and the held prop tracked onto your rig | [First person](docs/first_person.md) |
| [`fp_grip_take.json`](workflows/fp_grip_take.json) | One grip take, then actions that start from that grip (the hand-off step is in progress) | [First person](docs/first_person.md#the-grip-take) |

Also: [install](docs/install.md) · [troubleshooting](docs/troubleshooting.md). The workflows are generated
from the installed nodes and checked by `python workflows/build_workflows.py --url http://127.0.0.1:8188`.

## Install

Details in [docs/install.md](docs/install.md).

1. Clone into `ComfyUI/custom_nodes/` and restart ComfyUI. The nodes use only what ComfyUI
   already ships (NumPy, SciPy, PyTorch, PyAV, Pillow).
2. Open an example workflow and click Run with **accept_licenses** on in the
   **MotionBridge Setup** node. The first run downloads, into `ComfyUI/models/motionbridge`:
   - GEM-X source and model weights from `nvidia/GEM-X` (~6.7 GB, no account needed)
   - GEM's own Python environment (~6 GB), built with uv so ComfyUI's packages are untouched
   - Blender 5.0 (~0.4 GB), only if no Blender is installed

   Later runs skip all of this. Downloads resume if interrupted.
3. H3 models: ComfyUI offers to download any missing ones when the workflow opens
   (official `Comfy-Org/MiniMax-H3` files plus the lightx2v turbo LoRA, ~42 GB).

Requirements: Windows x64, an NVIDIA GPU with 8 GB+ VRAM, ~60 GB free disk.
Video to Motion unloads H3 from VRAM before GEM runs, so both fit on 8 GB.

**Characters:** BLEND, GLB, GLTF or FBX with a humanoid skeleton, any bone names and
proportions. Pack textures into a BLEND (File > External Data > Pack Resources) before
uploading it; Load Character warns when textures are missing.

## Nodes

| Node | Does |
| --- | --- |
| MotionBridge Setup | One-time download and check of everything outside ComfyUI |
| Load Character | Finds hips, spine, neck, head, arms, legs, feet and fingers by shape |
| Render Starting Views | Matching front / three-quarter (/ hands) views of the character for H3 |
| Video to Motion (GEM) | 77-joint 3D motion from H3 output or any video |
| Retarget Motion | Motion onto the character, its proportions kept, planted feet pinned |
| Export Animated Character | BLEND + FBX + GLB, checked bone by bone, plus a preview video |
| Load FP Scene | First person: finds the FPS arms, hands and main prop; renders the (capture-mode) start frame |
| Track FP Hands | First person: arms, wrists, every finger and the held prop fitted to the H3 take |
| Export FP Animation | First person: BLEND (IK kept) + FBX + GLB, preview next to the source video |

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
