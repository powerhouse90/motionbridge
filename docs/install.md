# Install

## Requirements

- Windows x64 with an NVIDIA GPU, 8 GB VRAM or more (tested on an RTX 4060 8 GB)
- ComfyUI with MiniMax H3 support (tested on ComfyUI 0.42.1 portable)
- About 60 GB free disk (H3 models ~42 GB, GEM-X and its environment ~13 GB, Blender ~0.4 GB)
- Enough free system RAM for H3 to load its weights; close other heavy jobs while it runs
  ([troubleshooting](troubleshooting.md#out-of-memory))

## Steps

1. Clone this repository into `ComfyUI/custom_nodes/` and restart ComfyUI. The nodes only use what ComfyUI
   already ships (NumPy, SciPy, PyTorch, PyAV, Pillow). Nothing is pip-installed into ComfyUI's Python.
2. Drag an example workflow from [`workflows/`](../workflows) into ComfyUI:
   - `full_body.json`: your character performs an action ([guide](full_body.md))
   - `fp_hands.json`: first-person arms follow an H3 grip take ([guide](first_person.md))
   - `fp_grip_take.json`: one grip take, then actions that start from that grip
3. **H3 models.** When the workflow opens, ComfyUI offers to download any missing model. Each loader node
   carries its download link:

   | Folder | File | Source |
   | --- | --- | --- |
   | `diffusion_models` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
   | `text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
   | `vae` | `minimax_h3_video_vae_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
   | `loras` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) |

4. **MotionBridge Setup.** Turn on `accept_licenses` in the **MotionBridge Setup** node and click Run. The first
   run downloads into `ComfyUI/models/motionbridge`:
   - GEM-X source and weights from `nvidia/GEM-X` (~6.7 GB, no account needed), including the SAM 3D Body weights
   - GEM's own Python environment (~6 GB), built with uv so ComfyUI's packages are untouched
   - SAM 2.1 (hiera small) for the first-person arm and prop masks
   - Blender 5.0 (~0.4 GB), only when no Blender is installed

   Later runs skip all of this, and downloads resume if interrupted. The node's `status` output lists anything
   still missing. Connect it to the `setup` input of **Load Character** / **Load FP Scene**, so one Run installs
   first and then animates.

## Licenses

MotionBridge is Apache-2.0. What Setup downloads keeps its own license, and you accept these by turning on
`accept_licenses`:

- GEM-X code (Apache-2.0) and weights (NVIDIA Open Model License)
- SAM 3D Body weights (SAM License)
- SAM 2 (Apache-2.0)
- SOMA body model (Apache-2.0)
- Blender (GPL)

H3 is covered by its own model license.

## Updating

`git pull` in `custom_nodes/motionbridge`, then restart ComfyUI. Results are cached under
`ComfyUI/output/motionbridge/` by content hash, so a changed input recomputes and an unchanged one is reused.
