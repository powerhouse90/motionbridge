# Troubleshooting

## Out of memory

- **H3 fails while loading, e.g. `hostbuf_file_reader_read failed`.** System RAM ran out while H3 read its
  weights. Close other heavy jobs (another ComfyUI, training, big parallel scripts) and run again.
- **CUDA out of memory in GEM / SAM.** Video to Motion and Track FP Hands unload H3 from VRAM before they start.
  If something else holds VRAM (a second ComfyUI, a game), close it.
- **Windows "paging file is too small".** Raise the page file size (System > Advanced system settings >
  Performance > Advanced > Virtual memory), or run fewer jobs at once.

## Ports and servers

- ComfyUI listens on **8188** by default. A second install needs its own port (`--port 8189`).
- Two ComfyUI installs that both load MotionBridge (or the older `ComfyUI-MotionBridge` pack, which uses the same
  node IDs) must not share a `custom_nodes` folder. Disable the old pack.

## Blender

Setup uses, in order:

1. the `blender_executable` you give it (Blender 4.2+);
2. `BLENDER_PATH`, then `blender` on `PATH`, then `C:/Program Files/Blender Foundation/...`;
3. otherwise it downloads Blender 5.0.1 into `ComfyUI/models/motionbridge/blender`.

`blender: always download` skips the search and always downloads. Every Blender step writes a log
(e.g. `fp_apply.log`) next to its output in `ComfyUI/output/motionbridge/`. Read that first when a Blender step
fails.

## Characters and scenes

- **Missing textures:** pack them into the BLEND (File > External Data > Pack Resources) before uploading.
  Load Character warns; Load FP Scene skips images it can't find.
- **Wrong front view:** set `front_direction` on Render Starting Views.
- **Several armatures in one file:** set `armature_name`.
- **First person, hand not tracked:** the hand must be in view in the start frame. A hand out of view is left as
  it is.
- **First person, prop not tracked:** set `main_prop` to the prop's object name if no hand holds a prop in the
  start pose.

## H3 takes

- **The camera moves or cuts:** say "locked-off camera" / "no camera movement, no cuts" in the prompt.
- **First person, colours lost:** name them ("the right hand wears an orange glove, the left a blue glove; the
  rifle is clear see-through glass").
- **H3 lifts or moves a prop you wanted still:** the main prop is tracked from the video, so its motion is kept.
  Re-roll the seed if it matters.

## Where things are

- Models and tools: `ComfyUI/models/motionbridge/`
- Intermediate results (cached by content hash): `ComfyUI/output/motionbridge/`
  (`fp_scene/`, `fp_motion/`, `fp_export/`, ...)
- Exported files: `ComfyUI/output/<filename_prefix>_*.blend / .fbx / .glb`
