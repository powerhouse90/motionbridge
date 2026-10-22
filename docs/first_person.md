# First-person hands

FPS arms that follow an H3 video: wrists, every finger, and the main prop the hands hold. Workflows:
[`fp_hands.json`](../workflows/fp_hands.json) and [`fp_grip_take.json`](../workflows/fp_grip_take.json).

## The scene

One `.blend` with:

- **FPS arms**: any rig with finger bones. Hands and fingers are found by shape, not by name. A full arm
  (upper arm, forearm, hand, ideally with an IK control) follows H3 much better than a stub arm that starts at
  the wrist.
- **The scene camera**: the player's view. The start frame is rendered from it.
- **Props**: the **main prop** is tracked. By default it's the one a hand holds in the start pose; otherwise set
  `main_prop` on **Load FP Scene**. Other props stay in the scene for H3 but are not tracked.

Pack textures into the BLEND (File > External Data > Pack Resources). Load FP Scene skips images it can't find.

## The grip take

H3 copies whatever grip is in the start image. A grip you posed by hand ends up frozen into the video, and the
tracker can't tell your grip from H3's. So let H3 do the grab:

1. Put the arms in a **neutral pose on either side** of the view.
2. Put the main prop **floating in the middle**, in reach.
3. **Don't** position the hands on the prop.
4. Prompt H3 to reach forward and take hold of it, then hold still.

The end of that take is a natural, settled grip. Use it as the start of every later animation (aim, reload,
inspect) so they all share one grip. See [`fp_grip_take.json`](../workflows/fp_grip_take.json): stage 2 loads a
scene made from the end of the grip take.

> **In progress:** turning the end of a take into a start scene is a Blender script for now,
> `motionbridge/blender/fp_grip.py` (it averages the last frames: hands, fingers, prop). There's no node yet.
> Alternatively, open the exported BLEND, go to the last frame and save it as your start scene.

## Capture mode

`capture_mode` on **Load FP Scene** (on by default) dresses the *start frame only*, for tracking:

- **right arm orange, left arm blue**: H3 keeps the colours, so each arm separates cleanly from the other and
  from the prop, even when they cross;
- **main prop see-through** (35% opaque glass): the fingers stay visible *through* the prop, so the hand reader
  still sees the fingers wrapped around a grip, which an opaque prop would hide.

Name the colours and the glass prop in the prompt ("the right hand wears an orange glove... the rifle is clear
see-through glass"). The exported animation keeps your scene's own look either way.

## What the nodes do

1. **Load FP Scene**: finds the arms, hands and fingers, picks the main prop, freezes the start pose and renders
   the start frame (default 1280x720; H3 rounds to multiples of 32, e.g. 1152x640).
2. **H3**: 73 frames (3 s at 24 fps), 0.7 MP, turbo LoRA, 8 steps.
3. **Track FP Hands**:
   - SAM 2 masks each arm and the prop, prompted from the known start pose.
   - SAM 3D Body reads each hand.
   - The rig is fitted frame by frame from the known start frame: screen positions from the video, depth
     constrained so fingers never pass through the prop.
   - `smoothing` 0.7 removes frame-to-frame shake without softening the grab.
   - `hands` picks `both`, `right` or `left`.
4. **Export FP Animation**: BLEND (arm IK kept, editable), FBX (every bone baked), optionally GLB, and a preview
   next to the source video.

## In progress

These are tested as scripts on a rifle grip take and are being moved into **Track FP Hands**:

- **Point-tracked prop.** The prop's pose comes from hundreds of tracked surface points (optical flow and PnP,
  ~1 px reprojection) instead of its mask outline.
- **Grasp step.** Every finger near the prop either *wraps* (the handle sits between palm and fingertip),
  *presses* (fingertip on the surface: triggers, keys, buttons) or stays *free*. The hand's depth is solved so
  the grip closes around the handle while every joint stays where the video shows it.
- **Forearm roll** (pronation / supination). The hand's twist goes into the forearm, so the wrist bends instead
  of wringing.
- **Smooth, re-grasp, smooth.** Smoothing alone opens the grip; grasping alone brings the shake back. Doing both
  in turn keeps the grip closed and steady.
- **Elbow**: not fitted to the video yet; it follows the arm's IK.
