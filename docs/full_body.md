# Full body

Type an action and get it on your own rigged character, keeping its proportions. Workflow:
[`full_body.json`](../workflows/full_body.json).

## Pipeline

```
Setup -> Load Character -> Render Starting Views -> H3 -> Video to Motion (GEM) -> Smooth -> Retarget -> Export
```

1. **Load Character**: BLEND, GLB, GLTF or FBX with a humanoid skeleton. It finds hips, spine, neck, head,
   arms, legs, feet and fingers **by shape**, whatever they're named; names only confirm or contradict. The
   `bone map` output shows what it found. Set `armature_name` only when the file has several armatures.
2. **Render Starting Views**: matching front and three-quarter views of the character in its own pose, as one
   split-screen image for H3. `front + 3/4 + hands` adds a hand close-up panel. `front_direction` `auto`
   usually finds the character's front; set it if the front view shows the back or a side.
3. **H3** animates both panels as the same instant from two cameras. Keep the two-panel wording in the prompt
   and change only the action. Ask for a locked camera and the full body in view.
4. **Video to Motion (GEM)** extracts SOMA's 77-joint 3D motion from the front panel
   (`left half (front view of 2)`, or `left third` for three panels). `rendered reference (orthographic)`
   suits rendered starting views; use `real camera` for phone or webcam footage (connect a **Load Video** to
   `video` instead of H3's frames).
5. **Smooth Motion**: adaptive. Slow joints are smoothed hard and fast ones barely, so strikes keep their snap,
   with no lag. 0.5 suits most clips.
6. **Animation Spacing** gives each move an animator's timing, keeping its poses and length: a slow ease in,
   fastest past the middle, a shorter ease out onto the extreme, a small settle. Each limb is timed on its own.
   - `ease_in` / `ease_out`: how slowly a move leaves its start and arrives on its extreme (`ease_out` under ~0.5 is
     a dead stop, which usually reads as a mistake);
   - `overshoot`: how far it carries past the extreme before settling back;
   - `skip_frames` / `skip_above`: where a hand or foot moves faster than `skip_above` m/s, that share of frames is
     cut so the limb jumps, for impact. Slower moves are only respaced. `skip_frames` 0 turns it off.
7. **Make Loop** (optional, for walks, runs and idles a game plays on repeat) cuts one seamless cycle out of the
   take: it finds the cycle whose end best matches its start and crossfades the cut into the take's own
   continuation. Retarget Motion then exports just that cycle, with clean feet across the seam, ending on a copy
   of its first frame as engines expect of a loop. Pair it with `travel` `in place` or `root bone` for locomotion.
   - `cycle_min` / `cycle_max`: seconds searched. A walk cycle is two steps, about 1-1.6 s;
   - `crossfade`: frames blended at the seam;
   - `start_frame` / `end_frame` (advanced): pin the cycle instead of searching.

   The report gives how well the cut matches its start and warns when the take has no clean cycle (a one-off
   action like an attack). Use it after Animation Spacing, which retimes frames.
8. **Retarget Motion** onto your character, keeping its bone lengths:
   - `foot_ik` pins planted feet so they don't slide or sink;
   - `head` `face` aims the face where the performer's face points, `carriage` keeps your character's built-in
     head pose and adds the change;
   - `travel` `keep` leaves the travel on the hips. `in place` keeps walks and runs on the spot, for locomotion
     driven by code. `root bone` moves the travel and turn onto the bone above the hips, for root motion in
     Unreal or Unity Generic rigs (Unity Humanoid makes its own root motion from `keep`). A rig without a bone
     above the hips keeps the travel on the hips; no bone is added, so the export still matches the character
     already in your engine.
9. **Export Animated Character**: BLEND + FBX (+ GLB), checked bone by bone against the retarget. Your source
   file is never modified. `preview` renders the textured character (`follow`, `front`, `profile`, `whole path`).

## Tips

- One action per take, 3 seconds (73 frames). A short, concrete prompt beats a long one.
- Full body, hands and feet inside the frame for the whole take, with margin. A cut-off foot is guessed.
- **8 GB GPU:** Video to Motion unloads H3 from VRAM before GEM runs, so both fit.
- Results are cached by content: change only the retarget settings and only the retarget re-runs.
