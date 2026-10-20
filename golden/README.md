# Golden motions

Transforms-only SOMA motions (`motionbridge.soma.v1`) used by `python -m tests.golden`.
All are GEM front-panel extractions of H3 clips, 73 frames at 24 fps. The `h3_*` takes come from
`golden/h3_takes.toml` via `python -m tests.h3_takes queue|collect` (H3 test model: Eros fp8 + turbo LoRA, 8 steps).

| File | Action | Why it's here |
| --- | --- | --- |
| `arthur_idle` | Relaxed standing idle | Baseline: both feet planted, almost no root motion |
| `arthur_handwash` | Hands rubbing in front of the chest | Close hand-to-hand work; one foot planted |
| `arthur_trunk_open` | Reach up and lift a car trunk | Big reach, weight shift, partial contacts |
| `comfy_test_front` | Full ComfyUI pipeline test clip | Visible source foot slide (~0.15 m/s) for cleanup to fix |
| `h3_walk` | Four steps toward the camera | 1.56 m of travel; source foot slide up to 0.35 m/s |
| `h3_crouch` | Deep squat and back up | Hips drop from 1.03 m to 0.35 m |
| `h3_floor_touch` | Palms flat on the floor | Hand-to-floor contact across arm lengths |
| `h3_jump` | Small vertical jump | Both feet ~15 cm off the floor at frame 27 |
| `h3_fingers` | Counting one to five, fists | Finger chains; its hand panel drifted to a face close-up |
