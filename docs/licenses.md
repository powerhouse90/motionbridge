# Licenses and commercial use

MotionBridge's own code is Apache-2.0. Everything below is downloaded by **MotionBridge Setup** (or, for H3, by
you) and keeps its own terms. This is a reading of those terms, not legal advice; check the linked texts for
your own use. Read on 2026-10-25.

| Component | Used for | License | Commercial use |
|---|---|---|---|
| H3 (MiniMax) | generating the video | H3 model license | **In the US, apply for MiniMax's commercial license** (the application is quick). |
| GEM-X code | full-body motion from video | [Apache-2.0](https://github.com/NVlabs/GEM-X/blob/main/LICENSE) | Yes. |
| GEM weights (motion model and its bundled 77-joint 2D pose model) | full-body motion | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Yes ("Models are commercially usable"). NVIDIA claims no ownership of outputs (§2.4). Redistributing the weights needs the agreement and a "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" notice (§3). Don't bypass guardrails (§2.1). |
| SOMA body model | skeleton and body | [Apache-2.0](https://github.com/NVlabs/SOMA-X) | Yes. |
| MHR (Momentum Human Rig) | body model inside GEM's features | [Apache-2.0](https://github.com/facebookresearch/MHR/blob/main/LICENSE) | Yes. |
| SAM 3D Body weights | GEM's features; first-person hand reading | [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE) | Yes (royalty-free, worldwide; you own your derivatives, §5a). **But §1.b.v forbids uses "related to military or warfare purposes ... or the development or use of guns or illegal weapons."** Animating guns in a video game is a depiction, not weapon development, but the clause is broad; get your own advice before selling gun animations made with it. Redistributing needs a copy of the license. |
| SAM 2 | first-person arm and prop masks | [Apache-2.0](https://github.com/facebookresearch/sam2) | Yes. |
| YOLOX-x person detector (`yolox_x_8xb8-300e_humanart`, OpenMMLab) | GEM's person boxes | code Apache-2.0; weights trained on COCO and [Human-Art](https://github.com/IDEA-Research/HumanArt) | **Unclear.** Human-Art is released for non-commercial purposes only. Whether that reaches weights trained on it is unsettled. It only finds the person's box in each frame, so it can be replaced (see below). |
| Blender | rendering, export | GPL | Yes; the GPL doesn't cover the animations and files you make with it. |
| OGA FPS arms (test rig) | first-person tests | CC0 | Yes. |

Never publish or sell anything made with the **Eros** H3 finetune used for MotionBridge's own test takes; the
example workflows use the official H3 weights.

## Open items

- **Person detector:** replace the Human-Art-trained YOLOX boxes with a commercially clean source. MotionBridge
  already knows where the character is in the first frame (the starting views), so SAM 2 (Apache-2.0), prompted
  from there as the first-person tracker does, can give the boxes.
- **SAM 3D Body and guns:** for first-person weapon animations sold commercially, decide whether the SAM
  License's weapons clause is a concern for you; there is no drop-in replacement for its hand reading yet.
