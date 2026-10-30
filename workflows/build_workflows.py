"""Build the example workflows (ComfyUI UI format, drag into ComfyUI) and check them against a running ComfyUI.

python workflows/build_workflows.py [--url http://127.0.0.1:8188 | --object-info object_info.json] [--check-only]

Widget order and socket types come from ComfyUI's /object_info, so the files match the installed nodes. Every
file is then re-read and checked: node types exist, every link joins matching types, every required socket is
connected, widget counts match. Model files use the official H3 release (Comfy-Org/MiniMax-H3) plus the
lightx2v turbo LoRA; each loader carries its download URL so ComfyUI offers to fetch a missing model.
"""
import argparse
import json
import sys
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTES = ('Note', 'MarkdownNote')  # frontend-only node types
WIDGETS = ('INT', 'FLOAT', 'STRING', 'BOOLEAN', 'COMBO', 'COMFY_DYNAMICCOMBO_V3')

H3 = 'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main'
MODELS = {  # official files (ComfyUI's own MiniMax H3 image-to-video template), not the test checkpoints
    'unet': ('minimax_h3_fl2va_pruned_int8_convrot.safetensors', f'{H3}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors', 'diffusion_models'),
    'clip': ('qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', f'{H3}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', 'text_encoders'),
    'vae': ('minimax_h3_video_vae_int8_convrot.safetensors', f'{H3}/vae/minimax_h3_video_vae_int8_convrot.safetensors', 'vae'),
    'lora': ('minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
             'https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors', 'loras'),
}

GROUP_COLOURS = {'setup': '#3f789e', 'h3': '#8A8', 'track': '#b06634', 'export': '#a1309b', 'note': '#444'}


def object_info(args):
    if args.object_info:
        return json.loads(Path(args.object_info).read_text(encoding='utf-8'))
    with urllib.request.urlopen(args.url.rstrip('/') + '/object_info', timeout=60) as r:
        return json.load(r)


def spec_type(spec):
    return 'COMBO' if isinstance(spec[0], list) else spec[0]


def is_widget(spec):
    opts = spec[1] if len(spec) > 1 else {}
    return spec_type(spec) in WIDGETS and not opts.get('forceInput')


def inputs_of(info, kind):
    """[(name, spec, required)] in the order the frontend lays them out."""
    d = info[kind]['input']
    order = info[kind].get('input_order') or {s: list(d.get(s, {})) for s in ('required', 'optional')}
    return [(n, d[s][n], s == 'required') for s in ('required', 'optional') for n in order.get(s, []) if n in d.get(s, {})]


class Graph:
    def __init__(self, info, title):
        self.info, self.title = info, title
        self.nodes, self.links, self.groups = [], [], []

    def node(self, kind, pos, title=None, size=None, widgets=None, models=None, **values):
        """values: widget values by name, or (node, output index) for a link."""
        if kind in NOTES:
            text = values['text']
            n = {'id': len(self.nodes) + 1, 'type': kind, 'pos': list(pos), 'size': list(size or (420, 200)), 'flags': {},
                 'order': 0, 'mode': 0, 'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': [text],
                 'color': '#432', 'bgcolor': '#653'}
            self.nodes.append(n)
            return n
        info = self.info[kind]
        n = {'id': len(self.nodes) + 1, 'type': kind, 'pos': list(pos), 'flags': {}, 'order': 0, 'mode': 0,
             'inputs': [], 'outputs': [], 'properties': {'Node name for S&R': kind}, 'widgets_values': []}
        if title:
            n['title'] = title
        if models:
            n['properties']['models'] = [{'name': m[0], 'url': m[1], 'directory': m[2]} for m in models]
        unknown = set(values) - {name for name, _, _ in inputs_of(self.info, kind)}
        if unknown:
            raise KeyError(f'{kind}: no inputs named {sorted(unknown)}')
        for name, spec, required in inputs_of(self.info, kind):
            v = values.get(name)
            linked = isinstance(v, tuple)
            if is_widget(spec):
                opts = spec[1] if len(spec) > 1 else {}
                if widgets is None:
                    n['widgets_values'].append(opts.get('default', '') if v is None or linked else v)
                    if opts.get('control_after_generate'):
                        n['widgets_values'].append('fixed')
                if linked:
                    n['inputs'].append({'name': name, 'type': spec_type(spec), 'widget': {'name': name}, 'link': None})
            else:
                n['inputs'].append({'name': name, 'type': spec_type(spec), 'link': None, **({} if required else {'shape': 7})})
            if linked:
                self.link(v[0], v[1], n, name)
        if widgets is not None:
            n['widgets_values'] = list(widgets)
        n['outputs'] = [{'name': nm, 'type': t, 'links': []} for t, nm in zip(info['output'], info['output_name'])]
        rows = len(n['inputs']) + len(n['widgets_values']) + len(n['outputs'])
        n['size'] = list(size or (340, 46 + 24 * rows))
        self.nodes.append(n)
        return n

    def link(self, src, slot, dst, name):
        i = next(k for k, s in enumerate(dst['inputs']) if s['name'] == name)
        t = src['outputs'][slot]['type'] if src['outputs'] else None
        lid = len(self.links) + 1
        self.links.append([lid, src['id'], slot, dst['id'], i, t or dst['inputs'][i]['type']])
        src['outputs'][slot]['links'].append(lid)
        dst['inputs'][i]['link'] = lid

    def group(self, title, box, colour):
        self.groups.append({'id': len(self.groups) + 1, 'title': title, 'bounding': list(box), 'color': GROUP_COLOURS[colour],
                            'font_size': 24, 'flags': {}})

    def save(self, path):
        # execution order: a node after everything feeding it
        by_id = {n['id']: n for n in self.nodes}
        deps = {n['id']: {l[1] for l in self.links if l[3] == n['id']} for n in self.nodes}
        done, order = set(), []
        while len(order) < len(self.nodes):
            for i in sorted(deps):
                if i not in done and deps[i] <= done:
                    done.add(i); order.append(i)
        for k, i in enumerate(order):
            by_id[i]['order'] = k
        wf = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'motionbridge/' + path.name)), 'revision': 0,
              'last_node_id': len(self.nodes), 'last_link_id': len(self.links), 'nodes': self.nodes, 'links': self.links,
              'groups': self.groups, 'config': {}, 'extra': {'ds': {'scale': 0.55, 'offset': [80, 120]}, 'title': self.title},
              'version': 0.4}
        path.write_text(json.dumps(wf, indent=1), encoding='utf-8')
        return path


# ---------------------------------------------------------------- shared pieces

def h3_chain(g, image, x, y, prompt, length=73, seed=21092631, prefix='motionbridge/h3_take', loaders=None):
    """Start image -> H3 image-to-video (0.7 MP, turbo LoRA, 8 steps) -> frames (IMAGE) and a saved video.
    The same graph MotionBridge's tests queue (runs/rifle/queue_grip.py), on the official weights.
    Returns (frames node, loaders); pass loaders back in to share one set of models between takes."""
    scale = g.node('ImageScaleToTotalPixels', (x, y), image=image, upscale_method='lanczos', megapixels=0.7, resolution_steps=32)
    size = g.node('GetImageSize', (x, y + 190), image=(scale, 0))
    if loaders is None:
        unet = g.node('UNETLoader', (x + 380, y), unet_name=MODELS['unet'][0], weight_dtype='default', models=[MODELS['unet']])
        lora = g.node('LoraLoaderModelOnly', (x + 380, y + 130), model=(unet, 0), lora_name=MODELS['lora'][0], strength_model=1.0,
                      models=[MODELS['lora']])
        clip = g.node('CLIPLoader', (x + 380, y + 270), clip_name=MODELS['clip'][0], type='minimax', device='default', models=[MODELS['clip']])
        vae = g.node('VAELoader', (x + 380, y + 420), vae_name=MODELS['vae'][0], models=[MODELS['vae']])
        loaders = (lora, clip, vae)
    lora, clip, vae = loaders
    i2v = g.node('MiniMaxH3ImageToVideo', (x + 760, y), size=(460, 420), clip=(clip, 0), vae=(vae, 0), prompt=prompt,
                 width=(size, 0), height=(size, 1), length=length, first_frame=(scale, 0))
    noise = g.node('RandomNoise', (x + 1260, y), noise_seed=seed)
    guider = g.node('BasicGuider', (x + 1260, y + 130), model=(lora, 0), conditioning=(i2v, 0))
    sampler = g.node('KSamplerSelect', (x + 1260, y + 250), sampler_name='res_multistep')
    sigmas = g.node('BasicScheduler', (x + 1260, y + 350), model=(lora, 0), scheduler='simple', steps=8, denoise=1.0)
    sample = g.node('SamplerCustomAdvanced', (x + 1640, y), noise=(noise, 0), guider=(guider, 0), sampler=(sampler, 0),
                    sigmas=(sigmas, 0), latent_image=(i2v, 1))
    decode = g.node('VAEDecode', (x + 1640, y + 230), samples=(sample, 0), vae=(vae, 0))
    video = g.node('CreateVideo', (x + 1640, y + 350), images=(decode, 0), fps=24.0)
    g.node('SaveVideo', (x + 2020, y), size=(420, 420), video=(video, 0), widgets=[prefix, 'mp4', 'auto'])
    return decode, loaders


H3_NOTE = """## H3 take
Turns the start image into a 73-frame (3 s) video. Settings MotionBridge is tested with: **0.7 MP**, turbo LoRA,
**8 steps**, `res_multistep`, fixed seed. Frames are rounded to multiples of 32 (1280x720 -> 1152x640); the tracker
reads the size from the frames.

**Models** (ComfyUI offers to download missing ones when the workflow opens):
- `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` (Comfy-Org/MiniMax-H3)
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (Comfy-Org/MiniMax-H3)
- `vae/minimax_h3_video_vae_int8_convrot.safetensors` (Comfy-Org/MiniMax-H3)
- `loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (lightx2v/Minimax-h3-Turbo)

The prompt matters most: describe the motion, keep the camera locked, and (first person) name the glove colours."""


# ---------------------------------------------------------------- (a) full body

FULL_BODY_PROMPT = (
    'Animate this split-screen reference as two synchronized fixed-camera views of the SAME person performing the SAME '
    'action at the SAME time. Left panel is a front view; right panel is a three-quarter view. Preserve both camera angles '
    'and the central divider throughout. The person walks naturally toward the front camera: four relaxed steps with clear '
    'heel strikes, toe-offs and alternating arm swing, then stops with both feet planted. Realistic planted soles and weight '
    'transfer, stable limb anatomy. Full body, hands and feet remain visible with margin throughout. Preserve identity, '
    'proportions, clothing, neutral background and lighting. No camera movement, cuts, extra limbs, props, or changes to the '
    'panel layout.')


def full_body(info):
    g = Graph(info, 'MotionBridge: full body (character -> H3 -> motion -> your rig)')
    g.node('MarkdownNote', (-460, -40), size=(420, 560), text="""# MotionBridge: full body
Type an action, get it on **your own rigged character**.

1. **Setup**: turn on `accept_licenses` once. The first Run downloads GEM-X, its Python environment and
   Blender (if none is installed) into `ComfyUI/models/motionbridge`. Later runs skip this.
2. **Load Character**: upload a BLEND / GLB / GLTF / FBX with a humanoid skeleton (any bone names and
   proportions). Pack textures into a BLEND first.
3. **Render Starting Views** renders matching front + three-quarter views for H3.
4. **H3** animates them. Edit the prompt: keep the two-panel wording, change the action.
5. **Video to Motion (GEM)** reads 77-joint 3D motion from the left (front) panel.
6. **Smooth**, **Animation Spacing** (ease in, fast middle, ease out, settle; frames skipped on the fastest stretches),
   **Retarget** onto your rig (feet pinned), **Export** BLEND + FBX + GLB and a preview.

8 GB GPU: Video to Motion unloads H3 before GEM runs.""")
    g.group('1. Setup and character', (0, -100, 1180, 640), 'setup')
    setup = g.node('MotionBridgeSetup', (30, -40), accept_licenses=False, blender='use installed, else download')
    char = g.node('MotionBridgeLoadCharacter', (30, 170), size=(360, 150), asset_file='', setup=(setup, 0))
    views = g.node('MotionBridgeRenderStartingViews', (420, -40), size=(380, 470), character=(char, 0), layout='front + 3/4',
                   framing='character only', elevation=15)
    g.node('PreviewImage', (830, -40), size=(320, 420), images=(views, 0))
    g.group('2. H3 take', (1220, -100, 2480, 640), 'h3')
    frames, _ = h3_chain(g, (views, 0), 1250, -40, FULL_BODY_PROMPT, prefix='motionbridge/full_body_take')
    g.node('MarkdownNote', (1250, 360), size=(700, 160), text=H3_NOTE.split('**Models**')[0])
    g.group('3. Motion onto your rig', (3740, -100, 1240, 640), 'export')
    motion = g.node('MotionBridgeVideoToMotion', (3770, -40), size=(380, 170), images=(frames, 0), fps=24.0,
                    panel='left half (front view of 2)', camera='rendered reference (orthographic)')
    smooth = g.node('MotionBridgeSmoothMotion', (3770, 160), motion=(motion, 0), strength=0.5)
    spaced = g.node('MotionBridgeAnimationSpacing', (3770, 280), size=(380, 230), motion=(smooth, 0), ease_in=3.0,
                    ease_out=1.2, overshoot=0.25, skip_frames=0.5, skip_above=5.0)
    retarget = g.node('MotionBridgeRetarget', (4180, -40), size=(360, 200), character=(char, 0), motion=(spaced, 0),
                      action_name='motion', foot_ik=True, head='face')
    g.node('MotionBridgeExport', (4570, -40), size=(380, 520), animation=(retarget, 0), filename_prefix='motionbridge/character',
           glb=True, preview='follow')
    return g


# ---------------------------------------------------------------- (b) first-person hands

FP_PROMPT = (
    'First-person view, locked-off camera. The right hand wears an orange glove and the left hand wears a blue glove; the '
    'rifle is clear see-through glass and floats in front. Both hands reach forward and take a firm, natural hold of the '
    'rifle: the orange right hand wraps around the pistol grip with the index finger along the side above the trigger, the '
    'blue left hand grips the handguard from below. Then both hands hold the rifle steady and still.')

FP_NOTE = """# MotionBridge: first-person hands
Rig arms that follow an H3 video, fingers and the held prop included.

**The scene** (one .blend): your FPS arms rig (any rig with finger bones), the scene camera as the player's view,
and the props. For a *grip take*, put the arms in a **neutral pose on either side** and the main prop **floating in
the middle**. Don't pose the hands on the prop: H3 copies whatever grip is in the start image, so let H3 animate
the grab.

**capture_mode** (on by default) dresses the start frame for tracking: right arm **orange**, left arm **blue**, main
prop **see-through**. H3 keeps the colours, so each arm and the prop separate cleanly, and the fingers stay visible
through the prop. Your exported animation keeps your own look either way.

1. **Load FP Scene** finds the hands and fingers by shape, picks the main prop, renders the start frame.
2. **H3** animates it: name the glove colours and the see-through prop in the prompt.
3. **Track FP Hands**: SAM 2 masks each arm and the prop, SAM 3D Body reads each hand, the rig is fitted frame
   by frame from the known start; fingers never pass through the prop.
4. **Export FP Animation**: BLEND (arm IK kept, editable) + FBX + GLB, and a side-by-side preview."""

FP_STATUS = """## In progress (not in these nodes yet)
Tested as scripts on a rifle grip take, being moved into Track FP Hands:
- prop placed by point tracking (sub-pixel) instead of its mask
- grasp step: each finger wraps, presses or stays free; depth adjusted so the handle sits inside the hand
- forearm roll (pronation / supination) so the wrist doesn't wring
- smooth -> re-grasp -> smooth, so the grip holds without jitter"""


def fp_stage(g, x, y, scene_file, prompt, prefix, setup=None, label='', loaders=None, muted=False):
    first = len(g.nodes)
    load = g.node('MotionBridgeLoadFPScene', (x, y), size=(360, 250), scene_file=scene_file, width=1280, height=720,
                  capture_mode=True, main_prop='', armature_name='', **({'setup': (setup, 0)} if setup else {}))
    g.node('PreviewImage', (x, y + 280), size=(360, 230), title=f'start frame{label}', images=(load, 1))
    frames, loaders = h3_chain(g, (load, 1), x + 400, y, prompt, prefix=prefix, loaders=loaders)
    track = g.node('MotionBridgeTrackFPHands', (x + 2860, y), size=(360, 200), fp_scene=(load, 0), images=(frames, 0), fps=24.0,
                   smoothing=0.7, hands='both')
    g.node('PreviewAny', (x + 2860, y + 230), size=(360, 200), title='tracking report', source=(track, 1))
    export = g.node('MotionBridgeExportFP', (x + 3260, y), size=(420, 460), fp_motion=(track, 0),
                    filename_prefix=prefix.replace('_take', ''), action_name='fp_motion', glb=True, preview='side by side')
    for n in g.nodes[first:]:
        n['mode'] = 2 if muted else 0  # 2 = muted: skipped on Run until unmuted (Ctrl+M)
    return loaders


def fp_hands(info):
    g = Graph(info, 'MotionBridge: first-person hands (FPS arms -> H3 -> tracked rig)')
    g.node('MarkdownNote', (-560, -40), size=(520, 620), text=FP_NOTE)
    g.node('MarkdownNote', (-560, 610), size=(520, 220), text=FP_STATUS)
    g.group('Setup', (0, -100, 400, 220), 'setup')
    setup = g.node('MotionBridgeSetup', (20, -40), accept_licenses=False, blender='use installed, else download')
    g.group('1. Scene and start frame', (0, 140, 400, 680), 'setup')
    g.group('2. H3 grip take', (420, 140, 2420, 680), 'h3')
    g.group('3. Track and export', (2860, 140, 860, 680), 'track')
    fp_stage(g, 20, 200, 'motionbridge/rifle_neutral.blend', FP_PROMPT, 'motionbridge/fp_grip_take', setup=setup)
    g.node('MarkdownNote', (440, 620), size=(700, 180), text=H3_NOTE.split('**Models**')[0])
    return g


# ---------------------------------------------------------------- (c) grip take -> start pose for later takes

GRIP_NOTE = """# Grip take -> start pose
Do the grab **once**, then start every later animation from the settled grip.

**Stage 1** is the first-person workflow: neutral arms, prop floating in the middle, H3 animates the grab,
Track FP Hands fits the rig. Its exported BLEND ends on the grip.

**Between the stages** (in progress, no node yet): turn the end of the take into a start scene. The script
averages the last frames (hands, fingers, prop) into one steady pose:
```
blender -b --python motionbridge/blender/fp_grip.py -- --scene <fp_scene folder>/scene.blend
        --track <fp_motion folder>/track.npz --out ComfyUI/input/motionbridge/rifle_grip.blend
        --frames 8 --render grip.png
```
Both folders are under `ComfyUI/output/motionbridge/` (Track FP Hands' report names them). Or open the exported
BLEND in Blender, go to the last frame and save it as your start scene.

**Stage 2** is muted until that file exists: select its nodes and press Ctrl+M to unmute, then Run (stage 1 is
cached, so only stage 2 runs). It loads the grip scene. The main prop is now the one a hand holds in the start pose, so H3 only has to
animate the action (aim, reload, inspect...), and every action starts from the same grip."""

ACTION_PROMPT = (
    'First-person view, locked-off camera. The right hand wears an orange glove and the left hand wears a blue glove; they '
    'hold a clear see-through glass rifle. Keeping that exact grip, both hands raise the rifle slightly and sweep it slowly '
    'from left to right as if scanning a room, then lower it back to the start position. Fingers stay wrapped on the grip '
    'and handguard throughout.')


def fp_grip_take(info):
    g = Graph(info, 'MotionBridge: grip take, then actions from that grip')
    g.node('MarkdownNote', (-620, -40), size=(580, 760), text=GRIP_NOTE)
    g.group('Setup', (0, -100, 400, 220), 'setup')
    setup = g.node('MotionBridgeSetup', (20, -40), accept_licenses=False, blender='use installed, else download')
    g.group('Stage 1: grip take (neutral arms, prop in the middle)', (0, 140, 3720, 680), 'h3')
    loaders = fp_stage(g, 20, 200, 'motionbridge/rifle_neutral.blend', FP_PROMPT, 'motionbridge/fp_grip_take', setup=setup,
                       label=' (neutral)')
    g.group('Stage 2 (muted until the grip scene exists): an action from the grip', (0, 860, 3720, 680), 'track')
    fp_stage(g, 20, 920, 'motionbridge/rifle_grip.blend', ACTION_PROMPT, 'motionbridge/fp_action_take', setup=setup,
             label=' (grip)', loaders=loaders, muted=True)
    return g


BUILDERS = {'full_body.json': full_body, 'fp_hands.json': fp_hands, 'fp_grip_take.json': fp_grip_take}


# ---------------------------------------------------------------- checks

def check(path, info):
    wf = json.loads(path.read_text(encoding='utf-8'))
    problems, warnings = [], []
    nodes = {n['id']: n for n in wf['nodes']}
    for n in wf['nodes']:
        if n['type'] in NOTES:
            continue
        if n['type'] not in info:
            problems.append(f"node {n['id']}: unknown type {n['type']}")
            continue
        ins = inputs_of(info, n['type'])
        widgets = [(name, spec) for name, spec, _ in ins if is_widget(spec)]
        expected = len(widgets) + sum(1 for _, s in widgets if (s[1] if len(s) > 1 else {}).get('control_after_generate'))
        if n['type'] != 'SaveVideo' and len(n['widgets_values']) != expected:
            problems.append(f"node {n['id']} {n['type']}: {len(n['widgets_values'])} widget values, expected {expected}")
        vals = iter(n['widgets_values'])
        for name, spec in widgets:
            v = next(vals, None)
            if (spec[1] if len(spec) > 1 else {}).get('control_after_generate'):
                next(vals, None)
            options = spec[0] if isinstance(spec[0], list) else (spec[1].get('options') if len(spec) > 1 and spec_type(spec) == 'COMBO' else None)
            if options is not None and v not in options:
                (warnings if n['properties'].get('models') else problems).append(
                    f"node {n['id']} {n['type']}.{name} = {v!r} not in this install's options" + (' (model not installed here)' if n['properties'].get('models') else ''))
        linked = {s['name'] for s in n['inputs'] if s.get('link') is not None}
        for name, spec, required in ins:
            if required and not is_widget(spec) and name not in linked:
                problems.append(f"node {n['id']} {n['type']}: required input {name} not connected")
    for lid, a, sa, b, sb, t in wf['links']:
        if a not in nodes or b not in nodes:
            problems.append(f'link {lid}: missing node'); continue
        out_t = nodes[a]['outputs'][sa]['type']
        in_slot = nodes[b]['inputs'][sb]
        if lid not in (nodes[a]['outputs'][sa]['links'] or []) or in_slot['link'] != lid:
            problems.append(f'link {lid}: not registered on both ends')
        in_t = in_slot['type']
        if not (out_t == t and (in_t in (t, '*') or (in_t == 'COMBO'))):
            problems.append(f"link {lid}: {nodes[a]['type']}.{nodes[a]['outputs'][sa]['name']} ({out_t}) -> "
                            f"{nodes[b]['type']}.{in_slot['name']} ({in_t}), link says {t}")
    return problems, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8188')
    ap.add_argument('--object-info')
    ap.add_argument('--check-only', action='store_true')
    args = ap.parse_args()
    info = object_info(args)
    bad = 0
    for name, build in BUILDERS.items():
        path = HERE / name
        if not args.check_only:
            build(info).save(path)
        problems, warnings = check(path, info)
        wf = json.loads(path.read_text(encoding='utf-8'))
        print(f'{name}: {len(wf["nodes"])} nodes, {len(wf["links"])} links, {len(problems)} problems, {len(warnings)} warnings')
        for p in problems:
            print('  PROBLEM', p)
        for w in warnings:
            print('  warning', w)
        bad += len(problems)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
