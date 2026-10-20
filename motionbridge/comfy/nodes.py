"""MotionBridge nodes: Setup, Load Character, Video to Motion, Retarget, Export.

Motion and characters pass between nodes as small dicts of file paths; the heavy work runs in
subprocesses (GEM in its own environment, Blender in the background) so ComfyUI's Python and
VRAM stay free. Every node caches by content, so re-running an unchanged graph is instant.
"""
import hashlib
import json
import shutil
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import io, ui, InputImpl

from motionbridge.comfy.common import (output_folder, digest, resolve_input, run_blender, runtime, free_vram)
from motionbridge.runtime.install import LICENSES, install

Character = io.Custom('MOTIONBRIDGE_CHARACTER')
Motion = io.Custom('MOTIONBRIDGE_MOTION')
Animation = io.Custom('MOTIONBRIDGE_ANIMATION')

PANELS = {'whole frame': (0, 1), 'left half (front view of 2)': (0, 2), 'left third (front view of 3)': (0, 3)}
LENSES = {'rendered reference (orthographic)': 8.0, 'real camera': 1.0}


class Setup(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeSetup', display_name='MotionBridge Setup', category='MotionBridge',
            description='One-time download of everything MotionBridge needs besides ComfyUI: GEM-X and its '
                        'model weights (~6.7 GB), GEM\'s own Python environment (~6 GB), and Blender if none is '
                        'installed. Safe to run again: it only fetches what is missing.\n\n' + LICENSES,
            is_output_node=True,
            inputs=[
                io.Boolean.Input('accept_licenses', default=False,
                                 tooltip='Required before anything is downloaded. Licenses are listed in the node description.'),
                io.Combo.Input('blender', options=['use installed, else download', 'always download'],
                               default='use installed, else download'),
                io.String.Input('blender_executable', default='', advanced=True,
                                tooltip='Use this Blender (4.2+) instead of finding or downloading one.'),
            ],
            outputs=[io.String.Output(display_name='status')])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')  # always re-check the disk

    @classmethod
    def execute(cls, accept_licenses, blender, blender_executable=''):
        rt = runtime(check=False)
        gaps = rt.missing()
        if gaps and not accept_licenses:
            text = (f'MotionBridge needs: {", ".join(gaps)}.\nTurn on accept_licenses and run again to download '
                    f'them into {rt.home}.\n\n{LICENSES}')
            return io.NodeOutput(text, ui=ui.PreviewText(text))
        if gaps:
            bar = comfy.utils.ProgressBar(100)
            choice = blender_executable or ('download' if blender == 'always download' else 'auto')
            install(rt.home, True, choice, progress=lambda i, n, text: bar.update_absolute(int(100 * i / n)))
            bar.update_absolute(100)
        text = f'MotionBridge is ready.\nGEM: {rt.gem_root}\nBlender: {rt.blender()}'
        return io.NodeOutput(text, ui=ui.PreviewText(text))


class LoadCharacter(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeLoadCharacter', display_name='Load Character', category='MotionBridge',
            description='Read a rigged character (BLEND, GLB, GLTF or FBX) and find its humanoid bones by '
                        'shape: hips, spine, neck, head, arms, legs, feet and fingers, whatever they are named.',
            inputs=[
                io.String.Input('asset_file', default='', tooltip='Use the upload button, or a path relative to ComfyUI/input.'),
                io.String.Input('armature_name', default='', advanced=True,
                                tooltip='Only needed when the file holds more than one armature.'),
                io.String.Input('setup', optional=True, force_input=True,
                                tooltip='Connect MotionBridge Setup here so one Run installs first, then animates.'),
            ],
            outputs=[Character.Output(display_name='character'), io.String.Output(display_name='bone map')])

    @classmethod
    def fingerprint_inputs(cls, asset_file, armature_name='', setup=None):
        try:
            return digest(resolve_input(asset_file), armature_name)
        except (OSError, ValueError):
            return float('nan')

    @classmethod
    def execute(cls, asset_file, armature_name='', setup=None):
        from motionbridge.core import rigmap
        asset = resolve_input(asset_file)
        folder = output_folder('characters', f'{asset.stem}_{digest(asset, armature_name)}')
        dump_file = folder / 'rig.json'
        if not dump_file.exists():
            args = ['--file', asset, '--out', dump_file] + (['--armature', armature_name] if armature_name else [])
            run_blender('dump_rig.py', args, folder / 'dump_rig.log')
        dump = json.loads(dump_file.read_text())
        m = rigmap.build(dump)
        missing = [r for r in rigmap.REQUIRED if r not in m.roles]
        if missing:
            raise ValueError(f'{asset.name}: could not find {", ".join(missing)}. Notes: {"; ".join(m.notes)}')
        lines = [f'{asset.name}: armature "{dump["armature"]}", {len(dump["bones"])} bones, '
                 f'{sum(1 for k, v in m.roles.items() if "Hand" in k and isinstance(v, list))} finger chains']
        lines += [f'  {role:>14}: {bone if isinstance(bone, str) else " > ".join(bone)}  ({m.confidence.get(role, 1.0):.1f})'
                  for role, bone in m.roles.items()]
        lines += [f'  note: {n}' for n in m.notes]
        if dump.get('missing_images'):
            lines.append(f"  WARNING: {len(dump['missing_images'])} texture(s) not found, so the character renders "
                         f"untextured. In Blender use File > External Data > Pack Resources, or use a GLB/FBX "
                         f"with embedded textures. Missing: {', '.join(dump['missing_images'][:5])}")
        character = {'file': str(asset), 'name': asset.stem, 'armature': dump['armature'], 'dump': str(dump_file)}
        text = '\n'.join(lines)
        return io.NodeOutput(character, text, ui=ui.PreviewText(text))


def frames_to_video(images, fps, path, crf=0):
    """IMAGE batch (B, H, W, 3 floats) -> H.264. crf 0 (lossless) feeds GEM; previews need crf > 0,
    because browsers cannot play lossless H.264."""
    frames = (images.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
    h, w = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=Fraction(fps).limit_denominator(1001))
        stream.width, stream.height, stream.pix_fmt = w, h, 'yuv420p'
        stream.options = {'crf': str(crf)}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame[:h, :w]), format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class VideoToMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeVideoToMotion', display_name='Video to Motion (GEM)', category='MotionBridge',
            description='Extract full-body 3D motion (77 joints: body, hands, face) from video with NVIDIA GEM. '
                        'Frees ComfyUI\'s models from VRAM first, so it runs on 8 GB cards right after H3.',
            inputs=[
                io.Image.Input('images', optional=True, tooltip='Frames, e.g. straight from H3\'s VAE Decode.'),
                io.Video.Input('video', optional=True, tooltip='Or a video (Load Video).'),
                io.Float.Input('fps', default=24.0, min=1, max=120, tooltip='Frame rate of images (ignored for video).'),
                io.Combo.Input('panel', options=list(PANELS), default='whole frame',
                               tooltip='Multi-view H3 references: use only the front view on the left.'),
                io.Combo.Input('camera', options=list(LENSES), default='rendered reference (orthographic)',
                               tooltip='Rendered references use a near-orthographic camera; GEM\'s default lens '
                                       'hunches the pose there. Use "real camera" for phone or webcam footage.'),
            ],
            outputs=[Motion.Output(display_name='motion')])

    @classmethod
    def execute(cls, fps, panel, camera, images=None, video=None):
        from motionbridge.core import soma
        from motionbridge.gem.extract import extract
        if video is not None:
            components = video.get_components()
            images, fps = components.images, float(components.frame_rate)
        if images is None:
            raise ValueError('Connect images or a video.')
        index, count = PANELS[panel]
        width = images.shape[2] // count
        images = images[:, :, index * width:(index + 1) * width]
        if images.shape[0] < 15:
            raise ValueError(f'GEM needs at least 15 frames; got {images.shape[0]}.')
        rt = runtime()
        sample = hashlib.sha256((images[::3, ::4, ::4].cpu().numpy() * 255).astype(np.uint8).tobytes()).hexdigest()
        key = digest(sample, list(images.shape), fps, LENSES[camera])
        folder = output_folder('motion', key)
        motion_file = folder / 'motion.npz'
        if not motion_file.exists():
            clip = folder / 'source.mp4'
            if not clip.exists():
                frames_to_video(images, fps, clip)
            free_vram()
            export = extract(clip, folder / 'gem', rt.gem_root, rt.gem_python, LENSES[camera], env=rt.gem_env(),
                             interrupted=comfy.model_management.throw_exception_if_processing_interrupted)
            soma.from_gem_export(export, fps).save(motion_file)
        return io.NodeOutput({'file': str(motion_file), 'fps': fps, 'source': str(folder / 'source.mp4')})


class Retarget(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeRetarget', display_name='Retarget Motion', category='MotionBridge',
            description='Transfer motion onto the character, keeping its own proportions. Planted feet are '
                        'pinned with leg IK so they do not slide or sink.',
            inputs=[
                Character.Input('character'),
                Motion.Input('motion'),
                io.String.Input('action_name', default='motion', tooltip='Name of the animation clip in the exported file.'),
                io.Boolean.Input('foot_ik', default=True, tooltip='Pin planted feet to the floor.'),
                io.Combo.Input('head', options=['face', 'carriage'], default='face', advanced=True,
                               tooltip='face: the face points where the performer\'s face points. carriage: keep '
                                       'the character\'s built-in head pose and add the performer\'s change.'),
            ],
            outputs=[Animation.Output(display_name='animation'), io.String.Output(display_name='report')])

    @classmethod
    def execute(cls, character, motion, action_name, foot_ik, head='face'):
        from motionbridge.core import retarget, rigmap, soma
        name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in action_name) or 'motion'
        dump = json.loads(Path(character['dump']).read_text())
        folder = output_folder('animation', digest(Path(character['dump']), Path(motion['file']), foot_ik, head,
                                                   Path(retarget.__file__)))
        result_file = folder / f"{character['name']}__{name}.npz"
        result = retarget.retarget(soma.load(motion['file']), dump, rigmap.build(dump), feet_ik=foot_ik, head=head)
        result.save(result_file)
        report = f"{character['name']} / {name}: {result.quaternions.shape[0]} frames at {result.fps:g} fps, " \
                 f"scale {result.scale:.2f}\n" + '\n'.join(result.notes or [])
        return io.NodeOutput({'character': character, 'clip': str(result_file), 'name': name}, report,
                             ui=ui.PreviewText(report))


class Export(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='MotionBridgeExport', display_name='Export Animated Character', category='MotionBridge',
            description='Key the animation onto the character in Blender and save BLEND + FBX (+ GLB), checked '
                        'bone by bone against the retarget. The source file is never modified. Optionally '
                        'renders a preview video of the textured character.',
            is_output_node=True,
            inputs=[
                Animation.Input('animation'),
                io.String.Input('filename_prefix', default='motionbridge/character'),
                io.Boolean.Input('glb', default=True, tooltip='Also save a GLB (opens in ComfyUI\'s 3D viewer).'),
                io.Combo.Input('preview', options=['front three-quarter', 'front', 'profile', 'none'],
                               default='front three-quarter'),
            ],
            outputs=[io.Video.Output(display_name='preview'), io.String.Output(display_name='files')])

    @classmethod
    def execute(cls, animation, filename_prefix, glb, preview):
        from motionbridge.core import rigmap
        character = animation['character']
        free_vram()
        folder = output_folder('export', digest(Path(animation['clip']), glb))
        blend, fbx = folder / f"{character['name']}.blend", folder / f"{character['name']}.fbx"
        report = folder / 'verify.json'
        args = ['--file', character['file'], '--out', blend, '--fbx', fbx, '--report', report,
                '--armature', character['armature'], '--motions', animation['clip']]
        if glb:
            args += ['--glb', folder / f"{character['name']}.glb"]
        run_blender('apply_motion.py', args, folder / 'export.log')
        worst = max(a['max_head_error_m'] for a in json.loads(report.read_text())['actions'].values())

        # Copy the deliverables to the requested prefix under ComfyUI/output.
        target_dir, stem, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        files = []
        for src in [blend, fbx] + ([folder / f"{character['name']}.glb"] if glb else []):
            dst = Path(target_dir) / f"{stem}_{animation['name']}{src.suffix}"
            shutil.copy2(src, dst)
            files.append(dst)
        text = '\n'.join(str(f) for f in files) + f'\nBlender vs retarget: worst bone error {worst * 1000:.3f} mm'

        video_out, ui_out = None, ui.PreviewText(text)
        if preview != 'none':
            view = {'front three-quarter': 'front34', 'front': 'front', 'profile': 'profile'}[preview]
            roles = rigmap.build(json.loads(Path(character['dump']).read_text())).roles
            frames_dir = folder / f'preview_{view}'
            run_blender('render_preview.py', ['--blend', blend, '--actions', animation['name'], '--outdir', frames_dir,
                                              '--view', view, '--size', 480, 640,
                                              '--hips-pair', roles['LeftLeg'], roles['RightLeg']],
                        folder / 'preview.log')
            pngs = sorted(next(frames_dir.iterdir()).glob('*.png'))
            mp4 = Path(target_dir) / f"{stem}_{animation['name']}_preview.mp4"
            images = np.stack([np.asarray(Image.open(p).convert('RGB')) for p in pngs])
            frames_to_video(torch.from_numpy(images.astype(np.float32) / 255), 24, mp4, crf=18)
            video_out = InputImpl.VideoFromFile(str(mp4))
            ui_out = ui.PreviewVideo([ui.SavedResult(mp4.name, subfolder, io.FolderType.output)])
        return io.NodeOutput(video_out, text, ui=ui_out)


NODES = [Setup, LoadCharacter, VideoToMotion, Retarget, Export]
