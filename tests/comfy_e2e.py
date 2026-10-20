"""End-to-end check of the node pack through a running ComfyUI: character -> starting views -> H3 ->
GEM -> retarget -> export, one take from golden/h3_takes.toml.

python -m tests.comfy_e2e --url http://127.0.0.1:8189 --character input/path.blend --take walk
       [--unet NAME] [--accept-licenses]

Prints each node's text output and the files written. Fails on any node error.
"""
import argparse
import json
import time
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def graph(character, take, defaults, unet, accept, seed):
    prompt = ' '.join([defaults['two_view_intro'], take['action'], defaults['outro']])
    return {
        'setup': {'class_type': 'MotionBridgeSetup', 'inputs': {'accept_licenses': accept, 'blender': 'use installed, else download'}},
        'character': {'class_type': 'MotionBridgeLoadCharacter', 'inputs': {'asset_file': character, 'setup': ['setup', 0]}},
        'views': {'class_type': 'MotionBridgeRenderStartingViews', 'inputs': {
            'asset_file': '', 'character': ['character', 0], 'layout': 'front + 3/4', 'frame': 1, 'panel_width': 528,
            'panel_height': 704, 'framing': 'character only', 'front_direction': 'auto', 'threequarter_angle': 50,
            'elevation': 15, 'armature_name': '', 'left_hand_bone': '', 'right_hand_bone': '', 'blender_executable': ''}},
        'scale': {'class_type': 'ImageScaleToTotalPixels', 'inputs': {'image': ['views', 0], 'upscale_method': 'lanczos',
                                                                      'megapixels': 0.7, 'resolution_steps': 32}},
        'size': {'class_type': 'GetImageSize', 'inputs': {'image': ['scale', 0]}},
        'unet': {'class_type': 'UNETLoader', 'inputs': {'unet_name': unet, 'weight_dtype': 'default'}},
        'lora': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['unet', 0], 'lora_name': 'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors', 'strength_model': 1.0}},
        'clip': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
                                                        'type': 'minimax', 'device': 'default'}},
        'vae': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}},
        'h3': {'class_type': 'MiniMaxH3ImageToVideo', 'inputs': {
            'clip': ['clip', 0], 'vae': ['vae', 0], 'prompt': prompt, 'width': ['size', 0], 'height': ['size', 1],
            'length': defaults['frames'], 'first_frame': ['scale', 0]}},
        'noise': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed}},
        'guider': {'class_type': 'BasicGuider', 'inputs': {'model': ['lora', 0], 'conditioning': ['h3', 0]}},
        'sampler': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'res_multistep'}},
        'sigmas': {'class_type': 'BasicScheduler', 'inputs': {'model': ['lora', 0], 'scheduler': 'simple', 'steps': 8, 'denoise': 1.0}},
        'sample': {'class_type': 'SamplerCustomAdvanced', 'inputs': {
            'noise': ['noise', 0], 'guider': ['guider', 0], 'sampler': ['sampler', 0], 'sigmas': ['sigmas', 0],
            'latent_image': ['h3', 1]}},
        'decode': {'class_type': 'VAEDecode', 'inputs': {'samples': ['sample', 0], 'vae': ['vae', 0]}},
        'video': {'class_type': 'CreateVideo', 'inputs': {'images': ['decode', 0], 'fps': 24.0}},
        'save_h3': {'class_type': 'SaveVideo', 'inputs': {'video': ['video', 0], 'filename_prefix': 'motionbridge/h3',
                                                          'format': 'mp4', 'codec': 'auto'}},
        'motion': {'class_type': 'MotionBridgeVideoToMotion', 'inputs': {
            'images': ['decode', 0], 'fps': 24.0, 'panel': 'left half (front view of 2)',
            'camera': 'rendered reference (orthographic)'}},
        'retarget': {'class_type': 'MotionBridgeRetarget', 'inputs': {
            'character': ['character', 0], 'motion': ['motion', 0], 'action_name': take['name'], 'foot_ik': True}},
        'export': {'class_type': 'MotionBridgeExport', 'inputs': {
            'animation': ['retarget', 0], 'filename_prefix': 'motionbridge/character', 'glb': True,
            'preview': 'follow'}},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8189')
    p.add_argument('--character', required=True, help='path relative to ComfyUI/input, or absolute')
    p.add_argument('--take', default='walk')
    p.add_argument('--unet', default='minimax_h3_fl2va_pruned_int8_convrot.safetensors')
    p.add_argument('--accept-licenses', action='store_true')
    a = p.parse_args()
    spec = tomllib.loads((ROOT / 'golden' / 'h3_takes.toml').read_text(encoding='utf-8'))
    take = next(t for t in spec['takes'] if t['name'] == a.take)
    prompt = graph(a.character, take, spec['defaults'], a.unet, a.accept_licenses, spec['defaults']['seed'])
    request = urllib.request.Request(a.url + '/prompt', json.dumps({'prompt': prompt}).encode(),
                                     {'Content-Type': 'application/json'})
    try:
        queued = json.load(urllib.request.urlopen(request))
    except urllib.error.HTTPError as e:
        raise SystemExit(f'ComfyUI rejected the graph: {e.read().decode()[:3000]}')
    pid, started = queued['prompt_id'], time.time()
    print('queued', pid, flush=True)
    while True:
        history = json.load(urllib.request.urlopen(f'{a.url}/history/{pid}'))
        if pid in history:
            break
        time.sleep(5)
    entry = history[pid]
    for kind, data in entry['status'].get('messages', []):
        if kind == 'execution_error':
            raise SystemExit(f"FAILED at {data['node_id']} ({data['node_type']}): {data['exception_message'][-3000:]}")
    print(f"{entry['status']['status_str']} in {time.time() - started:.0f} s")
    for node, out in entry['outputs'].items():
        for text in out.get('text', []):
            print(f'--- {node}\n{text}')
        for item in out.get('images', []):
            print(f"--- {node} file: output/{item.get('subfolder', '')}/{item['filename']}")


if __name__ == '__main__':
    main()
