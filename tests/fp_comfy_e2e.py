"""Drive the first-person graph through ComfyUI's API: Setup -> Load FP Scene -> Load Video ->
Track FP Hands -> Export FP Animation. python tests/fp_comfy_e2e.py URL SCENE_IN_INPUT VIDEO_IN_INPUT"""
import json, sys, time, urllib.request
url, scene, video = sys.argv[1:4]
W, H = (int(v) for v in sys.argv[4:6]) if len(sys.argv) > 5 else (1280, 720)
ACTION = sys.argv[6] if len(sys.argv) > 6 else 'grab'
PROP = sys.argv[7] if len(sys.argv) > 7 else ''
g = {
    'setup': {'class_type': 'MotionBridgeSetup', 'inputs': {'accept_licenses': True, 'blender': 'use installed, else download'}},
    'scene': {'class_type': 'MotionBridgeLoadFPScene', 'inputs': {'scene_file': scene, 'width': W, 'height': H, 'capture_mode': True, 'main_prop': PROP, 'setup': ['setup', 0]}},
    'video': {'class_type': 'LoadVideo', 'inputs': {'file': video}},
    'track': {'class_type': 'MotionBridgeTrackFPHands', 'inputs': {'fp_scene': ['scene', 0], 'video': ['video', 0], 'fps': 24.0, 'smoothing': 0.7, 'hands': 'both'}},
    'export': {'class_type': 'MotionBridgeExportFP', 'inputs': {'fp_motion': ['track', 0], 'filename_prefix': 'motionbridge/fp_test',
                                                                'action_name': ACTION, 'glb': True, 'preview': 'side by side'}},
}
req = urllib.request.Request(url + '/prompt', json.dumps({'prompt': g}).encode(), {'Content-Type': 'application/json'})
resp = json.load(urllib.request.urlopen(req))
if resp.get('node_errors'):
    sys.exit('INVALID: ' + json.dumps(resp['node_errors'])[:2000])
pid = resp['prompt_id']
print('queued', pid, flush=True); t0 = time.time()
while True:
    h = json.load(urllib.request.urlopen(f'{url}/history/{pid}'))
    if pid in h: break
    time.sleep(3)
e = h[pid]
for kind, data in e['status'].get('messages', []):
    if kind == 'execution_error':
        sys.exit(f"FAILED in {data['node_type']}: {data['exception_message'][-3000:]}")
print(f'done in {time.time() - t0:.0f}s')
for node, out in e['outputs'].items():
    for k, v in out.items():
        print(node, k, json.dumps(v)[:600])
