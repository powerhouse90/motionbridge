"""Starting views for H3: render matching front / three-quarter / hand views of a character, or assemble supplied ones."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import folder_paths
from comfy_api.latest import io,ui
import comfy.model_management

from motionbridge.comfy.common import BLENDER_SCRIPTS, runtime

WORKER=BLENDER_SCRIPTS/'reference_views.py'
VERSION='0.1.0'
LAYOUTS=['front + 3/4','front + 3/4 + hands']

def resolve_asset(value):
    p=Path(value).expanduser()
    if not p.is_absolute():p=Path(folder_paths.get_input_directory())/p
    p=p.resolve()
    if not p.is_file():raise ValueError('Upload a character/scene file or select an existing file.')
    if p.suffix.lower() not in ('.blend','.glb','.gltf','.fbx'):raise ValueError('Supported files: BLEND, GLB, GLTF, FBX.')
    return p

def blender_path(value=''):
    if value:
        if not Path(value).is_file():raise ValueError(f'Blender not found at {value}')
        return str(Path(value).resolve())
    return runtime(check=False).blender()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def cache_folder(asset,settings):
    key=hashlib.sha256(json.dumps({'asset':str(asset),'sha256':digest(asset),'settings':settings,'worker':digest(WORKER),'version':VERSION},sort_keys=True).encode()).hexdigest()[:24]
    return Path(folder_paths.get_output_directory())/'motionbridge'/'references'/key

def dependency_signature(paths):
    result=[]
    for value in paths:
        p=Path(value)
        result.append([str(p),p.stat().st_size,p.stat().st_mtime_ns] if p.is_file() else [str(p),None,None])
    return result

def read_tensor(path):
    with Image.open(path) as im:return torch.from_numpy(np.asarray(im.convert('RGB'),dtype=np.float32).copy()/255)[None]

def assemble(front,threequarter,hands=None):
    images=[front,threequarter]+([hands] if hands is not None else [])
    shape=front.shape
    if len(shape)!=4 or shape[0]!=1 or shape[-1]!=3:raise ValueError('References must be single RGB images, not video batches.')
    if any(x.shape!=shape for x in images):raise ValueError('All starting views must have identical dimensions. Resize the input images first.')
    return torch.cat(images,dim=2)

class RenderStartingViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MotionBridgeRenderStartingViews',display_name='Render Starting Views',category='MotionBridge/References',description='Render matching front, elevated three-quarter and optional hand views from one native 3D pose. Upload a rigged GLB/FBX or a BLEND scene.',is_output_node=True,
            inputs=[io.String.Input('asset_file',default='',tooltip='Use Upload character / scene, or a path relative to ComfyUI input. Ignored when a character is connected.'),io.Custom('MOTIONBRIDGE_CHARACTER').Input('character',optional=True,tooltip='From Load Character: render that file and armature.'),io.Combo.Input('layout',options=LAYOUTS,default=LAYOUTS[0]),io.Int.Input('frame',default=1,min=0,max=100000),io.Int.Input('panel_width',default=528,min=128,max=2048,step=16),io.Int.Input('panel_height',default=704,min=128,max=2048,step=16),io.Combo.Input('framing',options=['character and props','character only'],default='character and props'),io.Combo.Input('front_direction',options=['auto','-Y','+Y','-X','+X'],default='auto'),io.Float.Input('threequarter_angle',default=50,min=15,max=80,step=1),io.Float.Input('elevation',default=25,min=0,max=60,step=1),io.String.Input('armature_name',default='',advanced=True),io.String.Input('left_hand_bone',default='',advanced=True),io.String.Input('right_hand_bone',default='',advanced=True),io.String.Input('blender_executable',default='',advanced=True)],
            outputs=[io.Image.Output(display_name='H3 reference'),io.Image.Output(display_name='front'),io.Image.Output(display_name='threequarter'),io.Image.Output(display_name='hands'),io.String.Output(display_name='camera metadata')])

    @classmethod
    def fingerprint_inputs(cls,asset_file,character=None,**kwargs):
        if character:asset_file=character['file'];kwargs['armature_name']=character['armature']
        try:
            asset=resolve_asset(asset_file);out=cache_folder(asset,kwargs);manifest=out/'metadata.json'
            deps=json.loads(manifest.read_text()).get('dependencies',[]) if manifest.exists() else []
            return [digest(asset),digest(WORKER),dependency_signature(deps)]
        except (OSError,ValueError):return float('nan')

    @classmethod
    def execute(cls,asset_file,character=None,**settings):
        if character:asset_file=character['file'];settings['armature_name']=character['armature']
        asset=resolve_asset(asset_file);binary=blender_path(settings.get('blender_executable',''));out=cache_folder(asset,settings);out.mkdir(parents=True,exist_ok=True);manifest=out/'metadata.json'
        metadata=json.loads(manifest.read_text()) if manifest.exists() else None
        ready=metadata is not None and all((out/(v+'.png')).exists() for v in ('front','threequarter','hands')) and metadata.get('dependency_signature')==dependency_signature(metadata.get('dependencies',[]))
        if not ready:
            request={'asset':str(asset),'output':str(out),**settings};(out/'request.json').write_text(json.dumps(request,indent=2))
            command=[binary,'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(WORKER),'--',str(out/'request.json')]
            with (out/'render.log').open('w',encoding='utf-8') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                try:
                    while process.poll() is None:
                        comfy.model_management.throw_exception_if_processing_interrupted();time.sleep(.2)
                except BaseException:
                    process.terminate();process.wait(timeout=15);raise
            if process.returncode:
                tail=(out/'render.log').read_text(encoding='utf-8',errors='replace')[-2500:]
                raise RuntimeError('Starting-view render failed. '+tail)
            metadata=json.loads(manifest.read_text());metadata['dependency_signature']=dependency_signature(metadata.get('dependencies',[]));manifest.write_text(json.dumps(metadata,indent=2))
        front,quarter,hands=[read_tensor(out/(v+'.png')) for v in ('front','threequarter','hands')]
        reference=assemble(front,quarter,hands if settings['layout']==LAYOUTS[1] else None)
        Image.fromarray((reference[0].numpy()*255).round().astype('uint8')).save(out/'reference.png')
        return io.NodeOutput(reference,front,quarter,hands,json.dumps(metadata,indent=2),ui=ui.PreviewImage(reference,cls=cls))

class AssembleReferenceViews(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MotionBridgeAssembleReferenceViews',display_name='Assemble H3 Starting Views',category='MotionBridge/References',description='Combine supplied matching starting images, without resizing or changing their pose.',inputs=[io.Image.Input('front'),io.Image.Input('threequarter'),io.Combo.Input('layout',options=LAYOUTS,default=LAYOUTS[0]),io.Image.Input('hands',optional=True)],outputs=[io.Image.Output(display_name='H3 reference'),io.Int.Output(display_name='width'),io.Int.Output(display_name='height')])
    @classmethod
    def execute(cls,front,threequarter,layout,hands=None):
        if layout==LAYOUTS[1] and hands is None:raise ValueError('Connect a matching hand starting image for the three-view layout.')
        reference=assemble(front,threequarter,hands if layout==LAYOUTS[1] else None)
        return io.NodeOutput(reference,reference.shape[2],reference.shape[1])
