"""Adapter to the user's installed GEM-X runtime; no models bundled or installed.

Ported from the Midnight Rinse mocap harness. Runs one GEM stage per process
(detect, pose, features, motion, export) so each releases the GPU when done.
Run with the GEM environment Python; extract.py drives the stage loop.
"""
import argparse,os,sys,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gem-root',required=True);p.add_argument('--video',required=True);p.add_argument('--out',required=True);p.add_argument('--stage',choices=['detect','pose','features','motion','export'],required=True)
p.add_argument('--no-contact-postproc',action='store_true',help='Diagnostic ablation of GEM contact translation/IK cleanup')
p.add_argument('--seed',type=int,help='Optional fixed seed for paired inference experiments')
a=p.parse_args();root=Path(a.gem_root).resolve();out=Path(a.out).resolve();video=Path(a.video).resolve();out.mkdir(parents=True,exist_ok=True)
# Executing this file directly otherwise exposes the harness's profile.py as
# stdlib "profile", which breaks cProfile imports inside Torch/Transformers.
script_dir=Path(__file__).resolve().parent
sys.path[:]=[entry for entry in sys.path if Path(entry or os.curdir).resolve()!=script_dir]
sys.path.insert(0,str(root));os.chdir(root);os.environ['PYOPENGL_PLATFORM']='win32' if sys.platform=='win32' else 'egl';os.environ['WANDB_MODE']='disabled'
import torch,numpy as np
if a.seed is not None:
    torch.manual_seed(a.seed);np.random.seed(a.seed)
from hydra import compose,initialize_config_dir
import hydra
# Installed model assets are trusted local dependencies; this override is confined
# to the backend subprocess and is never used for user-supplied observation caches.
original_load=torch.load
def model_load(*args,**kwargs):
    kwargs.setdefault('weights_only',False);return original_load(*args,**kwargs)
torch.load=model_load
with initialize_config_dir(version_base='1.3',config_dir=str(root/'configs')):
    cfg=compose(config_name='demo_soma',overrides=['exp=gem_soma_regression','video_name=capture',f'video_path={video.as_posix()}',f'output_root={out.as_posix()}','static_cam=true','verbose=false','render_mhr=false','use_wandb=false','task=test','ckpt_path=inputs/pretrained/gem_soma.ckpt','sam3d_ckpt_path=inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt','sam3d_mhr_path=inputs/mhr_data/mhr_model.pt'])
Path(cfg.preprocess_dir).mkdir(parents=True,exist_ok=True)
from gem.utils.video_io_utils import read_video_np,get_video_lwh
L,W,H=get_video_lwh(cfg.video_path)
with torch.no_grad():
    if a.stage=='detect':
        from gem.utils.yolox_detector import YOLOXDetector,detect_and_track
        from gem.utils.kp2d_utils import smooth_bbx_xyxy
        from gem.utils.geo_transform import get_bbx_xys_from_xyxy
        boxes,_=detect_and_track(read_video_np(cfg.video_path),YOLOXDetector(device='cuda'))
        boxes=smooth_bbx_xyxy(torch.from_numpy(boxes).float(),window=3);boxes[:,[0,2]]=boxes[:,[0,2]].clamp(0,W-1);boxes[:,[1,3]]=boxes[:,[1,3]].clamp(0,H-1)
        torch.save({'bbx_xyxy':boxes,'bbx_xys':get_bbx_xys_from_xyxy(boxes,base_enlarge=1.2).float()},cfg.paths.bbx)
    elif a.stage=='pose':
        from gem.utils.vitpose_extractor import VitPoseExtractor
        result=VitPoseExtractor(device='cuda:0',pose_type='soma').extract(read_video_np(cfg.video_path),torch.load(cfg.paths.bbx)['bbx_xys'],img_ds=1.,batch_size=2,path_type='np');torch.save(result,cfg.paths.vitpose)
    elif a.stage=='features':
        from gem.utils.sam3db_extractor import SAM3DBExtractor
        from gem.utils.cam_utils import estimate_K,get_a_pred_cam
        boxes=torch.load(cfg.paths.bbx)['bbx_xys'];result=SAM3DBExtractor(checkpoint_path=cfg.sam3d_ckpt_path,mhr_path=cfg.sam3d_mhr_path,device='cuda:0').extract_video_features(cfg.video_path,boxes,batch_size=1)
        K=estimate_K(W,H).repeat(L,1,1);torch.save({'pose_tokens':result['pose_tokens'],'pred_cam':get_a_pred_cam(result['transls'],boxes,K)},cfg.paths.vit_features)
    elif a.stage=='motion':
        from scripts.demo.demo_soma import load_data_dict
        from gem.utils.net_utils import detach_to_cpu
        model=hydra.utils.instantiate(cfg.model,_recursive_=False);model.load_pretrained_model(cfg.ckpt_path);model=model.eval().cuda()
        torch.save(detach_to_cpu(model.predict(load_data_dict(cfg),static_cam=True,postproc=not a.no_contact_postproc)),cfg.paths.hpe_results)
        (out/'inference_options.json').write_text(json.dumps({'static_camera':True,'contact_postprocessing_requested':not a.no_contact_postproc},indent=2))
    else:
        from gem.utils.soma_utils.soma_layer import SomaLayer
        from scripts.demo.demo_soma import _get_body_params
        pred=torch.load(cfg.paths.hpe_results);soma=SomaLayer('inputs/soma_assets',low_lod=True,device='cuda',identity_model_type='mhr',mode='warp');captured={};original=soma.soma.batched_skinning.pose
        def capture(*args,**kwargs):
            result=original(*args,**kwargs);captured['transforms']=result[1].detach().cpu().numpy();return result
        soma.soma.batched_skinning.pose=capture;result={}
        for space in ['incam','global']:
            params={k:v.cuda() for k,v in _get_body_params(pred,'body_params_'+space).items()};body=soma(**params)
            result[space+'_joints']=body['joints'].cpu().numpy();result[space+'_transforms']=captured['transforms']
            if 'vertices' in body:result[space+'_vertices']=body['vertices'].cpu().numpy().astype(np.float32)  # body mesh for visual review
        ident=params['identity_coeffs'].reshape(-1,params['identity_coeffs'].shape[-1]).mean(0,keepdim=True);scales=params['scale_params'].reshape(-1,params['scale_params'].shape[-1]).mean(0,keepdim=True);soma.get_skeleton(ident,scales);result['reference_transforms']=captured['transforms']
        skeleton=np.load(root/'inputs/soma_assets/SOMA_neutral.npz',allow_pickle=True);result['names']=np.asarray(skeleton['joint_names'],dtype=str);result['parents']=np.asarray(skeleton['joint_parent_ids']);result['K']=pred['K_fullimg'].cpu().numpy()
        if hasattr(soma,'faces'):result['faces']=np.asarray(soma.faces.cpu() if hasattr(soma.faces,'cpu') else soma.faces)
        np.savez_compressed(out/'motion_export.npz',**result)
        (out/'extraction.json').write_text(json.dumps({'motion':str(out/'motion_export.npz'),'keypoints':str(Path(cfg.paths.vitpose).resolve()),'video':str(video),'camera_assumption':'static inference; output fitting remains camera-relative'},indent=2))
print('BACKEND_STAGE_DONE',a.stage,flush=True)
