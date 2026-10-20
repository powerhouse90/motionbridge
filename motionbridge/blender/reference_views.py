"""Headless, read-only source-scene renderer. Run only inside Blender."""
import bpy,sys,json,math,hashlib,re
from pathlib import Path
from mathutils import Vector
request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text());asset=Path(request['asset']);out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
if asset.suffix.lower()=='.blend':bpy.ops.wm.open_mainfile(filepath=str(asset),load_ui=False,use_scripts=False)
else:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if asset.suffix.lower() in ('.glb','.gltf'):bpy.ops.import_scene.gltf(filepath=str(asset))
    elif asset.suffix.lower()=='.fbx':bpy.ops.import_scene.fbx(filepath=str(asset))
    else:raise ValueError('Unsupported scene format')
scene=bpy.context.scene;scene.frame_set(request['frame']);bpy.context.view_layer.update()
rigs=[o for o in scene.objects if o.type=='ARMATURE'];selected=request.get('armature_name','')
if selected:
    rig=scene.objects.get(selected)
    if rig is None or rig.type!='ARMATURE':raise ValueError('Selected armature was not found')
elif len(rigs)==1:rig=rigs[0]
elif not rigs:raise ValueError('This file has no armature. Supply a rigged character.')
else:raise ValueError('Multiple armatures found; choose one with armature_name: '+', '.join(o.name for o in rigs))
def rig_signature():
    data={b.name:{'parent':b.parent.name if b.parent else None,'rest':[list(row) for row in b.matrix_local],'length':b.length} for b in rig.data.bones}
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
signature=rig_signature();pose_before={b.name:[list(row) for row in b.matrix_basis] for b in rig.pose.bones};world_before=[list(row) for row in rig.matrix_world]
skin=[o for o in scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' and m.object==rig for m in o.modifiers)]
if not skin:raise ValueError('No skinned meshes linked to the armature')
def bounds(objects):
    points=[]
    deps=bpy.context.evaluated_depsgraph_get()
    for o in objects:
        evaluated=o.evaluated_get(deps)
        points.extend(evaluated.matrix_world@Vector(corner) for corner in evaluated.bound_box)
    return points
character_points=bounds(skin);character_low=Vector(tuple(min(p[i] for p in character_points) for i in range(3)));character_high=Vector(tuple(max(p[i] for p in character_points) for i in range(3)));height=max(character_high.z-character_low.z,.01)
def normalized(name):return re.sub('[^a-z0-9]','',name.lower())
def bone_for(side,part,override=''):
    if override:
        if override not in rig.pose.bones:raise ValueError('Unknown bone '+override)
        return rig.pose.bones[override]
    short='l' if side=='left' else 'r';tokens=[side+part,short+part,part+short]
    candidates=[b for b in rig.pose.bones if any(normalized(b.name).endswith(t) for t in tokens)]
    if len(candidates)==1:return candidates[0]
    return None
up=Vector((0,0,1));facing=request['front_direction'];direction={'-Y':Vector((0,-1,0)),'+Y':Vector((0,1,0)),'-X':Vector((-1,0,0)),'+X':Vector((1,0,0))}.get(facing)
facing_method=facing
if direction is None:
    left=bone_for('left','upperarm') or bone_for('left','arm');right=bone_for('right','upperarm') or bone_for('right','arm')
    if left and right:
        across=rig.matrix_world.to_3x3()@(right.bone.head_local-left.bone.head_local);across.z=0;direction=up.cross(across).normalized();facing_method='rest shoulder axis'
    else:direction=Vector((0,-1,0));facing_method='fallback -Y; set front_direction if needed'
meshes=[o for o in scene.objects if o.type=='MESH' and not o.hide_render];framing=skin.copy();excluded=[]
if request['framing']=='character and props':
    for obj in meshes:
        if obj in skin:continue
        dims=obj.dimensions
        # Exclude broad flat floors/backdrops from framing, keep them rendered.
        if max(dims)>height*8 or (dims.z<height*.03 and max(dims.x,dims.y)>height*2):excluded.append(obj.name);continue
        framing.append(obj)
else:
    for obj in meshes:
        if obj not in skin:obj.hide_render=True
points=bounds(framing);lo=Vector(tuple(min(p[i] for p in points) for i in range(3)));hi=Vector(tuple(max(p[i] for p in points) for i in range(3)));center=(lo+hi)/2
camera_data=bpy.data.cameras.new('MotionBridge_ReferenceCamera');camera=bpy.data.objects.new('MotionBridge_ReferenceCamera',camera_data);scene.collection.objects.link(camera);scene.camera=camera;camera_data.type='ORTHO';camera_data.sensor_fit='AUTO';camera_data.clip_start=max(height*.001,.001);camera_data.clip_end=height*200
width=request['panel_width'];imgheight=request['panel_height'];aspect=width/imgheight
def camera_rotation(azimuth,elevation):
    theta=math.radians(azimuth);d=Vector((direction.x*math.cos(theta)-direction.y*math.sin(theta),direction.x*math.sin(theta)+direction.y*math.cos(theta),0));d=d*math.cos(math.radians(elevation))+up*math.sin(math.radians(elevation));return d,(-d).to_track_quat('-Z','Y')
specs={'front':camera_rotation(0,0),'threequarter':camera_rotation(request['threequarter_angle'],request['elevation'])}
def scale_for(pts,target,quat):
    inv=quat.to_matrix().transposed();local=[inv@(p-target) for p in pts];spanx=2*max(abs(v.x) for v in local);spany=2*max(abs(v.y) for v in local)
    vertical=max(spany,spanx/aspect)*1.15
    return vertical*max(aspect,1)
common_scale=max(scale_for(points,center,q) for d,q in specs.values())
# EEVEE is fast and doesn't load another inference model.
try:scene.render.engine='BLENDER_EEVEE_NEXT'
except TypeError:scene.render.engine='BLENDER_EEVEE'
if hasattr(scene,'eevee'):scene.eevee.taa_render_samples=32
scene.render.resolution_x=width;scene.render.resolution_y=imgheight;scene.render.resolution_percentage=100;scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
if not scene.world:scene.world=bpy.data.worlds.new('MotionBridge_World')
scene.world.use_nodes=True;scene.world.node_tree.nodes.get('Background').inputs[0].default_value=(.22,.22,.22,1);scene.world.node_tree.nodes.get('Background').inputs[1].default_value=.6
if not any(o.type=='LIGHT' and not o.hide_render for o in scene.objects):
    for name,offset,power in [('Key',(2,-3,4),700),('Fill',(-3,-1,2),450),('Rim',(0,3,3),500)]:
        light=bpy.data.lights.new('MotionBridge_'+name,'AREA');light.energy=power*(height/1.7)**2;light.shape='DISK';light.size=height*2;o=bpy.data.objects.new(light.name,light);scene.collection.objects.link(o);o.location=center+Vector(offset)*(height/1.7);o.rotation_euler=(center-o.location).to_track_quat('-Z','Y').to_euler()
metadata={'version':1,'asset':str(asset),'frame':request['frame'],'armature':rig.name,'native_rest_sha256':signature,'native_bone_lengths_preserved':True,'front_direction':list(direction),'front_direction_method':facing_method,'framed_objects':[o.name for o in framing],'excluded_from_framing':excluded,'cameras':{},'panel_size':[width,imgheight]}
def render(name,target,d,quat,scale):
    camera.location=target+d*height*10;camera.rotation_euler=quat.to_euler();camera_data.ortho_scale=scale;bpy.context.view_layer.update();scene.render.filepath=str(out/(name+'.png'));bpy.ops.render.render(write_still=True)
    metadata['cameras'][name]={'matrix_world':[list(row) for row in camera.matrix_world],'projection':'orthographic','ortho_scale':scale,'target':list(target),'width':width,'height':imgheight}
for name,(d,q) in specs.items():render(name,center,d,q,common_scale)
hands=[bone_for('left','hand',request.get('left_hand_bone','')),bone_for('right','hand',request.get('right_hand_bone',''))]
if all(hands):
    hand_points=[]
    for hand in hands:
        for bone in [hand]+list(hand.children_recursive):hand_points.extend((rig.matrix_world@bone.head,rig.matrix_world@bone.tail))
    hlow=Vector(tuple(min(p[i] for p in hand_points) for i in range(3)));hhigh=Vector(tuple(max(p[i] for p in hand_points) for i in range(3)));hcenter=(hlow+hhigh)/2;d,q=specs['front'];hscale=max(scale_for(hand_points,hcenter,q)*1.2,height*.18)
    render('hands',hcenter,d,q,hscale);metadata['hand_bones']=[b.name for b in hands]
elif request['layout']=='front + 3/4 + hands':raise ValueError('Could not identify both hand bones. Set left_hand_bone and right_hand_bone in advanced options.')
else:
    # Unused output remains a valid image for the two-view workflow.
    import shutil
    shutil.copy2(out/'front.png',out/'hands.png');metadata['hand_view_available']=False
deps=[]
for image in bpy.data.images:
    if image.source=='FILE' and not image.packed_file and image.filepath:
        p=Path(bpy.path.abspath(image.filepath)).resolve()
        if p.is_file():deps.append(str(p))
if asset.suffix.lower()=='.gltf':
    gltf=json.loads(asset.read_text())
    for entry in gltf.get('buffers',[])+gltf.get('images',[]):
        uri=entry.get('uri','')
        if uri and not uri.startswith('data:'):deps.append(str((asset.parent/uri).resolve()))
metadata['dependencies']=sorted(set(deps));assert rig_signature()==signature;assert pose_before=={b.name:[list(row) for row in b.matrix_basis] for b in rig.pose.bones};assert world_before==[list(row) for row in rig.matrix_world]
(out/'metadata.json').write_text(json.dumps(metadata,indent=2));print('MOTIONBRIDGE_REFERENCES_READY',out,flush=True)
