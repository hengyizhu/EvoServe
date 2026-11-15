"""Real simulator camera orbit / table texture. No image-space substitute."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

def apply_scene(env,cfg,upload=True):
 sim=env.sim;model=sim.model._model;data=sim.data._data
 cache=getattr(env,'_followon_scene_base',None)
 if cache is None or cache['model'] is not model:
  cache={'model':model,'cam_pos':model.cam_pos.copy(),'cam_quat':model.cam_quat.copy(),'mat_rgba':model.mat_rgba.copy(),'tex_data':model.tex_data.copy()};env._followon_scene_base=cache
 kind=cfg['kind']
 def geom_names(i):
  names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or ''];body=int(model.geom_bodyid[i])
  while body:
   names.append(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body) or '');body=int(model.body_parentid[body])
  return ' '.join(names).lower()
 table=[i for i in range(model.ngeom) if 'table' in geom_names(i) and 'leg' not in geom_names(i)]
 assert table,'no table geoms found'
 if kind=='camera_orbit':
  cid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,'agentview');assert cid>=0
  gid=max(table,key=lambda i:float(np.prod(model.geom_size[i,:2])));center=data.geom_xpos[gid].copy();rot=Rotation.from_euler('z',cfg['degrees'],degrees=True).as_matrix();body=int(model.cam_bodyid[cid]);bodyrot=data.xmat[body].reshape(3,3);bodypos=data.xpos[body]
  worldpos=bodypos+bodyrot@cache['cam_pos'][cid];newpos=center+rot@(worldpos-center);q=cache['cam_quat'][cid];oldrot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();newrot=bodyrot.T@rot@bodyrot@oldrot;qnew=Rotation.from_matrix(newrot).as_quat()
  model.cam_pos[cid]=bodyrot.T@(newpos-bodypos);model.cam_quat[cid]=qnew[[3,0,1,2]];sim.forward()
  return dict(kind=kind,degrees=cfg['degrees'],camera='agentview',wrist_changed=False,orbit_center=center.tolist(),parent_pos=worldpos.tolist(),new_pos=newpos.tolist(),physics_unchanged=True)
 if kind=='table_texture':
  mids=sorted({int(model.geom_matid[i]) for i in table if model.geom_matid[i]>=0});tids=sorted({int(model.mat_texid[m,int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]) for m in mids if model.mat_texid[m,int(mujoco.mjtTextureRole.mjTEXROLE_RGB)]>=0});assert tids,'table has no RGB texture'
  colors=np.asarray(cfg['colors'],dtype=np.uint8);cells=int(cfg['cells']);changed=[]
  for t in tids:
   at=int(model.tex_adr[t]);h=int(model.tex_height[t]);w=int(model.tex_width[t]);ch=int(model.tex_nchannel[t]);assert ch in [3,4]
   grid=(np.arange(h)[:,None]*cells//h+np.arange(w)[None,:]*cells//w)%len(colors);arr=cache['tex_data'][at:at+h*w*ch].reshape(h,w,ch).copy();arr[...,:3]=colors[grid];model.tex_data[at:at+h*w*ch]=arr.ravel();changed.append(dict(texture_id=t,height=h,width=w))
  for m in mids:model.mat_rgba[m]=[1.,1.,1.,1.]
  if upload:
   ctx=sim._render_context_offscreen;ctx.gl_ctx.make_current()
   for t in tids:mujoco.mjr_uploadTexture(model,ctx.con,t)
  sim.forward();return dict(kind=kind,colors=colors.tolist(),cells=cells,textures=changed,physics_unchanged=True)
 raise ValueError(kind)

def repair_xml(xml):
 from pathlib import Path
 import xml.etree.ElementTree as ET
 import robosuite
 from libero.libero import get_libero_path
 tree=ET.fromstring(xml)
 for node in tree.iter():
  old=node.get('file')
  if not old:continue
  p=Path(old)
  if p.exists():continue
  parts=p.parts
  if 'robosuite' in parts:
   i=max(i for i,x in enumerate(parts) if x=='robosuite');new=Path(robosuite.__file__).parent.joinpath(*parts[i+1:])
  elif 'assets' in parts:
   i=max(i for i,x in enumerate(parts) if x=='assets');new=Path(get_libero_path('assets')).joinpath(*parts[i+1:])
  else:raise FileNotFoundError(old)
  assert new.exists(),str(new);node.set('file',str(new))
 return ET.tostring(tree,encoding='unicode')
