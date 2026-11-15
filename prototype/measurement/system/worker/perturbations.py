import numpy as np,json
from pathlib import Path
def transform_image(image,cfg):
 kind=cfg['kind']
 if kind=='exposure':return (image.astype(np.float32)*cfg['factor']).clip(0,255).astype(np.uint8)
 if kind=='colorcast':return (image.astype(np.float32)*np.asarray(cfg['channels'],dtype=np.float32)).clip(0,255).astype(np.uint8)
 if kind=='camera_shift':
  out=np.zeros_like(image);shift=round(image.shape[-2]*cfg['fraction']);out[...,shift:,:]=image[...,:image.shape[-2]-shift,:];return out
 raise ValueError(kind)
def install(model,cfg):
 old=model._input_transform_fn;target=cfg.get('target_prompt') or json.loads(Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020/data_split.json').read_text())['target']
 def fn(raw):
  if raw['prompt']==target:
   raw=dict(raw)
   for key in ['observation/image','observation/wrist_image']:raw[key]=transform_image(raw[key],cfg)
  return old(raw)
 model._input_transform_fn=fn
