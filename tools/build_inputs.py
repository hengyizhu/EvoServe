"""Recreate the fixed serving corpus from demonstration IDs; no GPU needed."""
import argparse,io,json,hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--work',required=True);a=p.parse_args();w=Path(a.work);cfg=json.loads((w/'workspace.json').read_text());manifest=json.loads((w/'vla_followup_20261020/benchmark/input_manifest.json').read_text());tables={};images=[];wrists=[];states=[];prompts=[]
for x in manifest['samples']:
 ep=x['episode']
 if ep not in tables:tables[ep]=pq.read_table(Path(cfg['data'])/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet')
 row=tables[ep].slice(x['frame'],1).to_pylist()[0]
 images.append(np.array(Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')));wrists.append(np.array(Image.open(io.BytesIO(row['wrist_image']['bytes'])).convert('RGB')));states.append(row['state']);prompts.append(x['task'])
arrays={'images':np.stack(images),'wrists':np.stack(wrists),'states':np.asarray(states,dtype=np.float32),'prompts':np.asarray(prompts)}
out=w/'vla_followup_20261020/benchmark/inputs.npz';np.savez(out,**arrays)
print(json.dumps({k:{'shape':list(v.shape),'dtype':str(v.dtype),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for k,v in arrays.items()},indent=2))
