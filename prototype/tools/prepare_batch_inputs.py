"""Small real demonstration corpus; matching raw observations and 10-action windows."""
import argparse,io,json,pickle
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--split',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=json.loads(Path(a.split).read_text());rows=[]
for category in ['target','history']:
 for ep in cfg[category+'_train'][:4]:
  eid=ep['episode_index'];table=pq.read_table(Path(cfg['data_root'])/f'data/chunk-{eid//1000:03d}/episode_{eid:06d}.parquet')
  for start in np.linspace(0,len(table)-10,8,dtype=int):
   row=table.slice(int(start),1).to_pylist()[0]
   image=lambda key:np.array(Image.open(io.BytesIO(row[key]['bytes'])).convert('RGB'))
   raw={'observation/image':image('image'),'observation/wrist_image':image('wrist_image'),'observation/state':np.array(row['state'],dtype=np.float32),'prompt':cfg[category]}
   if category=='target':
    for key in ['observation/image','observation/wrist_image']:raw[key]=np.clip(raw[key].astype(np.float32)*.05,0,255).astype(np.uint8)
   rows.append(dict(raw=raw,actions=np.array(table['actions'].slice(int(start),10).to_pylist(),dtype=np.float32),episode=eid,frame=int(start),category=category))
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(pickle.dumps(rows));print(json.dumps(dict(windows=len(rows),path=str(out))))
