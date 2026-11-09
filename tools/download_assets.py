"""Pinned public model, tokenizer and required demonstration episodes; CPU/network only."""
import argparse,json,hashlib,urllib.request
from pathlib import Path
from huggingface_hub import hf_hub_download
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--only',choices=['model','data','tokenizer','all'],default='all');a=p.parse_args();w=Path(a.work);cfg=json.loads((w/'workspace.json').read_text());manifest=json.loads((R/'experiments/vla_followup_20261020/audit/parent_download.json').read_text())
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
if a.only in ['model','all']:
 for item in manifest['files']:
  dest=hf_hub_download(manifest['repo'],item['path'],revision=manifest['revision'],local_dir=cfg['model']);assert digest(dest)==item['sha256'],item['path'];print('verified',item['path'])
if a.only in ['data','all']:
 required=set()
 for path in (R/'experiments').rglob('*.json'):
  if path.name=='data_split.json' or path.parent.name=='splits':
   split=json.loads(path.read_text())
   for field in ['target_train','target_validation','history_train']:
    required.update(e['episode_index'] for e in split.get(field,[]))
   required.update(e['episode'] for e in split.get('replay_windows',[]))
 source=json.loads((R/'experiments/vla_followup_20261020/benchmark/input_manifest.json').read_text());required.update(x['episode'] for x in source['samples'])
 files=['meta/info.json','meta/tasks.jsonl','meta/episodes.jsonl','meta/stats.json']+[f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet' for ep in sorted(required)]
 rev='9dfa69510ea9e1613fc54112bc706444b686a231'
 for file in files:hf_hub_download('physical-intelligence/libero',file,repo_type='dataset',revision=rev,local_dir=cfg['data'])
 print('dataset revision',rev,'episode count',len(required))
if a.only in ['tokenizer','all']:
 dest=w/'vla_followup_20261020/assets/.cache/openpi/big_vision/paligemma_tokenizer.model';dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():
  temp=dest.with_suffix('.partial');urllib.request.urlretrieve('https://storage.googleapis.com/big_vision/paligemma_tokenizer.model',temp);temp.replace(dest)
 expected=json.loads((R/'provenance/assets.json').read_text())['tokenizer_sha256'];assert digest(dest)==expected,'tokenizer mismatch'
 print('tokenizer verified')
