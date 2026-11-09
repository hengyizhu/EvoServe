import json,pathlib,csv
R=pathlib.Path('/home/hengyi/pi05_measure_20261107');checks=[];warnings=[]
def check(name,ok,detail=None):checks.append({'check':name,'passed':bool(ok),'detail':detail})
for batch in [1,4,8,16]:
 p=R/f'inference/batch{batch}_raw.jsonl'
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 check(f'inference batch{batch}: 50 samples',len(rows)==50,len(rows));check(f'inference batch{batch}: positive/finite times and ten denoise steps',all(r['finite'] and r['prefix_ms']>0 and len(r['denoise_ms'])==10 and min(r['denoise_ms'])>0 for r in rows))
 err=max(abs(r['prefix_ms']+sum(r['denoise_ms'])-r['model_ms']) for r in rows);check(f'inference batch{batch}: CUDA partition sum',err<.02,err)
for tier in ['action','action_vision_lora']:
 p=R/f'training/{tier}/raw.jsonl'
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines()];check(tier+': 10 warmup + 50 measured',len(rows)==60 and sum(r['warmup'] for r in rows)==10,len(rows));check(tier+': memory reserved >= allocated',all(r['peak_reserved_bytes']>=r['peak_allocated_bytes']>0 for r in rows));check(tier+': positive phase times',all(min(r[k] for k in ['prefix_ms','action_ms','backward_ms','optimizer_ms'])>0 for r in rows));
 parity=json.load(open(R/f'training/{tier}/parity.json'));warnings.append({'configuration':tier,'issue':'native joint vs split arithmetic differs','gradient_relative_l2':parity['gradient_relative_l2'],'loss_abs_diff':parity['loss_abs_diff']})
for p in (R/'interference').glob('*/*/summary.json'):
 s=json.loads(p.read_text())
 if 'p99_ms' not in s:continue
 check(str(p.relative_to(R))+': 240 requests',s.get('requests',s.get('n_requests'))==240)
 check(str(p.relative_to(R))+': SLO bounds',0<=s['slo_rate']<=1)
 if p.parent.name.startswith('mps'):
  dev=json.load(open(p.parent/'train_device.json'));check(str(p.relative_to(R))+': capped SM visibility',dev['visible_sm_count']<84,dev)
result={'checks':checks,'warnings':warnings,'all_available_checks_pass':all(x['passed'] for x in checks),'scope':'Checks only completed/present artifacts; not proof every requested experiment is complete.'};(R/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
