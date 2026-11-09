import pathlib,json,csv,numpy as np
R=pathlib.Path('/home/hengyi/pi05_optimized_20261107');checks=[]
def check(name,passed,detail=None):checks.append({'check':name,'passed':bool(passed),'detail':detail})
for batch in [1,4,8,16]:
 p=R/f'inference/batch{batch}_raw.jsonl'
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines()];eq=json.load(open(p.with_name(f'batch{batch}_correctness.json')));check(f'graph batch{batch} 100 finite samples',len(rows)==100 and all(x['finite'] for x in rows));check(f'graph batch{batch} exact original eager outputs',len(eq)==4 and all(x['raw_action_max_abs']==0 for x in eq));err=max(abs(x['I_PREFIX_ms']+sum(x['denoise_ms'])-x['model_ms']) for x in rows);check(f'graph batch{batch} phase sum',err<.02,err)
for tier in ['action','action_vision_lora']:
 p=R/'training_mb4'/tier/'raw.jsonl'
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines()];check(tier+' training records',len(rows)==60 and sum(x['warmup'] for x in rows)==10);check(tier+' positive phases/memory',all(min(x[k] for k in ['prefix_ms','action_ms','backward_ms','optimizer_ms'])>0 and x['peak_reserved_bytes']>=x['peak_allocated_bytes']>0 for x in rows))
 n=json.load(open(p.with_name('microbatch_parity.json')));check(tier+' microbatch numerical difference preserved',n['gradient_relative_l2']>0,n)
for p in sorted(R.glob('matrix*/*/*/*/summary.json')):
 s=json.loads(p.read_text());folder=p.parent;rows=[json.loads(x) for x in (folder/'requests.jsonl').read_text().splitlines()];good=[x for x in rows if x.get('finish_ns') is not None];lat=[(x['finish_ns']-x['nominal_release_ns'])/1e6 for x in good];p99=float(np.percentile(lat,99)) if lat else None;slo=sum(x['finish_ns']<=x['deadline_ns'] for x in good)/len(rows);expected=int(30/.25)*s['clients']
 ok=len(rows)==expected and len({x['request_id'] for x in rows})==expected and len(good)==s['completed'] and abs(slo-s['slo_rate'])<1e-12 and (p99==s['p99_completed_ms'] or abs(p99-s['p99_completed_ms'])<1e-9)
 if len(good)<len(rows):ok=ok and s['p99_all_ms'] is None
 else:ok=ok and abs(s['p99_all_ms']-p99)<1e-9
 cfg=next(json.load(open(x)) for x in folder.parent.glob('command_*.json') if json.load(open(x)).get('name')==folder.name);a=cfg['origin_ns'];b=a+30_000_000_000;within=[x for x in good if x['start_ns']>=a and x['finish_ns']<=b];ok=ok and len(within)==s['phase_samples_within_window']
 for k in ['I_PREFIX_ms','I_ACTION_ms']:
  if within:ok=ok and abs(float(np.percentile([x[k] for x in within],50))-s['phases'][k]['p50'])<1e-9
 updates=0
 if (folder/'segments.csv').exists():
  seg=list(csv.DictReader(open(folder/'segments.csv')));updates=sum(int(x['optimizer_completed']) for x in seg if int(x['cpu_submit_ns'])>=a and int(x['gpu_done_ns'])<=b)
 ok=ok and abs(updates/30-s['steps_per_s'])<1e-12
 check(str(p.relative_to(R)),ok,{'requests':expected,'unfinished':len(rows)-len(good),'complete_updates':updates})
result={'all_available_checks_pass':all(x['passed'] for x in checks),'conditions_checked':sum('summary.json' in x['check'] for x in checks),'checks':checks,'scope':'Only completed artifacts checked; not a statement that all experiments are complete.'};(R/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
