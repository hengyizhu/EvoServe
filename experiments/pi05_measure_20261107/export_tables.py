import pathlib,json,csv
R=pathlib.Path('/home/hengyi/pi05_measure_20261107')
def write(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (R/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
rows=[]
for tier in ['action','action_vision_lora']:
 p=R/'training'/tier/'summary.json'
 if not p.exists():continue
 d=json.loads(p.read_text());cfg=json.loads(p.with_name('config.json').read_text());row={'configuration':tier,'gpu':cfg['physical_gpu'],'microbatch':1,'accumulation':8,'lora_rank':cfg['lora_rank'],'trainable_parameters':sum(x['numel'] for x in cfg['updated_parameters'])}
 row.update({k+'_median':v['p50'] for k,v in d.items()});row.update({k+'_max':d[k]['max'] for k in ['peak_allocated_bytes','peak_reserved_bytes']});rows.append(row)
write('experiment0_training.csv',rows)
rows=[]
for d in json.load(open(R/'inference/summary.json')):
 row={'batch':d['batch'],'gpu':3,'denoise_steps':10}
 for k in ['prefix_ms','denoise_total_ms','end_to_end_ms','model_ms']:row[k+'_median']=d[k]['p50']
 for k in ['prefix_flops','one_denoise_flops','prefix_tflops','denoise_tflops']:row[k]=d[k]
 row.update({f'denoise_{j+1}_median_ms':v for j,v in enumerate(d['denoise_each_median_ms'])});row['cycle_gap_ms']=250-d['end_to_end_ms']['p50'];row['signed_gap_fraction']=row['cycle_gap_ms']/250;rows.append(row)
write('experiment1_inference.csv',rows)
rows=[]
for p in (R/'interference').glob('*/*/summary.json'):
 d=json.loads(p.read_text());row={'configuration':p.parent.parent.name,'mode':p.parent.name};row.update({k:d.get(k) for k in ['p99_ms','slo_rate','steps_per_s']});row['requests']=d.get('requests',d.get('n_requests'));row['gpu']=d.get('gpu',d.get('gpu_uuid'));row['duration_s']=120;row['client_period_s']=1.;row['phase_offset_s']=.5;row['slo_s']=1.;row['repeats']=1;rows.append(row)
write('experiment2_interference.csv',rows)
