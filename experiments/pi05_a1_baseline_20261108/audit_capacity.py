"""CPU-only audit; preserves the historical sample order and uses measured KV bytes."""
import json, hashlib, collections
from pathlib import Path
R=Path(__file__).resolve().parent
source=Path('/home/hengyi/pi05_cache_20261107/learning/b_seed20261107/sequence.json')
layout=Path('/home/hengyi/pi05_cache_20261107/cache_cost/kv_layout.json')
sequence=json.loads(source.read_text())[:1000]
assert len(sequence)==1000 and all(len(x)==8 for x in sequence)
flat=[tuple(x) for step in sequence for x in step]
measured=json.loads(layout.read_text()); size=measured['bytes_per_sample']+measured['mask_bytes']
seen=set(); steps=[]; hits=0
for i,step in enumerate(sequence):
 h=0
 for item in step:
  key=tuple(item)
  if key in seen:h+=1
  else:seen.add(key)
 hits+=h
 steps.append({'step':i+1,'hits':h,'misses':8-h,'cumulative_hits':hits,'resident_samples':len(seen),'resident_kv_and_mask_bytes':len(seen)*size})
curves=[]
for capacity in [512,1024,1536,1792,2048]:
 cache=collections.OrderedDict();h=0
 for key in flat:
  if key in cache:h+=1;cache.move_to_end(key)
  else:
   cache[key]=None
   if len(cache)>capacity:cache.popitem(last=False)
 curves.append({'capacity_samples':capacity,'kv_and_mask_gib':capacity*size/2**30,'lru_hit_rate':h/len(flat),'note':'CPU simulation only, not a measured GPU result; excludes model and training memory'})
audit={'state':'blocked_by_requested_device_capacity','sequence_source':str(source),'sequence_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'layout_source':str(layout),'steps':1000,'samples':len(flat),'unique_samples':len(seen),'k':len(flat)/len(seen),'a1_no_eviction_hits':hits,'a1_no_eviction_hit_rate':hits/len(flat),'bytes_per_prefix_kv':measured['bytes_per_sample'],'bytes_per_mask':measured['mask_bytes'],'full_kv_gib':len(seen)*measured['bytes_per_sample']/2**30,'full_kv_and_mask_gib':len(seen)*size/2**30,'device_capacity_nominal_gib':48,'model_optimizer_activations_included':False,'lru_alternatives_simulated':curves,'gpu_work_started':False,'learning_jobs_untouched':True,'allowed_gpus':[0,1],'pending_user_choice':'host cache, bounded device cache, or larger GPU'}
(R/'capacity_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
(R/'a1_cold_cache_trace.json').write_text(json.dumps(steps))
(R/'sequence.json').write_text(json.dumps(sequence))
print(json.dumps(audit,ensure_ascii=False,indent=2))
