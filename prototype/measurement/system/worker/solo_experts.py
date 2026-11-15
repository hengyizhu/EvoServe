import os,time,json,sys
import interleaved_train as it
from cache_engine import torch,obs_map,save,append,OUT
from experts import install
tr=it.Streaming('A2');tr.total_steps=4000;tr.current_parts=[];tr.parity();bank=install(tr,it)
# Prepare cached prefixes for the independence check before the streaming pool resets.
for k in set(tr.micros[0]):
 with torch.no_grad():
  _,mask,kv=tr.model.build_prefix_cache(obs_map(lambda x:x.cuda(),tr.data[k][0]));tr.store(k,(mask,kv))
bank.validate();tr.init_stream()
for i,source in enumerate(tr.new_sources[:512]):
 key=(*source,i);tr.data[key]=tr.data[source];tr.recent[key]=dict(source=source,key=key,sample_count=0)
 with torch.no_grad():
  _,mask,kv=tr.model.build_prefix_cache(obs_map(lambda x:x.cuda(),tr.data[key][0]));tr.store(key,(mask,kv))
tr.copy.synchronize();torch.cuda.reset_peak_memory_stats()
if os.environ.get('ONE_STEP')=='1':
 row=tr.next_step();append(OUT/'steps.jsonl',row)
 from pathlib import Path
 parent=json.loads(Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020/audit/parent_download.json').read_text())
 torch.save(dict(parameters={n:e[0].detach().to(dtype=p.dtype).cpu() for (n,p),e in zip(tr.pairs,bank.ema)},family='lora32_heads',parent=parent,weights_kind='EMA',eval_task_id=3,step=1),OUT/'ema_1.pt');save(OUT/'result.json',dict(state='complete',one_step=True,experts=bank.n,seconds=row['seconds']));sys.exit(0)
for _ in range(5):tr.next_step()
duration=120;start=time.monotonic_ns();end=start+int(duration*1e9);rows=[]
while time.monotonic_ns()<end:
 row=tr.next_step();append(OUT/'steps.jsonl',row)
 if start<=row['start_ns'] and row['finish_ns']<=end:rows.append(row)
save(OUT/'result.json',dict(state='complete',mode='solo',experts=bank.n,seconds=duration,group_steps_per_s=len(rows)/duration,expert_steps_per_s=bank.n*len(rows)/duration,mean_step_s=sum(x['seconds'] for x in rows)/len(rows),peak_cuda_bytes=torch.cuda.max_memory_allocated(),effective_batch_per_expert=8))
