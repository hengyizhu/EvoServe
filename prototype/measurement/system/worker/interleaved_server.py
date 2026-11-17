import os,sys,argparse,time,json,importlib.util,traceback,dataclasses,math,types
from pathlib import Path
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--mode',required=True);ap.add_argument('--inputs',required=True);args=ap.parse_args();OUT=Path(args.out)
if os.environ.get('PI_MPS')=='1':
 os.environ['PI_MPS_UUID']=os.environ['CUDA_VISIBLE_DEVICES'];spec=importlib.util.spec_from_file_location('core',R/'core_mps.py');mod=importlib.util.module_from_spec(spec);sys.modules['core']=mod;spec.loader.exec_module(mod)
sys.path.insert(0,'/home/hengyi/Workspace/01_VLA-Serving/Experiments/pi05_review_20261107')
from review_graph_serving import GraphServing as Parent,torch,np,save,stamp,GPU,stats,model_module
from packed import PackedObservation
from ipc import Export
os.sched_setaffinity(0,set(map(int,os.environ.get('SERVER_CPU_LIST',','.join(map(str,range(GPU*8,GPU*8+4)))).split(','))));torch.set_num_threads(int(os.environ.get('SERVER_TORCH_THREADS','4')))
class Serving(Parent):
 def __init__(self):
  raw=torch.load(args.inputs,weights_only=False,map_location='cpu');self.inputs=[(x['key'],x['raw']) for x in raw];self.packer=PackedObservation();super().__init__(1);torch.set_num_threads(int(os.environ.get('SERVER_TORCH_THREADS','4')))
 def source_index(self,i):return int(np.random.default_rng(20261109+i).integers(len(self.inputs)))
 def batch_input(self,i):
  if not getattr(self,'input_transform_installed',False):
   from perturbations import install
   install(self.model,json.loads(Path('/home/hengyi/Workspace/01_VLA-Serving/Experiments/pi05_extended_20261109/tasks/drawer_lowlight_0.json').read_text()));self.input_transform_installed=True
  from torch.utils._pytree import tree_map
  marks={};start=stamp();cpu=time.thread_time_ns()
  def mark(name):
   nonlocal start,cpu
   now=stamp();c=time.thread_time_ns();marks[name+'_wall_ms']=(now-start)/1e6;marks[name+'_cpu_ms']=(c-cpu)/1e6;start=now;cpu=c
  processed=self.model._input_transform_fn(dict(self.inputs[self.source_index(i)][1]));mark('transform')
  batch=tree_map(lambda x:torch.as_tensor(np.asarray(x).copy())[None],processed);batch.pop('actions',None);obs=model_module.Observation.from_dict(batch);obs=dataclasses.replace(obs,state=obs.state.float());mark('tensorize')
  obs=model_module.preprocess_observation(obs,train=False);mark('preprocess')
  obs=model_module._observation_to_dtype(obs,self.model.embed_dtype);mark('dtype')
  obs=self.packer.transfer(obs);mark('packed_transfer')
  generator=torch.Generator(device='cuda');mark('generator_construct')
  generator.manual_seed(71000+i);mark('generator_seed')
  noise=torch.randn((1,self.model.action_horizon,self.model.action_dim),device='cuda',generator=generator);mark('noise_launch')
  self.input_parts=marks;return obs,noise
 def prefix(self,obs):
  value=super().prefix(obs);self.cache_value=(value[1],value[2]);return value
 def infer(self,i):
  from gil_probe import sample
  gil_wait=sample();out,row=super().infer(i);row['gil_acquire_sample_ms']=gil_wait;row.update(self.packer.stats);row.update(self.input_parts);row['input_h2d_cuda_ms']=self.packer.h2d_start.elapsed_time(self.packer.h2d_end);return out,row

from arrival_plan import schedule
def wait_file(p):
 while not p.exists():time.sleep(.02)
 return json.loads(p.read_text())
def main():
 s=Serving();save(OUT/'graph_correctness.json',s.verify_graph());export=Export(OUT,s.cache_value);ident=0;source_counts={};prefix_checked=False
 for _ in range(10):s.infer(0)
 for tid in Path('/proc/self/task').iterdir():
  if os.environ.get('SINGLE_PROCESS')=='1' and int(tid.name)!=os.getpid():continue
  try:os.sched_setaffinity(int(tid.name),set(map(int,os.environ.get('SERVER_CPU_LIST',','.join(map(str,range(GPU*8,GPU*8+4)))).split(','))))
  except ProcessLookupError:pass
 save(OUT/'server_ready.json',{'ready':True,'affinity':sorted(os.sched_getaffinity(0)),'num_steps':s.model.num_steps,'stream_priority':s.stream.priority,'torch_threads':torch.get_num_threads(),'mps_pipe':os.environ.get('CUDA_MPS_PIPE_DIRECTORY'),'visible_sm_count':torch.cuda.get_device_properties(0).multi_processor_count})
 try:
  for index in range(200):
   cfg=wait_file(OUT/f'command_{index:03d}.json')
   if cfg['op']=='stop':break
   if cfg['op']=='profile':save(OUT/f'server_done_{index:03d}.json',{'complete':True});continue
   if cfg['op']=='activate':save(OUT/f'server_done_{index:03d}.json',{'complete':True,'next_id':ident});continue
   if cfg['op']=='calibrate':
    rows=[s.infer(i)[1] for i in range(cfg.get('samples',100))];save(OUT/'calibration.json',{k:stats([r[k] for r in rows]) for k in ['service_ms','model_ms','I_PREFIX_ms','I_ACTION_ms']});save(OUT/f'server_done_{index:03d}.json',{'complete':True});continue
   if cfg.get('reset_expert_states'):source_counts[cfg['variant']]=0
   mode=cfg['mode'];variant=cfg['variant'];source_counts.setdefault(variant,0);out=OUT/cfg['name'];out.mkdir(exist_ok=True);origin=cfg['origin_ns'];end=origin+round(cfg['duration_s']*1e9);plan=schedule(cfg);rows=[]
   publisher=None;arrivals=None
   if cfg.get('queue_guard'):
    from arrival_state import Arrivals
    arrivals=Arrivals(OUT,out,plan,origin)
   if os.environ.get('SLO_FEEDBACK')=='1' or os.environ.get('DYNAMIC_BANK')=='1':
    from deadline_feedback import Publisher
    publisher=Publisher(OUT,out,plan,origin,end,arrivals=arrivals)
   for req in plan:
    now=stamp()
    if now>end+5_000_000_000:break
    if now<req['nominal_release_ns']:time.sleep((req['nominal_release_ns']-now)/1e9)
    if arrivals:arrivals.before_start(req['request_id'])
    source_id=cfg.get('source_offset',0)+cfg.get('source_stride',1)*source_counts[variant];_,row=s.infer(source_id);row.update(req)
    if not prefix_checked and mode=='A2':
     prefix_checked=True
     from torch.utils._pytree import tree_map,tree_flatten
     with torch.inference_mode():
      _,em,ek=s.model.build_prefix_cache(s.static_obs)
     graph=tree_map(lambda x:x.detach().cpu().clone(),s.cache_value);eager=tree_map(lambda x:x.detach().cpu().clone(),(em,ek))
     checks=[{'shape':list(a.shape),'equal':torch.equal(a,b),'max_abs':float((a.float()-b.float()).abs().max())} for a,b in zip(tree_flatten(graph)[0],tree_flatten(eager)[0])]
     assert all(x['equal'] for x in checks),'graph prefix differs from eager service prefix'
     save(OUT/'server_prefix_check.json',{'graph_vs_eager':checks,'tf32':torch.backends.cuda.matmul.allow_tf32})
     torch.save({'graph':graph,'eager':eager,'obs':{f.name:tree_map(lambda x:x.cpu().clone() if torch.is_tensor(x) else x,getattr(s.static_obs,f.name)) for f in dataclasses.fields(s.static_obs)}},OUT/'service_reference.pt')
    if mode!='baseline':
     rec={'id':ident,'mode':mode,'variant':variant,'source':s.inputs[s.source_index(source_id)][0],'nominal_ns':req['nominal_release_ns'],'service_finish_ns':row['finish_ns']}
     if mode=='A2':rec=export.publish(ident,s.cache_value,rec);row.update(export_d2h_ms=rec['export_d2h_ms'],export_wall_ms=rec['export_wall_ms'],export_queue_wait_ms=rec['export_queue_wait_ms'],export_host_copy_ms=rec['export_host_copy_ms'],export_bytes=rec['export_bytes'])
     row['finish_ns']=stamp();row['service_ms']=(row['finish_ns']-row['start_ns'])/1e6;rec['available_ns']=row['finish_ns'];save(OUT/'ipc'/f'{ident:08d}.json',rec)
    ident+=1;source_counts[variant]+=1;rows.append(row)
    if arrivals:arrivals.complete()
    if publisher:publisher.add(row)
    with (out/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if publisher:publisher.join()
   if arrivals:arrivals.close()
   lat=[(x['finish_ns']-x['nominal_release_ns'])/1e6 for x in rows];passed=sum(x['finish_ns']<=x['deadline_ns'] for x in rows)
   # Graph CUDA events capture elapsed graph duration (including MPS interference), not SM utilization.
   busy=sum(max(0,min(end,x['start_ns']+round((x['input_wall_ms']+x['copy_submit_wall_ms']+x['model_ms'])*1e6))-max(origin,x['start_ns']+round((x['input_wall_ms']+x['copy_submit_wall_ms'])*1e6))) for x in rows)/1e9/cfg['duration_s']
   wallbusy=sum(max(0,min(end,x['finish_ns'])-max(origin,x['start_ns'])) for x in rows)/1e9/cfg['duration_s']
   save(out/'server_summary.json',{'requests':len(plan),'completed':len(rows),'unfinished':len(plan)-len(rows),'slo_successes':passed,'slo_rate':passed/len(plan),'p99_ms':float(np.percentile(lat,99)) if lat else None,'p99_is_completed_only':len(rows)<len(plan),'busy_fraction':busy,'service_wall_busy_fraction':wallbusy,'rate':cfg['rate'],'new_samples_in_window':sum(origin<=x['finish_ns']<end for x in rows) if mode!='baseline' else 0,'new_samples_per_s':sum(origin<=x['finish_ns']<end for x in rows)/cfg['duration_s'] if mode!='baseline' else 0,'clients':cfg.get('clients',4),'client_period_s':cfg.get('clients',4)/cfg.get('global_rate',cfg['rate']),'stagger_s':1/cfg.get('global_rate',cfg['rate']),'input_h2d_ms':stats([x['input_h2d_cuda_ms'] for x in rows]) if rows else None,'phases':{k:stats([x[k] for x in rows]) for k in ['I_PREFIX_ms','I_ACTION_ms','service_ms']} if rows else None})
   save(OUT/f'server_done_{index:03d}.json',{'complete':True,'next_id':ident})
  save(OUT/'server_status.json',{'state':'complete'})
 finally:
  if export:export.close()
if __name__=='__main__':
 try:main()
 except Exception:save(OUT/'server_status.json',{'state':'failed','traceback':traceback.format_exc()});raise
