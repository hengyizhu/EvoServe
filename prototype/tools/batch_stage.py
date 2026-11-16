"""Measured batch capacity gates; no M2/M3 launch before required repairs pass."""
import argparse,json,math,os,sys,time,traceback,subprocess,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.campaign import Campaign
from evoserve.io import read,save

class Stage(Campaign):
 def report(self):
  super().report();rows=[read(p) for p in sorted(self.states.glob('*.json'))]
  fields=['item','state','alignment','clients','equivalent_robots','rate','batch_sizes','batch_wait_ms','seconds','p99_ms','slo','steps_s','training_steps','mean_duty','raw','reason']
  with (self.root/'batch_capacity_and_co.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
  lines=['# 复位与动态 batching（持续更新）','225 W；250 ms 时限、SLO≥99%。GPU2/3外部负载持续记录。复位/空动作不发请求，正式执行仍为250 ms节拍。','', '|项目|状态|P99 ms|SLO|训练步/秒|','|---|---|---:|---:|---:|']
  for x in rows:
   if x['item'].endswith('_10min') or x['item'] in ['M1_long','M2','M2_long','M3']:
    lines.append('|'+ '|'.join(str(x.get(k,'')) for k in ['item','state','p99_ms','slo','steps_s'])+'|')
  calibration=next((x for x in rows if x['item']=='batch_calibration'),{})
  if calibration.get('batches'):
   lines+=['','|静态 batch|中位耗时 ms|P99 ms|前缀 ms|去噪 ms|','|---|---:|---:|---:|---:|']
   for x in calibration['batches']:lines.append('|'+ '|'.join(str(x[k]) for k in ['batch','median_ms','p99_ms','prefix_median_ms','denoise_median_ms'])+'|')
  cap=read(self.root/'batch_capacity.json',{})
  if cap:lines+=['','错峰最大已验证机器人数：'+str(cap['maximum_verified_robots'])+'；同步到达：'+str(cap['synchronized_maximum_verified_robots'])+'。每台4 Hz；70%/85%为等效总请求负载。']
  bad=[x['item']+': '+x.get('reason','') for x in rows if x['state'] in ['skipped','blocked','failed']]
  if bad:lines+=['','异常/未启动：'+'；'.join(bad)]
  lines+=['','原始CSV：batch_capacity_and_co.csv，以及各raw目录下 batch_latency_raw.csv / batch_latency.csv / feedback_windows.csv。训练吞吐用真实示范标签和服务前缀测量，不作为学习质量结果。']
  (self.root/'复位与batching汇总.md').write_text('\n'.join(lines)+'\n')
 def external(self,name,tool,config=None,extra=(),seconds=60):
  old=read(self.states/(name+'.json'))
  if old:return old
  for attempt in [1,2]:
   out=self.out/f'{name}_{attempt}_{time.time_ns()}'
   try:
    self.wait_idle([0,1]);self.status(name,attempt=attempt)
    cmd=[sys.executable,str(self.repo/'prototype/tools'/tool),'--out',str(out/'run')]
    if config is not None:
     path=out.with_suffix('.json');save(path,config);cmd+=['--config',str(path)]
    cmd+=list(extra)
    env=dict(self.env,CUDA_VISIBLE_DEVICES='0') if tool=='batch_calibrate.py' else self.env
    self.execute(cmd,out,env,seconds+1800)
    result=read(out/'run/summary.json');assert result and result['state'] in ['passed','complete'],result
    return self.record(name,'complete',raw=str(out/'run'),**{k:v for k,v in result.items() if k not in ['state','power_limit_w']})
   except Exception:
    error=traceback.format_exc();save(self.root/'failures'/f'{self.namespace}_{name}_{attempt}.json',dict(error=error,raw=str(out)))
  return self.record(name,'skipped',reason=error.splitlines()[-1],raw=str(out))
 def load(self,name,n,seconds,alignment='staggered',training=False,fraction=1.):
  cfg=self.setting('ours' if training else 'no_learning',seconds=seconds)
  cfg.update(operator_gpu=None,operators={},robot_experts=[],expert_ids=[3] if training else [3,5,9,1][:min(n,4)],export_cache=True,batch_sizes=self.sizes,batch_wait_ms=10,inputs=str(self.root/'batch_inputs.pkl'),clients=n,rate=4*n*fraction,alignment=alignment)
  return self.external(name,'batch_load.py',cfg,seconds=seconds)
 def run(self):
  calibration=self.external('batch_calibration','batch_calibrate.py',extra=['--assets',self.cfg['assets'],'--inputs',str(self.root/'batch_inputs.pkl'),'--operators',str(self.operator_manifest)],seconds=600)
  verified=calibration['state']=='complete';cap=calibration.get('selected_max_batch',1);self.sizes=[b for b in [1,2,4,8] if b<=cap]
  batches=calibration.get('batches',[dict(batch=1,median_ms=170)])
  peak=max(x['batch']*1000/x['median_ms'] for x in batches if x['batch']<=cap)
  upper=min(8,max(2,math.ceil(peak/4)+1));screens=[];formal=[]
  for alignment in ['staggered','synchronized']:
   for n in range(1,upper+1):
    result=self.load(f'batch_capacity_{alignment}_n{n}_screen',n,60,alignment)
    screens.append(dict(robots=n,alignment=alignment,passed=result.get('slo',0)>=.99,raw=result.get('raw'),slo=result.get('slo'),p99_ms=result.get('p99_ms')))
   eligible=[x['robots'] for x in screens if x['alignment']==alignment and x['passed']];best=max(eligible,default=0)
   # Long validation of best screen and the next larger offered load.
   candidates=sorted({max(1,best),min(upper,best+1)})
   for n in candidates:
    result=self.load(f'batch_capacity_{alignment}_n{n}_10min',n,600,alignment)
    formal.append(dict(robots=n,alignment=alignment,passed=result.get('slo',0)>=.99,raw=result.get('raw'),slo=result.get('slo'),p99_ms=result.get('p99_ms')))
   save(self.root/'batch_capacity_progress.json',dict(screens=screens,formal=formal))
  maximum=max([x['robots'] for x in formal if x['alignment']=='staggered' and x['passed']],default=0)
  synchronized=max([x['robots'] for x in formal if x['alignment']=='synchronized' and x['passed']],default=0)
  capacity=dict(state='complete',maximum_verified_robots=maximum,synchronized_maximum_verified_robots=synchronized,searched_robots=list(range(1,upper+1)),batch_sizes=self.sizes,batch_wait_ms=10,screens=screens,formal=formal,deadline_ms=250,slo_threshold=.99,power_limit_w=225,definition='4 Hz per robot; staggered capacity sets M2; synchronized capacity is supplemental; maximum verified in tested range')
  save(self.root/'batch_capacity.json',capacity)
  co=[]
  if maximum:
   for fraction in [.70,.85]:
    name=f'batch_co_u{round(fraction*100)}'
    smoke=self.load(name+'_smoke',maximum,60,training=True,fraction=fraction)
    if smoke['state']=='complete' and smoke.get('training_steps',0)>0:
     co.append(self.load(name+'_10min',maximum,600,training=True,fraction=fraction))
    else:co.append(self.record(name+'_10min','blocked',reason='short test failed or had no actual training update'))
  resets=all(read(self.states/(name+'.json'),{}).get('state')=='complete' for name in ['reset_inference_10min','reset_M1_10min'])
  ready=verified and resets and maximum>0 and len(co)==2 and all(x.get('slo',0)>=.99 for x in co)
  save(self.root/'batch_revision_result.json',dict(ready_to_resume=ready,reset_tests_passed=resets,batch_graph_validated=verified,capacity_robots=maximum,concurrent=co,power_limit_w=225))
  if not ready:
   for name in ['M1_long','M2','M3']:self.record(name,'blocked',reason='reset/batching/capacity/concurrent gates incomplete; thresholds unchanged')
   return
  self.resume(maximum)
 def resume(self,capacity):
  save(self.root/'milestone_hold.json',dict(M2='waiting_M1_hour',M3='waiting_M2',reset_and_batching_passed=True,time=time.time()))
  extra=dict(batch_sizes=self.sizes,batch_wait_ms=10)
  lowlight=[dict(expert=3,at_s=0,perturb=dict(kind='exposure',factor=.05))]
  short=self.trial('M1_batch_smoke',self.setting(changes=lowlight,**extra),lambda x:self.health(x,learning=True,slo=True))
  if short['state']!='complete':self.record('M1_long','blocked',reason='M1 batched smoke failed');return
  long=self.trial('M1_long',self.setting(seconds=3600,changes=lowlight,**extra),lambda x:self.health(x,learning=True,slo=True))
  save(self.root/'milestones/M1.json',dict(state='passed' if long['state']=='complete' and long.get('promotions') else 'not_demonstrated',raw=long.get('raw'),promotions=long.get('promotions',0),revision=self.namespace))
  if long['state']!='complete' or not long.get('promotions'):
   for name in ['M2','M3']:self.record(name,'blocked',reason='M1 did not demonstrate validated improvement');return
  n=min(capacity,4);active=[3,5,9,1][:n]
  m2cfg=self.setting(robots=active,**extra);m2cfg['expert_ids']=[3,5,9,1]
  m2cfg['inactive_expert_policy']='No robots: retain registered slot; train only if existing cache pool has samples; no synthetic/offline data injected.'
  save(self.root/'M2_capacity_configuration.json',m2cfg)
  m2=self.trial('M2_smoke',m2cfg,lambda x:self.health(x,learning=True,slo=True))
  if m2['state']!='complete':self.record('M3','blocked',reason='capacity-adjusted M2 smoke failed');return
  m2cfg['seconds']=3600;m2=self.trial('M2_long',m2cfg,lambda x:self.health(x,learning=True,slo=True))
  if m2['state']!='complete':self.record('M3','blocked',reason='capacity-adjusted M2 long run failed');return
  active=[3,5,9,1][:min(4,capacity*2)];operator=1
  for gpu in [2,3]:
   if not subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():operator=gpu;break
  save(self.root/'M3_operator_placement.json',dict(operator_gpu=operator,stream_priority=0,same_placement_for_all_methods=True))
  ours=False
  for method in ['ours','no_learning','recompute','split']:
   if method!='ours' and not ours:self.record('M3_'+method,'blocked',reason='ours failed');continue
   def config(seconds):
    changes=[dict(expert=e,at_s=seconds/2,perturb=dict(kind='exposure',factor=.05) if e==3 else dict(kind='camera_shift',fraction=.25,both_cameras=True)) for e in active[:len(active)//2]]
    c=self.setting(method,active,[0,1],seconds,changes,operator_gpu=operator,**extra);c['expert_ids']=[3,5,9,1];return c
   smoke=self.trial('M3_'+method+'_smoke',config(600),lambda x:self.health(x,learning=method!='no_learning',slo=True))
   if smoke['state']!='complete':continue
   result=self.trial('M3_'+method,config(10800))
   if method=='ours':ours=result['state']=='complete' and result.get('slo',0)>=.99

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);a=p.parse_args();stage=Stage(a.root,read(a.config))
 try:stage.run()
 finally:stage.report()
if __name__=='__main__':main()
