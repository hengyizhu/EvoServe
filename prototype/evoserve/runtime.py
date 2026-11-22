"""Launch real LIBERO robots, graph servers and learning workers for one setting."""
from pathlib import Path
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import traceback
from .core import DeadlineLedger,ExpertState,LearningPrior,Scheduler,TwoKnobController
from .io import RPCClient,RPCServer,save,read,append,wait_file


def gpu_status():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,power.limit,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for line in raw.splitlines():
        i,u,p,m,v=[x.strip() for x in line.split(',')];rows.append(dict(gpu=int(i),uuid=u,power_limit_w=float(p),memory_mib=int(m),util_percent=int(v)))
    return dict(time=time.time(),monotonic=time.monotonic(),cards=rows,host_load=list(os.getloadavg()),cpu_pressure=Path('/proc/pressure/cpu').read_text())


class Run:
    def __init__(self,cfg,out):
        self.cfg=cfg;self.operator_gpu=cfg.get('operator_gpu');self.used_gpus=sorted(set(cfg['gpus']+([] if self.operator_gpu is None else [self.operator_gpu])));self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
        self.closed=False;self.children=[];self.mps=[];self.stop=threading.Event();self.origin=None;self.deadline=None
        self.period=cfg.get('request_period_s',.25);self.ledger=DeadlineLedger(self.period);self.episodes=[];self.takeovers=[];self.sample_sources=defaultdict(int);self.event_cursor=0;self.error=None
        self.lock=threading.RLock();self.production_paths={};self.candidate_counter=defaultdict(int)
        self.pending_activation={};self.service={};self.trainer={};self.service_out={};self.plans={};self.expert_server={};self.expert_trainer={}
        self.pool_sample_counter=defaultdict(int);self.seen_change=set();self.corpus={};self.corpus_count=defaultdict(int)
        self.learning=cfg['method']!='no_learning' and not cfg.get('capacity_only',False)
        self.export=cfg.get('export_cache',cfg['method'] not in ['recompute','no_learning'])
        prior=read(cfg['prior']);self.prior=LearningPrior(prior['points'],prior['default_amplitude'])
        robots=defaultdict(list)
        for i,e in enumerate(cfg['robot_experts']):robots[e].append(i)
        from .paired import PairedScheduler
        schedule=PairedScheduler if cfg.get('paired') else Scheduler
        self.scheduler=schedule([ExpertState(e,robots[e],f'{e}:v0') for e in cfg['expert_ids']],self.prior)
        self.sockets=Path('/tmp')/('ep_'+hashlib.sha256(str(self.out).encode()).hexdigest()[:12]);self.sockets.mkdir(mode=0o700)
        save(self.out/'protocol.json',cfg)
        before=gpu_status();save(self.out/'hardware_before.json',before)
        assert all(x['power_limit_w']==225 for x in before['cards'] if x['gpu'] in self.used_gpus)
        for gpu in self.used_gpus:
            active=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if active:raise RuntimeError(f'GPU{gpu} occupied: {active}; no process interrupted')
        self.monitor=threading.Thread(target=self.monitor_loop,daemon=True);self.monitor.start()

    def monitor_loop(self):
        while not self.stop.is_set():
            try:append(self.out/'host_gpu_monitor.jsonl',gpu_status())
            except Exception:append(self.out/'monitor_errors.jsonl',dict(error=traceback.format_exc()))
            self.stop.wait(2)

    def spawn(self,module,cfg,label,env):
        file=self.out/(label+'_config.json');save(file,cfg)
        with (self.out/(label+'.log')).open('w') as f:
            cmd=[sys.executable,'-u','-m',module,'--config',str(file)]
            if module.endswith('workers'):cmd+=['--kind',cfg.get('worker_kind','serve' if label.startswith('serve') else 'train')]
            p=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.children.append(p);save(self.out/'worker_pids.json',[p.pid for p in self.children]);return p

    def base_env(self):
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',JAX_PLATFORMS='cpu')
        for k in list(env):
            if k.startswith('CUDA_MPS_') or k in ['PI_MPS','EXPERT_COUNT','SLO_FEEDBACK','TRACE_CAPTURE']:env.pop(k,None)
        return env

    def gpu_env(self,gpu,co):
        env=self.base_env();env['CUDA_VISIBLE_DEVICES']=str(gpu)
        if co:
            directory=self.sockets/f'mps{gpu}';directory.mkdir();(directory/'pipe').mkdir();(directory/'log').mkdir()
            uuid=next(x['uuid'] for x in gpu_status()['cards'] if x['gpu']==gpu)
            env.update(CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(directory/'pipe'),CUDA_MPS_LOG_DIRECTORY=str(directory/'log'),CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='100')
            subprocess.run(['nvidia-cuda-mps-control','-d'],env=env,check=True,capture_output=True);self.mps.append(env)
        return env

    def start(self):
        cfg=self.cfg;split=cfg['method']=='split';gpu_experts=defaultdict(list);train_experts=defaultdict(list)
        for i,e in enumerate(cfg['expert_ids']):
            sg=cfg['gpus'][0] if split else cfg['gpus'][i%len(cfg['gpus'])]
            tg=cfg['gpus'][1] if split else sg
            gpu_experts[sg].append(e);self.expert_server[e]=sg;self.expert_trainer[e]=tg
            if self.learning or self.export:train_experts[tg].append(e)
        all_gpus=set(gpu_experts)|set(train_experts)|({self.operator_gpu} if self.operator_gpu is not None else set())
        environments={g:self.gpu_env(g,(g in gpu_experts and g in train_experts) or (g==self.operator_gpu and (g in gpu_experts or g in train_experts))) for g in all_gpus}
        if self.operator_gpu is not None:
            self.operator_out=self.out/f'operator{self.operator_gpu}';self.operator_out.mkdir()
        for g,experts in gpu_experts.items():
            out=self.out/f'serve{g}';out.mkdir();self.service_out[g]=out;address=self.sockets/f's{g}'
            c=dict(assets=cfg['assets'],out=str(out),socket=str(address),experts=experts,export_cache=self.export,cpus=list(range(g*8,g*8+4)))
            c.update(queue_guard=cfg.get('feedforward',False),period=self.period,batch_sizes=cfg.get('batch_sizes',[1]),batch_wait_ms=cfg.get('batch_wait_ms',0))
            c['operators']={str(e):cfg.get('operators',{})[str(e)] for e in experts if str(e) in cfg.get('operators',{})}
            if self.operator_gpu is not None:c['operator_results']=str(self.operator_out/'results')
            self.spawn('evoserve.workers',c,f'serve{g}',environments[g]);self.service[g]=RPCClient(address)
        for g,experts in train_experts.items():
            out=self.out/f'train{g}';out.mkdir();address=self.sockets/f't{g}';plan=self.out/f'train_plan{g}.json';self.plans[g]=plan
            save(plan,dict(experts=[],m=1,duty=0))
            c=dict(assets=cfg['assets'],out=str(out),socket=str(address),plan=str(plan),experts=experts,service_outs=[str(self.service_out[self.expert_server[e]]) for e in experts],cpus=list(range(g*8+4,g*8+8)),mode=cfg['method'],seed=cfg['seed'],pool_samples=512,pool_bytes=12*1024**3)
            c['queue_guard']=cfg.get('feedforward',False)
            c['service_outs']=list(dict.fromkeys(c['service_outs']))
            if not self.learning:c['worker_kind']='sink'
            self.spawn('evoserve.workers',c,f'train{g}',environments[g]);self.trainer[g]=RPCClient(address)
        if self.operator_gpu is not None:
            c=dict(assets=cfg['assets'],out=str(self.operator_out),gpu=self.operator_gpu,cpus=[28,29],operators=cfg['operators'],sources=[str(p/'operator_export') for p in self.service_out.values()])
            self.spawn('evoserve.operator_worker',c,'operator'+str(self.operator_gpu),environments[self.operator_gpu])
        self.controllers={g:TwoKnobController(len(es)) for g,es in train_experts.items()}
        for g in gpu_experts:wait_file(self.out/f'serve{g}/ready.json',self.children)
        for g in train_experts:wait_file(self.out/f'train{g}/ready.json',self.children)
        if self.operator_gpu is not None:wait_file(self.operator_out/'ready.json',self.children)
        self.rpc=RPCServer(self.sockets/'coord',self.handle)
        for robot,e in enumerate([] if cfg.get('replay_source') else cfg['robot_experts']):
            out=self.out/f'robot{robot}';out.mkdir()
            c=dict(out=str(out),coordinator=str(self.sockets/'coord'),server=str(self.sockets/f's{self.expert_server[e]}'),expert=e,task_id=cfg.get('operators',{}).get(str(e),{}).get('task_id',e),robot=robot,seed=cfg['seed'],control_hz=cfg.get('control_hz',20),action_budget=cfg['action_budget'],cpus=[16+2*robot,17+2*robot],hg=cfg.get('hg',False),scene=next((x['perturb'] for x in cfg['changes'] if x['expert']==e and x['at_s']==0),None),period=self.period,phase_s=self.period*robot/len(cfg['robot_experts']))
            c['operator_enabled']=str(e) in cfg.get('operators',{});c['stagnation_distance_m']=cfg.get('stagnation_distance_m',.002)
            env=self.base_env();env.update(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='osmesa',PYOPENGL_PLATFORM='osmesa',LIBERO_CONFIG_PATH=str(Path(cfg['assets'])/'libero_config'),LP_NUM_THREADS='1',GALLIUM_DRIVER='llvmpipe',LD_LIBRARY_PATH=cfg['osmesa_lib']+':'+env.get('LD_LIBRARY_PATH',''))
            self.spawn('evoserve.robot',c,f'robot{robot}',env)
        for robot in range(0 if cfg.get('replay_source') else len(cfg['robot_experts'])):wait_file(self.out/f'robot{robot}/ready.json',self.children)
        if cfg.get("resume_pairs"):
            from .resume_pairs import restore
            restore(self,cfg["resume_pairs"])
        self.origin=time.monotonic()+1;self.deadline=self.origin+cfg['seconds']
        save(self.out/'clock.json',dict(origin=self.origin,end=self.deadline,wall_time=time.time(),seconds=cfg['seconds']))
        if cfg.get('replay_source'):
            from .replay import ArrivalReplay
            self.replay=ArrivalReplay(self,cfg['replay_source']);self.replay.start()

    def handle(self,msg):
        op=msg['op'];e=msg.get('expert')
        if op=='begin_episode':
            with self.lock:
                if self.stop.is_set() or (self.deadline and time.monotonic()>=self.deadline):return dict(stop=True)
                if self.origin is None or time.monotonic()<self.origin:return dict(wait=True,origin=self.origin)
                version=self.scheduler.begin_episode(e,msg['robot'],msg['episode'])
                pair=self.scheduler.experts[e].active_episodes.get(msg['episode'],{})
                if version and self.pending_activation.get(e)==version:
                    append(self.out/'activations.jsonl',dict(expert=e,version=version,episode=msg['episode'],robot=msg['robot'],activation_time=time.monotonic()))
                    self.pending_activation.pop(e)
                    if self.cfg.get("resume_pairs"):(self.out/"stop_requested").touch()
                return dict(wait=version is None,state=pair.get('state'),seed=pair.get('seed'),version=version,production_version=self.scheduler.experts[e].production,epoch=self.scheduler.experts[e].epoch,origin=self.origin,end=self.deadline)
        if op=='environment':
            return dict(stop=self.stop.is_set() or time.monotonic()>=self.deadline,epoch=self.scheduler.experts[e].epoch,perturb=self.perturbation(e))
        if op=='arrive':
            self.ledger.arrive(msg['request_id'],msg['release'],e,msg['robot'])
            # Small fixed corpus for matched-arrival replay without simulator CPU.
            if msg.get('raw') is not None and self.corpus_count[e]<16:
                self.corpus[(e,self.corpus_count[e])]=msg['raw'];self.corpus_count[e]+=1
            append(self.out/'arrivals.jsonl',dict(request_id=msg['request_id'],release=msg['release'],expert=e,robot=msg['robot'],seed=msg.get('seed'),operator=msg.get('operator',False),corpus_index=max(0,self.corpus_count[e]-1)))
            return True
        if op=='finish':self.ledger.finish(msg['request_id'],msg['finish']);return True
        if op=='window':
            if not self.export and not self.learning:return dict(accepted=False)
            result=self.trainer[self.expert_trainer[e]].call('window',**{k:v for k,v in msg.items() if k!='op'})
            self.pool_sample_counter[e]+=int(result['accepted']);self.sample_sources[f'{e}:'+msg.get('data_source','self_success')]+=int(result['accepted'])
            append(self.out/'training_windows.jsonl',dict(expert=e,episode=msg['episode'],observation_frame=msg['frame'],action_frames=msg['action_frames'],action_sources=msg.get('action_sources'),data_source=msg.get('data_source','self_success'),cache_version=msg['version'],episode_success=msg['success'],accepted=result['accepted']))
            return result
        if op=='drop':
            if self.export:return self.trainer[self.expert_trainer[e]].call('drop',refs=msg['refs'])
            return True
        if op=='takeover':
            with self.lock:
                row={k:v for k,v in msg.items() if k!='op'};self.takeovers.append(row);append(self.out/'takeovers.jsonl',row)
            return True
        if op=='abort_episode':
            with self.lock:self.scheduler.experts[e].active_episodes.pop(msg['episode'],None)
            return True
        if op=='end_episode':
            with self.lock:
                event=(self.scheduler.end_scored_episode(e,msg['episode'],msg['autonomous_success'],msg['takeover_frames'],msg['steps']) if hasattr(self.scheduler,'end_scored_episode') else self.scheduler.end_episode(e,msg['episode'],msg.get('autonomous_success',msg['success'])));row={k:v for k,v in msg.items() if k!='op'};self.episodes.append(row);append(self.out/'episodes.jsonl',row)
                if event:self.apply_decision(event)
            return True
        raise ValueError(op)

    def perturbation(self,expert):
        elapsed=time.monotonic()-(self.origin or time.monotonic())
        selected=None
        for change in self.cfg['changes']:
            if change['expert']==expert and elapsed>=change['at_s']:selected=change['perturb']
        return selected

    def apply_decision(self,event):
        e=event['expert'];state=self.scheduler.experts[e]
        if event['decision']=='promote':
            self.pending_activation[e]=state.production
            self.production_paths[e]=self.version_paths[event['candidate_version']]
            self.service[self.expert_server[e]].call('remove',expert=e,version=event['old_version'])
            self.trainer[self.expert_trainer[e]].call('resume',expert=e,version=state.production)
        else:
            self.service[self.expert_server[e]].call('remove',expert=e,version=event['candidate_version'])
            self.trainer[self.expert_trainer[e]].call('resume',expert=e,version=state.production,discard=True,production_path=self.production_paths.get(e))
            state.production_step=state.step
        append(self.out/'validation_results.jsonl',dict(event,decision_time=time.monotonic()))
        if self.cfg.get('resume_pairs') and not self.pending_activation and all(s.trial is None for s in self.scheduler.experts.values()):
            (self.out/'stop_requested').touch()

    def refresh_training(self):
        if not self.learning or self.cfg.get("resume_pairs"):return
        for gpu,client in self.trainer.items():
            st=client.call('status')
            if st['error']:raise RuntimeError(st['error'])
            for eid,steps in st['expert_steps'].items():self.scheduler.experts[int(eid)].step=steps

    def decide(self):
        if not self.learning or self.cfg.get("resume_pairs"):return
        with self.lock:
            for eid in self.scheduler.experts:
                if not self.scheduler.should_validate(eid):continue
                self.candidate_counter[eid]+=1;version=f'{eid}:c{self.candidate_counter[eid]}'
                snapshot=self.trainer[self.expert_trainer[eid]].call('snapshot',expert=eid,version=version)
                self.scheduler.experts[eid].step=snapshot['step'];self.version_paths[version]=snapshot['path']
                self.service[self.expert_server[eid]].call('install',expert=eid,version=version,path=snapshot['path'])
                self.scheduler.start_validation(eid,version,time.monotonic())
            self.ranking=self.scheduler.ranked()
            append(self.out/'scheduler.jsonl',dict(time=time.monotonic(),ranking=self.ranking,predictions={str(e):self.scheduler.prediction(s) for e,s in self.scheduler.experts.items()},priorities={str(e):self.scheduler.priority(s) for e,s in self.scheduler.experts.items()}))
        self.write_plans()

    def write_plans(self):
        for gpu,plan in self.plans.items():
            c=self.controllers[gpu]
            eligible=[] if self.cfg.get('resume_pairs') else [e for e in self.ranking if self.expert_trainer[e]==gpu and not self.scheduler.experts[e].trial]
            save(plan,dict(experts=eligible,m=c.m,duty=c.duty if self.learning else 0,time=time.monotonic()))

    def run(self):
        self.version_paths={};self.ranking=list(self.cfg['expert_ids']);self.start();window=0;next_decision=self.origin;next_status=self.origin
        try:
            while time.monotonic()<self.deadline:
                if (self.out/'stop_requested').exists():
                    self.deadline=time.monotonic()
                    self.cfg['seconds']=max(1,math.ceil(self.deadline-self.origin))
                    break
                for p in self.children:
                    if p.poll() is not None:raise RuntimeError(f'worker {p.pid} exited {p.returncode}')
                now=time.monotonic()
                if hasattr(self,'replay') and self.replay.errors:raise RuntimeError(self.replay.errors[-1])
                for i,change in enumerate(self.cfg['changes']):
                    if i not in self.seen_change and now>=self.origin+change['at_s']:
                        with self.lock:self.scheduler.change_environment(change['expert']);self.seen_change.add(i)
                        append(self.out/'environment_changes.jsonl',dict(**change,actual_elapsed_s=now-self.origin))
                if now>=next_status:
                    self.refresh_training();next_status=now+1
                    save(self.out/'status.json',dict(state='running',elapsed_s=now-self.origin,episodes=len(self.episodes),successes=sum(x['success'] for x in self.episodes),expert_steps={str(e):s.step for e,s in self.scheduler.experts.items()},accepted_windows=dict(self.pool_sample_counter)))
                if now>=next_decision:self.decide();next_decision+=30
                while now>=self.origin+(window+1)*5+self.period:
                    lo=self.origin+window*5;hi=lo+5
                    for gpu,controller in self.controllers.items():
                        es=[e for e in self.cfg['expert_ids'] if self.expert_trainer[e]==gpu]
                        stat=self.ledger.window(lo,hi,now,es)
                        if self.cfg.get('feedforward'):
                            rate=stat['requests']/5
                            load=rate*self.cfg['service_ms']/1000
                            safe=next((m for bound,m in self.cfg['safe_table'] if load<=bound),1)
                            before=dict(m=controller.m,duty=controller.duty)
                            controller.m=min(len(es),safe)
                            if stat['timeout_rate'] is not None:
                                controller.duty=max(0,min(1,controller.duty-5*(stat['timeout_rate']-.01)))
                            change=dict(before=before,after=dict(m=controller.m,duty=controller.duty),measured_load=load)
                        else:change=controller.update(stat)
                        append(self.out/'controller.jsonl',dict(window=window,gpu=gpu,**stat,**change))
                    append(self.out/'slo_windows.jsonl',dict(window=window,**self.ledger.window(lo,hi,now)))
                    self.write_plans();window+=1
                    if window%60==0 and self.cfg.get('system_metrics'):
                        from .system_metrics import record
                        record(self,hi)
                if len(self.scheduler.events)>self.event_cursor:
                    with self.lock:
                        for event in self.scheduler.events[self.event_cursor:]:append(self.out/'events.jsonl',event)
                        self.event_cursor=len(self.scheduler.events)
                time.sleep(.01)
            # No new requests; allow the final deadline to mature.
            time.sleep(self.period+.05)
            if self.cfg.get('system_metrics'):
                from .system_metrics import record
                last_recorded=self.origin+300*(window//60)
                if self.deadline>last_recorded:
                    record(self,self.deadline,lo=last_recorded)
            self.refresh_training()
            return self.summarize()
        finally:self.close()

    def summarize(self):
        stats=self.ledger.window(self.origin,self.deadline,time.monotonic())
        with self.lock:episodes=[dict(x) for x in self.episodes if self.origin<=x['finished']<self.deadline]
        per_expert={str(e):dict(episodes=sum(x['expert']==e for x in episodes),successes=sum(x['expert']==e and x['success'] for x in episodes),autonomous_successes=sum(x['expert']==e and x.get('autonomous_success',x['success']) for x in episodes),takeovers=sum(x['expert']==e for x in self.takeovers),takeovers_per_hour=sum(x['expert']==e for x in self.takeovers)*3600/self.cfg['seconds'],production_version=s.production,training_steps=s.step) for e,s in self.scheduler.experts.items()}
        validations=[x for x in self.scheduler.events if x['event']=='validation_finished']
        robot_stats={}
        for i in range(len(self.cfg['robot_experts'])):
            path=self.out/f'robot{i}'
            rows=[json.loads(x) for x in (path/'requests.jsonl').read_text().splitlines()] if (path/'requests.jsonl').exists() else []
            cycles=[json.loads(x) for x in (path/'cycles.jsonl').read_text().splitlines()] if (path/'cycles.jsonl').exists() else []
            dispatches=[json.loads(x) for x in (path/'dispatches.jsonl').read_text().splitlines()] if (path/'dispatches.jsonl').exists() else []
            sent=sum(self.origin<=x['release']<self.deadline and x['dispatched']<self.deadline for x in dispatches)
            # Nominal periods absent during reset/barriers remain deadline misses.
            robot_stats[str(i)]=dict(sent_requests=sent,actual_request_rate=sent/self.cfg['seconds'],simulation_overrun_cycles=sum(x['simulation_overrun'] for x in cycles),stale_observation_cycles=sum(x.get('observation_stale',False) for x in cycles),request_period_s=self.period)
            reset_path=self.out/f'robot{i}/resets.jsonl'
            resets=[json.loads(x) for x in reset_path.read_text().splitlines()] if reset_path.exists() else []
            reset_seconds=sum(max(0,min(x.get('reset_completed',x['ready']),self.deadline)-max(x['reset_started'],self.origin)) for x in resets)
            excluded_seconds=sum(max(0,min(x.get('excluded_until',x['first_request']),self.deadline)-max(x['reset_started'],self.origin)) for x in resets)
            robot_stats[str(i)].update(reset_seconds=reset_seconds,reset_and_admission_excluded_seconds=excluded_seconds,resets=len(resets),reset_excluded_from_slo=True,active_request_rate=sent/max(.001,self.cfg['seconds']-excluded_seconds))
        simulation_overruns=sum(x['simulation_overrun_cycles'] for x in robot_stats.values())
        summary=dict(state='complete',accepted_windows_by_source=dict(self.sample_sources),autonomous_successes=sum(x.get('autonomous_success',x['success']) for x in episodes),autonomous_successful_tasks_per_hour=sum(x.get('autonomous_success',x['success']) for x in episodes)*3600/self.cfg['seconds'],validation_success_definition='completed successfully without any operator takeover',operator_placement=dict(gpu=self.operator_gpu,serving_gpus=self.cfg['gpus'],shares_experiment_gpu=self.operator_gpu in self.cfg['gpus'] if self.operator_gpu is not None else True,production_does_not_wait_for_operator=self.operator_gpu is not None),replay=bool(self.cfg.get('replay_source')),per_robot=robot_stats,simulation_overrun_cycles=simulation_overruns,method=self.cfg['method'],seconds=self.cfg['seconds'],robots=len(self.cfg['robot_experts']),episodes=len(episodes),successes=sum(x['success'] for x in episodes),successful_tasks_per_hour=sum(x['success'] for x in episodes)*3600/self.cfg['seconds'],expert_results=per_expert,promotions=sum(x['decision']=='promote' for x in validations),validations=validations,accepted_cache_windows=dict(self.pool_sample_counter),power_limits_w={x['gpu']:x['power_limit_w'] for x in gpu_status()['cards']},nominal_request_rate=stats['requests']/self.cfg['seconds'],actual_request_rate=(self.replay.dispatched if hasattr(self,'replay') else sum(x['sent_requests'] for x in robot_stats.values()))/self.cfg['seconds'],**stats)
        curves=[]
        for e in self.cfg['expert_ids']:
            for begin in range(0,self.cfg['seconds'],300):
                end=min(self.cfg['seconds'],begin+300);subset=[x for x in episodes if x['expert']==e and self.origin+begin<=x['finished']<self.origin+end]
                curves.append(dict(expert=e,begin_s=begin,end_s=end,episodes=len(subset),successes=sum(x['success'] for x in subset),autonomous_successes=sum(x.get('autonomous_success',x['success']) for x in subset),autonomous_success_rate=sum(x.get('autonomous_success',x['success']) for x in subset)/len(subset) if subset else None,takeover_episodes=sum(x.get('intervened',False) for x in subset),power_limit_w=225))
        with (self.out/'expert_success_curves.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
        summary['hourly']=[dict(hour=i//3600+1,completed_tasks=sum(self.origin+i<=x['finished']<self.origin+min(i+3600,self.cfg['seconds']) for x in episodes),successful_tasks=sum(self.origin+i<=x['finished']<self.origin+min(i+3600,self.cfg['seconds']) and x['success'] for x in episodes)) for i in range(0,self.cfg['seconds'],3600)]
        operator_rows=[]
        if hasattr(self,'operator_out') and (self.operator_out/'requests.jsonl').exists():operator_rows=[json.loads(x) for x in (self.operator_out/'requests.jsonl').read_text().splitlines()]
        serving_rows=[]
        for source in self.service_out.values():
            if (source/'requests.jsonl').exists():serving_rows.extend(json.loads(x) for x in (source/'requests.jsonl').read_text().splitlines())
        operator_hourly=[]
        for e in self.cfg['expert_ids']:
            for begin in range(0,self.cfg['seconds'],3600):
                end=min(self.cfg['seconds'],begin+3600);count=sum(x['expert']==e and self.origin+begin<=x['time']<self.origin+end for x in self.takeovers)
                jobs=[x for x in operator_rows if x['expert']==e and self.origin+begin<=x['deadline']-self.period<self.origin+end]
                exports=[x for x in serving_rows if x['expert']==e and self.origin+begin<=x['finish_ns']/1e9<self.origin+end and x.get('operator_requested')]
                operator_hourly.append(dict(operator_export_d2h_seconds=sum(x.get('operator_export_d2h_ms',0) for x in exports)/1000,operator_export_host_seconds=sum(x.get('operator_export_host_copy_ms',0) for x in exports)/1000,expert=e,begin_s=begin,end_s=end,takeovers=count,takeovers_per_hour=count*3600/(end-begin),operator_gpu=self.operator_gpu,operator_requests=len(jobs),operator_gpu_seconds=sum(x.get('operator_gpu_ms',0)+x.get('operator_h2d_ms',0) for x in jobs)/1000,operator_h2d_seconds=sum(x.get('operator_h2d_ms',0) for x in jobs)/1000,operator_late_or_dropped=sum(x['status']!='complete' or x['finished']>x['deadline'] for x in jobs)))
        with (self.out/'operator_hourly.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(operator_hourly[0]));w.writeheader();w.writerows(operator_hourly)
        summary['operator_hourly']=operator_hourly
        save(self.out/'summary.json',summary)
        with (self.out/'replay_corpus.pkl').open('wb') as f:pickle.dump(self.corpus,f)
        return summary

    def close(self):
        if self.closed:return
        self.closed=True
        self.stop.set()
        for p in self.children:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for p in self.children:
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        for env in self.mps:
            subprocess.run(['nvidia-cuda-mps-control'],env=env,input='quit\n',text=True,capture_output=True,timeout=20)
        for path in self.service_out.values():
            for layout in path.rglob('ipc_layout.pkl'):Path(pickle.loads(layout.read_bytes())[0]).unlink(missing_ok=True)
        if hasattr(self,'rpc'):self.rpc.close()
        self.monitor.join(timeout=5)
        save(self.out/'hardware_after.json',gpu_status())


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=read(a.config);os.sched_setaffinity(0,{30,31});run=None
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('terminated')))
    try:run=Run(cfg,a.out);run.run()
    except BaseException:
        save(Path(a.out)/'failure.json',dict(error=traceback.format_exc()))
        if run:run.close()
        raise


if __name__=='__main__':main()
