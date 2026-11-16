"""Resumable measured milestones; failed gates are reported, never invented."""
from pathlib import Path
import argparse,csv,fcntl,json,os,signal,subprocess,sys,time,traceback
from .io import read,save

class Campaign:
    def __init__(self,root,cfg):
        self.root=Path(root);self.cfg=cfg;self.repo=Path(__file__).resolve().parents[2]
        self.out=self.root/('prototype_runs'+('_'+cfg['namespace'] if cfg.get('namespace') else ''));self.out.mkdir(exist_ok=True)
        self.namespace=cfg.get('namespace','');self.states=self.root/('prototype_task_state'+('_'+self.namespace if self.namespace else ''));self.states.mkdir(exist_ok=True)
        self.graph_available=False;self.capacity_path=self.root/('capacity'+('_'+cfg['namespace'] if cfg.get('namespace') else '')+'.json')
        self.operator_manifest=Path(cfg.get('operator_manifest',self.repo/'prototype/configs/operators.machine_a.json'))
        self.operators=read(self.operator_manifest)['operators']
        self.env=dict(os.environ,PYTHONPATH=str(self.repo/'prototype'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        for key in list(self.env):
            if key.startswith('CUDA_MPS_') or key in ['CUDA_VISIBLE_DEVICES','EXPERT_COUNT','SLO_FEEDBACK','PI_MPS']:self.env.pop(key,None)
    def report(self):
        rows=[read(p) for p in sorted(self.states.glob('*.json'))]
        fields=['item','state','reason','method','robots','seconds','p99_ms','slo','successes','successful_tasks_per_hour','autonomous_successful_tasks_per_hour','promotions','actual_request_rate','simulation_overrun_cycles','power_limit_w','raw']
        with (self.root/'prototype_summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
        lines=['# 原型 v0（持续更新）','功耗上限225 W；时限250 ms、SLO≥99%；GPU2/3外部负载见各项监控。每台机器人固定250 ms周期，仿真迟到不延长周期。',
               '|项目|状态|P99 ms|SLO|完成任务/h（含接管）|自主成功/h|上线次数|仿真超期周期|','|---|---|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            lines.append('|'+ '|'.join(str(r.get(k,'')) for k in ['item','state','p99_ms','slo','successful_tasks_per_hour','autonomous_successful_tasks_per_hour','promotions','simulation_overrun_cycles'])+'|')
        bad=[r['item']+': '+r.get('reason','') for r in rows if r['state'] not in ['complete','passed']]
        lines+=['','异常/未完成：'+'；'.join(bad),'容量、每机器人请求率及验证明细：prototype_summary.csv、capacity.json、各项summary.json / validation_results.jsonl。']
        (self.root/'原型汇总.md').write_text('\n'.join(lines)+'\n')
        if (self.root/'report.py').exists():subprocess.run([sys.executable,str(self.root/'report.py')],timeout=60,check=True)
    def record(self,name,state,**kw):
        row=dict(item=name,state=state,power_limit_w=225,updated=time.time(),**kw)
        save(self.states/(name+'.json'),row);self.report();return row
    def status(self,name,**kw):save(self.root/'status.json',dict(state='prototype_running',item=name,time=time.time(),**kw))
    def wait_idle(self,gpus):
        while True:
            busy={g:subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in gpus}
            if not any(busy.values()):
                time.sleep(30)
                again={g:subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip() for g in gpus}
                if not any(again.values()):return
            self.status('waiting_for_own_gpus',busy=busy);time.sleep(30)
    def execute(self,cmd,out,env=None,timeout=1800):
        out.mkdir(parents=True,exist_ok=True)
        with (out/'process.log').open('w') as f:
            p=subprocess.Popen(cmd,env=env or self.env,cwd=self.repo,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        save(self.root/'prototype_active.json',dict(pid=p.pid,cmd=cmd,out=str(out),time=time.time()))
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise TimeoutError(str(cmd))
        if rc:raise RuntimeError(f'exit {rc}: {out}/process.log')
    def preflight(self):
        name='gpu_preflight';old=read(self.states/(name+'.json'))
        if old:
            self.graph_available=old.get('graph_available',old['state']=='passed')
            return old['state']=='passed'
        for attempt in [1,2]:
            out=self.out/f'{name}_{attempt}_{time.time_ns()}'
            try:
                self.wait_idle([0,1]);self.status(name,attempt=attempt)
                env=dict(self.env,CUDA_VISIBLE_DEVICES='0')
                self.execute([sys.executable,str(self.repo/'prototype/tools/gpu_preflight.py'),'--assets',self.cfg['assets'],'--operators',str(self.operator_manifest),'--out',str(out/'data')],out,env,900)
                result=read(out/'data/summary.json');assert result['state']=='passed'
                if self.cfg.get('operator_gpu') is not None:
                    self.execute([sys.executable,str(self.repo/'prototype/tools/remote_operator_preflight.py'),'--assets',self.cfg['assets'],'--operators',str(self.operator_manifest),'--out',str(out/'remote')],out/'remote_process',env,900)
                    assert read(out/'remote/summary.json')['state']=='passed'
                self.record(name,'passed',raw=str(out));return True
            except Exception:
                self.graph_available=self.graph_available or read(out/'data/graph_summary.json',{}).get('state')=='passed'
                save(self.root/'failures'/f'prototype_{name}_{attempt}.json',dict(error=traceback.format_exc(),raw=str(out)))
        self.record(name,'skipped',reason='GPU preflight failed twice; dependent GPU milestones not started',graph_available=self.graph_available,raw=str(out));return False
    def setting(self,method='ours',robots=None,gpus=None,seconds=600,changes=None,**extra):
        robots=robots or [3]
        if self.cfg.get('operator_gpu') is not None:extra.setdefault('operator_gpu',self.cfg['operator_gpu'])
        return dict(assets=self.cfg['assets'],prior=str(self.repo/'prototype/configs/learning_prior.json'),osmesa_lib=self.cfg['osmesa_lib'],gpus=gpus or [0],expert_ids=list(dict.fromkeys(robots)),robot_experts=robots,method=method,seconds=seconds,seed=self.cfg['seed'],action_budget=600,changes=changes or [],request_period_s=.25,operators={str(e):self.operators[str(e)] for e in set(robots)},stagnation_distance_m=.002,**extra)
    def trial(self,name,cfg,gate=None):
        old=read(self.states/(name+'.json'))
        if old:return old
        for attempt in [1,2]:
            out=self.out/f'{name}_{attempt}_{time.time_ns()}'
            try:
                self.wait_idle(sorted(set([0,1]+([] if cfg.get('operator_gpu') is None else [cfg['operator_gpu']]))));self.status(name,attempt=attempt)
                config=out.with_suffix('.json');save(config,cfg)
                self.execute([sys.executable,'-u','-m','evoserve.runtime','--config',str(config),'--out',str(out/'run')],out,timeout=cfg['seconds']+1200)
                result=read(out/'run/summary.json');assert result['state']=='complete'
                reason=gate(result) if gate else None
                if reason:raise ValueError('measurement gate: '+reason)
                return self.record(name,'complete',**{k:v for k,v in result.items() if k!='state'},raw=str(out/'run'))
            except Exception:
                error=traceback.format_exc();save(self.root/'failures'/f'prototype_{name}_{attempt}.json',dict(error=error,raw=str(out)))
        result=read(out/'run/summary.json',{})
        return self.record(name,'skipped',reason=error.splitlines()[-1],**{k:v for k,v in result.items() if k not in ['state','reason']},raw=str(out/'run'))
    @staticmethod
    def health(result,learning=False,slo=False):
        if not result.get('requests'):return 'no inference requests'
        if not result.get('episodes') and not result.get('replay'):return 'no completed real simulator episode'
        if result.get('simulation_overrun_cycles',0):return 'simulator missed fixed-period cycles; reduce robot count'
        if learning and not sum(x['training_steps'] for x in result['expert_results'].values()):return 'no actual training update from successful serving episodes'
        if slo and (result.get('slo') or 0)<.99:return 'SLO below 99%'
    def capacity(self):
        existing=read(self.capacity_path)
        if existing:return existing
        rows=[]
        # Screen all N=1..6 at fixed 4 requests/s/robot, then verify largest
        # passing screen and the next count for ten minutes with matched replay.
        for n in range(1,7):
            live=self.trial(f'capacity_screen_n{n}',self.setting('no_learning',[3]*n,seconds=60))
            if live['state']!='complete':rows.append(dict(robots=n,passed=False,reason=live.get('reason')));continue
            replay=self.trial(f'capacity_replay_screen_n{n}',self.setting('no_learning',[3]*n,seconds=60,replay_source=live['raw']))
            p=live.get('p99_ms');b=replay.get('p99_ms')
            passed=(live.get('slo') or 0)>=.99 and (replay.get('slo') or 0)>=.99 and not live.get('simulation_overrun_cycles') and p is not None and b is not None and p<=1.05*b
            rows.append(dict(robots=n,stage='screen',passed=passed,live=live['raw'],replay=replay['raw'],p99_ms=p,baseline_p99_ms=b,slo=live.get('slo'),simulation_overrun_cycles=live.get('simulation_overrun_cycles')))
            save(self.root/'capacity_progress.json',rows)
        candidates=[r['robots'] for r in rows if r['passed']];best=max(candidates,default=0)
        verified=[]
        for n in sorted(set([best,min(6,best+1)])-{0}):
            live=self.trial(f'capacity_formal_n{n}',self.setting('no_learning',[3]*n,seconds=600))
            if live['state']!='complete':continue
            replay=self.trial(f'capacity_replay_formal_n{n}',self.setting('no_learning',[3]*n,seconds=600,replay_source=live['raw']))
            p=live.get('p99_ms');b=replay.get('p99_ms')
            passed=(live.get('slo') or 0)>=.99 and (replay.get('slo') or 0)>=.99 and not live.get('simulation_overrun_cycles') and p is not None and b is not None and p<=1.05*b
            verified.append(dict(robots=n,passed=passed,live=live['raw'],replay=replay['raw'],p99_ms=p,baseline_p99_ms=b,slo=live.get('slo')))
        result=dict(screens=rows,verification=verified,largest_verified_robots=max([r['robots'] for r in verified if r['passed']],default=0),searched_robots=list(range(1,7)),request_rate_per_robot=4,p99_ratio_limit=1.05,slo_threshold=.99,power_limit_w=225)
        save(self.capacity_path,result);return result
    def run(self):
        assert read(self.root/'measurements_complete.json') is not None,'Measurements must finish before prototype GPU work'
        if not self.preflight():
            if self.graph_available:self.capacity()
            for name in ['M1','M2','M3']:self.record(name,'blocked',reason='GPU training preflight failed; dependent learning milestone not started')
            return
        lowlight=[dict(expert=3,at_s=0,perturb=dict(kind='exposure',factor=.05))]
        m1=self.trial('M1_smoke',self.setting(changes=lowlight),lambda x:self.health(x,learning=True,slo=True))
        if m1['state']=='complete':
            long=self.trial('M1_long',self.setting(seconds=3600,changes=lowlight),lambda x:self.health(x,learning=True,slo=True))
            verified=long.get('promotions',0)>0 and long['state']=='complete'
            save(self.root/'milestones/M1.json',dict(state='passed' if verified else 'not_demonstrated',promotions=long.get('promotions',0),raw=long.get('raw'),seconds=3600,autonomous_successes=long.get('autonomous_successes'),operator_hourly=long.get('operator_hourly'),reason=None if verified else 'one-hour autonomous improvement not demonstrated; operator-assisted successes are not counted as policy gains'))
        else:save(self.root/'milestones/M1.json',dict(state='not_demonstrated',reason=m1.get('reason')))
        capacity=self.capacity()
        # M2 remains an explicit real short test even when capacity predicts failure.
        robots=[3,5,9,1]
        m2=self.trial('M2_smoke',self.setting(robots=robots),lambda x:self.health(x,learning=True,slo=True))
        if m2['state']=='complete':
            m2=self.trial('M2_long',self.setting(robots=robots,seconds=3600),lambda x:self.health(x,learning=True,slo=True))
        save(self.root/'milestones/M2.json',dict(state='passed' if m2['state']=='complete' else 'not_demonstrated',raw=m2.get('raw'),reason=m2.get('reason'),capacity_robots=capacity['largest_verified_robots']))
        self.report()
        if read(self.root/'milestones/M1.json')['state']!='passed' or m2['state']!='complete':
            self.record('M3','blocked',reason='M1/M2 prerequisites not demonstrated; no 12-hour run started');return
        # Fix auxiliary placement once for all four M3 conditions, without evicting anyone.
        operator_choice=None
        if self.cfg.get('operator_gpu') is not None:
            placement_path=self.root/'M3_operator_placement.json';placement=read(placement_path)
            if placement:operator_choice=placement['operator_gpu']
            else:
                operator_choice=1
                for g in [2,3]:
                    active=subprocess.check_output(['nvidia-smi',f'--id={g}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                    if not active:operator_choice=g;break
                save(placement_path,dict(operator_gpu=operator_choice,stream_priority=0,same_placement_for_all_methods=True,separate_accounting=['takeovers','operator_gpu_seconds','operator_h2d_seconds']))
        ours_ok=False
        for method in ['ours','no_learning','recompute','split']:
            if method!='ours' and not ours_ok:
                self.record('M3_'+method,'blocked',reason='ours did not pass; baseline long run not started');continue
            def changes(at):return [dict(expert=3,at_s=at,perturb=dict(kind='exposure',factor=.05)),dict(expert=5,at_s=at,perturb=dict(kind='camera_shift',fraction=.25,both_cameras=True))]
            smoke=self.trial('M3_'+method+'_smoke',self.setting(method,robots,[0,1],600,changes(300),operator_gpu=operator_choice),lambda x:self.health(x,learning=method!='no_learning',slo=True))
            if smoke['state']!='complete':continue
            long=self.trial('M3_'+method,self.setting(method,robots,[0,1],10800,changes(5400),operator_gpu=operator_choice))
            if method=='ours':ours_ok=long['state']=='complete' and (long.get('slo') or 0)>=.99
            save(self.root/'milestones'/('M3_'+method+'.json'),dict(state=long['state'],raw=long.get('raw'),seconds=10800,slo=long.get('slo')))
            self.report()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--config',required=True);a=p.parse_args();root=Path(a.root)
    with (root/'prototype_campaign.lock').open('w') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);campaign=Campaign(root,read(a.config))
        try:campaign.run()
        except Exception:
            campaign.record('campaign','failed',reason=traceback.format_exc());raise
        finally:campaign.report()
