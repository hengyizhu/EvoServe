"""Fixed 250ms arrivals and 20Hz physics. No inference-dependent pacing."""
from pathlib import Path
import os,random,sys,time,traceback,queue,threading
from concurrent.futures import ThreadPoolExecutor
from .io import RPCClient,save,append,read
from .core import ActionWindowAssembler,PeriodicActions,TakeoverRule,window_data_source,request_key,cycle_timing_flags
from .simulation import AsyncRenderer

PERIOD=.250

def run(cfg):
    PERIOD=cfg.get("period",.25)
    import numpy as np
    repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'vendor/RLinf'))
    sys.path.insert(0,str(repo/'experiments/pi05_machine_b_20261115'))
    from perturbations import transform_image
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from rlinf.envs.sim.libero.utils import quat2axisangle
    out=Path(cfg['out']);out.mkdir(exist_ok=True,parents=True)
    suite=benchmark.get_benchmark_dict()['libero_10'](task_order_index=0);task=suite.get_task(cfg['task_id']);states=suite.get_task_init_states(cfg['task_id'])
    bddl=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file)
    env=ControlEnv(bddl_file_name=bddl,use_camera_obs=False,has_offscreen_renderer=False,hard_reset=False)
    renderer=AsyncRenderer(bddl,cfg.get('scene'));coord=RPCClient(cfg['coordinator']);pool=ThreadPoolExecutor(max_workers=4)
    index=0;next_release=None;requests_total=0;requests_sent=0;sim_misses=0;holds=0;last_report=time.monotonic()
    def raw_observation(view,perturb):
        obs=view['obs']
        raw={'observation/image':np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),'observation/wrist_image':np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),'observation/state':np.concatenate([obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]).astype(np.float32),'prompt':task.language}
        if perturb:
            for key in ['observation/image','observation/wrist_image']:raw[key]=transform_image(raw[key],perturb)
        return raw
    try:
        while True:
            episode=f"r{cfg['robot']}:e{index}";seed=cfg['seed']+cfg['robot']*100000+index
            reset_start=time.monotonic();np.random.seed(seed);env.seed(seed);env.reset();env.set_init_state(states[(index+cfg['robot']*19)%len(states)])
            for _ in range(10):env.step([0.,0.,0.,0.,0.,0.,-1.])
            renderer.submit(episode,0,env.get_sim_state().copy())
            while renderer.get(episode) is None:time.sleep(.002)
            # Refresh after renderer construction; startup is outside the clock.
            if index==0:
                fresh=time.monotonic();renderer.submit(episode,0,env.get_sim_state().copy())
                while renderer.get(episode)['captured']<fresh:time.sleep(.002)
            if index==0:
                save(out/'ready.json',dict(pid=os.getpid(),renderer='independent OSMesa state',request_period_s=PERIOD,physics_hz=20,returned_actions=10,normal_actions_per_cycle=5,cpu_affinity=sorted(os.sched_getaffinity(0))))
            reset_completed=time.monotonic()
            primed_origin=None
            while True:
                assignment=coord.call('begin_episode',expert=cfg['expert'],robot=cfg['robot'],episode=episode)
                if assignment.get('stop'):return
                if not assignment.get('wait'):break
                planned=assignment.get('origin')
                if planned is not None and primed_origin!=planned and time.monotonic()>=planned-.20:
                    renderer.submit(episode,0,env.get_sim_state().copy());primed_origin=planned
                time.sleep(.002)
            version=assignment['version'];production_version=assignment.get('production_version',version);origin=assignment['origin'];end=assignment['end']
            if next_release is None:next_release=origin+cfg['phase_s']
            # An episode owns its own fixed-rate clock. Reset, settling and
            # validation barriers are outside the request/SLO population.
            if assignment.get('state') is not None:
                seed=assignment['seed'];env.seed(seed);np.random.seed(seed);env.reset();env.set_init_state(states[assignment['state']%len(states)])
                for _ in range(10):env.step([0.,0.,0.,0.,0.,0.,-1.])
                fresh=time.monotonic();renderer.submit(episode,0,env.get_sim_state().copy())
                while renderer.get(episode)['captured']<fresh:time.sleep(.001)
            reset_ready=time.monotonic()
            next_release=max(reset_ready+.02,origin+cfg['phase_s'] if index==0 else 0.)
            append(out/'resets.jsonl',dict(episode=episode,reset_started=reset_start,ready=reset_ready,first_request=next_release,reset_completed=reset_completed,reset_seconds=reset_completed-reset_start,admission_wait_seconds=reset_ready-reset_completed,excluded_until=next_release,excluded_from_slo=True))
            assembler=ActionWindowAssembler(episode,version);actions=PeriodicActions();completed=queue.Queue();futures=[];seen_frames=set();reservoir=[];rng=random.Random(seed+211);eligible=0;success=False;stopped=False;cycle=0;episode_time=next_release;tick=0;physics_cpu=0.;episode_misses=0;episode_holds=0
            deviation=threading.Event();operator_active=False;takeover_event=None;action_sources=[]
            detector=TakeoverRule(cfg.get('stagnation_distance_m',.002),time_fraction=float('inf') if cfg.get('hg') else .7)
            detector.update(env.env._get_observations()['robot0_eef_pos'],0,cfg['action_budget']/20)
            def infer(cycle,release,rid,raw,frame,captured,operator_requested):
                client=RPCClient(cfg['server']);c=RPCClient(cfg['coordinator'])
                try:
                    dispatched=time.monotonic()
                    append(out/'dispatches.jsonl',dict(request_id=rid,release=release,dispatched=dispatched,expert=cfg['expert'],robot=cfg['robot']))
                    cache_version=production_version if operator_requested and not cfg.get('hg') else version
                    reply=client.call('infer',raw=raw,expert=cfg['expert'],version=cache_version,operator=operator_requested or cfg.get('hg',False),release=release,seed=seed*10000+cycle,episode=episode,frame=frame,request_id=rid)
                    finish=time.monotonic()
                    # Production SLO ends when production actions arrive, independently of the simulated human.
                    c.call('finish',request_id=rid,finish=finish)
                    selected=reply['operator_actions'] if operator_requested else reply['actions'];action_finished=finish
                    if (operator_requested or cfg.get('hg')) and reply.get('operator_ref'):
                        from .operators import wait_operator_result
                        operator_result=wait_operator_result(reply['operator_ref'],release+PERIOD)
                        action_finished=time.monotonic()
                        selected=operator_result.get('actions') if operator_result.get('status')=='complete' else None
                        append(out/'operator_responses.jsonl',dict(request_id=rid,status=operator_result.get('status'),production_finish=finish,action_received=action_finished,deadline=release+PERIOD,ontime=selected is not None and action_finished<=release+PERIOD))
                        if cfg.get('hg') and not operator_requested:
                            if selected is not None:
                                ctl=env.env.robots[0].controller
                                delta=ctl.scale_action(np.asarray(selected[0])[:ctl.control_dim])[:3]-ctl.scale_action(np.asarray(reply['actions'][0])[:ctl.control_dim])[:3]
                                if np.linalg.norm(delta)>.02:deviation.set()
                            selected=reply['actions'];action_finished=finish
                    elif operator_requested and selected is None:raise RuntimeError('operator action missing')
                    if selected is not None and action_finished<=release+PERIOD:actions.offer(cycle,action_finished,selected,'operator' if operator_requested else 'production')
                    completed.put((cycle,release,finish,frame,raw,reply['cache_ref'],cache_version))
                    append(out/'requests.jsonl',dict(request_id=rid,expert=cfg['expert'],version=cache_version,episode_version=version,operator_requested=operator_requested,release=release,finish=finish,latency_ms=(finish-release)*1000,observation_frame=frame,observation_age_ms=(dispatched-captured)*1000,observation_after_nominal_ms=(captured-release)*1000,sent=True))
                except Exception:completed.put(('error',traceback.format_exc()))
                finally:client.close();c.close()
            def accept_ready():
                nonlocal eligible
                for key,source,executed in assembler.collect_ready():
                    sources=action_sources[key.frame:key.frame+10];data_source=window_data_source(sources,operator_active)
                    if data_source is None:
                        coord.call('drop',expert=cfg['expert'],refs=[source['cache_ref']]);continue
                    eligible+=1;j=rng.randrange(eligible)
                    sample=dict(episode=episode,frame=key.frame,version=key.version,data_source=data_source,action_sources=sources,expert=cfg['expert'],raw=source['raw'],cache_ref=source['cache_ref'],actions=np.asarray(executed,dtype=np.float32),action_frames=list(range(key.frame,key.frame+10)))
                    discarded=None
                    if len(reservoir)<8:reservoir.append(sample)
                    elif j<8:discarded=reservoir[j];reservoir[j]=sample
                    else:discarded=sample
                    if discarded:coord.call('drop',expert=cfg['expert'],refs=[discarded['cache_ref']])
            def drain():
                while not completed.empty():
                    value=completed.get()
                    if value[0]=='error':raise RuntimeError(value[1])
                    _,release,finish,frame,raw,ref,cache_version=value
                    if frame in seen_frames or finish>release+PERIOD:
                        coord.call('drop',expert=cfg['expert'],refs=[ref]);continue
                    seen_frames.add(frame);assembler.observe(frame,dict(raw=raw,cache_ref=ref),version=cache_version);accept_ready()
            while len(assembler.actions)<cfg['action_budget'] and not success:
                due=episode_time+tick/20
                time.sleep(max(0,due-time.monotonic()))
                if time.monotonic()>=end:stopped=True;break
                drain()
                if due+1e-7>=next_release:
                    if cycle:actions.boundary(cycle-1,next_release,allowed_source='operator' if operator_active else 'production')
                    state=coord.call('environment',expert=cfg['expert'])
                    if state['stop']:stopped=True;break
                    rid=request_key(cfg['robot'],next_release);view=renderer.get(episode)
                    if cfg.get('hg'):
                        # Prefer a frame at this actual action boundary for aligned online labels.
                        frame=len(assembler.actions)
                        # The preceding physics tick already submitted this state.
                        until=next_release+PERIOD/3
                        while time.monotonic()<until and (view is None or view['frame']!=frame):
                            time.sleep(.0005);view=renderer.get(episode)
                    raw=raw_observation(view,state.get('perturb')) if view else None
                    lag=time.monotonic()-next_release
                    timing=cycle_timing_flags(lag,None if view is None else time.monotonic()-view['captured'])
                    behind=lag>=PERIOD;observation_stale=view is None or time.monotonic()-view['captured']>=PERIOD
                    episode_misses+=int(behind);sim_misses+=int(behind)
                    coord.call('arrive',request_id=rid,release=next_release,expert=cfg['expert'],robot=cfg['robot'],raw=raw,seed=seed*10000+cycle,operator=operator_active)
                    requests_total+=1;futures=[f for f in futures if not f.done()]
                    if raw is not None and len(futures)<8:
                        futures.append(pool.submit(infer,cycle,next_release,rid,raw,view['frame'],view['captured'],operator_active));requests_sent+=1
                    else:append(out/'requests.jsonl',dict(request_id=rid,release=next_release,finish=None,reason='simulator_unready' if raw is None else 'request_backlog_limit',sent=False))
                    append(out/'cycles.jsonl',dict(cycle=cycle,nominal_release=next_release,dispatch_lag_ms=lag*1000,simulation_overrun=behind,observation_stale=observation_stale,observation_age_ms=None if view is None else (time.monotonic()-view['captured'])*1000))
                    next_release+=PERIOD;cycle+=1
                action,hold=actions.next();action=np.asarray(action,dtype=np.float32);holds+=int(hold);episode_holds+=int(hold)
                t=time.thread_time();env.step(action.tolist());physics_cpu+=time.thread_time()-t;assembler.actions.append(action.copy());action_sources.append(actions.last_source);accept_ready()
                append(out/'executed_actions.jsonl',dict(episode=episode,frame=len(assembler.actions)-1,source=actions.last_source,action=action.tolist()))
                success=bool(env.check_success())
                if cfg.get('operator_enabled') and not operator_active:
                    trigger=detector.update(env.env._get_observations()['robot0_eef_pos'],time.monotonic()-episode_time,cfg['action_budget']/20,success)
                    if cfg.get('hg'):
                        if trigger and trigger['reason']=='time_budget_70pct':trigger=None
                        if deviation.is_set():trigger=dict(reason='deviation_2cm')
                    if trigger:
                        operator_active=True;actions.interrupt()
                        takeover_event=dict(**trigger,episode=episode,expert=cfg['expert'],robot=cfg['robot'],frame=len(assembler.actions),time=time.monotonic(),episode_version=version,production_version=production_version)
                        coord.call('takeover',**takeover_event);append(out/'takeovers.jsonl',takeover_event)
                        coord.call('drop',expert=cfg['expert'],refs=[x['cache_ref'] for x in reservoir]);reservoir.clear();eligible=0
                next_tick=episode_time+(tick+1)/20
                render_due=(next_tick+1e-7>=next_release) if cfg.get('hg') else (tick%5==1)
                if render_due:
                    renderer.submit(episode,len(assembler.actions),env.get_sim_state().copy())
                tick+=1
                if time.monotonic()-last_report>=5:
                    save(out/'status.json',dict(requests_total=requests_total,requests_sent=requests_sent,simulation_overrun_cycles=sim_misses,hold_ticks=holds,wall_elapsed_s=time.monotonic()-origin,actual_request_rate=requests_sent/max(1e-6,time.monotonic()-origin)))
                    last_report=time.monotonic()
            for f in futures:f.result()
            drain();coord.call('drop',expert=cfg['expert'],refs=[v['cache_ref'] for v in assembler.pending.values()])
            for sample in reservoir:
                if (sample['data_source']=='operator_correction' or (success and not operator_active)) and not stopped:coord.call('window',**sample,success=success)
                else:coord.call('drop',expert=cfg['expert'],refs=[sample['cache_ref']])
            if stopped:coord.call('abort_episode',expert=cfg['expert'],episode=episode);break
            row=dict(episode=episode,episode_index=index,expert=cfg['expert'],robot=cfg['robot'],version=version,success=success,autonomous_success=success and not operator_active,intervened=operator_active,takeover_frames=action_sources.count('operator'),takeover=takeover_event,steps=len(assembler.actions),seconds=time.monotonic()-episode_time,sim_cpu_seconds=physics_cpu,simulation_overrun_cycles=episode_misses,hold_ticks=episode_holds,seed=seed,epoch_at_start=assignment['epoch'],finished=time.monotonic(),reset_seconds=episode_time-reset_start)
            coord.call('end_episode',**row);append(out/'episodes.jsonl',row);index+=1
    finally:
        save(out/'final_counters.json',dict(requests_total=requests_total,requests_sent=requests_sent,simulation_overrun_cycles=sim_misses,hold_ticks=holds))
        pool.shutdown(wait=False,cancel_futures=True);renderer.close();env.close();coord.close()


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=read(a.config);os.sched_setaffinity(0,set(cfg['cpus']))
    try:run(cfg)
    except Exception:save(Path(cfg['out'])/'failure.json',dict(error=traceback.format_exc()));raise

if __name__=='__main__':main()
