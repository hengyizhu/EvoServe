"""Emit wall-clock system metrics and the request population around each promotion."""
import json
from .io import append,save

def records(path):
    if not path.exists():return []
    with path.open() as f:return [json.loads(line) for line in f if line.endswith('\n')]

def record(run,hi):
    lo=hi-300
    controls=records(run.out/'controller.jsonl')
    host=[r for r in records(run.out/'host_gpu_monitor.jsonl') if lo<=r['monotonic']<hi]
    external={g: [c for r in host for c in r['cards'] if c['gpu']==g] for g in (2,3)}
    for gpu in run.plans:
        es=[e for e,g in run.expert_trainer.items() if g==gpu]
        steps=[r for r in records(run.out/f'train{gpu}/steps.jsonl') if lo<=r['finish_ns']/1e9<hi]
        cs=[r['before'] for r in controls if r['gpu']==gpu and lo<=run.origin+5*r['window']<hi]
        episodes=[r for r in run.episodes if r['expert'] in es and lo<=r['finished']<hi]
        frames=sum(r['steps'] for r in episodes)
        append(run.out/'system_5min.jsonl',dict(gpu=gpu,end_s=hi-run.origin,
            **run.ledger.window(lo,hi,hi+run.period,es),expert_steps_s=sum(len(r['experts']) for r in steps)/300,
            mean_merged_experts=sum(len(r['experts']) for r in steps)/len(steps) if steps else 0,
            mean_duty=sum(r['duty'] for r in cs)/len(cs) if cs else None,
            takeover_frame_fraction=sum(r['takeover_frames'] for r in episodes)/frames if frames else None,
            external_util_pct={g:sum(x['util_percent'] for x in v)/len(v) if v else None for g,v in external.items()}))
    save(run.out/'overall_slo.json',run.ledger.window(run.origin,hi,hi+run.period))
    switches=records(run.out/'activations.jsonl')
    sample=requests(run)
    save(run.out/'switch_latency.json',[dict(**r,window_complete=hi>=r['activation_time']+30,
        requests=[q for q in sample if q['expert']==r['expert'] and abs(q['release']-r['activation_time'])<=30]) for r in switches])

def requests(run):
    with run.ledger.lock:
        return [dict(request_id=i,**q,latency_ms=(q['finish']-q['release'])*1000 if q['finish'] is not None else None) for i,q in run.ledger.requests.items()]
