"""Export recorded CUDA kernels without collapsing actual process/stream identities."""
from pathlib import Path
import sqlite3,json,csv,argparse,collections
p=argparse.ArgumentParser();p.add_argument('--sqlite',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.out);db=sqlite3.connect(a.sqlite);db.row_factory=sqlite3.Row
names={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")};assert 'CUPTI_ACTIVITY_KIND_KERNEL' in names,'No recorded CUDA kernels'
strings={r[0]:r[1] for r in db.execute('SELECT id,value FROM StringIds')};markers=[]
for x in db.execute('SELECT * FROM NVTX_EVENTS'):
 x=dict(x);text=x.get('text') or strings.get(x.get('textId'),'')
 if text=='evoserve_trace_5s' and x.get('end') is not None:markers.append(x)
assert markers,'5second NVTX capture marker missing';marker=max(markers,key=lambda x:x['end']-x['start']);lo,hi=marker['start'],marker['end'];duration=(hi-lo)/1e9;assert 4.9<=duration<=5.5,('capture duration',duration)
pids=json.loads((run/'worker_pids.json').read_text());assert len(pids)==2,pids;roles={pids[0]:'training',pids[1]:'inference'};result=json.loads((run/'result.json').read_text());records=[]
for x in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start < ? AND end > ? ORDER BY start',(hi,lo)):
 x=dict(x);global_pid=x.get('globalPid');pid=((global_pid>>24)&0x00FFFFFF) if global_pid is not None else x.get('processId');name=strings.get(x.get('demangledName')) or strings.get(x.get('shortName')) or strings.get(x.get('mangledName')) or x.get('name');assert name,'Missing kernel name'
 records.append(dict(start_ns=x['start'],end_ns=x['end'],duration_ns=x['end']-x['start'],kernel_name=name,pid=pid,process_role=roles.get(pid,'unmapped'),global_pid=global_pid,device_id=x.get('deviceId'),context_id=x.get('contextId'),stream_id=x.get('streamId'),correlation_id=x.get('correlationId'),start_ms_from_capture=(x['start']-lo)/1e6,end_ms_from_capture=(x['end']-lo)/1e6,power_limit_w=result['power_limits_before_w']['0']))
assert records,'Empty kernel capture'
with out.open('w') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
seen=collections.Counter(x['process_role'] for x in records);streams=collections.Counter((x['pid'],x['stream_id']) for x in records)
meta=dict(kernels=len(records),processes=len({x['pid'] for x in records}),role_kernel_counts=dict(seen),streams=[dict(pid=p,stream_id=s,kernels=n) for (p,s),n in streams.items()],capture_start_ns=lo,capture_end_ns=hi,capture_seconds=duration,power_limits_w=result['power_limits_before_w'],power_limits_during_w=result['power_limits_during_w'],raw_sqlite=a.sqlite,raw_run=a.run,timing_note='Nsight common timestamp domain; actual context/stream IDs retained. CUDA graph internal streams, if present, are not merged.',exported_full_overlapping_kernels=True)
meta['state']='passed' if seen['training'] and seen['inference'] and not seen['unmapped'] else 'failed_process_mapping';out.with_suffix('.json').write_text(json.dumps(meta,indent=2));assert meta['state']=='passed',meta
print(json.dumps(meta,indent=2))
