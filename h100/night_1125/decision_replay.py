"""Replay the last 30 minutes of the deployed request ledger at wall-clock cadence."""
import csv
import json
from pathlib import Path
import sys
import time

from evoserve.core import DeadlineLedger

root, out = map(Path, sys.argv[1:3])
cfg = json.loads((root/'protocol.json').read_text())
origin = json.loads((root/'clock.json').read_text())['origin']
def records(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)

logged = list(records(root/'controller.jsonl'))
last = logged[-1]['window']
first = last - 359
ledger = DeadlineLedger()
finished = {r['request_id']: r['finish_ns']/1e9 for g in (0,1)
            for r in records(root/f'serve{g}/requests.jsonl')}
arrivals = sorted(records(root/'arrivals.jsonl'),key=lambda x:x['release'])
cursor = 0
def advance(now):
    global cursor
    while cursor<len(arrivals) and arrivals[cursor]['release']<=now:
        r=arrivals[cursor]
        key=r['request_id']
        ledger.arrive(key,r['release'],r['expert'],r['robot'])
        if key in finished:
            ledger.finish(key,finished[key])
        cursor+=1
advance(origin+first*5)
start=time.monotonic()
with out.open('w',buffering=1) as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['window','gpu','history_requests','ledger_plus_policy_us','policy_us','matches_recorded_command'])
    for row in logged:
        w=row['window']
        if w<first: continue
        due=start+(w-first+1)*5
        time.sleep(max(0,due-time.monotonic()))
        hi=origin+(w+1)*5
        advance(hi+.25)
        t=time.perf_counter_ns()
        stat=ledger.window(hi-5,hi,hi+.25,[row['gpu']])
        p=time.perf_counter_ns()
        load=stat['requests']/5*cfg['service_ms']/1000
        width=next((m for bound,m in cfg['safe_table'] if load<=bound),1)
        duty=row['before']['duty']
        if stat['timeout_rate'] is not None:
            duty=max(0,min(1,duty-5*(stat['timeout_rate']-.01)))
        after=dict(m=min(1,width),duty=duty)
        end=time.perf_counter_ns()
        writer.writerow([w,row['gpu'],cursor,(end-t)/1000,(end-p)/1000,after==row['after']])
