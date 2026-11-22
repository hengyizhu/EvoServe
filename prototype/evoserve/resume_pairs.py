"""Recover the first paired validation of a frozen smoke-test candidate."""
import hashlib
import time
from pathlib import Path

from .io import save
from .system_metrics import records


def restore(run, directory):
    source = Path(directory)
    events = records(source / 'events.jsonl')
    starts = [r for r in events if r['event'] == 'validation_started']
    episodes = records(source / 'episodes.jsonl')
    if len(starts) != len(run.scheduler.experts) or any(r['event'] == 'validation_finished' for r in events):
        raise ValueError('Expected one unfinished first validation for each smoke-test expert')
    manifest = []
    for i, change in enumerate(run.cfg['changes']):
        assert change['at_s'] == 0
        run.scheduler.change_environment(change['expert'])
        run.seen_change.add(i)
    for v in starts:
        eid = v['expert']
        e = run.scheduler.experts[eid]
        assert v['production'] == e.production and e.epoch == v['epoch']
        e.step = v['step']
        # Use the original production success rate for the validation trigger.
        before = [r for r in episodes if r['expert']==eid and r['finished']<v['started'] and r['version']==e.production]
        e.production_recent.extend(int(r['autonomous_success']) for r in before)
        run.scheduler.start_validation(eid, v['candidate'], time.monotonic())
        t = e.trial
        t['prediction'] = v['prediction']
        t['metric'] = 'takeover_fraction' if v['prediction']['production_success'] < .2 else 'autonomous'
        path = source / f"train{run.expert_trainer[eid]}" / (v['candidate'].replace(':','_')+'.pt')
        run.version_paths[v['candidate']] = str(path)
        run.service[run.expert_server[eid]].call('install', expert=eid, version=v['candidate'], path=str(path))
        for r in episodes:
            if r['expert'] != eid or r['finished']-r['seconds'] < v['started']:
                continue
            arm = 'candidate' if r['version']==v['candidate'] else 'production'
            assert r['version']==v[arm] and r['seed']==t['seed']+t['assigned'][arm]
            pair = t['assigned'][arm]
            assert pair<t['limit']
            t['pairs'].setdefault(pair,{})[arm] = dict(autonomous=int(r['autonomous_success']), takeover_fraction=r['takeover_frames']/r['steps'])
            t['assigned'][arm] += 1
            t[arm+'_outcomes'].append(int(r['autonomous_success']))
        with path.open('rb') as f:
            digest = hashlib.file_digest(f,'sha256').hexdigest()
        manifest.append(dict(expert=eid,candidate=v['candidate'],sha256=digest,assigned=t['assigned'].copy(),step=e.step,source=str(source)))
    save(run.out/'restored_pairs.json',manifest)
