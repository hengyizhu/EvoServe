"""Attribute time from a Kineto timeline without double-counting overlapping GPU intervals."""
import bisect
import collections
import json
import gzip


def intervals(events, key):
    return [(e['ts'], e['ts']+e['dur']) for e in events if key(e)]


def analyze(path, compact_path=None):
    with open(path,encoding='utf8') as fp:
        trace=json.load(fp)
    events=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('dur',0)>0]
    annotation=lambda e:e.get('cat')=='user_annotation'
    windows=sorted(intervals(events,lambda e:annotation(e) and e['name']=='evo.step'))
    if not windows:
        raise RuntimeError('missing host step markers')
    phases={'forward','backward','optimizer'}
    scopes=sorted((e['ts'],e['ts']+e['dur'],e['name'][4:]) for e in events
                  if annotation(e) and e['name'].startswith('evo.') and e['name'][4:] in phases)
    starts=[s[0] for s in scopes]
    cache_scopes=sorted(intervals(events,lambda e:annotation(e) and e['name']=='evo.cache_transfer'))
    cache_starts=[a for a,_ in cache_scopes]
    runtime={}
    external={}
    for e in events:
        args=e.get('args',{})
        if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in args:
            runtime[args['correlation']]=e['ts']
        if e.get('cat')=='cpu_op' and 'External id' in args:
            external[args['External id']]=e['ts']
    spans=[]
    for e in events:
        cat=e.get('cat','')
        if cat not in ('kernel','gpu_memcpy','gpu_memset'):
            continue
        args=e.get('args',{})
        label='other_gpu'
        launch=runtime.get(args.get('correlation'),external.get(args.get('External id')))
        if cat=='gpu_memcpy' and ('htod' in e['name'].lower() or 'h2d' in e['name'].lower()):
            idx=bisect.bisect_right(cache_starts,launch)-1 if launch is not None else -1
            label='kv_h2d' if idx>=0 and launch<=cache_scopes[idx][1] else 'h2d'
        else:
            if launch is not None:
                j=bisect.bisect_right(starts,launch)-1
                if j>=0 and launch<=scopes[j][1]:
                    label=scopes[j][2]
        spans.append((e['ts'],e['ts']+e['dur'],label))
    spans.extend((a,b,'cache_lookup') for a,b in intervals(events,lambda e:annotation(e) and e['name']=='evo.cache_lookup'))
    if compact_path is not None:
        with gzip.open(compact_path,'wt',encoding='utf8',compresslevel=3) as fp:
            json.dump(dict(windows=windows,host_phase_ranges=scopes,attributed_intervals=spans),fp,separators=(',',':'))
    # CUDA work takes precedence over host lookup; all disjoint buckets sum to the step wall.
    # The independent H2D/compute intersection is reported even when assigned only once.
    priority=['kv_h2d','h2d','backward','forward','optimizer','other_gpu','cache_lookup']
    rows=[]
    for lo,hi in windows:
        changes=[(lo,None,0),(hi,None,0)]
        for a,b,k in spans:
            if b>lo and a<hi:
                changes.extend([(max(a,lo),k,1),(min(b,hi),k,-1)])
        changes.sort(key=lambda item:item[0])
        active=collections.Counter()
        totals=collections.Counter()
        overlap=0.
        last=lo
        for t,k,delta in changes:
            elapsed=t-last
            label=next((p for p in priority if active[p]),'other_wall')
            totals[label]+=elapsed/1000
            if (active['h2d'] or active['kv_h2d']) and any(active[x] for x in phases):
                overlap+=elapsed/1000
            if k:
                active[k]+=delta
            last=t
        row={k:totals[k] for k in priority+['other_wall']}
        row['h2d']+=row['kv_h2d']
        row.update(wall_ms=(hi-lo)/1000,h2d_compute_overlap_ms=overlap)
        rows.append(row)
    if not sum(r['forward']+r['backward'] for r in rows)>0:
        raise RuntimeError('CUDA kernels did not link to forward/backward CPU scopes')
    return rows
