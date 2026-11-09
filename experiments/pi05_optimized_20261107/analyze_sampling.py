import pathlib,json,collections,csv
R=pathlib.Path('/home/hengyi/pi05_optimized_20261107');s=json.load(open('/home/hengyi/vla_followup_20261020/data_split.json'));seq=json.load(open('/home/hengyi/vla_ladder_20261020/sample_sequence.json'));counts=collections.Counter(tuple(x) for batch in seq for x in batch)
target={(e['episode_index'],i) for e in s['target_train'] for i in range(e['length']-9)};replay=[(x['episode'],x['start']) for x in s['replay_windows']];assert not target.intersection(replay)
out={}
for name,domain in [('target',target),('replay',set(replay)),('combined',target|set(replay))]:
 c={x:counts[x] for x in domain};draws=sum(c.values());unique=sum(v>0 for v in c.values());hist=collections.Counter(c.values());out[name]={'candidate_unique_windows':len(domain),'draws':draws,'unique_observed_windows':unique,'k_per_candidate':draws/len(domain),'k_per_observed_window':draws/unique,'histogram_including_zero':dict(sorted(hist.items())),'within_training_exact_window_repeat_fraction':(draws-unique)/draws}
out['replay_candidate_list_length']=len(replay);out['previous_steps']=1000;out['effective_batch']=8;out['warning']='An observation window identity is not an augmented input identity. With stochastic image augmentation, A1 and A2 can both miss; these k values do not prove cacheability.'
rows=[]
for tier,source in [('microbatch1','/home/hengyi/pi05_measure_20261107/training/action/summary.json'),('microbatch4',str(R/'training_mb4/action/summary.json'))]:
 p=pathlib.Path(source)
 if not p.exists():continue
 d=json.loads(p.read_text());C=d['prefix_ms']['p50']/8;rest=sum(d[k]['p50'] for k in ['action_ms','backward_ms','optimizer_ms'])/8
 for k in [1,5,10,out['combined']['k_per_observed_window'],out['target']['k_per_observed_window'],out['replay']['k_per_observed_window']]:rows.append({'training_shape':tier,'k':k,'prefix_ms_per_sample':C,'other_compute_ms_per_sample':rest,'A2_vs_A1_ideal_saving_fraction':C/(C+k*rest),'formula':'C/(C+kR); excludes augmentation, cache storage/lookup and serving cost; assumes A1 computes once per distinct input and A2 always hits service cache'})
(R/'sampling_k.json').write_text(json.dumps(out,indent=2));(R/'cache_sensitivity.json').write_text(json.dumps(rows,indent=2))
with (R/'sampling_windows.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['episode','start','population','draw_count']);w.writerows((a,b,'target' if (a,b) in target else 'replay',counts[(a,b)]) for a,b in sorted(target|set(replay)))
print(json.dumps({k:{n:v for n,v in d.items() if 'histogram' not in n} for k,d in out.items() if isinstance(d,dict)},indent=2))
