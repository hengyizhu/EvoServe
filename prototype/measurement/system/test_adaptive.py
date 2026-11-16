import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'worker'))
from adaptive_feedback import Rule,Feedback
from arrival_plan import schedule

class Tests(unittest.TestCase):
 def test_halving_then_duty(self):
  r=Rule('adaptive',8)
  for expected in [4,2,1]:r.observe(.04);self.assertEqual((r.experts,r.duty),(expected,1.))
  r.observe(.04);self.assertAlmostEqual(r.duty,.85)
  for _ in range(10):r.observe(1.)
  self.assertEqual(r.duty,0.)
  r.observe(0.);self.assertAlmostEqual(r.duty,.05)
 def test_recovery_streak(self):
  r=Rule();r.observe(0);r.observe(0);self.assertEqual(r.experts,1);r.observe(0);self.assertEqual(r.experts,2)
  r.observe(0);r.observe(.005);r.observe(0);r.observe(0);self.assertEqual(r.experts,2)
  r.observe(0);self.assertEqual(r.experts,4)
  r.duty=.95;r.zero_streak=0;r.observe(0);self.assertEqual(r.zero_streak,0)
 def test_exact_threshold_and_fixed(self):
  r=Rule('adaptive',8);r.observe(.01);self.assertEqual(r.experts,8)
  r=Rule('fixed',8);r.observe(.04);self.assertEqual(r.experts,8);self.assertAlmostEqual(r.duty,.85)
 def test_all_windows_consumed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);rs=[dict(window_end_ns=i,timeout_rate=0.,p99_ms=100,requests=25,timeouts=0) for i in range(1,7)]
   (p/'feedback_windows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs));f=Feedback(p);f.update(p);self.assertEqual(f.experts,4);f.update(p);self.assertEqual(len((p/'controller.jsonl').read_text().splitlines()),6)
 def test_arrival_transition_and_deadlines(self):
  cfg=dict(origin_ns=1_000_000_000,duration_s=1800,clients=4,phases=[dict(seconds=600,rate=x/.14249900977,load=x) for x in [.7,.85,.7]])
  p=schedule(cfg);self.assertEqual(len(set(r['request_id'] for r in p)),len(p));self.assertTrue(all(b['nominal_release_ns']>a['nominal_release_ns'] for a,b in zip(p,p[1:])))
  for stage,t in [(1,0),(2,600),(3,1200)]:self.assertEqual(next(r for r in p if r['stage']==stage)['nominal_release_ns'],cfg['origin_ns']+t*1_000_000_000)
  self.assertTrue(all(r['deadline_ns']-r['nominal_release_ns']==250_000_000 for r in p));self.assertEqual(p,schedule(cfg))
 def test_summary_missing_responses_and_expert_accounting(self):
  from adaptive_campaign import summarize_run
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);q=p/'pair0/measurement';q.mkdir(parents=True)
   cfg=dict(origin_ns=1_000_000_000,duration_s=15,global_rate=1,phases=[dict(seconds=5,rate=1,load=x) for x in [.7,.85,.7]])
   (p/'measurement_schedule.json').write_text(json.dumps(cfg));(p/'config.json').write_text(json.dumps(dict(configuration='fixed1',initial_experts=1)));(p/'calibrations.json').write_text('[]')
   plan=schedule(cfg);req=[dict(r,finish_ns=r['nominal_release_ns']+100_000_000) for r in plan[:-1]]
   (q/'requests.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in req))
   ss=[];counts=[0]*8
   for i,ids in enumerate([[0],[1,2],[3,4,5,6]]):
    counts=[n+int(e in ids) for e,n in enumerate(counts)]
    ss.append(dict(start_ns=cfg['origin_ns']+int((i*5+.1)*1e9),finish_ns=cfg['origin_ns']+int((i*5+1)*1e9),experts=len(ids),expert_steps=len(ids),expert_ids=ids,optimizer_steps=list(counts),ema_steps=list(counts),loss=.01))
   (q/'steps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ss));r=summarize_run(p)
   self.assertEqual(r['total_expert_steps'],7);self.assertEqual([s['expert_steps'] for s in r['stages']],[1,2,4]);self.assertEqual(r['stages'][2]['slo_rate'],.8);self.assertIsNone(r['stages'][2]['p99_ms'])
if __name__=='__main__':unittest.main()
