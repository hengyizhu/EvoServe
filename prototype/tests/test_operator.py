import unittest
from evoserve.core import TakeoverRule,PeriodicActions,ActionWindowAssembler,Reservoir,Window,CacheKey,window_data_source,Scheduler,ExpertState,LearningPrior

class OperatorTests(unittest.TestCase):
    def test_confirmed_two_mm_twenty_intervals(self):
        r=TakeoverRule(.002)
        self.assertIsNone(r.update([0,0,0],0,30))
        for i in range(1,20):self.assertIsNone(r.update([.0015*i/20,0,0],i*.05,30))
        self.assertEqual(r.update([.0015,0,0],1,30)['reason'],'stagnation')
        self.assertIsNone(r.update([1,0,0],2,30))
    def test_uses_max_displacement_not_just_endpoints(self):
        r=TakeoverRule(.002)
        for i in range(21):self.assertIsNone(r.update([.004 if i==10 else 0,0,0],i*.05,30))
    def test_seventy_percent_wall_time_and_success(self):
        r=TakeoverRule(.002)
        self.assertIsNone(r.update([0,0,0],20.999,30))
        self.assertEqual(r.update([1,0,0],21,30)['reason'],'time_budget_70pct')
        self.assertIsNone(TakeoverRule(.002).update([0,0,0],21,30,success=True))
    def test_old_production_block_cannot_override_operator(self):
        a=PeriodicActions();block=[[0,0,0,0,0,0,-1]]*10
        a.offer(0,.1,block,'production');a.interrupt()
        self.assertFalse(a.boundary(0,.25,allowed_source='operator'));self.assertTrue(a.next()[1])
        a.offer(1,.4,block,'operator');self.assertTrue(a.boundary(1,.5,allowed_source='operator'))
        self.assertFalse(a.next()[1]);self.assertEqual(a.last_source,'operator')
    def test_cache_is_actual_production_version_during_candidate_rescue(self):
        a=ActionWindowAssembler('ep','candidate');a.observe(0,{},version='production')
        for i in range(9):a.execute(i)
        [(key,_,labels)]=a.execute(9)
        self.assertEqual(key.version,'production');self.assertEqual(labels,list(range(10)))
    def test_never_train_mixed_transition_or_hold_actions(self):
        self.assertIsNone(window_data_source(['production']*5+['operator']*5,True))
        self.assertIsNone(window_data_source(['operator']*9+['hold'],True))
        self.assertIsNone(window_data_source(['production']*10,True))
        self.assertEqual(window_data_source(['operator']*10,True),'operator_correction')
        self.assertEqual(window_data_source(['production']*10,False),'self_success')
    def test_corrective_samples_do_not_require_fabricated_success(self):
        p=Reservoir(8,100,1);p.activate(3,'v')
        self.assertTrue(p.add(Window(CacheKey('corrected',0,'v'),3,{},1,tuple(range(10)),False,'operator_correction')))
        self.assertFalse(p.add(Window(CacheKey('failed',0,'v'),3,{},1,tuple(range(10)),False,'self_success')))
        self.assertFalse(p.add(Window(CacheKey('wrong_version',0,'old'),3,{},1,tuple(range(10)),False,'operator_correction')))

    def test_runtime_scores_assisted_completion_as_policy_failure(self):
        import tempfile,threading
        from pathlib import Path
        from evoserve.runtime import Run
        runtime=Run.__new__(Run)
        runtime.scheduler=Scheduler([ExpertState(3,[0],'v0')],LearningPrior([(0,0),(1000,1)],.5))
        runtime.scheduler.begin_episode(3,0,'episode')
        runtime.lock=threading.RLock();runtime.episodes=[]
        with tempfile.TemporaryDirectory() as folder:
            runtime.out=Path(folder)
            runtime.handle(dict(op='end_episode',expert=3,episode='episode',success=True,autonomous_success=False,intervened=True))
            self.assertEqual(list(runtime.scheduler.experts[3].production_recent),[0])
            self.assertTrue(runtime.episodes[0]['success'])
            self.assertFalse(runtime.episodes[0]['autonomous_success'])
