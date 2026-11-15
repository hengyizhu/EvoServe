import math
import unittest
from evoserve.core import *


class ControlTests(unittest.TestCase):
    def test_missing_requests_count_as_timeouts(self):
        d=DeadlineLedger();d.arrive('a',0,1,1);d.arrive('b',.1,1,2);d.finish('a',.1)
        with self.assertRaises(ValueError):d.window(0,1,1.1)
        r=d.window(0,1,1.3)
        self.assertEqual(r['timeouts'],1);self.assertEqual(r['slo'],.5);self.assertTrue(r['p99_infinite'])

    def test_reduce_group_first_and_pause_allowed(self):
        c=TwoKnobController(4,m=4)
        for n in [3,2,1]:
            c.update({'timeout_rate':.5});self.assertEqual(c.m,n);self.assertEqual(c.duty,1)
        c.update({'timeout_rate':.5});self.assertEqual(c.duty,0);self.assertTrue(math.isinf(c.pause_after(.3)))
        c.update({'timeout_rate':0});self.assertAlmostEqual(c.duty,.05)

    def test_exact_observation_action_alignment(self):
        a=ActionWindowAssembler('episode','v1');a.observe(0,'prefix_at_0')
        for i in range(5):self.assertFalse(a.execute(i))
        a.observe(5,'prefix_at_5')
        for i in range(5,9):self.assertFalse(a.execute(i))
        [(key,prefix,actions)]=a.execute(9)
        self.assertEqual((key.frame,prefix,actions),(0,'prefix_at_0',list(range(10))))
        self.assertIn(5,a.pending)

    def test_eviction_version_and_owned_batch(self):
        p=Reservoir(2,20,1);p.activate(0,'v1')
        for i in range(2):self.assertTrue(p.add(Window(CacheKey(str(i),0,'v1'),0,{'x':i},10,tuple(range(10)))))
        batch=p.sample(0);self.assertEqual(len(batch),8)
        p.activate(0,'v2');self.assertEqual(p.bytes,0);self.assertEqual(len(batch),8)
        self.assertFalse(p.add(Window(CacheKey('old',0,'v1'),0,{},10,tuple(range(10)))))
        with self.assertRaises(ValueError):p.add(Window(CacheKey('bad',0,'v2'),0,{},10,(1,2)))

    def scheduler(self):
        return Scheduler([ExpertState(0,[0],'v0')],LearningPrior([(0,0),(1000,.5),(4000,1)],.2))

    def test_noise_gate_and_fallback(self):
        self.assertAlmostEqual(validation_margin(50),.10606601717798213)
        s=self.scheduler();e=s.experts[0];e.production_recent.extend([0]*50);e.step=999
        self.assertFalse(s.should_validate(0));e.step=1000;self.assertTrue(s.should_validate(0))

    def test_fifty_real_episodes_each_arm_and_episode_boundary(self):
        s=self.scheduler();e=s.experts[0];e.step=1000;s.start_validation(0,'candidate',10)
        result=None
        for i in range(100):
            v=s.begin_episode(0,0,str(i));result=s.end_episode(0,str(i),v=='candidate')
            if i<99:self.assertIsNone(result);self.assertEqual(e.production,'v0')
        self.assertEqual(result['decision'],'promote');self.assertEqual(e.production,'candidate')
        self.assertEqual(result['episodes_per_arm'],50)

    def test_inflight_episode_keeps_version_on_decision(self):
        s=Scheduler([ExpertState(0,[0,1],'v0')],LearningPrior([(0,0),(1000,1)],.2),validation_episodes=1)
        e=s.experts[0];e.step=1000;s.start_validation(0,'candidate',10)
        s.begin_episode(0,0,'c');s.begin_episode(0,1,'p');s.begin_episode(0,1,'inflight')
        s.end_episode(0,'c',True);s.end_episode(0,'p',False)
        self.assertEqual(e.production,'v0');self.assertIsNone(s.begin_episode(0,0,'blocked'))
        result=s.end_episode(0,'inflight',False);self.assertEqual(result['decision'],'promote')

    def test_prior_fit_no_loss_or_future_data(self):
        p=LearningPrior([(0,0),(1000,.5),(2000,1)],.4)
        pred=p.predict(.2,1000,[]);self.assertAlmostEqual(pred['success'],.4)
        amp,scale=p.fit([(1000,.1)]);self.assertAlmostEqual(scale,1);self.assertAlmostEqual(amp,.2)
        amp,scale=p.fit([(500,.1),(1000,.2),(2000,.4)])
        self.assertAlmostEqual(amp,.4);self.assertAlmostEqual(scale,1)

    def test_priority_is_best_marginal_gain_per_step(self):
        s=Scheduler([ExpertState(0,[0,1],'v0')],LearningPrior([(0,0),(500,.1),(1000,.8),(4000,1)],.5))
        e=s.experts[0];e.production_recent.extend([0,1]*25)
        score=s.priority(e)
        self.assertEqual(score['selected_budget'],1000)
        self.assertAlmostEqual(score['score'],2*.4/1000)
        e.step=1000
        score=s.priority(e)
        # Already-predicted gains must not be counted again for ranking.
        self.assertLess(score['score'],2*.4/1000)


class RankingEdgeTests(unittest.TestCase):
    def test_all_budgets_and_production_floor(self):
        class Prior:
            def predict(self, production, steps, validations):
                values={100:.1,350:.15,600:.3,1100:.4,2100:.5,4100:.6}
                return {'success':values[steps],'production_success':production}
        e=ExpertState(7,[0,1,2],'parent');e.step=100;e.production_recent.extend([True,False,False,False,False])
        scheduler=Scheduler([e],Prior());r=scheduler.priority(e)
        self.assertEqual([v['budget'] for v in r['budgets']],[250,500,1000,2000,4000])
        self.assertEqual(r['budgets'][0]['average_gain_per_step'],0)
        self.assertAlmostEqual(r['score'],3*.1/500)
        self.assertFalse(scheduler.should_validate(7))




class PeriodicTests(unittest.TestCase):
    def test_timeout_uses_remaining_actions_then_holds(self):
        a=PeriodicActions();block=[[i,0,0,0,0,0,1] for i in range(10)]
        a.offer(0,.2,block);self.assertTrue(a.boundary(0,.25))
        self.assertEqual([a.next()[0][0] for _ in range(5)],list(range(5)))
        self.assertFalse(a.boundary(1,.5))
        self.assertEqual([a.next()[0][0] for _ in range(5)],list(range(5,10)))
        hold,held=a.next();self.assertTrue(held);self.assertEqual(hold,[0,0,0,0,0,0,1])
        a.offer(2,.751,block);self.assertFalse(a.boundary(2,.75))
        self.assertTrue(a.next()[1])
    def test_async_observation_keeps_exact_action_labels(self):
        a=ActionWindowAssembler('e','v')
        for i in range(12):a.execute(i)
        a.observe(1,'rendered_frame_1')
        [(key,cache,labels)]=a.collect_ready()
        self.assertEqual(key.frame,1);self.assertEqual(labels,list(range(1,11)))

if __name__=='__main__':unittest.main()
