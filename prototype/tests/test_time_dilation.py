from evoserve.core import DeadlineLedger, ExpertState, LearningPrior
from evoserve.paired import PairedScheduler


def test_dilated_deadline_applies_to_unanswered_requests():
    ledger=DeadlineLedger(.75)
    for i in range(3):
        ledger.arrive(str(i),0,0,i)
    ledger.finish('0',.749)
    ledger.finish('1',.751)
    result=ledger.window(0,1,1.75)
    assert result['requests']==3 and result['timeouts']==2


def test_single_robot_runs_both_arms_on_identical_state_and_seed():
    state=ExpertState(0,[0],'parent')
    sched=PairedScheduler([state],LearningPrior([(0,0),(1000,1)],.8))
    state.step=1000
    sched.start_validation(0,'candidate',0)
    for pair in range(10):
        settings=[]
        for arm in ['candidate','production']:
            key=f'{pair}:{arm}'
            assert sched.begin_episode(0,0,key)==('candidate' if arm=='candidate' else 'parent')
            row=state.active_episodes[key]
            assert row['arm']==arm and row['pair']==pair
            settings.append((row['state'],row['seed']))
            decision=sched.end_scored_episode(0,key,arm=='candidate',0 if arm=='candidate' else 10,10)
        assert settings[0]==settings[1]
    assert decision['decision']=='promote' and state.production=='candidate'
