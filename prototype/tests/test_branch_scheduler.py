import unittest
from evoserve.core import ExpertState, LearningPrior
from evoserve.branch_scheduler import BranchScheduler


class BranchValidation(unittest.TestCase):
    def test_select_then_compare_to_production(self):
        e = ExpertState(0, [0, 1, 2, 3], "parent")
        e.step = 1000
        e.production_recent.extend([0] * 10)
        sched = BranchScheduler([e], LearningPrior([(0, 0), (1000, 1)], 0.5))
        sched.start_contest(0, ["r32", "r128"], 0)
        for round_id in range(2):
            for i in range(10):
                for arm, r in enumerate((i % 2, 2 + i % 2)):
                    key = f"{round_id}-{i}-{arm}"
                    version = sched.begin_episode(0, r, key)
                    expected = ("r32", "r128") if round_id == 0 else ("r128", "parent")
                    self.assertEqual(version, expected[arm])
                    event = sched.end_scored_episode(
                        0, key, False, 0 if version == "r128" else 5, 10
                    )
            if round_id == 0:
                self.assertEqual(
                    (event["decision"], event["winner"], e.production),
                    ("advance", "r128", "parent"),
                )
        self.assertEqual((event["decision"], e.production), ("promote", "r128"))
