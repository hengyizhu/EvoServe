import unittest
from evoserve.core import ExpertState, LearningPrior
from evoserve.paired import PairedScheduler


class PairedTest(unittest.TestCase):
    def run_trial(self, production_recent, better, fixed50=False, only_last=False):
        e = ExpertState(0, [0, 1, 2, 3], "parent")
        e.step = 1000
        e.production_recent.extend(production_recent)
        sched = PairedScheduler(
            [e], LearningPrior([(0, 0), (1000, 1)], 0.5), fixed50=fixed50
        )
        sched.start_validation(0, "candidate", 0)
        seen = {}
        for i in range(50):
            for arm, r in enumerate((i % 2, 2 + i % 2)):
                key = f"{i}.{arm}"
                sched.begin_episode(0, r, key)
                a = e.active_episodes[key]
                seed = (a["state"], a["seed"])
                if arm == 0:
                    seen[i] = seed
                else:
                    self.assertEqual(seen[i], seed)
                result = sched.end_scored_episode(
                    0,
                    key,
                    int(better and not arm),
                    0 if better and not arm and (not only_last or i == 49) else 7,
                    10,
                )
                if result:
                    return result
        self.fail("no decision")

    def test_early_takeover(self):
        r = self.run_trial([0] * 10, True)
        self.assertEqual(
            (r["pairs_used"], r["metric"], r["decision"]),
            (10, "takeover_fraction", "promote"),
        )

    def test_at_threshold(self):
        self.assertEqual(self.run_trial([0] * 4 + [1], True)["metric"], "autonomous")

    def test_tie_at_cap(self):
        r = self.run_trial([0], False)
        self.assertEqual(
            (r["pairs_used"], r["decision"], len(r["checks"])), (50, "discard", 5)
        )

    def test_positive_but_not_significant_at_cap(self):
        for fixed in (False, True):
            r = self.run_trial([0], True, fixed50=fixed, only_last=True)
            self.assertEqual((r["pairs_used"], r["decision"]), (50, "discard"))

    def test_fixed_waits_for_50_despite_clear_gain(self):
        r = self.run_trial([0], True, fixed50=True)
        self.assertEqual(
            (r["pairs_used"], r["decision"], len(r["checks"])), (50, "promote", 1)
        )
