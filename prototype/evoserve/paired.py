"""Paired validation with balanced ten-pair looks in the accelerated simulator."""

import math
import random
import statistics

from .core import Scheduler


class PairedScheduler(Scheduler):
    def __init__(self, *args, fixed50=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed50 = fixed50
        self.trial_number = {e: 0 for e in self.experts}

    def start_validation(self, expert, candidate, now):
        super().start_validation(expert, candidate, now)
        trial = self.experts[expert].trial
        self.trial_number[expert] += 1
        seed = 20261108 + expert * 100000 + self.trial_number[expert]
        states = list(range(50))
        random.Random(seed).shuffle(states)
        trial.update(
            states=states,
            seed=seed,
            assigned={"candidate": 0, "production": 0},
            pairs={},
            limit=self.n if self.fixed50 else 10,
            checks=[],
            metric=(
                "takeover_fraction"
                if trial["prediction"]["production_success"] < 0.20
                else "autonomous"
            ),
        )

    def begin_episode(self, expert, robot, episode):
        e = self.experts[expert]
        t = e.trial
        if not t:
            return super().begin_episode(expert, robot, episode)
        arm = "candidate" if robot in t["candidate_robots"] else "production"
        if len(e.robots)==1:
            arm=min(t["assigned"],key=t["assigned"].get)
        i = t["assigned"][arm]
        if i >= t["limit"] or e.pending_decision:
            return None
        version = super().begin_episode(expert, robot, episode)
        version=e.candidate if arm=="candidate" else e.production
        e.active_episodes[episode].update(version=version,
            pair=i, arm=arm, state=t["states"][i], seed=t["seed"] + i
        )
        t["assigned"][arm] += 1
        return version

    def end_scored_episode(self, expert, episode, success, corrections, frames):
        e = self.experts[expert]
        assignment = e.active_episodes.pop(episode)
        if assignment["version"] == e.production:
            e.production_recent.append(int(success))
        t = e.trial
        if "pair" in assignment:
            i, arm = assignment["pair"], assignment["arm"]
            t["pairs"].setdefault(i, {})[arm] = dict(
                autonomous=int(success), takeover_fraction=corrections / frames
            )
            t[arm + "_outcomes"].append(int(success))
            n = t["limit"]
            if all(len(t["pairs"].get(j, {})) == 2 for j in range(n)):
                metric = t["metric"]
                differences = [
                    p["candidate"][metric] - p["production"][metric]
                    for p in t["pairs"].values()
                ]
                gain = statistics.mean(differences) * (
                    -1 if metric == "takeover_fraction" else 1
                )
                se = statistics.stdev(differences) / math.sqrt(n)
                t["checks"].append(
                    dict(pairs=n, favorable_difference=gain, paired_se=se)
                )
                if abs(gain) > 2 * se or n == self.n:
                    e.pending_decision = "promote" if gain > 2 * se else "discard"
                else:
                    t["limit"] += 10
        if e.pending_decision and not e.active_episodes:
            # The base scheduler fits future success predictions to autonomous outcomes.
            # Use the actual number of pairs in both means on an early decision.
            event = self.finish_validation(e)
            event.update(
                metric=t["metric"],
                checks=t["checks"],
                pairs_used=len(t["pairs"]),
                episodes_used=2 * len(t["pairs"]),
                baseline_autonomous=t["prediction"]["production_success"],
            )
            return event
