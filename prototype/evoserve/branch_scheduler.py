"""Head-to-head validation followed by a separate comparison against production."""

import random
from .paired import PairedScheduler


class BranchScheduler(PairedScheduler):
    def start_contest(self, e, versions, now):
        self.start_validation(e, versions[0], now)
        t = self.experts[e].trial
        t["production_trial_seed"] = t["seed"]
        if len(versions) > 1:
            t["seed"] += 100000000
            random.Random(t["seed"]).shuffle(t["states"])
        t.update(
            opponents=list(versions[1:]),
            reference=(
                versions[1] if len(versions) > 1 else self.experts[e].production
            ),
        )

    def begin_episode(self, expert, robot, episode):
        v = super().begin_episode(expert, robot, episode)
        st = self.experts[expert]
        if (
            v is not None
            and st.trial
            and st.active_episodes[episode].get("arm") == "production"
        ):
            v = st.trial["reference"]
            st.active_episodes[episode]["version"] = v
        return v

    def finish_validation(self, e):
        t = e.trial
        if not t["opponents"]:
            return super().finish_validation(e)
        ref = t["reference"]
        challenger = t["candidate"]
        last = t["checks"][-1]
        gain = last["favorable_difference"]
        se = last["paired_se"]
        winner = (challenger if gain > 0 else ref) if abs(gain) > 2 * se else None
        event = dict(
            event="branch_comparison",
            expert=e.expert,
            step=e.step,
            candidate_version=challenger,
            reference_version=ref,
            old_version=e.production,
            decision="advance" if winner else "discard",
            winner=winner,
            final_branch_winner=bool(winner and len(t["opponents"]) == 1),
            episodes_per_arm=last["pairs"],
        )
        e.pending_decision = None
        e.candidate = winner
        if winner is None:
            e.trial = None
            e.last_validation_step = e.step
        else:
            remaining = t["opponents"][1:]
            seed = t["seed"] + 100000000 if remaining else t["production_trial_seed"]
            states = list(range(50))
            random.Random(seed).shuffle(states)
            e.trial = dict(
                t,
                seed=seed,
                states=states,
                candidate=winner,
                reference=remaining[0] if remaining else e.production,
                opponents=remaining,
                assigned={"candidate": 0, "production": 0},
                pairs={},
                limit=10,
                checks=[],
                candidate_outcomes=[],
                production_outcomes=[],
            )
        return event
