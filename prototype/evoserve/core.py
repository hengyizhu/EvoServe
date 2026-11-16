"""CPU control plane; simulation, CUDA and transport are separate modules."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import random
import threading


DEADLINE_S = .250
SLO = .99


def validation_margin(episodes=50):
    return max(.05, 1.5 * math.sqrt(.25 / episodes))


def percentile(values, q=.99):
    return sorted(values)[max(0, math.ceil(q * len(values))-1)] if values else None


class DeadlineLedger:
    """Count nominal arrivals, including requests never completed."""
    def __init__(self):
        self.requests = {}
        self.lock = threading.RLock()

    def arrive(self, key, release, expert, robot):
        with self.lock:
            if key in self.requests:
                raise ValueError('duplicate request')
            self.requests[key] = dict(release=release, finish=None, expert=expert, robot=robot)

    def finish(self, key, timestamp):
        with self.lock:
            if self.requests[key]['finish'] is not None:
                raise ValueError('duplicate completion')
            self.requests[key]['finish'] = timestamp

    def window(self, lo, hi, observed, experts=None):
        if observed < hi + DEADLINE_S:
            raise ValueError('deadlines have not matured')
        with self.lock:
            rows = [dict(x) for x in self.requests.values() if lo <= x['release'] < hi and (experts is None or x['expert'] in experts)]
        latencies = [(x['finish']-x['release']) if x['finish'] is not None else math.inf for x in rows]
        misses = sum(v > DEADLINE_S for v in latencies)
        p99 = percentile(latencies)
        return dict(requests=len(rows), timeouts=misses,
                    timeout_rate=misses/len(rows) if rows else None,
                    slo=1-misses/len(rows) if rows else None,
                    p99_ms=p99*1000 if p99 is not None and math.isfinite(p99) else None,
                    p99_infinite=p99 is not None and not math.isfinite(p99))


@dataclass
class TwoKnobController:
    max_experts: int
    m: int = 1
    duty: float = 1.

    def update(self, window):
        before = dict(m=self.m, duty=self.duty)
        miss = window['timeout_rate']
        reason = 'no_mature_requests'
        if miss is not None:
            if miss > 1-SLO + 1e-12:
                if self.m > 1:
                    self.m -= 1
                    reason = 'reduce_group_before_duty'
                else:
                    self.duty = max(0., self.duty-5*(miss-.01))
                    reason = 'reduce_duty'
            elif self.duty < 1:
                self.duty = min(1., self.duty+5*(.01-miss))
                reason = 'restore_duty'
            elif miss < .01 and self.m < self.max_experts:
                self.m += 1
                reason = 'restore_group'
            else:
                reason = 'hold'
        return dict(before=before, after=dict(m=self.m, duty=self.duty), reason=reason,
                    target_timeout_rate=.01, deadline_ms=250)

    def pause_after(self, step_seconds):
        return math.inf if self.duty == 0 else step_seconds*(1-self.duty)/self.duty


@dataclass(frozen=True)
class CacheKey:
    episode: str
    frame: int
    version: str


@dataclass
class Window:
    key: CacheKey
    expert: int
    payload: object
    nbytes: int
    action_frames: tuple
    successful_episode: bool = True
    data_source: str = 'self_success'

    def check(self):
        if self.data_source not in ['self_success','operator_correction','demonstration']:raise ValueError('unknown data source')
        if self.action_frames != tuple(range(self.key.frame, self.key.frame+10)):
            raise ValueError('cache/action window misalignment or incomplete window')


class Reservoir:
    """Bounded random replacement, with owned batch references across eviction."""
    def __init__(self, capacity, byte_limit, seed):
        self.capacity, self.byte_limit = capacity, byte_limit
        self.rng = random.Random(seed)
        self.items = {}
        self.versions = {}
        self.recent = {}
        self.bytes = 0
        self.seen = 0
        self.lock = threading.RLock()

    def activate(self, expert, version):
        with self.lock:
            self.versions[expert] = version
            for key in list(self.items):
                row = self.items[key]
                if row.expert == expert and key.version != version:
                    self.bytes -= row.nbytes
                    del self.items[key]
            self.recent[expert] = deque(maxlen=64)
            self.seen = len(self.items)

    def add(self, row):
        row.check()
        with self.lock:
            if self.versions.get(row.expert) != row.key.version or (not row.successful_episode and row.data_source!='operator_correction'):
                return False
            if row.nbytes > self.byte_limit or row.key in self.items:
                return False
            self.seen += 1
            if len(self.items) >= self.capacity:
                # Algorithm R avoids silently turning the pool into a sliding window.
                j = self.rng.randrange(self.seen)
                if j >= self.capacity:
                    return False
                victim = list(self.items)[j]
                self.bytes -= self.items.pop(victim).nbytes
            while self.items and self.bytes+row.nbytes > self.byte_limit:
                victim = self.rng.choice(list(self.items))
                self.bytes -= self.items.pop(victim).nbytes
            self.items[row.key] = row
            self.bytes += row.nbytes
            self.recent.setdefault(row.expert, deque(maxlen=64)).append(row.key)
            return True

    def sample(self, expert):
        with self.lock:
            all_rows = [v for v in self.items.values() if v.expert == expert]
            recent = [self.items[k] for k in self.recent.get(expert, ()) if k in self.items]
            if not recent or not all_rows:
                return None
            # Four recent + four historical; sampling with replacement is explicit.
            return [self.rng.choice(recent) for _ in range(4)] + [self.rng.choice(all_rows) for _ in range(4)]


class ActionWindowAssembler:
    """Cache arrives before its actual ten executed actions. Never pad labels."""
    def __init__(self, episode, version):
        self.episode, self.version = episode, version
        self.pending = {}
        self.actions = []
        self.versions = {}

    def observe(self, frame, cache, version=None):
        if frame < 0 or frame > len(self.actions) or frame in self.pending:
            raise ValueError('observation frame is invalid or duplicated')
        self.pending[frame] = cache
        self.versions[frame] = version or self.version

    def execute(self, action):
        self.actions.append(action)
        return self.collect_ready()

    def collect_ready(self):
        ready = []
        for start in list(self.pending):
            if start+10 <= len(self.actions):
                ready.append((CacheKey(self.episode, start, self.versions.pop(start)),
                              self.pending.pop(start), self.actions[start:start+10]))
        return ready


class LearningPrior:
    """Fixed measured curve; fit only amplitude and time scale after validation."""
    def __init__(self, points, amplitude):
        self.points = sorted((float(s), float(v)) for s, v in points)
        self.default_amplitude = float(amplitude)

    def curve(self, steps):
        if steps <= self.points[0][0]:
            return self.points[0][1]
        for (a, x), (b, y) in zip(self.points, self.points[1:]):
            if steps <= b:
                return x+(y-x)*(steps-a)/(b-a)
        return self.points[-1][1]

    def fit(self, validations):
        if not validations:
            return self.default_amplitude, 1.
        # One point cannot identify two parameters: retain prior time scale.
        scales = [1.] if len({s for s, _ in validations}) < 2 else [math.exp(i/20) for i in range(-46,47)]
        choices = []
        for scale in scales:
            xs = [self.curve(s/scale) for s, _ in validations]
            den = sum(x*x for x in xs)
            amp = sum(x*y for x, (_,y) in zip(xs,validations))/den if den else self.default_amplitude
            amp = min(1., max(-1., amp))
            error = sum((amp*x-y)**2 for x, (_,y) in zip(xs,validations))
            choices.append((error, abs(math.log(scale)), amp, scale))
        _, _, amp, scale = min(choices)
        return amp, scale

    def predict(self, production_success, steps, validations):
        amplitude, scale = self.fit(validations)
        raw = production_success+amplitude*self.curve(steps/scale)
        return dict(success=min(1.,max(0.,raw)), amplitude=amplitude, time_scale=scale,
                    steps=steps, production_success=production_success,
                    validation_points=len(validations))


@dataclass
class ExpertState:
    expert: int
    robots: list[int]
    production: str
    candidate: str | None = None
    step: int = 0
    production_step: int = 0
    last_validation_step: int = 0
    production_recent: deque = field(default_factory=lambda: deque(maxlen=50))
    validations: list = field(default_factory=list)
    trial: dict | None = None
    active_episodes: dict = field(default_factory=dict)
    pending_decision: str | None = None
    episode_counter: int = 0
    epoch: int = 0


class Scheduler:
    def __init__(self, experts, prior, validation_episodes=50):
        self.experts = {x.expert:x for x in experts}
        self.prior = prior
        self.n = validation_episodes
        self.events = []
        self.lock = threading.RLock()

    def prediction(self, e):
        p = sum(e.production_recent)/len(e.production_recent) if e.production_recent else 0.
        return self.prior.predict(p, e.step-e.production_step, e.validations)

    def priority(self, e):
        current = self.prediction(e)
        production = current['production_success']
        values = []
        for budget in [250,500,1000,2000,4000]:
            future = self.prior.predict(production,e.step-e.production_step+budget,e.validations)
            gain = max(future['success'],production)-max(current['success'],production)
            values.append(dict(budget=budget,average_gain_per_step=gain/budget,predicted_future=future['success']))
        best = max(values,key=lambda x:x['average_gain_per_step'])
        return dict(score=len(e.robots)*best['average_gain_per_step'],selected_budget=best['budget'],budgets=values,current=current)

    def ranked(self):
        with self.lock:
            rows = []
            for e in self.experts.values():
                if e.trial or e.pending_decision:
                    continue
                rows.append((self.priority(e)['score'],e.step,e.expert))
            return [e for _,_,e in sorted(rows, key=lambda x:(-x[0],x[1],x[2]))]

    def should_validate(self, expert):
        e = self.experts[expert]
        if not e.robots or e.trial or e.pending_decision or e.step == e.production_step:
            return False
        pred = self.prediction(e)
        return (pred['success']-pred['production_success'] >= validation_margin(self.n)
                or e.step-e.last_validation_step >= 1000)

    def start_validation(self, expert, candidate, now):
        with self.lock:
            e = self.experts[expert]
            if not self.should_validate(expert):
                raise ValueError('candidate has not reached a validation trigger')
            pred = self.prediction(e)
            e.candidate = candidate
            e.trial = dict(candidate=candidate, production=e.production, prediction=pred,
                           started=now, step=e.step, epoch=e.epoch, candidate_outcomes=[],
                           production_outcomes=[], candidate_robots=e.robots[:max(1,len(e.robots)//2)])
            e.last_validation_step = e.step
            self.events.append(dict(event='validation_started', expert=expert, **e.trial))

    def begin_episode(self, expert, robot, episode):
        with self.lock:
            e = self.experts[expert]
            if e.pending_decision:
                return None
            version = e.production
            if e.trial:
                candidate = (e.episode_counter % 2 == 0) if len(e.robots)==1 else robot in e.trial['candidate_robots']
                if candidate:
                    version = e.candidate
            e.episode_counter += 1
            e.active_episodes[episode] = dict(version=version, epoch=e.epoch, trial_started=e.trial['started'] if e.trial else None)
            return version

    def end_episode(self, expert, episode, success):
        with self.lock:
            e = self.experts[expert]
            assignment = e.active_episodes.pop(episode)
            version, epoch = assignment['version'], assignment['epoch']
            if epoch == e.epoch and version == e.production:
                e.production_recent.append(int(success))
            t = e.trial
            if t and epoch == t['epoch'] and assignment['trial_started'] == t['started']:
                key = 'candidate_outcomes' if version==e.candidate else 'production_outcomes'
                if len(t[key]) < self.n:
                    t[key].append(int(success))
                if len(t['candidate_outcomes']) >= self.n and len(t['production_outcomes']) >= self.n:
                    e.pending_decision = 'promote' if sum(t['candidate_outcomes'])>sum(t['production_outcomes']) else 'discard'
            if e.pending_decision and not e.active_episodes:
                return self.finish_validation(e)
            return None

    def finish_validation(self, e):
        t = e.trial
        p, c = sum(t['production_outcomes'])/self.n, sum(t['candidate_outcomes'])/self.n
        e.validations.append((t['step']-e.production_step,c-p))
        event = dict(event='validation_finished',expert=e.expert,decision=e.pending_decision,
                     predicted=t['prediction']['success'], predicted_gain=t['prediction']['success']-t['prediction']['production_success'],
                     actual_candidate=c,actual_production=p,episodes_per_arm=self.n,step=t['step'],
                     old_version=e.production,candidate_version=e.candidate,epoch=e.epoch)
        if e.pending_decision=='promote':
            e.production=e.candidate
            e.production_step=e.step
            e.production_recent=deque(t['candidate_outcomes'],maxlen=50)
        e.candidate=None
        e.trial=None
        e.pending_decision=None
        self.events.append(event)
        return event

    def change_environment(self, expert):
        with self.lock:
            e=self.experts[expert]
            e.epoch+=1
            e.production_recent.clear()
            if e.trial:
                e.trial.update(epoch=e.epoch,candidate_outcomes=[],production_outcomes=[])
            self.events.append(dict(event='environment_changed',expert=expert,epoch=e.epoch))


class PeriodicActions:
    """At each 250ms boundary install only that cycle's on-time result.

    Keep all ten returned actions: normally execute the first five; on a missed
    update continue the remaining five, then hold pose. Never apply a late block.
    """
    def __init__(self):
        self.replies={};self.remaining=deque();self.gripper=-1.;self.lock=threading.Lock();self.source='production';self.last_source='hold'
    def offer(self,cycle,finished,actions,source='production'):
        with self.lock:self.replies[cycle]=(finished,actions,source)
    def boundary(self,cycle,deadline,allowed_source=None):
        with self.lock:
            reply=self.replies.pop(cycle,None)
            for old in [k for k in self.replies if k<cycle]:self.replies.pop(old)
            ontime=reply is not None and reply[0]<=deadline and (allowed_source is None or reply[2]==allowed_source)
            if ontime:self.remaining=deque(reply[1]);self.source=reply[2]
            return ontime
    def next(self):
        with self.lock:
            if self.remaining:
                action=self.remaining.popleft();self.gripper=float(action[-1]);self.last_source=self.source;return action,False
            self.last_source='hold'
            return [0.,0.,0.,0.,0.,0.,self.gripper],True
    def interrupt(self):
        with self.lock:self.remaining.clear()


class TakeoverRule:
    """Latched takeover: 20 control intervals below displacement threshold or 70% time."""
    def __init__(self, distance_m, steps=20, time_fraction=.7):
        self.distance_m=distance_m;self.steps=steps;self.time_fraction=time_fraction
        self.positions=deque(maxlen=steps+1);self.reason=None
    def update(self,position,elapsed,time_limit,success=False):
        self.positions.append(tuple(float(x) for x in position))
        if self.reason or success:return None
        span=None
        if len(self.positions)==self.steps+1:
            origin=self.positions[0]
            span=max(math.sqrt(sum((x-y)**2 for x,y in zip(p,origin))) for p in self.positions)
            if span<self.distance_m:self.reason='stagnation'
        if not self.reason and elapsed>=self.time_fraction*time_limit:self.reason='time_budget_70pct'
        return dict(reason=self.reason,max_displacement_m=span,elapsed_s=elapsed) if self.reason else None


def window_data_source(action_sources, intervened):
    if len(action_sources)!=10:return None
    if all(x=='operator' for x in action_sources):return 'operator_correction'
    if not intervened and all(x=='production' for x in action_sources):return 'self_success'
    return None


def request_key(robot, release):
    """Absolute release IDs avoid banker-rounding collisions at half-period phases."""
    return f'r{robot}:release_ns:{round(release*1_000_000_000)}'
