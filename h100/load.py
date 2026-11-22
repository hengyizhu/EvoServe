"""Absolute open-loop arrivals. Every release contributes one latency observation."""

import bisect
import math
import multiprocessing as mp
import queue
import os
import subprocess
import threading
import time
import traceback
import numpy as np
from model import Server, Work


def worker(mode, rank, maximum, shared, ready, errors):
    try:
        w = Work(mode, rank, maximum)
        w.step(maximum)
        ready.set()
        deadline = 0.0
        while not shared["stop"].is_set():
            with shared["state"].get_lock():
                n, duty, pending, protected, enabled = shared["state"][:]
            if (
                not enabled
                or not n
                or not duty
                or (protected and pending)
                or time.monotonic() < deadline
            ):
                time.sleep(0.001)
                continue
            start = time.monotonic()
            w.step(int(n))
            finish = time.monotonic()
            shared["completed"].put((start, finish, int(n)))
            deadline = finish + (finish - start) * (1 / duty - 1)
    except BaseException:
        errors.put(traceback.format_exc())
        ready.set()


class Load:
    def __init__(self, mode=None, maximum=1, rank=32, sizes=(1, 2, 4, 8, 16)):
        self.server = Server(sizes)
        self.sizes = sizes
        ctx = mp.get_context("spawn")
        self.shared = {
            "state": ctx.Array("d", [1, 1, 0, 0, 0]),
            "stop": ctx.Event(),
            "completed": ctx.Queue(),
        }
        self.errors = ctx.Queue()
        self.job = None
        use_mps = mode == "cache" and os.environ.get("EVOSERVE_TRANSPORT") == "mps"
        if mode:
            ready = ctx.Event()
            cls = ctx.Process if use_mps else threading.Thread
            self.job = cls(
                target=worker,
                args=(mode, rank, maximum, self.shared, ready, self.errors),
                daemon=True,
            )
            self.job.start()
            if not ready.wait(180):
                raise TimeoutError("training worker initialization")
            self.check()
        self.mode = mode

        if use_mps:

            def control(cmd):
                return subprocess.run(
                    ["nvidia-cuda-mps-control"],
                    input=cmd + "\n",
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=True,
                ).stdout

            servers = control("get_server_list").split()
            clients = set()
            for pid in servers:
                if pid.isdigit():
                    clients.update(control("get_client_list " + pid).split())
            if not {str(os.getpid()), str(self.job.pid)}.issubset(clients):
                self.close()
                raise RuntimeError(f"MPS did not confirm both client PIDs: {clients}")

    def check(self):
        if self.job and not self.job.is_alive():
            try:
                msg = self.errors.get(timeout=2)
            except queue.Empty:
                msg = "worker exited"
            raise RuntimeError(msg)

    def close(self):
        self.shared["stop"].set()
        if self.job:
            self.job.join(30)
            if self.job.is_alive() and hasattr(self.job, "terminate"):
                self.job.terminate()

    def run(self, rates, seconds, n=1, sync=1, table=None, service=0.0, feedback=False):
        state = self.shared["state"]
        # Each phase has a deterministic arrival process. sync>1 releases a burst
        # from that many robots every 250 ms at capacity.
        releases = []
        for phase, rate in enumerate(rates):
            for i in range(math.ceil(seconds * rate - 1e-9)):
                releases.append(phase * seconds + (i // sync) * sync / rate)
        total = len(releases)
        duration = seconds * len(rates)
        origin = time.monotonic() + 0.05
        complete = [None] * total
        records = []
        controls = []
        steps = []
        index = 0
        next_control = 5.0
        next_rate = 1.0
        duty = 1.0
        m = n

        def update_pending():
            # Independently publish newly queued requests while GPU kernels run.
            while not done.wait(0.0008):
                with state.get_lock():
                    state[2] = max(
                        0,
                        bisect.bisect_right(releases, time.monotonic() - origin)
                        - dispatched[0],
                    )

        done = threading.Event()
        dispatched = [0]
        with state.get_lock():
            state[:] = [m, duty, 0, int(feedback), 0]
        watcher = threading.Thread(target=update_pending, daemon=True)
        watcher.start()
        try:
            time.sleep(max(0, origin - time.monotonic()))
            with state.get_lock():
                state[4] = 1
            while index < total:
                self.check()
                now = time.monotonic() - origin
                with state.get_lock():
                    state[4] = int(now < duration)
                if feedback and now >= next_control + 0.25:
                    lo = bisect.bisect_left(releases, next_control - 5)
                    hi = bisect.bisect_left(releases, next_control)
                    if hi > lo:
                        miss = sum(
                            complete[j] is None or complete[j] - releases[j] > 0.250
                            for j in range(lo, hi)
                        ) / (hi - lo)
                        duty = max(0.0, min(1.0, duty - 5 * (miss - 0.01)))
                    next_control += 5
                if table and now >= next_rate:
                    rate = bisect.bisect_right(releases, now) - bisect.bisect_right(
                        releases, now - 1
                    )
                    load = rate * service
                    keys = sorted(table)
                    k = bisect.bisect_left(keys, load)
                    m = table[keys[min(k, len(keys) - 1)]]
                    if 0 < k < len(keys) and abs(keys[k] - load) > 1e-8:
                        m = min(m, table[keys[k - 1]])
                    next_rate = math.floor(now) + 1
                with state.get_lock():
                    state[0] = m
                    state[1] = duty
                if not controls or controls[-1][1:] != [m, duty]:
                    controls.append([max(0.0, now), m, duty])
                if now < releases[index]:
                    time.sleep(min(0.002, releases[index] - now))
                    continue
                limit = max(self.sizes)
                end = bisect.bisect_right(
                    releases, now, index, min(total, index + limit)
                )
                if (
                    end - index < limit
                    and now - releases[index] < 0.010
                    and end < total
                ):
                    time.sleep(0.0005)
                    continue
                dispatched[0] = end
                self.server.serve(end - index)
                finish = time.monotonic() - origin
                for j in range(index, end):
                    complete[j] = finish
                records.append([index, end, now, finish])
                index = end
            time.sleep(max(0.0, origin + duration - time.monotonic()))
        finally:
            with state.get_lock():
                state[4] = 0
            done.set()
            watcher.join()
        # Let an in-flight update finish; use completion timestamps for throughput.
        time.sleep(0.02)
        while True:
            try:
                s, f, count = self.shared["completed"].get_nowait()
                steps.append((s - origin, f - origin, count))
            except queue.Empty:
                break
        latency = 1000 * (np.array(complete) - np.array(releases))
        phases = []
        for phase, rate in enumerate(rates):
            lo = phase * seconds
            hi = lo + seconds
            ids = slice(
                bisect.bisect_left(releases, lo), bisect.bisect_left(releases, hi)
            )
            ms = latency[ids]
            active = sum(c for _, f, c in steps if lo <= f < hi)
            duty_time = sum(
                max(0, min(hi, end) - max(lo, start)) * d
                for (start, _, d), end in zip(
                    controls, [x[0] for x in controls[1:]] + [duration]
                )
            )
            phases.append(
                dict(
                    rate_hz=rate,
                    requests=len(ms),
                    p99_ms=float(np.percentile(ms, 99)),
                    slo=float(np.mean(ms <= 250)),
                    expert_steps_s=active / seconds if self.mode != "rebuild" else None,
                    samples_s=4 * active / seconds if self.mode == "rebuild" else None,
                    mean_duty=duty_time / seconds if self.job else 0,
                )
            )
        return dict(
            phases=phases,
            dispatch=records,
            controls=controls,
            seconds_per_phase=seconds,
        )
