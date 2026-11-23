"""Sample eight requests without replacement per robot episode."""

import random
import torch


def selected_requests(robots, periods, episode_length=104):
    chosen = set()
    for robot in range(robots):
        for start in range(0, periods, episode_length):
            span = min(episode_length, periods - start)
            # Sample the full episode; a truncated observation window only sees
            # the selections that actually arrive before its end.
            rng = random.Random(20261108 + robot + start * robots)
            for j in rng.sample(range(episode_length), 8):
                if j < span:
                    chosen.add((start + j) * robots + robot)
    return chosen


class SparseExport:
    def __init__(self, model):
        self.stream = torch.cuda.Stream()
        tensors = model.leaves(model.buckets[1][-1])
        self.buffers = [
            [torch.empty_like(t[:1], device="cpu", pin_memory=True) for t in tensors]
            for _ in range(16)
        ]
        self.events = [torch.cuda.Event() for _ in self.buffers]
        self.last = {}
        self.cursor = 0

    def before_graph(self, size, stream):
        if size in self.last:
            stream.wait_event(self.last[size])

    def submit(self, size, indices, tensors, prefix_ready):
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(prefix_ready)
            for index in indices:
                slot = self.cursor % len(self.buffers)
                for buf, src in zip(self.buffers[slot], tensors, strict=True):
                    buf.copy_(src[index : index + 1], non_blocking=True)
                self.events[slot].record(self.stream)
                self.last[size] = self.events[slot]
                self.cursor += 1
