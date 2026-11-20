"""Serialized inference requests from both loops share one operator model."""

import argparse
import threading
from pathlib import Path
import torch
from .batch_backend import BatchedGraphModel
from .io import RPCServer, read
from .operators import load_operator


def serve(socket):
    root = Path(__file__).resolve().parents[1] / "configs"
    op = BatchedGraphModel(read(root / "machine_a.example.json")["assets"], (1, 2, 4))
    for e, spec in enumerate(read(root / "accelerated.machine_a.json")[:2]):
        weights, _ = load_operator(spec["checkpoint"], op.pairs, spec["task_id"])
        op.slots[e, "operator"] = {
            k: v.to(device="cuda", dtype=op.pairs[k].dtype) for k, v in weights.items()
        }
    lock = threading.Lock()

    def handle(request):
        assert request["op"] == "actions"
        with lock, torch.cuda.device(0):
            return op.infer_batch(request["messages"])[0]

    server = RPCServer(socket, handle)
    threading.Event().wait()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("socket")
    serve(p.parse_args().socket)
