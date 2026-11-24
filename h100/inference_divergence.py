"""Summarize actual low-light trajectories for discordant paired episodes."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read(path):
    return {
        r["episode"]: r["trace"] for r in map(json.loads, path.read_text().splitlines())
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("optimized", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    base, opt = read(a.baseline), read(a.optimized)
    assert base.keys() == opt.keys()
    fields = [
        "episode",
        "step",
        "action_mean_abs",
        "action_max_abs",
        "state_max_abs",
        "first_state_difference",
    ]
    with a.output.open("w") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for ep in sorted(base):
            first = True
            for b, o in zip(base[ep], opt[ep]):
                assert b["step"] == o["step"]
                delta = np.abs(np.asarray(b["action"]) - o["action"])
                state = float(np.max(np.abs(np.asarray(b["state"]) - o["state"])))
                writer.writerow(
                    {
                        "episode": ep,
                        "step": b["step"],
                        "action_mean_abs": delta.mean(),
                        "action_max_abs": delta.max(),
                        "state_max_abs": state,
                        "first_state_difference": first and state > 0,
                    }
                )
                if state > 0:
                    first = False


if __name__ == "__main__":
    main()
