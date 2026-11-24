"""Exact two-sided McNemar test on matched initial states; preserve discordant episodes."""

import argparse
import json
import math
from pathlib import Path


def read(path):
    rows = list(map(json.loads, path.read_text().splitlines()))
    episodes = {r["episode"]: r for r in rows}
    if len(rows) != 50 or set(episodes) != set(range(50)):
        raise ValueError(f"Expected 50 unique episodes in {path}")
    return episodes


def compare(a, b):
    base, opt = read(a), read(b)
    flips = []
    for i, row in base.items():
        other = opt[i]
        assert (row["seed"], row["state_sha256"]) == (
            other["seed"],
            other["state_sha256"],
        )
        if row["success"] != other["success"]:
            flips.append(
                {
                    "episode": i,
                    "baseline": row["success"],
                    "optimized": other["success"],
                }
            )
    wins = sum(r["optimized"] for r in flips)
    losses = len(flips) - wins
    p = min(
        1,
        2
        * sum(math.comb(len(flips), j) for j in range(min(wins, losses) + 1))
        / 2 ** len(flips),
    )
    return {
        "baseline": sum(r["success"] for r in base.values()),
        "optimized": sum(r["success"] for r in opt.values()),
        "discordant": flips,
        "improved": wins,
        "regressed": losses,
        "mcnemar_exact_p": p,
        "passed": abs(wins - losses) <= 4 and p > 0.05,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("baseline", type=Path)
    p.add_argument("optimized", type=Path)
    p.add_argument("result", type=Path)
    a = p.parse_args()
    r = compare(a.baseline, a.optimized)
    a.result.write_text(json.dumps(r, indent=2))
    print(json.dumps(r))
    raise SystemExit(0 if r["passed"] else 1)
