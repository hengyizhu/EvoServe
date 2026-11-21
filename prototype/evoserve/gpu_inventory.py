"""Snapshot the physical cards and record compute processes for placement decisions."""

import csv
import subprocess
from pathlib import Path
import pwd


def query(fields, kind):
    text = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-{kind}=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [list(map(str.strip, row)) for row in csv.reader(text.splitlines()) if row]


def inventory():
    cards = {
        int(i): dict(index=int(i), uuid=u, free_mib=int(m), processes=[])
        for i, u, m in query(["index", "uuid", "memory.free"], "gpu")
    }
    by_uuid = {c["uuid"]: c for c in cards.values()}
    for u, pid in query(["gpu_uuid", "pid"], "compute-apps"):
        try:
            owner = pwd.getpwuid(Path("/proc", pid).stat().st_uid).pw_name
        except (FileNotFoundError, KeyError):
            owner = "unknown"
        by_uuid[u]["processes"].append(dict(pid=int(pid), owner=owner))
    return cards
