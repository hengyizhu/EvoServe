"""Download only the episodes used here, and construct fixed labeled action windows."""

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_REV = "9dfa69510ea9e1613fc54112bc706444b686a231"


def prepare(work, local):
    assets = work / "vla_followup_20261020"
    bench = assets / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    data = local if local else work / "data"
    if not local:
        from huggingface_hub import hf_hub_download

        (work / "workspace.json").write_text(
            json.dumps({"model": str(assets / "parent"), "data": str(data)})
        )
        for kind in ("model", "tokenizer"):
            subprocess.run(
                [
                    sys.executable,
                    ROOT / "tools/download_assets.py",
                    "--work",
                    work,
                    "--only",
                    kind,
                ],
                check=True,
            )
    manifest = json.loads(
        (
            ROOT / "experiments/vla_followup_20261020/benchmark/input_manifest.json"
        ).read_text()
    )
    images, wrists, states, prompts, actions, ids = [], [], [], [], [], []
    for ep in sorted({x["episode"] for x in manifest["samples"]}):
        file = f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        if not local:
            hf_hub_download(
                "physical-intelligence/libero",
                file,
                repo_type="dataset",
                revision=DATA_REV,
                local_dir=data,
            )
        table = pq.read_table(
            data / file, columns=["image", "wrist_image", "state", "actions"]
        )
        for sample in manifest["samples"]:
            frame = sample["frame"]
            if sample["episode"] != ep or frame + 10 > len(table):
                continue
            row = table.slice(frame, 1).to_pylist()[0]
            for key, output in [("image", images), ("wrist_image", wrists)]:
                output.append(
                    np.asarray(Image.open(io.BytesIO(row[key]["bytes"])).convert("RGB"))
                )
            states.append(row["state"])
            prompts.append(sample["task"])
            actions.append(table["actions"].slice(frame, 10).to_pylist())
            ids.append([ep, frame])
    assert len(actions) >= 16
    np.savez(
        bench / "inputs.npz",
        images=images,
        wrists=wrists,
        states=np.asarray(states, np.float32),
        prompts=prompts,
        actions=np.asarray(actions, np.float32),
        indices=ids,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("work", type=Path)
    p.add_argument("--local-data", type=Path)
    a = p.parse_args()
    prepare(a.work.resolve(), a.local_data)
