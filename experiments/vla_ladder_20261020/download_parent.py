"""Download the fixed parent once, with upstream LFS hash verification."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parent
repo = "RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT"
def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)

revision = get_json(f"https://huggingface.co/api/models/{repo}")["sha"]
files = get_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true")
target = root / "parent"
target.mkdir(exist_ok=True)
manifest = {"repo": repo, "revision": revision, "files": []}
(root / "audit/parent_source.json").write_text(json.dumps(
    {"repo": repo, "revision": revision, "upstream_files": files}, indent=2))
for entry in files:
    name = entry["path"]
    if name not in ("model.safetensors", "physical-intelligence/libero/norm_stats.json"):
        continue
    path = target / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with urllib.request.urlopen(
        f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=120
    ) as response, temp.open("wb") as output:
        while block := response.read(8 * 1024 * 1024):
            digest.update(block)
            output.write(block)
    assert temp.stat().st_size == entry["size"], name
    sha = digest.hexdigest()
    if "lfs" in entry:
        assert sha == entry["lfs"]["oid"], name
    temp.rename(path)
    path.chmod(0o444)
    manifest["files"].append({"path": name, "bytes": entry["size"], "sha256": sha})
    (root / "audit/parent_download.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified {name}: {entry['size']} bytes", flush=True)
