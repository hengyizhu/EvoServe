"""Run the RLinf install driver with paths and hardware adapted to this host."""
import importlib.util
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
repo = root / "RLinf"
os.chdir(repo)
spec = importlib.util.spec_from_file_location(
    "install_driver", repo / ".agents/skills/test-install/driver.py"
)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
job = next(j for j in driver.load_jobs()
           if j.name == "embodied-openpi-maniskill-libero-test")
job.install_run = job.install_run.replace(
    "/workspace/dataset/.uv_cache", str(root / "install_cache")
).replace("/workspace/dataset/.uv_python", "/home/hengyi/.local/share/uv/python"
).replace("/workspace/dataset/.uv", "/home/hengyi/.local/bin"
).replace("UV_LINK_MODE=symlink", "UV_LINK_MODE=hardlink"
).replace("--env maniskill_libero", "--env libero --no-root")
os.environ["LIBERO_CONFIG_PATH"] = str(root / "libero_config")
os.environ["DOWNLOAD_DIR"] = str(root / "assets")
os.environ["MAX_JOBS"] = "4"
script = driver._install_script(job, str(root / "venv"), mirror=False)
(root / "audit/install_command.sh").write_text(script)
sys.exit(driver._run_shell(script, "--dry-run" in sys.argv))
