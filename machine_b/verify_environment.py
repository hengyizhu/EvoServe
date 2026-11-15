"""Verify source pins and intentional transformer overlay; no model/GPU allocation."""
from pathlib import Path
import json,platform,importlib.metadata as md,subprocess,sys,argparse
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();ref=json.loads((R/'environment/manifest.json').read_text())
wrong=[]
for name,v in ref['packages'].items():
 try:got=md.version(name)
 except md.PackageNotFoundError:got=None
 if got!=v:wrong.append(dict(package=name,expected=v,actual=got))
import torch,transformers,numpy,mujoco,robosuite
check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True);expected=set((R/'environment/pip-check-source.txt').read_text().splitlines());unexpected=sorted(set(check.stdout.splitlines())-expected)
try:gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,power.limit','--format=csv'],text=True)
except Exception as e:gpu=str(e)
x=dict(python=platform.python_version(),version_mismatches=wrong,torch=torch.__version__,torch_cuda=torch.version.cuda,transformers_import=transformers.__version__,unexpected_dependency_conflicts=unexpected,source_known_dependency_conflicts=sorted(expected),gpu_inventory=gpu,sets_power_limit=False)
Path(a.out).write_text(json.dumps(x,indent=2)+'\n');assert x['python']==ref['python'];assert not wrong,wrong;assert not unexpected,unexpected;assert torch.__version__==ref['torch_import'] and transformers.__version__==ref['transformers_import'];print('Version pins and effective transformer import match. Known source dependency conflicts recorded.')
