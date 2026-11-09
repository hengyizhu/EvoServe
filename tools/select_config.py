"""Set explicit runtime inputs, never claim a learning gate passed without evaluation."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--family',choices=['full','lora32_heads']);p.add_argument('--lr',type=float);p.add_argument('--gate-evaluation');p.add_argument('--brightness',type=float,default=.05);a=p.parse_args();w=Path(a.work)
if a.family:
 assert a.lr is not None
 target=w/'pi05_lora_lr_20261108/selection.json'
 if target.exists():raise FileExistsError(target)
 target.write_text(json.dumps({'state':'selected','family':a.family,'lr':a.lr,'source':'explicit reproduction configuration; not a stability result'},indent=2))
if a.gate_evaluation:
 rows=[json.loads(x) for x in Path(a.gate_evaluation).read_text().splitlines()];rows=[x for x in rows if x['task']==3];assert sorted(x['index'] for x in rows)==list(range(50));k=sum(x['success'] for x in rows);assert k<25,'parent must be below50%'
 target=w/'pi05_stability_round2/task_gate.json'
 if target.exists():raise FileExistsError(target)
 target.write_text(json.dumps({'state':'passed','brightness':a.brightness,'parent_successes':k,'episodes':50,'parent_rate':k/50,'source':str(Path(a.gate_evaluation).resolve())},indent=2))
