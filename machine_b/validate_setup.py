from pathlib import Path
import sys,json,hashlib,tempfile,subprocess,ast
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'tools'));from init_workspace import materialize
for row in json.loads((R/'machine_b/source_manifest.json').read_text()):assert hashlib.sha256((R/row['destination']).read_bytes()).hexdigest()==row['sha256']
for p in (R/'machine_b').glob('*.py'):ast.parse(p.read_text())
with tempfile.TemporaryDirectory() as d:
 w=materialize(Path(d)/'work',Path(d)/'model',Path(d)/'data',Path(sys.executable).absolute());n=w/'pi05_machine_b_20261115'
 for p in n.rglob('*.py'):
  assert '/home/hengyi/' not in p.read_text(),str(p)
 subprocess.run([sys.executable,str(R/'machine_b/reproduce_book.py'),'--work',str(w),'--dry-run'],check=True,capture_output=True)
 s=json.loads((n/'sequences/book_1000.json').read_text());ref=json.loads((w/'pi05_extended_20261109/sequences/book_20261108_4000.json').read_text());assert s==ref[:1000]
 assert 'a.schedule_horizon or a.steps' in (n/'train_book.py').read_text()
import vla_portability as vp,os
calls=[];old=os.sched_setaffinity;os.sched_setaffinity=lambda pid,cpus:calls.append(cpus);os.environ.update(VLA_NODE_CPUS='[16,17]',VLA_CPU_POLICY='shared');vp.set_affinity(0,range(8,16));os.sched_setaffinity=old;assert calls==[{16,17}]
print(json.dumps(dict(source_integrity=True,relocation=True,book_first1000_sequence=True,cosine_horizon4000=True,small_cpu_allocation=True,gpu_training_executed=False)))
