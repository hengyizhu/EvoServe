"""Explicit CPU allocation translation for materialized experiment scripts."""
import os,json

def allowed_gpus():return [int(x) for x in os.environ['VLA_ALLOWED_GPUS'].split(',')]
def set_affinity(pid,virtual_cpus):
 pool=json.loads(os.environ['VLA_NODE_CPUS']);desired=sorted(set(virtual_cpus))
 if not desired or min(desired)<0 or max(desired)>=len(pool):raise RuntimeError(f'CPU allocation has {len(pool)} slots, requested virtual slots {desired}; reserve a whole node or choose smaller GPU ordinals')
 os.sched_setaffinity(pid,{pool[i] for i in desired})
