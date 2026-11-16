"""Frozen, full-demonstration action experts used as simulated human operators."""
from pathlib import Path
import hashlib

REVISION='6222623f635769bfc73c9472e29fab9b7fd8e027'


def load_operator(path, expected_keys=None, expert=None):
    import torch
    checkpoint=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if checkpoint['parent']['revision']!=REVISION:raise ValueError('Operator parent revision mismatch')
    if checkpoint['weights_kind']!='EMA':raise ValueError('Expected operator EMA weights')
    if expert is not None and checkpoint['eval_task_id']!=expert:raise ValueError('Operator task mismatch')
    weights=checkpoint['parameters'];action={n:v for n,v in weights.items() if not n.startswith('img.')}
    extras={n:v for n,v in weights.items() if n.startswith('img.')}
    for name,value in extras.items():
        if not (name.endswith('.A') or name.endswith('.B')):raise ValueError('Unexpected visual parameter: '+name)
        if name.endswith('.B') and torch.count_nonzero(value).item():raise ValueError('Operator changed vision; cannot reuse production prefix: '+name)
    if expected_keys is not None and set(action)!=set(expected_keys):raise ValueError('Operator action module keys differ')
    if sum(v.numel() for v in action.values())!=20876320:raise ValueError('Operator is not the matching action LoRA r32 + heads')
    for name,value in action.items():
        if not torch.isfinite(value).all():raise ValueError('Nonfinite operator weight: '+name)
    return action,checkpoint


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def operator_token(source, ident):
    return hashlib.sha256(f'{source}:{ident}'.encode()).hexdigest()[:32]


def wait_operator_result(reference, deadline):
    import time,json
    path=Path(reference['path'])
    while time.monotonic()<=deadline:
        if path.exists():
            result=json.loads(path.read_text())
            if result['request_id']!=reference['request_id']:raise ValueError('Operator result/request mismatch')
            path.unlink(missing_ok=True);return result
        time.sleep(.0005)
    return dict(status='operator_deadline_miss')
