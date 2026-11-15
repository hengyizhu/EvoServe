"""Post-measurement smoke: actual graph -> exported host prefix -> real LoRA update."""
import argparse, json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evoserve.io import save


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--out',required=True);p.add_argument('--operators');a=p.parse_args()
    os.sched_setaffinity(0,{0,1,2,3});out=Path(a.out);out.mkdir(parents=True)
    from evoserve.backend import GraphModel,TrainModel
    import torch
    graph=GraphModel(a.assets);graph.install(3,'3:v0')
    operator_check=None
    if a.operators:
        operators=json.loads(Path(a.operators).read_text())['operators'];graph.enable_operators({'3':operators['3']})
    raw=graph.dummy();actions,_=graph.infer(raw,3,'3:v0',20261108)
    assert actions.shape==(10,7),actions.shape
    if a.operators:
        operator_actions,operator_timing=graph.infer_operator(3,20261108+7919)
        graph.install(3,'operator_reference',operators['3']['checkpoint'])
        reference,_=graph.infer(raw,3,'operator_reference',20261108+7919)
        graph.np.testing.assert_allclose(operator_actions,reference,rtol=1e-5,atol=1e-5)
        restored,_=graph.infer(raw,3,'3:v0',20261108)
        graph.np.testing.assert_array_equal(actions,restored)
        operator_check=dict(prefix_reuse_max_abs_error=float(abs(operator_actions-reference).max()),production_restored_exactly=True,timing=operator_timing)
        graph.remove(3,'operator_reference')
    save(out/'graph_summary.json',dict(state='passed',returned_actions=10,operator_checked=operator_check is not None))
    # Smoke labels only: explicitly excluded from all learning/quality results.
    labels=actions
    prefix=graph.map(lambda q:q.detach().cpu().pin_memory(),graph.prefix)
    train=TrainModel(a.assets,[3,5],20261108)
    batch=[dict(raw=raw,actions=labels,prefix=prefix) for _ in range(8)]
    # Test independent experts and vmap; no gradients or optimizer steps leak.
    before={e:{n:q.detach().clone() for n,q in train.experts[e]['masters'].items()} for e in [3,5]}
    first=train.step({3:batch})
    assert train.experts[3]['step']==1 and train.experts[5]['step']==0
    assert all(torch.equal(before[5][n],q) for n,q in train.experts[5]['masters'].items())
    assert any(not torch.equal(before[3][n],q) for n,q in train.experts[3]['masters'].items())
    second=train.step({3:batch,5:batch})
    assert train.experts[3]['step']==2 and train.experts[5]['step']==1
    path=out/'smoke_candidate.pt';train.snapshot(3,path);graph.install(3,'3:c1',path)
    result,timing=graph.infer(raw,3,'3:c1',20261108)
    assert graph.np.isfinite(result).all()
    save(out/'summary.json',dict(state='passed',operator_check=operator_check,graph_actions_shape=list(result.shape),single=first,merged=second,candidate_timing=timing,independent_updates_verified=True,synthetic_labels_smoke_only=True))


if __name__=='__main__':main()
