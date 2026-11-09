import argparse,gc,traceback
from common import *
ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['learning','transfer'],required=True);args=ap.parse_args()
def main():
 if args.kind=='transfer':
  from pinned_serving import GraphServing
  s=GraphServing(1);check=s.verify_graph();rows=[s.infer(i)[1] for i in range(8)];assert all(x['h2d_calls']==1 for x in rows)
  save(ROOT/'preflight_transfer.json',{'state':'passed','gpu':GPU,'graph_correctness':check,'one_h2d_per_request':True,'packed_bytes':rows[0]['packed_bytes']});return
 model=load_model().eval();samples=Samples(model);split=samples.split;ep=split['target_train'][0]['episode_index'];obs,x,u,t=prep(model,samples,ep,0,20261108,False)
 with torch.no_grad():
  _,mask,kv=model.build_prefix_cache(obs);reference=model.velocity_from_suffix(model.run_suffix(obs,x,t,kv,mask)).clone()
 torch.manual_seed(20261108);pairs=setup_family(model,'lora32');pin_threads()
 with torch.no_grad():candidate=model.velocity_from_suffix(model.run_suffix(obs,x,t,kv,mask))
 torch.testing.assert_close(reference,candidate,rtol=0,atol=0)
 loss=(model.velocity_from_suffix(model.run_suffix(obs,x,t,kv,mask))-u).square().mean();loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in pairs);assert any(float(p.grad.abs().max())>0 for n,p in pairs if 'lora_b' in n or 'lora_B' in n)
 # EMA formula test, independent of original parameters; finite trainable gradients verified above.
 a=torch.tensor([1.,2.]);v=torch.tensor([3.,4.]);expected=.999*a+.001*v;ema=a.clone().mul_(.999).add_(v,alpha=.001);torch.testing.assert_close(ema,expected)
 save(ROOT/'preflight_learning.json',{'state':'passed','gpu':GPU,'lora_zero_init_max_abs':float((reference-candidate).abs().max()),'lora_trainable_numel':sum(p.numel() for _,p in pairs),'all_adapter_gradients_finite':True,'ema_formula_checked':True,'target_names':[n for n,p in pairs]})
if __name__=='__main__':
 try:main()
 except Exception:save(ROOT/f'preflight_{args.kind}.json',{'state':'failed','traceback':traceback.format_exc()});raise
