import os,sys,json,math,time,dataclasses
from pathlib import Path
ROOT=Path('/home/hengyi/pi05_machine_b_20261115')
sys.path.insert(0,'/home/hengyi/vla_overnight_20261104')
from core import load_model,configure,Samples,fingerprint,model_module,torch,np,GPU
from torch.utils._pytree import tree_map,tree_leaves
from rlinf.models.embodiment.openpi.modules.lora import FeedForward as AdapterFFN,LoRAConfig
def save(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(path)
def cat_obs(items):
 return type(items[0])(**{f.name:tree_map(lambda *x:torch.cat(x,0) if torch.is_tensor(x[0]) else x[0],*[getattr(o,f.name) for o in items]) for f in dataclasses.fields(items[0])})
class AdapterLinear(torch.nn.Module):
 def __init__(self,base,rank=32):
  super().__init__();self.base=base;self.rank=rank;self.scale=1.
  self.lora_A=torch.nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype));self.lora_B=torch.nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype));torch.nn.init.normal_(self.lora_A,std=.01)
 @property
 def weight(self):return self.base.weight
 @property
 def bias(self):return self.base.bias
 def forward(self,x):
  out=self.base(x);delta=torch.nn.functional.linear(torch.nn.functional.linear(x.to(self.lora_A.dtype),self.lora_A),self.lora_B);return out+delta.to(out.dtype)*self.scale
def setup_family(model,family):
 if family=='full':configure(model,'L2')
 elif family in ['lora32','lora32_heads']:
  for p in model.parameters():p.requires_grad_(False)
  for layer in model.llm.layers:
   for name in ['q_proj','k_proj','v_proj','o_proj']:
    modules=getattr(layer.attn,name)
    if modules[1] is not None:modules[1]=AdapterLinear(modules[1])
   old=layer.mlps[1];new=AdapterFFN(old.features,old.hidden_dim,LoRAConfig(rank=32,alpha=32)).to(device=old.w_gating.device,dtype=old.w_gating.dtype);new.w_gating=old.w_gating;new.w_linear=old.w_linear;layer.mlps[1]=new
  if family=='lora32_heads':
   norms=[norms[1] for layer in model.llm.layers for norms in [layer.pre_attention_norms,layer.pre_ffw_norms]]+[model.llm.final_norms[1]]
   for norm in norms:norm.ada_modulation=AdapterLinear(norm.ada_modulation)
  for name,p in model.named_parameters():p.requires_grad_('lora_' in name or (family=='lora32_heads' and name.split('.')[0] in ['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out']))
 else:raise ValueError(family)
 # Prefix runtime mode is fixed to serving mode. Augmentation is explicit in preprocessing.
 model.eval();pairs=[(n,p) for n,p in model.named_parameters() if p.requires_grad];assert pairs
 return pairs
def exposure_transform(model,brightness):
 original=model._input_transform_fn;split=json.loads(Path('/home/hengyi/vla_followup_20261020/data_split.json').read_text())
 def transform(raw):
  if raw['prompt']==split['target']:
   raw=dict(raw)
   for key in ['observation/image','observation/wrist_image']:raw[key]=(raw[key].astype(np.float32)*brightness).clip(0,255).astype(np.uint8)
  return original(raw)
 model._input_transform_fn=transform
def prep(model,samples,ep,start,seed,augment):
 ob,ac=samples.get(ep,start);ac=ac.float();ob=model_module.preprocess_observation(ob,train=augment,rng=torch.Generator().manual_seed(seed));ob=model_module._observation_to_dtype(ob,model.embed_dtype)
 noise=torch.randn(ac.shape,generator=torch.Generator().manual_seed(seed+100000)).cuda();t=torch.tensor(np.random.default_rng(seed+200000).beta(1.5,1.,size=1)*.999+.001,device='cuda',dtype=torch.float32)
 return ob,t[:,None,None]*noise+(1-t[:,None,None])*ac,noise-ac,t
def pin_threads(cpus=None):
 cpus=set(range(GPU*8,GPU*8+8)) if cpus is None else set(cpus)
 for p in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(p.name),cpus)
  except ProcessLookupError:pass
