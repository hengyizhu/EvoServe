import os,sys,json,math,time,dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_overnight_20261104')
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
 elif family in ['lora32','lora32_heads','lora128_heads']:
  rank=128 if family=='lora128_heads' else 32
  for p in model.parameters():p.requires_grad_(False)
  for layer in model.llm.layers:
   for name in ['q_proj','k_proj','v_proj','o_proj']:
    modules=getattr(layer.attn,name)
    if modules[1] is not None:modules[1]=AdapterLinear(modules[1],rank)
   old=layer.mlps[1];new=AdapterFFN(old.features,old.hidden_dim,LoRAConfig(rank=rank,alpha=rank)).to(device=old.w_gating.device,dtype=old.w_gating.dtype);new.w_gating=old.w_gating;new.w_linear=old.w_linear;layer.mlps[1]=new
  if family in ['lora32_heads','lora128_heads']:
   norms=[norms[1] for layer in model.llm.layers for norms in [layer.pre_attention_norms,layer.pre_ffw_norms]]+[model.llm.final_norms[1]]
   for norm in norms:norm.ada_modulation=AdapterLinear(norm.ada_modulation,rank)
  for name,p in model.named_parameters():p.requires_grad_('lora_' in name or (family in ['lora32_heads','lora128_heads'] and name.split('.')[0] in ['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out']))
 else:raise ValueError(family)
 # Prefix runtime mode is fixed to serving mode. Augmentation is explicit in preprocessing.
 model.eval();pairs=[(n,p) for n,p in model.named_parameters() if p.requires_grad];assert pairs
 return pairs
def exposure_transform(model,brightness):
 original=model._input_transform_fn;split=json.loads(Path('/home/hengyi/Workspace/01_VLA-Serving/Environment-and-Earlier-Runs/vla_followup_20261020/data_split.json').read_text())
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

# Identical rank16/alpha16 SigLIP MLP adapter used by the training breakdown.
class VisionLoRA(torch.nn.Module):
 def __init__(self,base,rank=16):
  super().__init__();self.base=base;self.rank=rank
  self.A=torch.nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=base.weight.dtype));torch.nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
  self.B=torch.nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=base.weight.dtype))
 def forward(self,x):return self.base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,self.A),self.B)
setup_action_family=setup_family
def setup_family(model,family):
 if family not in ['lora32_vision16','lora128_vision16']:return setup_action_family(model,family)
 setup_action_family(model,'lora128_heads' if family=='lora128_vision16' else 'lora32_heads');modules=[]
 for name,module in list(model.img.named_modules()):
  if isinstance(module,torch.nn.Linear) and name.endswith(('mlp.fc1','mlp.fc2')):
   parent,leaf=name.rsplit('.',1);setattr(model.img.get_submodule(parent),leaf,VisionLoRA(module));modules.append('img.'+name)
 assert modules;model.vision_lora_modules=modules
 model.gradient_checkpointing_enable({'use_reentrant':False});model.llm.train();model.img.encoder.train()
 assert not any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in model.modules()),'unexpected stochastic dropout'
 return [(n,p) for n,p in model.named_parameters() if p.requires_grad]
