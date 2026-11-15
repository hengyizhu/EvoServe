"""Actual pinned π0.5 model, CUDA Graph serving and independently optimized LoRAs."""
from pathlib import Path
import dataclasses
import inspect
import math
import os
import sys
import textwrap
import time
import types


def bootstrap(assets, threads):
    repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo/'vendor/RLinf'))
    assets=Path(assets)
    os.environ.setdefault('TOKENIZER_DIR',str(assets/'assets/.cache/openpi'))
    os.environ.setdefault('LIBERO_CONFIG_PATH',str(assets/'libero_config'))
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    import torch
    torch.set_num_threads(threads)
    return repo,assets,torch


def setup_action(model):
    import torch
    from rlinf.models.embodiment.openpi.modules.lora import FeedForward,LoRAConfig
    class Linear(torch.nn.Module):
        def __init__(self,base):
            super().__init__();self.base=base
            self.lora_A=torch.nn.Parameter(torch.empty(32,base.in_features,device=base.weight.device,dtype=base.weight.dtype))
            self.lora_B=torch.nn.Parameter(torch.zeros(base.out_features,32,device=base.weight.device,dtype=base.weight.dtype))
            torch.nn.init.normal_(self.lora_A,std=.01)
        @property
        def weight(self):return self.base.weight
        @property
        def bias(self):return self.base.bias
        def forward(self,x):
            out=self.base(x)
            return out+torch.nn.functional.linear(torch.nn.functional.linear(x.to(self.lora_A.dtype),self.lora_A),self.lora_B).to(out.dtype)
    for p in model.parameters():p.requires_grad_(False)
    for layer in model.llm.layers:
        for name in ['q_proj','k_proj','v_proj','o_proj']:
            modules=getattr(layer.attn,name)
            if modules[1] is not None:modules[1]=Linear(modules[1])
        old=layer.mlps[1]
        new=FeedForward(old.features,old.hidden_dim,LoRAConfig(rank=32,alpha=32)).to(device=old.w_gating.device,dtype=old.w_gating.dtype)
        new.w_gating=old.w_gating;new.w_linear=old.w_linear;layer.mlps[1]=new
    for norm in [n[1] for layer in model.llm.layers for n in [layer.pre_attention_norms,layer.pre_ffw_norms]]+[model.llm.final_norms[1]]:
        norm.ada_modulation=Linear(norm.ada_modulation)
    pairs={n:p for n,p in model.named_parameters() if 'lora_' in n or n.split('.')[0] in ['action_in_proj','action_out_proj','time_mlp_in','time_mlp_out']}
    for p in model.parameters():p.requires_grad_(False)
    assert sum(p.numel() for p in pairs.values())==20876320
    return pairs


class Model:
    def __init__(self,assets,threads=1):
        repo,self.assets,self.torch=bootstrap(assets,threads)
        import numpy as np
        from omegaconf import OmegaConf
        from rlinf.models.embodiment.openpi import get_model
        from rlinf.models.embodiment.openpi.modules import model as mod
        from torch.utils._pytree import tree_map,tree_leaves
        self.np,self.mod,self.map,self.leaves=np,mod,tree_map,tree_leaves
        cfg=OmegaConf.load(repo/'vendor/RLinf/examples/embodiment/config/model/pi0_5.yaml')
        cfg.model_path=str(self.assets/'parent');cfg.num_action_chunks=5;cfg.num_steps=10
        cfg.openpi.task='eval';cfg.openpi.action_horizon=10;cfg.openpi.action_chunk=10;cfg.openpi.discrete_state_input=False;cfg.openpi.torch_compile=False
        cfg.openpi_data={'norm_stats_path':str(self.assets/'parent/physical-intelligence/libero/norm_stats.json')}
        self.torch.manual_seed(20261108)
        self.model=get_model(cfg).cuda().eval();self.pairs=setup_action(self.model)

    def obs_map(self,fn,obs):
        return type(obs)(**{f.name:self.map(lambda x:fn(x) if self.torch.is_tensor(x) else x,getattr(obs,f.name)) for f in dataclasses.fields(obs)})

    def prepare(self,raw,actions=None):
        t=self.torch;np=self.np
        item=dict(raw)
        if actions is not None:item['actions']=np.asarray(actions,dtype=np.float32)
        processed=self.model._input_transform_fn(item)
        batch=self.map(lambda x:t.as_tensor(np.asarray(x).copy())[None],processed)
        labels=batch.pop('actions',None)
        obs=self.mod.Observation.from_dict(batch);obs=dataclasses.replace(obs,state=obs.state.float())
        obs=self.mod.preprocess_observation(obs,train=False)
        obs=self.mod._observation_to_dtype(obs,self.model.embed_dtype)
        return self.obs_map(lambda x:x.contiguous().pin_memory(),obs),labels.float().pin_memory() if labels is not None else None

    def cat_obs(self,items):
        t=self.torch
        return type(items[0])(**{f.name:self.map(lambda *v:t.cat(v,0) if t.is_tensor(v[0]) else v[0],*[getattr(x,f.name) for x in items]) for f in dataclasses.fields(items[0])})

    def dummy(self):
        with self.np.load(self.assets/'benchmark/inputs.npz',allow_pickle=False) as z:
            return {'observation/image':z['images'][0], 'observation/wrist_image':z['wrists'][0], 'observation/state':z['states'][0], 'prompt':str(z['prompts'][0])}


class GraphModel(Model):
    def __init__(self,assets):
        super().__init__(assets,1);t=self.torch
        self.stream=t.cuda.Stream(priority=-1)
        obs,_=self.prepare(self.dummy());self.static=self.obs_map(lambda x:x.to('cuda'),obs)
        self.noise=t.randn((1,10,32),device='cuda');self.graph=t.cuda.CUDAGraph()
        self.events=[t.cuda.Event(enable_timing=True,external=True) for _ in range(12)]
        def replace(module,name,old,new):
            method=getattr(module,name);code=textwrap.dedent(inspect.getsource(method))
            if old not in code:
                assert new in code,(name,'unexpected graph adaptation source');return
            ns=dict(method.__func__.__globals__);exec(compile(code.replace(old,new),'<graph_constants>','exec'),ns)
            setattr(module,name,types.MethodType(ns[name],module))
        replace(self.model,'embed_prefix','torch.tensor(ar_mask, device=tokens.device)','torch.zeros(len(ar_mask), dtype=torch.bool, device=tokens.device)')
        replace(self.model,'embed_suffix','ar_mask[0] = True','ar_mask[:1].fill_(True)')
        for layer in self.model.llm.layers:
            replace(layer.attn,'forward','torch.tensor(big_neg, dtype=logits.dtype, device=logits.device)','torch.full((), big_neg, dtype=logits.dtype, device=logits.device)')
        self.stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(self.stream),t.inference_mode():
            eager=self.model.sample_actions(self.static,num_steps=10,noise=self.noise).clone()
            for _ in range(3):self.work()
        self.stream.synchronize()
        with t.cuda.graph(self.graph,stream=self.stream),t.inference_mode():self.result=self.work()
        with t.cuda.stream(self.stream),t.inference_mode():self.graph.replay()
        self.stream.synchronize();t.testing.assert_close(eager,self.result,rtol=1e-6,atol=1e-6)
        self.slots={};self.loaded=None
        self.initial={n:p.detach().clone() for n,p in self.pairs.items()}

    def enable_operators(self, operators):
        from .operators import load_operator,sha256
        t=self.torch;self.operator_slots={}
        for expert,record in operators.items():
            expert=int(expert)
            if sha256(record['checkpoint'])!=record['sha256']:raise ValueError('Operator checksum changed')
            delta,_=load_operator(record['checkpoint'],self.pairs,expert)
            self.operator_slots[expert]={n:q.to(device='cuda',dtype=self.pairs[n].dtype) for n,q in delta.items()}
        self.operator_noise=t.randn((1,10,32),device='cuda');self.operator_graph=t.cuda.CUDAGraph()
        self.operator_events=[t.cuda.Event(enable_timing=True,external=True) for _ in range(2)]
        self.stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(self.stream),t.inference_mode():
            for _ in range(3):self.operator_work()
        self.stream.synchronize()
        with t.cuda.graph(self.operator_graph,stream=self.stream),t.inference_mode():self.operator_result=self.operator_work()
        self.stream.synchronize()

    def operator_work(self):
        t=self.torch;self.operator_events[0].record();mask,kv=self.prefix;x=self.operator_noise
        for j in range(10):
            tt=t.full((1,),1-j/10,device='cuda',dtype=t.float32)
            x=x-.1*self.model.velocity_from_suffix(self.model.run_suffix(self.static,x,tt,kv,mask))
        self.operator_events[1].record();return x

    def infer_operator(self,expert,seed):
        t=self.torch;start=time.monotonic_ns()
        with t.cuda.stream(self.stream),t.inference_mode():
            for n,p in self.pairs.items():p.copy_(self.operator_slots[expert][n])
            self.loaded=(expert,'operator')
            self.operator_noise.copy_(t.randn((1,10,32),device='cuda',generator=t.Generator(device='cuda').manual_seed(seed)))
            self.operator_graph.replay()
            result=self.model.decode_actions(self.operator_result,self.static.state).cpu()
            self.operator_events[-1].synchronize()
        return result[0].numpy(),dict(operator_gpu_ms=self.operator_events[0].elapsed_time(self.operator_events[1]),operator_wall_ms=(time.monotonic_ns()-start)/1e6,operator_recomputed_prefix=False)

    def work(self):
        t=self.torch;self.events[0].record()
        _,mask,kv=self.model.build_prefix_cache(self.static);self.prefix=(mask,kv);self.events[1].record();x=self.noise
        for j in range(10):
            tt=t.full((1,),1-j/10,device='cuda',dtype=t.float32)
            x=x-.1*self.model.velocity_from_suffix(self.model.run_suffix(self.static,x,tt,kv,mask));self.events[j+2].record()
        return x

    def install(self,expert,version,path=None):
        t=self.torch
        delta=t.load(path,map_location='cpu',weights_only=True)['parameters'] if path else self.initial
        assert set(delta)==set(self.pairs)
        with t.cuda.stream(self.stream),t.no_grad():
            self.slots[(expert,version)]={n:q.detach().to(device='cuda',dtype=self.pairs[n].dtype).clone() for n,q in delta.items()}
        self.stream.synchronize()

    def remove(self,expert,version):
        self.slots.pop((expert,version),None)
        if self.loaded==(expert,version):self.loaded=None

    def infer(self,raw,expert,version,seed):
        t=self.torch;start=time.monotonic_ns()
        obs,_=self.prepare(raw)
        with t.cuda.stream(self.stream),t.inference_mode():
            if self.loaded!=(expert,version):
                for n,p in self.pairs.items():p.copy_(self.slots[(expert,version)][n])
                self.loaded=(expert,version)
            static_fields={f.name:getattr(self.static,f.name) for f in dataclasses.fields(self.static)}
            fields={f.name:getattr(obs,f.name) for f in dataclasses.fields(obs)}
            for a,b in zip(self.leaves(static_fields),self.leaves(fields)):
                if t.is_tensor(a):a.copy_(b,non_blocking=True)
            self.noise.copy_(t.randn((1,10,32),device='cuda',generator=t.Generator(device='cuda').manual_seed(seed)))
            self.graph.replay()
            actions=self.model.decode_actions(self.result,self.static.state).cpu()
            self.events[-1].synchronize()
        return actions[0].numpy(),dict(start_ns=start,finish_ns=time.monotonic_ns(),prefix_ms=self.events[0].elapsed_time(self.events[1]),denoise_ms=self.events[1].elapsed_time(self.events[-1]))


class TrainModel(Model):
    def __init__(self,assets,experts,seed):
        super().__init__(assets,4);t=self.torch;self.stream=t.cuda.Stream(priority=0);self.copy=t.cuda.Stream(priority=0)
        self.experts={};self.seed=seed
        class Loss(t.nn.Module):
            def __init__(self,model):super().__init__();self.model=model
            def forward(self,fields,actions,noise,times,mask,kv):
                obs=self_outer.mod.Observation(**fields);x=times[:,None,None]*noise+(1-times[:,None,None])*actions
                v=self.model.velocity_from_suffix(self.model.run_suffix(obs,x,times,kv,mask))
                return (v-(noise-actions)).square().mean()
        self_outer=self;self.wrapper=Loss(self.model)
        for expert in experts:
            masters={n:t.nn.Parameter(p.detach().float().clone()) for n,p in self.pairs.items()}
            self.experts[expert]=dict(masters=masters,ema={n:p.detach().clone() for n,p in masters.items()},optim=t.optim.AdamW(list(masters.values()),lr=5e-5,betas=(.9,.95),eps=1e-8,weight_decay=1e-10,foreach=True),step=0,lr_step=0)

    def step(self,batches,recompute=False):
        t=self.torch;ids=list(batches);start=time.monotonic_ns();prepared={}
        for eid,items in batches.items():
            obs,labels=zip(*(self.prepare(x['raw'],x['actions']) for x in items))
            with t.cuda.stream(self.copy):
                prepared[eid]=(self.cat_obs([self.obs_map(lambda q:q.to('cuda',non_blocking=True),o) for o in obs]),t.cat([x.to('cuda',non_blocking=True) for x in labels],0),None if recompute else self.map(lambda *v:t.cat(v,0),*[self.map(lambda q:q.to('cuda',non_blocking=True),x['prefix']) for x in items]))
        ready=t.cuda.Event();ready.record(self.copy)
        losses=[]
        with t.cuda.stream(self.stream):
            self.stream.wait_event(ready)
            for micro in range(2):
                fields=[];acs=[];noises=[];times=[];masks=[];kvs=[]
                for eid in ids:
                    obs,ac,prefix=prepared[eid];sl=slice(micro*4,(micro+1)*4);o=self.obs_map(lambda q:q[sl],obs);ac=ac[sl]
                    if recompute:
                        with t.no_grad():_,mask,kv=self.model.build_prefix_cache(o)
                    else:mask,kv=self.map(lambda q:q[sl],prefix)
                    seed=self.seed+eid*1000000+self.experts[eid]['step']*2+micro
                    fields.append({f.name:getattr(o,f.name) for f in dataclasses.fields(o)});acs.append(ac)
                    noises.append(t.randn(ac.shape,device='cuda',generator=t.Generator(device='cuda').manual_seed(seed)))
                    times.append(t.tensor(self.np.random.default_rng(seed).beta(1.5,1.,4)*.999+.001,device='cuda',dtype=t.float32));masks.append(mask);kvs.append(kv)
                stack=lambda *v:t.stack(v,0) if t.is_tensor(v[0]) else v[0]
                fs=self.map(stack,*fields);params={'model.'+n:t.stack([self.experts[e]['masters'][n].to(p.dtype) for e in ids],0) for n,p in self.pairs.items()}
                def loss(p,f,a,n,tt,mask,kv):return t.func.functional_call(self.wrapper,p,(f,a,n,tt,mask,kv),strict=False)
                dims=self.map(lambda q:0 if t.is_tensor(q) else None,fs)
                values=t.vmap(loss,in_dims=(0,dims,0,0,0,0,0),randomness='error')(params,fs,t.stack(acs),t.stack(noises),t.stack(times),t.stack(masks),self.map(stack,*kvs))
                (values.sum()/2).backward();losses.append(values.detach())
            for eid in ids:
                e=self.experts[eid];e['step']+=1;e['lr_step']+=1;s=e['lr_step']
                lr=5e-5*(s/50 if s<=50 else .1+.9*.5*(1+math.cos(math.pi*min(1.,(s-50)/3950))))
                e['optim'].param_groups[0]['lr']=lr;t.nn.utils.clip_grad_norm_(list(e['masters'].values()),1.,error_if_nonfinite=True);e['optim'].step();e['optim'].zero_grad(set_to_none=True)
                with t.no_grad():
                    for n,p in e['masters'].items():e['ema'][n].mul_(.999).add_(p,alpha=.001)
            end=t.cuda.Event();end.record()
        end.synchronize()
        return dict(start_ns=start,finish_ns=time.monotonic_ns(),seconds=(time.monotonic_ns()-start)/1e9,experts=ids,expert_steps={e:self.experts[e]['step'] for e in ids},losses=t.stack(losses).mean(0).cpu().tolist(),peak_cuda_bytes=t.cuda.max_memory_allocated())

    def snapshot(self,expert,path):
        t=self.torch;self.stream.synchronize();e=self.experts[expert]
        t.save(dict(parameters={n:p.detach().cpu() for n,p in e['ema'].items()},step=e['step'],weights_kind='EMA',family='lora32_heads'),path)

    def discard(self,expert,production_path=None):
        t=self.torch;e=self.experts[expert]
        source=t.load(production_path,map_location='cpu',weights_only=True)['parameters'] if production_path else self.pairs
        with t.cuda.stream(self.stream),t.no_grad():
            for n,p in e['masters'].items():p.copy_(source[n]);e['ema'][n].copy_(p)
        e['optim'].state.clear();e['lr_step']=0;self.stream.synchronize()
