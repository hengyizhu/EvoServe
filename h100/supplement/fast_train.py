"""Production math, batched input assembly, and CUDA Graph training for fixed-shape expert batches."""
import dataclasses
import inspect
import math
import textwrap
import time
import types

import torch
from torch.utils import _pytree
from evoserve.backend import GraphModel,TrainModel
import stages


def numerical_functions():
    src=inspect.getsource(TrainModel.step)
    begin=src.index('            for micro in range(2):')
    optim=src.index("            for eid in ids:\n                e=self.experts",begin)
    finish=src.index('            end=t.cuda.Event()',optim)
    forward=src[begin:optim]
    a="t.randn(ac.shape,device='cuda',generator=t.Generator(device='cuda').manual_seed(seed))"
    b="t.tensor(self.np.random.default_rng(seed).beta(1.5,1.,4)*.999+.001,device='cuda',dtype=t.float32)"
    assert forward.count(a)==forward.count(b)==1
    forward=forward.replace(a,'noise_buffers[eid,micro]').replace(b,'time_buffers[eid,micro]')
    code="def fwd_bwd(self,prepared,ids,recompute,noise_buffers,time_buffers):\n    t=self.torch; losses=[]\n"
    code+=textwrap.indent(textwrap.dedent(forward),'    ')+'    return losses\n'
    code+='def update(self,ids):\n    t=self.torch\n'+textwrap.indent(textwrap.dedent(src[optim:finish]),'    ')
    env=dict(TrainModel.step.__globals__)
    exec(compile(code,'<pinned EvoServe training math>','exec'),env)
    return env['fwd_bwd'],env['update']


forward_backward,original_update=numerical_functions()


class PackedInputs:
    def __init__(self,model,safe_inputs=False):
        self.m=model
        self.cache={}
        self.layout=None
        self.safe_inputs=safe_inputs
        self.image_lut=None

    def pack_host_kv(self,item):
        key=id(item['prefix'])
        if key in self.cache:
            return self.cache[key][1]
        leaves,spec=_pytree.tree_flatten(item['prefix'])
        offsets={}
        layout=[]
        grouped={}
        for q in leaves:
            assert q.device.type=='cpu' and q.shape[0]==1
            dtype=q.dtype
            start=offsets.get(dtype,0)
            offsets[dtype]=start+q.numel()
            layout.append((dtype,start,q.numel(),tuple(q.shape[1:])))
            grouped.setdefault(dtype,[]).append(q.reshape(-1))
        if self.layout is None:self.layout=(layout,spec)
        else:assert self.layout==(layout,spec)
        data={k:torch.cat(xs).pin_memory() for k,xs in grouped.items()}
        self.cache[key]=(item['prefix'],data)
        return data

    def prefix(self,items):
        banks=[self.pack_host_kv(item) for item in items]
        # Each sample issues one transfer per dtype. Across-sample assembly is on the GPU.
        batch={dtype:torch.stack([b[dtype].to('cuda',non_blocking=True) for b in banks])
               for dtype in banks[0]}
        layout,spec=self.layout
        leaves=[batch[dtype][:,start:start+n].reshape(len(items),*shape)
                for dtype,start,n,shape in layout]
        return _pytree.tree_unflatten(leaves,spec)

    def observations(self,items):
        m=self.m
        if self.safe_inputs:
            pairs=[m.prepare(x['raw'],x['actions']) for x in items]
            obs=m.cat_obs([m.obs_map(lambda q:q.cuda(non_blocking=True),o) for o,_ in pairs])
            return obs,torch.cat([a.cuda(non_blocking=True) for _,a in pairs])
        rows=[]
        for item in items:
            raw=dict(item['raw'])
            raw['actions']=m.np.asarray(item['actions'],dtype=m.np.float32)
            rows.append(m.model._input_transform_fn(raw))
        def stack_on_gpu(*parts):
            return torch.stack([torch.as_tensor(m.np.asarray(x).copy()).pin_memory().to('cuda',non_blocking=True) for x in parts])
        buf=m.map(stack_on_gpu,*rows)
        # Preserve the exact CPU float32 normalization for all 256 input codes.
        if self.image_lut is None:
            self.image_lut=(torch.arange(256,dtype=torch.float32)/255.0*2.0-1.0).cuda()
        for key,image in buf['image'].items():
            if image.dtype==torch.uint8:buf['image'][key]=self.image_lut[image.long()]
        labels=buf.pop('actions').float()
        obs=m.mod.Observation.from_dict(buf)
        obs=dataclasses.replace(obs,state=obs.state.float())
        obs=m.mod.preprocess_observation(obs,train=False)
        obs=m.mod._observation_to_dtype(obs,m.model.embed_dtype)
        return m.obs_map(lambda x:x.contiguous(),obs),labels

    def prepare(self,batches,recompute):
        prepared={};unique={}
        with torch.cuda.stream(self.m.copy):
            for eid,items in batches.items():
                key=tuple(id(x) for x in items)
                if key not in unique:
                    obs,labels=self.observations(items)
                    kv=None if recompute else self.prefix(items)
                    unique[key]=(obs,labels,kv)
                prepared[eid]=unique[key]
            ready=torch.cuda.Event()
            ready.record()
        return prepared,ready


class TrainingGraph:
    def __init__(self,m,ids,recompute,full):
        self.m=m;self.ids=ids;self.recompute=recompute;self.full=full
        self.graph=None
        self.statics={};self.noise={};self.times={}
        self.grad_refs={}
        self.lr={}
        self.norms=[]
        self.setup_ms=0

    def static_inputs(self,prepared):
        m=self.m
        allocated={}
        for eid in self.ids:
            obs,action,kv=prepared[eid]
            if id(obs) not in allocated:
                allocated[id(obs)]=(m.obs_map(torch.empty_like,obs),torch.empty_like(action),
                                     None if kv is None else m.map(torch.empty_like,kv))
            self.statics[eid]=allocated[id(obs)]
            for micro in range(2):
                self.noise[eid,micro]=torch.empty(action[:4].shape,device='cuda')
                self.times[eid,micro]=torch.empty(4,device='cuda',dtype=torch.float32)
        self.alias=tuple(next(y for y in self.ids if prepared[e][0] is prepared[y][0]) for e in self.ids)

    def fill(self,prepared):
        m=self.m
        alias=tuple(next(y for y in self.ids if prepared[e][0] is prepared[y][0]) for e in self.ids)
        assert self.alias==alias
        for eid,first in zip(self.ids,alias):
            src,ac,kv=prepared[eid]
            dst,actions,static_kv=self.statics[eid]
            if first==eid:
                for a,b in zip(m.leaves(vars(dst)),m.leaves(vars(src)),strict=True):
                    if torch.is_tensor(a):a.copy_(b)
                actions.copy_(ac)
                if kv is not None:
                    for a,b in zip(m.leaves(static_kv),m.leaves(kv),strict=True):a.copy_(b)
            for micro in range(2):
                seed=m.seed+eid*1000000+m.experts[eid]['step']*2+micro
                noise=torch.randn(ac[:4].shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                self.noise[eid,micro].copy_(noise)
                values=m.np.random.default_rng(seed).beta(1.5,1.,4)*.999+.001
                self.times[eid,micro].copy_(torch.tensor(values,device='cuda',dtype=torch.float32))

    def replay_body(self):
        return forward_backward(self.m,self.statics,self.ids,self.recompute,self.noise,self.times)

    def init_capturable_optimizer(self):
        for eid in self.ids:
            e=self.m.experts[eid]
            opt=e['optim'];group=opt.param_groups[0]
            lr=self.lr[eid]=torch.zeros((),device='cuda',dtype=torch.float32)
            group.update(lr=lr,capturable=True)
            for p in e['masters'].values():
                state=opt.state[p]
                if not state:
                    state.update(step=torch.zeros((),device=p.device),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))

    def optimizer_body(self):
        for eid in self.ids:
            e=self.m.experts[eid]
            self.norms.append(torch.nn.utils.clip_grad_norm_(list(e['masters'].values()),1.,error_if_nonfinite=False))
            e['optim'].step()
            with torch.no_grad():
                for name,p in e['masters'].items():e['ema'][name].mul_(.999).add_(p,alpha=.001)

    def run(self,prepared,ready):
        m=self.m
        with torch.cuda.stream(m.stream):
            m.stream.wait_event(ready)
            fresh=self.graph is None
            if fresh:self.static_inputs(prepared)
            self.fill(prepared)
            if fresh:
                started=time.perf_counter()
                # A dry eager forward/backward initializes CUDA libraries without updating any weights.
                self.replay_body()
                for eid in self.ids:m.experts[eid]['optim'].zero_grad(set_to_none=True)
                if self.full:self.init_capturable_optimizer()
                m.stream.synchronize()
                self.graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph,stream=m.stream,capture_error_mode='thread_local'):
                    self.losses=self.replay_body()
                    if self.full:self.optimizer_body()
                for eid in self.ids:
                    self.grad_refs[eid]=[p.grad for p in m.experts[eid]['masters'].values()]
                self.setup_ms=1000*(time.perf_counter()-started)
            if self.full:
                for eid in self.ids:
                    e=m.experts[eid];e['step']+=1;e['lr_step']+=1;s=e['lr_step']
                    factor=s/50 if s<=50 else .1+.9*.5*(1+math.cos(math.pi*min(1.,(s-50)/3950)))
                    self.lr[eid].fill_(m.peak_lr*factor)
            self.graph.replay()
            if self.full:
                if not torch.stack(self.norms).isfinite().all().item():raise RuntimeError('non-finite graph gradient norm')
            else:
                for eid in self.ids:
                    for p,g in zip(m.experts[eid]['masters'].values(),self.grad_refs[eid],strict=True):p.grad=g
                original_update(m,self.ids)
            end=torch.cuda.Event();end.record()
        end.synchronize()
        return torch.stack(self.losses).mean(0).cpu().tolist()


def attach(m,path):
    stages.install(m)
    if 'graph' in path:
        # Reuse the exact prefix, suffix and attention constant adaptations from the
        # repository's serving graph, without constructing a separate serving policy.
        base=inspect.getsource(GraphModel.__init__)
        block=base[base.index('        def replace('):base.index('        self.stream.wait_stream(')]
        assert all(x in block for x in ('embed_suffix','embed_prefix','big_neg'))
        env=dict(GraphModel.__init__.__globals__)
        src='def install_constants(self):\n'+textwrap.indent(textwrap.dedent(block),'    ')
        exec(compile(src,'<EvoServe graph constants>','exec'),env)
        env['install_constants'](m)
    packer=PackedInputs(m,safe_inputs=path.startswith('exact'))
    graphs={}
    def step(self,batches,recompute=False):
        start=time.monotonic_ns()
        ids=list(batches)
        prepared,ready=packer.prepare(batches,recompute)
        if path in ('packed','exactpack'):
            return stages.compute(self,prepared,ready,ids,start,recompute)
        key=(tuple(ids),recompute)
        if key not in graphs:graphs[key]=TrainingGraph(self,ids,recompute,path=='fullgraph')
        losses=graphs[key].run(prepared,ready)
        return dict(start_ns=start,finish_ns=time.monotonic_ns(),experts=ids,losses=losses)
    m.step=types.MethodType(step,m)
    m.fast_graphs=graphs
