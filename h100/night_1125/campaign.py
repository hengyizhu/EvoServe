import csv
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
OUT=REPO/'results/night_1125_A'
RAW=Path('/var/tmp/night1125/raw')
LAYOUT=Path('/var/tmp/evo-layout-654017a/raw')
CAL=json.loads((LAYOUT/'1-service-ms.json').read_text())
lock=threading.Lock()
cells=[]
def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    tmp=OUT/(name+'.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    tmp.replace(OUT/name)

def plan(gpu):
    out=[]
    for rep in range(1,(3 if gpu<2 else 2)+1):
        if gpu<2: choices=itertools.product([70,75,80,85,90],[1,2,4,8],['A2'],[True])
        elif gpu==2: choices=itertools.product([50,70,85],[1],['A0','A1','A2'],[True])
        else:choices=itertools.product([70,85],[1],['A2'],[False,True])
        choices=list(choices)
        random.Random(112500+rep).shuffle(choices)
        for load,m,method,graph in choices:
            out.append(dict(group=['r32','r128','cache','graph'][gpu],gpu=gpu,rank=128 if gpu==1 else 32,
                       load=load,m=m,method=method,graph=graph,repeat=rep,seed=20261125+rep,seconds=300))
    return out

def cell(cfg,smoke=False):
    tag='{group}-{load}-{m}-{method}-{graph}-{repeat}'.format(**cfg)
    folder=RAW/(('smoke-' if smoke else '')+tag)
    folder.mkdir(exist_ok=True)
    saved=folder/'outcome.json'
    if saved.exists():
        old=json.loads(saved.read_text())
        if not smoke or old['status']!='ok':
            with lock:
                cells.append(old);report()
        return old['status']=='ok'
    (folder/'config.json').write_text(json.dumps(dict(cfg,service_ms=CAL,seconds=8 if smoke else 300)))
    status='ok'
    try:
        with (folder/'log').open('w') as f:
            cmd=[str(HERE/'run_cell'),str(cfg['gpu']),str(folder)]
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                if proc.wait(timeout=760):
                    raise RuntimeError(f'cell exited {proc.returncode}')
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                proc.wait()
                raise
        result=json.loads((folder/'result.json').read_text())
        if cfg['group']=='graph':
            k=json.loads((folder/'kernel.json').read_text());result.update(kernel_ms=k['ms'],kernel_name=k['name'])
    except Exception as e:
        status='error'
        lines=(folder/'log').read_text(errors='replace').splitlines()
        detail=next((line for line in reversed(lines) if any(t in line for t in ('Error:', 'Exception:', 'No CUDA'))),str(e))
        result=dict(error=detail[-400:])
    saved.write_text(json.dumps(dict(cfg,status=status,smoke=smoke,**result)))
    print(tag,'smoke' if smoke else '',status,result.get('error',''),flush=True)
    if not smoke or status!='ok':
        with lock:
            cells.append(dict(cfg,status=status,smoke=smoke,**result))
            report()
    return status=='ok'

def fmt(x,n=2):
    return '—' if x is None else f'{x:.{n}f}'

def report():
    write_csv('cells.csv',cells)
    text=['# A6000 执行层复测', '', '基于 a3146654；四卡功耗上限均为 225 W。请求时限 250 ms，训练步前保护 200 ms。每个正式点独立启动进程和 MPS，固定每卡 8 个物理核。按原布局标定 '+str(round(float(CAL['1']),3))+' ms 换算到达率；重复间改变周期到达的初相，同轮各方法共用种子。','', '## 并发预检','','|GPU|8 Hz P99 ms|历史单卡 ms|变化|','|---:|---:|---:|---:|']
    text += [f"|{r['gpu']}|{r['p99_ms']:.2f}|{r['baseline']:.2f}|{r['change']:+.2%}|" for r in pre]
    text+=['',('串行' if serial else '并行')+'执行。任一卡 P99 较单卡增加超过 5% 则串行。','', '## 前馈表','','|rank|负载|重复 1|重复 2|重复 3|最大差（档）|','|---:|---:|---:|---:|---:|---:|']
    ok=[c for c in cells if c['status']=='ok' and not c['smoke']]
    for rank,load in itertools.product((32,128),(70,75,80,85,90)):
        widths=[]
        for rep in (1,2,3):
            matches=[c for c in ok if c['group']==f'r{rank}' and c['load']==load and c['repeat']==rep]
            widths.append(max((r['m'] for r in matches if r['slo']>=.995 and r['p99_ms']<=220),default=0) if len(matches)==4 else None)
        levels=[0,1,2,4,8]
        diff=max(map(levels.index,widths))-min(map(levels.index,widths)) if None not in widths else None
        text.append('|'+ '|'.join(map(str,[rank,load,*['—' if v is None else v for v in widths],('**'+str(diff)+'**') if diff else ('0' if diff==0 else '—')]))+'|')
    text+=['','0 表示四个宽度均未通过，横线表示测点不足。档位按 0/1/2/4/8 排列。','','## 训练端缓存','','|负载|方式|重复|SLO|P99 ms|专家步/s|训练 VLM 前向/唯一样本|训练 VLM 前向/抽样|','|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in sorted(ok,key=lambda c:(c['load'],c['method'],c['repeat'])):
        if c['group']=='cache':
            text.append(f"|{c['load']}|{c['method']}|{c['repeat']}|{c['slo']:.2%}|{c['p99_ms']:.2f}|{c['expert_steps_s']:.2f}|{fmt(c['vlm_per_unique_training_sample'])}|{fmt(c['train_vlm_per_draw'],5)}|")
    text+=['','沿用最终执行层的 16 个固定样本、每步 8 个；A1 从空缓存开始，首次取样时在训练进程计算。分子按训练实际消费的样本前向数统计，服务前向另列于 CSV。这组测量覆盖高复用样本池。','','## 训练 CUDA Graph','','|负载|图|重复|SLO|P99 ms|专家步/s|单步均值 ± SD ms|最长 kernel ms|','|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in sorted(ok,key=lambda c:(c['load'],c['graph'],c['repeat'])):
        if c['group']=='graph':
            text.append(f"|{c['load']}|{c['graph']}|{c['repeat']}|{c['slo']:.2%}|{c['p99_ms']:.2f}|{c['expert_steps_s']:.2f}|{fmt(c['step_mean_ms'])} ± {fmt(c['step_sd_ms'])}|{fmt(c['kernel_ms'],3)}|")
    text+=['','两组均在 GPU 组 batch，优化器在图外。最长 kernel 取自独立共跑的 profiler 窗口。','','## 决策开销','','|范围|决策数|中位数 µs|P99 µs|','|---|---:|---:|---:|']
    cpu=OUT/'controller_raw.csv'
    if cpu.exists():
        with cpu.open() as f: rows=list(csv.DictReader(f))
        if rows:
            for key in ('policy_us','ledger_plus_policy_us'):
                vs=sorted(float(r[key]) for r in rows)
                text.append(f'|{key}|{len(vs)}|{statistics.median(vs):.2f}|{vs[math.ceil(.99*len(vs))-1]:.2f}|')
            text+=['',f'原实时日志未记录耗时；单独按 5 秒节拍回放最后 30 分钟。复现原命令 {sum(r["matches_recorded_command"]=="True" for r in rows)}/{len(rows)}。计时截至决策生成，另列含请求账本扫描的开销。']
    if len(ok)==146:
        varied=0
        for rank,load in itertools.product((32,128),(70,75,80,85,90)):
            safe=[max((r['m'] for r in ok if r['group']==f'r{rank}' and r['load']==load and r['repeat']==rep and r['slo']>=.995 and r['p99_ms']<=220),default=0) for rep in (1,2,3)]
            varied+=len(set(safe))>1
        graph_rows=[r for r in ok if r['group']=='graph']
        ratios=[]
        for on in graph_rows:
            if on['graph']:
                off=next(r for r in graph_rows if r['load']==on['load'] and r['repeat']==on['repeat'] and not r['graph'])
                if off['expert_steps_s']:ratios.append(on['expert_steps_s']/off['expert_steps_s'])
        text+=['',f'10 个 rank × 负载格中，{varied} 个格的安全宽度在重复间有变化。']
        if ratios:text.append(f'同档同重复下，开图/关图的训练吞吐比为 {min(ratios):.2f}–{max(ratios):.2f}。')
    err=[c for c in cells if c['status']!='ok']
    text+=['',f'正式点成功 {len(ok)}/146；失败或冒烟异常 {len(err)}。逐点记录见 cells.csv。']
    for e in err:text.append(f"- {e['group']} {e['load']}% m={e['m']} {e['method']} rep={e['repeat']}：{e['error']}")
    (OUT/'REPORT.tmp').write_text('\n'.join(text)+'\n')
    (OUT/'REPORT.tmp').replace(OUT/'REPORT.md')

def run_card(gpu):
    configs=plan(gpu)
    # An error removes only the offending cell; later points run independently.
    seen=set()
    for cfg in configs:
        signature=(cfg['method'],cfg['graph'])
        if signature not in seen:
            seen.add(signature)
            cell(cfg,smoke=True)
        cell(cfg)

def publish():
    subprocess.run(['git','add','-f','results/night_1125_A','h100/night_1125/*.py','h100/night_1125/run_cell'],cwd=REPO,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m','Measure A6000 execution-layer repeatability, cache methods and controller costs'],cwd=REPO,check=True)
    subprocess.run(['git','push','origin','HEAD:night-1125-a'],cwd=REPO,check=True)

if __name__=='__main__':
    for i in range(4):
        while not (RAW/f'pre-{i}.json').exists():
            if subprocess.run(['systemctl','--user','is-active','--quiet',f'night1125-pre-{i}']).returncode:
                raise RuntimeError(f'precheck {i} failed')
            time.sleep(2)
    ref=json.loads((LAYOUT/'cap-2.json').read_text())['phases'][0]['p99_ms']
    pre=[]
    for i in range(4):
        phase=json.loads((RAW/f'pre-{i}.json').read_text())['phases'][0]
        pre.append(dict(gpu=i,baseline=ref,change=phase['p99_ms']/ref-1,**phase))
    serial=any(r['change']>.05 for r in pre)
    write_csv('preflight.csv',pre)
    report()
    if serial:
        for g in range(4):run_card(g)
    else:
        with ThreadPoolExecutor(4) as ex:list(ex.map(run_card,range(4)))
    report()
    publish()
