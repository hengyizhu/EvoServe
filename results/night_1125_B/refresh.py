"""Refresh the report from immutable per-round JSON and episode records."""
import csv, gzip, json, io, os, sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

here=Path(__file__).resolve().parent
root=Path.home()/'evoserve-machine-b'
online=root/'work/machine_b_learning/online'
def readcsv(name):
    return list(csv.DictReader((here/name).open()))
def readlines(p):
    return list(map(json.loads,p.read_text().splitlines()))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,r))+' |' for r in rows])
def put(name,s):
    (here/name).write_text(s)
def dumpcsv(name,rows):
    with (here/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

runs={r['run'] for r in readcsv('b1_raw_rounds.csv')}
status=[]; observed={}
for t,i in [('camera',1),('texture',1),('camera',4),('camera',6)]:
    label=f'night1125_{t}_s10_i{i}'
    stfile=root/'night_1125_b'/f'{label}.json'
    s=json.loads(stfile.read_text()) if stfile.exists() else {'state':'queued'}
    rr=online/label/'rounds.jsonl'
    obs=readlines(rr) if rr.exists() else []
    if obs:runs.add(label)
    progress=online/label/'status.json'
    d=json.loads(progress.read_text()) if progress.exists() else {}
    state=s['state']
    if (here/'cutoff.json').exists() and state in ('running','queued'):state='截止未完成'
    if state=='complete':
        assert len(obs)==12 and all(r['rollout_episodes']==50 for r in obs)
        if i>1:
            for row in obs:
                j=row['round']
                assert row['deployed_after_round']==i*((j-1)//i)
                key=lambda r:(r['episode'],r['state_index'],r['seed'],r['state_sha256'],r['initial_observation_sha256'])
                rel=Path(f'round_{j:02d}/rollouts/episodes.jsonl')
                assert sorted(map(key,readlines(online/label/rel)))==sorted(map(key,readlines(online/'night1125_camera_s10_i1'/rel)))
        observed[t,i]=obs
    status.append(dict(task=t,interval=i,state=state,completed=len(obs),phase=d.get('state',''),round=d.get('round',''),episodes=d.get('episodes',''),exit_code=s.get('exit_code',''),reason=s.get('reason','')))
dumpcsv('b2_status.csv',status)
rawdir=here/'raw'
rawdir.mkdir(exist_ok=True)
for name in sorted(runs):
    folder=online/name
    rows=readlines(folder/'rounds.jsonl')
    dest=rawdir/(name+'.jsonl.gz')
    # Compress original observations without environment images, actions or checkpoint tensors.
    with dest.open('wb') as out, gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as z:
        for row in rows:
            rel=folder/f"round_{row['round']:02d}/rollouts/episodes.jsonl"
            for tag,values in [('round',[row]),('episode',readlines(rel))]:
                for v in values:
                    z.write((json.dumps(dict(round=row['round'],type=tag,record=v),separators=(',',':'))+'\n').encode())
    if (folder/'config.json').is_file():
        (rawdir/(name+'_config.json')).write_bytes((folder/'config.json').read_bytes())
m=readcsv('b1_mae.csv')
def scores(rows):
    return table(['任务','预测数','调度器 MAE','沿用上次','历史平均'],[(r['task'],r.get('n',r.get('targets')),*[f"{float(r[k]):.2f}" for k in ('prior','last','mean')]) for r in rows])
b3file=here/'b3_verified.md'
b3=b3file.read_text() if b3file.exists() else '''截至本次更新未取得目标 24 小时运行及其对照的逐轮日志。检索了本机实验目录及 EvoServe 的 main、prototype、h100-suite、machine-b-setup 分支。仓库中 `results/paired_validation_20261120/final_summary.md` 覆盖 18 个半小时窗口，末窗相机为 13/26 和 9/21；这组 9 小时序贯验证实验不能用于归属 46% 或计算训练步数比。46% 对应轮次、“1/4”的分子与分母、对照成功率和接管差值均待核实。B1 同样缺少这一来源。'''
summary=[]
for task in ('camera','texture'):
    read=lambda seed:readlines(online/(f'pure_hg_{task}_s{seed:02d}_a1' if task!='texture' or seed!=8 else 'texture_hg_online08_a1')/'rounds.jsonl')
    groups=[read(8),read(9),observed.get((task,1))]
    for metric,label,factor in [('takeovers','末轮接管次数',1),('takeover_frame_fraction','末轮接管帧占比 %',100),('collection_success_rate','末轮含接管成功率 %',100),('unassisted_completion_rate','末轮自主完成率 %',100)]:
        v=[r[-1][metric]*factor if r else None for r in groups]
        summary.append([task,label,*[f'{n:.2f}' if n is not None else '—' for n in v],f'{mean(v):.2f} [{min(v):.2f}, {max(v):.2f}]' if None not in v else '待齐'])
intervals={}
for interval in (1,4,6):
    vals=[]
    for seed in (8,9):
        label=f'pure_hg_camera_s{seed:02d}_a1' if interval==1 else f'period{interval}_hg_camera_s{seed:02d}_a1'
        vals.append(sum(r['takeover_frames'] for r in readlines(online/label/'rounds.jsonl')))
    vals.append(sum(r['takeover_frames'] for r in observed['camera',interval]) if ('camera',interval) in observed else None)
    intervals[interval]=vals
conclusions=[]
for i in (4,6):
    v=intervals[i][2];base=intervals[1][2]
    if None not in (v,base):
        conclusions.append(f'间隔 {i} 在 seed 10 相比间隔 1 相差 {v-base:+d} 帧；'+('保留' if v>base else '未复现')+'前两个 seed 的方向。')
if all((t,1) in observed for t in ('camera','texture')):
    diff=observed['texture',1][-1]['unassisted_completion_rate']-observed['camera',1][-1]['unassisted_completion_rate']
    conclusions.append(f'纹理相对相机的末轮自主完成率在 seed 10 相差 {100*diff:+.1f} 个百分点。')
now=datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M %Z')
ranks=readcsv('b1_rolling_rank.csv')
ranktext=[]
for model in ('prior','last','mean'):
    vals=[float(r['spearman']) for r in ranks if r['model']==model and r['spearman']]
    ranktext.append(f"{model}：{mean(vals):.3f}（{len(vals)}/60）" if vals else f"{model}：无定义（0/60）")
report=f'''# 11 月 25 日夜间学习实验 B

更新于 {now}，纽约时间。

## B1　前 k 次验证预测下一次

主指标为每轮 50 回合中完全无需接管且成功的比例。严格前缀 k=1、2、3 分别预测第 2、3、4 轮。四任务 × 两个 seed 基线共 8 条，另有 7 条定期部署轨迹；相同的间隔 1 只计一次。总体按目标等权。不同策略的首轮共用父模型，样本并非独立。24 小时日志的覆盖情况见 B3。

调用仓库 `prototype/evoserve/core.py::LearningPrior.predict` 和固定的 `learning_prior.json`。以最近一次部署成功率为起点，以已完成训练且准备上线的步数差为预测跨度；已观测的相邻验证构成（部署模型步数差，自主完成率差）拟合点。k=1 时还没有增益对。相邻轮的增益是针对现有 HG 日志的近似；缺少同时验证的候选/生产双臂。每次只用过去的成绩和已知部署步数。两个简单方法分别取最后一个值和这 k 个值的算术平均。

{scores([r for r in m if r['k']=='all'])}

单位均为百分点。总体按 k 分别为：

{table(['k','目标数','调度器','沿用上次','历史平均'],[(r['k'],r['n'],*[f"{float(r[t]):.2f}" for t in ('prior','last','mean')]) for r in m if r['task']=='ALL' and r['k']!='all'])}

总体 MAE 2.50 低于给定的验证噪声参考 10.6，但两个简单方法均为 0。早期目标自主完成率全部为零，不能据此证明模型有预测优势。10.6 作为外部比较尺度，不是从这些记录估计的 MAE。按任务和 k 的完整数值见 [b1_mae.csv](b1_mae.csv)。

**任务排序。** 在同一 seed、同一 k 的四任务间隔 1 轨迹内比较下一轮的预测与实际增益。真实增益六组均全并列，Spearman 均无定义，汇总也无可用系数；任务内部不存在跨任务排名。不同部署策略没有凑成四任务组。[分组排名](b1_rank.csv)

**补充滚动检验。** 保持同一预测器和指标，每次只用最近 k 次，预测后面各轮；同一目标可出现在不同 k 中。这与上面的首 k 次检验分开计数。

{scores(readcsv('b1_rolling_mae.csv'))}

按 seed、k 和目标轮次分块跨四任务计算 Spearman，以下是有定义分块的平均值和覆盖数：{'；'.join(ranktext)}。因并列而被排除的分块按模型而异。[逐次预测](b1_rolling.csv) · [逐块排名](b1_rolling_rank.csv)

## B2　第三个 seed

seed 10（20261110），相机与纹理纯在线 r32 / lr 5e-5。每轮 50 回合，共 12 轮；末轮只采集。相机间隔 1 复用于两项比较，4 和 6 使用同 seed 基线的逐回合初始状态及随机种子。每项只尝试一次。仅完整 12 轮可进入跨 seed 比较。

{table(['任务','部署间隔','状态','已落盘轮数','当前阶段'],[(r['task'],r['interval'],r['state'],r['completed'],r['phase']) for r in status])}

{table(['任务','指标','08','09','10','三 seed 均值 [范围]'],summary)}

{table(['相机部署间隔','08 累计帧','09 累计帧','10 累计帧','三 seed 均值 [范围]'],[(i,*[n if n is not None else '—' for n in v],f'{mean(v):,.1f} [{min(v):,}, {max(v):,}]' if None not in v else '待齐') for i,v in intervals.items()])}

{' '.join(conclusions) or 'seed 10 的结论待完整轨迹核验。'} 两个既有 seed 均表现为相机每轮部署比间隔 4、6 的累计接管帧更少。单个 seed 的结果用于核对方向，不能代替不确定性估计。

## B3　24 小时闭环日志核查

{b3}

## 数据与复算

[原始逐回合、逐轮记录及配置](raw/)；[B1 输入](b1_raw_rounds.csv)、[首前缀逐次预测](b1_predictions.csv)、[B2 运行状态](b2_status.csv)。每个压缩包按 JSONL 保存日志原文记录，外层标注轮次和记录类型。B1 使用的代码与固定先验随本分支提供。新增 seed 不进入 B1，防止事后改变预测数据集。
'''
put('REPORT.md',report)
