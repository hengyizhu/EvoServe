# RTX A6000 推理优化实测

RTX A6000，225 W。固定父模型、10 步去噪。计时采用 10 次 GPU Event 中位数，误差统计覆盖 12 个真实观测批次和固定噪声。

| Batch | 前缀 / 去噪实现 | 原前缀 / 去噪 ms | 新前缀 / 去噪 ms | 归一化动作平均 / 最大绝对误差 |
|---:|---|---:|---:|---:|
| 1 | graph / graph | 74.23 / 84.02 | 69.26 / 72.03 | 0 / 0 |
| 2 | graph / graph | 140.91 / 76.50 | 133.29 / 65.06 | 0 / 0 |
| 4 | graph / graph | 270.33 / 82.18 | 255.92 / 70.71 | 0 / 0 |
| 8 | graph / graph | 540.06 / 97.29 | 516.21 / 84.88 | 0 / 0 |

低照度父模型配对评估，原版 24/50，优化版 24/50。失败转成功 0，成功转失败 0，精确双侧 McNemar p=1。

不一致的回合（索引从 0 开始）：[]

250 ms、99% SLO，4 Hz 错峰，5 分钟正式测量。最高达标 1 台（4 请求/秒）。

| 机器人 | 训练 | 合并专家 | 请求 | SLO | P99 ms | 专家步/秒 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | infer | 1 | 1200 | 100.00% | 161.36 | 0.000 |
| 2 | infer | 1 | 2400 | 44.75% | 358.35 | 0.000 |
| 3 | infer | 1 | 3600 | 0.06% | 1265.51 | 0.000 |
| 4 | infer | 1 | 4800 | 0.04% | 81324.20 | 0.000 |
| 5 | infer | 1 | 6000 | 0.02% | 175629.03 | 0.000 |
| 1 | cache | 1 | 1200 | 100.00% | 203.96 | 8.513 |
| 1 | cache | 2 | 1200 | 100.00% | 214.50 | 13.440 |
| 1 | cache | 4 | 1200 | 100.00% | 229.59 | 19.853 |
| 1 | cache | 8 | 1200 | 100.00% | 242.60 | 25.680 |
| 1 | recompute | 1 | 1200 | 91.92% | 509.27 | 0.083 |

编译前缀配对验证：原版 24/50，编译前缀 24/50；改善 8，退化 8，McNemar p=1。通过：True。B1 平均误差 0.000686716，最大误差 0.0200408。
不一致回合：[{'episode': 0, 'baseline': False, 'optimized': True}, {'episode': 2, 'baseline': True, 'optimized': False}, {'episode': 6, 'baseline': False, 'optimized': True}, {'episode': 7, 'baseline': False, 'optimized': True}, {'episode': 13, 'baseline': True, 'optimized': False}, {'episode': 15, 'baseline': True, 'optimized': False}, {'episode': 17, 'baseline': True, 'optimized': False}, {'episode': 19, 'baseline': False, 'optimized': True}, {'episode': 21, 'baseline': False, 'optimized': True}, {'episode': 27, 'baseline': True, 'optimized': False}, {'episode': 28, 'baseline': False, 'optimized': True}, {'episode': 31, 'baseline': False, 'optimized': True}, {'episode': 33, 'baseline': True, 'optimized': False}, {'episode': 35, 'baseline': True, 'optimized': False}, {'episode': 37, 'baseline': False, 'optimized': True}, {'episode': 40, 'baseline': True, 'optimized': False}]
