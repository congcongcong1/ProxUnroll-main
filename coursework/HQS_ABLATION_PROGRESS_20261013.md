# 本轮续训和消融的进度入口

目标仍在进行：继续 HQS 微调，完成配对消融、公平评测、负例分析和新报告/PPT。
本文件记录已经执行的工作，不代表训练或测试已完成。以远端当前进程、退出码、checkpoint 和实际 CSV 为准。

## 已执行

- 220/244 身份、对应容器、共享挂载、可写空间、CUDA、GPU UUID 与占用已重新核验。
- 220 在正式启动前被另一任务占用，改用 244 `luozc_mlvc` 空闲 GPU4/5/6。
- 四个训练臂的真实 CUDA 短跑通过；2+1 与连续 3 步恢复逐位一致；旧/新入口 5→6 逐位一致。
- 本地 16 个测试通过，包括噪声开关不改变采样/随机配对、阶段权重归一化、因素效应与等权聚合。
- 训练、验证与主要测试来源哈希互不相交；52 张主要测试图未发现此前完成的项目推理记录。
- 三个训练队列已用 nohup/timeout 实际启动，并确认参数更新、有限损失、GPU 进程和 startup checkpoint。
- 配置和源码已锁定；正式训练新增预算 8400 步，预检另有 14 步。
- 后台 finalize 依赖门控已启动，只在三个训练/审计队列成功后进行推理及浮点核验。

## 当前具体句柄（不可仅凭 PID 文件认定仍运行）

根目录：`/workspace/ProxUnroll-main/runs/hqs_ablation_20261013`。

| 队列 | 初始 shell PID | GPU UUID | 目标 |
|---|---:|---|---|
| continuation | 144521 | GPU-5c29e362-93ff-356b-1b27-82ef64ea1327 | 原 10000 步完整状态 → 12000 步，最多 65 分钟 |
| seed_20261013 | 144536 | GPU-16f6ac58-4d8c-8845-0886-1dc146a0327e | 同种子四臂顺序执行，每臂 800 步 / 35 分钟 |
| seed_20261014 | 144552 | GPU-62160106-d537-698b-2e4a-94b4196592a9 | 另一个种子，相同四臂和预算 |
| finalize | 144773 | 训练结束后才使用 GPU4/5 | 最小推理、主要52图、回归110图、浮点核验与分析 |

状态文件：`workflow_status.json`。日志与退出码：`logs/{队列}.log/.exit`。
它们必须结合 `/proc/{PID}/cmdline` 或 `ps`、实际 GPU 进程和最新 train.csv 判断；观察超时不等于停止。

## 后续必须完成的工作

1. 逐臂检查 800 次更新、续训 12000 总更新、Adam 步数、固定六个测量因子、完整恢复状态及验证选模。
2. 检查两个种子的 pairing JSON，确保原图顺序、裁剪、采样率和潜在噪声逐步相同。
3. 检查 `evaluation_lock.json` 与所有推理前固定的权重哈希。
4. 检查 `evaluation_smoke/verification.json`、`evaluation_primary/verification.json`、`evaluation_regression/verification.json` 与全部退出码。
5. 阅读主要52图与回归110图的 `ablation_analysis.json`、`paired_summary.csv`、`factorial_effects.csv` 和 `seed_effect_stability.csv`。
6. 分开报告验证 best 与相同步数 last、原配方续训与官方起点消融、干净与噪声、两个训练种子与测量噪声重复。
7. 保留所有负例；选取真实失败案例，记录图名/采样率/噪声/测量种子和比较对象。
8. 只取回 CSV、JSON、日志和必要显示图；大数组、原图和完整恢复权重继续保留远端，避免重新堆积本地空间。
9. 另存英文 4–6 页技术报告、PPT 扩展版和中文讲稿，原版本不覆盖；实际渲染并逐页检查。
10. 更新复现说明和冻结证据，完成远端非删除同步；若 Git 同步目录再次回退或出现 SFConflict，先对比具体内容，保护已有修改。
11. 按目标逐项核验全部完成后才将 goal 标为 complete；失败或暂未完成如实报告并继续处理。

## 辅助命令

远端只读查询示例：

```bash
.venv/bin/python -m coursework.remote244 \
  'cat runs/hqs_ablation_20261013/workflow_status.json; tail -10 runs/hqs_ablation_20261013/logs/finalize.log'
```

取回轻量结果（默认不取回 NPY 和权重）：

```bash
.venv/bin/python -m coursework.fetch_lab_run --server 244 \
  --run runs/hqs_ablation_20261013 \
  --parts lock.json,evaluation_lock.json,workflow_status.json,logs,continuation,arms,evaluation_primary,evaluation_regression
```

不要在现有目录上重新 launch 或重新执行 prepare；先确认任务实际终止及缺失部分。训练源码锁定期间不要修改 `lock.json` 内列出的源码。

## 材料技能

PDF 和 Presentations 技能已读取。PDF 操作 marker 已成功调用一次，输出预期为一份新 PDF；不要重跑 marker。
PPT 操作 marker 也已成功调用一次；新的生成入口已经准备好，最终材料仍须真实结果和逐页渲染后交付，不要重跑 marker。
已有英文报告、12 页 PPT、原中文讲稿均是之前的10000步结果，不能提前改写成这轮结论。
