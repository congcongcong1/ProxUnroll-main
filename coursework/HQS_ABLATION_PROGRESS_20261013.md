# 本轮续训和消融的进度入口

当前实验与材料已完成核验，最终远端/GitHub同步正在收尾。下文早期进度按时间保留，不是当前运行状态。
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

## 训练完成、评测运行中的复查

2026-10-13恢复goal后实际核验：三个正式训练队列退出码0，续训12000步，验证选中11500；八组消融各800步，配对与训练审计通过。新增正式更新总计8400，累计212.42 GPU分钟，峰值分配10.84 GiB。
最小推理20状态、20浮点重建与120阶段指标复核通过。
主要/回归评测句柄为145389/145390，已经核查进程及GPU实际运行；未完成前不概括测试收益。
另有CPU等待/整理队列`collect_facts`，初始shell PID145551，确认finalize实际退出0后生成report_facts.json和experiment_completion_audit.json。
这个队列不重复训练或推理，依赖句柄消失/失败时停止，原日志保留。
GitHub master已推送训练/代码/审计阶段提交0e33c2a。当前报告/PPT/中文说明生成代码已就绪，但新版材料尚未生成、渲染或交付。
下一次只读查询应查`logs/collect_facts.exit`与`logs/finalize.exit`，成功后取回report_facts.json、experiment_completion_audit.json、illustration、完整评测CSV/JSON及必要PNG，再生成材料并完成独立交付审计。
若原评测达到硬时间上限失败，先保留部分CSV/浮点重建和原失败记录，只修复缺失评测，不能重新训练，也不能把旧失败退出码改成0。

## 独立 20000 步续训（不阻塞本 goal）

用户于2026-10-13新增授权：累计续训到20000，启动后报告ETA，完成后再查看效果。
已另建`runs/hqs_20000_20261013`，从12000完整状态再训练8000步，GPU6；shell PID145696，实际训练145698。
12001步恢复checkpoint已重新加载核查真实参数更新、Adam步数、有限损失/梯度和六个冻结因子；startup_verified.json已取回。
预计北京时间21:00–21:30，训练上限300分钟；不对它做本轮测试或报告，不等待它完成。
入口finetune_20000.py为独立预算扩展，原锁定finetune_ablation.py未改。详情见HQS_20000_20261013.md。

材料进展：修复中文结果表连续行格式；新版技术报告补原四方法实际PSNR数值，并留来源哈希。
增加verify_ablation_materials.py，最终重开原生图表并对照完整CSV；交付审核现要求图表数值核验与6页/12页逐页视觉记录。
这些检查尚未对最终新版材料执行，须等完整测试事实生成。私有tmp中的排版演练含明确标记的合成数值，不能进入交付/结果。

## 评测边界与补测准备

两路原推理有5400秒硬上限；实际回归速度提示可能不足以完成110张，但尚无失败退出记录时不能提前重启。
recover_ablation_evaluation.py已准备并通过2个分区测试：必须有原任务非零exit、原句柄消失且无原推理孤儿后才允许执行。
只复用完整原图的全部条件/方法；计数相同但重复/缺条件的图不算完成。其余原图在独立missing清单重新推理，最多35图、默认45分钟且不得超过60分钟。
补测先独立浮点复核，再保留terminal_partial原目录和失败exit，合并完整覆盖并重新浮点复核/分析；不得把原失败exit改成0。
入口目前未启动。若确有超时，需先实际核验，再同步这份入口；完成后让事实生成和交付审计识别真实恢复来源，而非要求原失败日志为0。
材料生成脚本的Markdown表也已修复，PDF和源文档均要正确显示表格。
GitHub推送14682bf遇到慢连接/45秒超时；本地提交和远端训练入口/启动审计均保存成功。不能把该新提交称为已推送，后续重新核对并完成。

## 原评测终止后的实际恢复

主要52图已全部完成：11440主指标、68640阶段指标、572配对条件，全部浮点重新计算最大差0，退出0。
回归原推理达到90分钟上限，退出1；88原图完整，其余22张另行补测。原finalize与collect_facts也退出1，全部原日志/exit保留。
244的3090已由其他任务占用，未干扰其进程。使用空闲GPU2 RTX2080Ti，补测实际启动shell146180、推理146192；45分钟推理预算/5400秒整体预算。
复用终止CSV前检查逐图全部条件和12状态完整，补测只使用缺失原图。恢复后保留evaluation_regression_terminal_partial，合并目录链接原数组与新数组并全量重新复核。
96条部分图跨3090/2080Ti的重叠记录核验：同测量哈希，PSNR最大差2.42e-6dB，SSIM5.57e-7，浮点重建5.30e-6，在指标容差内，未新增推理。
新CPU整理队列finalize_recovery PID146282，依赖真实补测退出0后才更新workflow，收集事实并做独立完结审计。原非零退出不改写。
事实收集/审计入口现在识别恢复完成记录、原失败、归档CSV哈希、缺失清单和完整覆盖；新增2个失败门控测试通过。
独立主要集干净10%MSE对官方+4.1158%，对10000选中模型+0.2084%；强噪声对官方+12.3336%，对10000+0.0897%。总体结论须等回归完结。
独立20000步续训已退出0并完成训练审计；该轮按用户指示暂不推理/分析效果，不进入本goal的12000/800步实验报告。

## 最终实验与材料核验

补测22/22已退出0，合并回归110图全部14520重建及87120阶段指标完成复核；主要52图11440重建和68640阶段指标完整。25960个浮点重建最大指标差0。
finalize_recovery退出0，workflow=experiments_complete；原回归/旧finalize/旧collect_facts退出1原样保留，归档CSV哈希和88+22原图分区通过。
完整结果已取回：report_facts.json、原始CSV、环境与数据/权重来源、所有负例；大数组和checkpoint继续在远端。
六页英文技术报告、十二页PPT、中文讲稿已生成。所有页已逐页检查；重做第6页原生图表及数据工作簿，修正第9/10页负值柱与标签重叠。
materials_numeric_audit.json把PPT四张原生图表和PDF三张结果表重新对照完整CSV；materials_visual_audit.json记录6+12页的预览与哈希。
delivery_audit.json通过完整实验/材料核验。旧repository_evidence.json的288文件仍全部哈希一致。
项目自动测试正确命令为`.venv/bin/python -m pytest tests -q`，20项通过。曾从根目录无限制收集pytest，误收官方CLI推理入口和tmp中的恢复checkout，因本地无albumentations和重名模块收集失败；不是上述20项测试失败，不据此声称根目录pytest通过。
课程要求与最新入口另存LATEST_COURSEWORK.md和coursework/REQUIREMENTS_STATUS_ABLATION_20261013.md。同期20000步训练已保存并退出0，但按用户要求不在本goal评测其效果。
