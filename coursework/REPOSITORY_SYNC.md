# GitHub 同步版说明（2026-10-09）

仓库：https://github.com/congcongcong1/ProxUnroll-main，默认分支 `master`。
本次版本保留原始实验与报告，并加入实验室数据评测、固定测量矩阵的 HQS 微调、
完整状态恢复、审计与 10000 步续训方案。

## 已完成的交付物

- 最新已完成英文技术报告：`output/pdf/technical_report_nanjing_20261009.pdf`，6 页。
- 最新已完成 PPT：`output/presentation/course_presentation_nanjing_20261009.pptx`，12 页、讲稿计划 600 秒。
- 旧技术报告、3 页文献报告、旧 PPT、实验室 100 步版本全部保留。
- `coursework/ROUND5000.md`、`DEFENSE_5000_ZH.md`、`PRESENTATION_GUIDE_5000.md`：复现、答辩和讲稿。
- `REQUIREMENTS_STATUS_20261009.md`：老师要求逐项核对及仍需本人完成的事项。

5000 步实际完成，验证选择第 4500 步。相对第一轮 100 步 HQS，32 张新 DIV2K 图像
在干净 10% 采样下平均 MSE 下降 1.84%，未达到 5–10% 目标；强噪声条件改善更明显。
逐图、逐视频和负例均保留，不将训练损失当成测试提升。
10000 步续训已实际完成，验证选择第 10000 步，训练审计与退出码 0 已核验。
新增南京网络场景测试：校园 18 张、景区 12 张；干净 10% 采样相对 100 步权重 MSE 下降 1.73% / 1.91%，相对 5000 预算模型仅 0.37% / 0.47%。
原定 DIV2K52 的独立测试尚未进行，不能用新场景结果替代它。

## 纳入 Git 的冻结证据

一般 `runs/` 仍被忽略；本次只显式收录选定的已完成证据以及续训的不可变启动材料。
没有把一个完整远端运行目录当作仓库可直接运行的数据集。

| 目录 | 内容 | 主要指标 / 阶段指标行数 |
|---|---|---|
| `coursework/results` | 原始模拟实验和图像 | 352 / 1056 |
| `runs/lab220_20261009/baseline` | Kodak24、8 段 HEVC 各 3 帧，四方法 | 2112 / 6336 |
| `runs/lab220_20261009/comparison` | 官方、100 步 best/last HQS 比较 | 1152 / 6912 |
| `runs/lab244_5000_20261009/comparison` | 80 图、官方/100 步/4500 步 HQS，11 条件 | 2640 / 15840 |
| `runs/lab244_5000_20261009/fresh_baseline` | 32 图的独立四方法控制 | 1408 / 4224 |
| `runs/nanjing_web30_20261009/comparison` | 30 张南京网络场景，四方法及三种微调状态 | 2310 / 9900 |

这些目录含完整原始 CSV、来源/测量/权重哈希、聚合与失败案例表、配置、
训练/验证历史和已有独立审计。报告图表及具体失败图的参考图/重建 PNG 也收录。
`runs/lab244_round2_20261009` 保存前 946 步的阶段信息和原图划分；
`runs/lab244_10000_20261009` 收录固定清单、环境、来源锁、启动/恢复证明和已完成训练历史/审计。
`runs/nanjing_web30_20261009` 收录精确灰度输入、来源/许可、预先锁定协议、原始 CSV、指标复核、全部负例和报告图。
全尺寸来源 PNG 保留在本地 `source_png/`，完整浮点重建保留在远端并已取回本地；它们不进入 Git。微调权重位置/哈希如下。

`repository_evidence.json` 列出上述冻结文件的 SHA256、CSV 覆盖和文档页数。
在仓库根目录执行以下检查，无需 CUDA、SSH 或实验室数据：

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m coursework.verify_repository_snapshot
.venv/bin/python -m coursework.verify_results \
  --source-snapshot coursework/archive_original/source \
  --output tmp/original-verification.json
```

这些本地检查验证冻结文件与覆盖；已有远端 `verification.json` 则记录原始浮点数组的独立指标重算。
新克隆要重新做浮点重算或全部推理，须先准备对应数据和权重。

## 外部数据和权重

服务器 220（`192.168.1.220`、`luozc_nvenc_new`）与服务器 244
（`192.168.1.244`、`luozc_mlvc`）共享项目 `/workspace/ProxUnroll-main` 和 `/datasets`。
各运行的 `data/*.json` 记录原图/视频、固定帧号、预处理、来源路径、源/处理后哈希与划分。
Kodak、DIV2K 与实验室已有 HEVC 属于已有数据，并非自己拍摄；视频帧仍逐张重建。

| 运行 | best checkpoint | SHA256 |
|---|---|---|
| 100 步 | `runs/lab220_20261009/finetune/best.pth` | `0252b7188b86e1f424a11ab65639e53bb769436ebec16555ea62938b26e7c1cf` |
| 5000 步（选中 4500） | `runs/lab244_5000_20261009/finetune/best.pth` | `3e85cbd683c10487104f39978abb325f3fb935052e9a0fc669aad1af6f7d95cb` |
| 10000 步验证选中 | `runs/lab244_10000_20261009/finetune/best.pth` | `78601d24cabff4dc884c1b129a0643491098cf84bf54dc5c93dbf02e38fd2ed3` |
| 完整状态恢复至 10000 | `runs/lab244_10000_20261009/finetune/last.pth` | `b84c743ebac18c4689d4ff86a95f28d06e3f0d5e093c700d60d0e536fe833c74` |
| 完整状态恢复至 5000 | `runs/lab244_5000_20261009/finetune/last.pth` | `497887636e72404b7ba472eac6001b055c3174c10364519d8b011690b5fd426d` |

上表相对于远端项目根目录。官方权重 `weight/*.pth` 和测量矩阵已在 Git 中。
新微调权重、完整数据、所有重建 PNG/浮点 NPY 及后台日志保留在远端运行目录；
新权重哈希亦记录在各轮 `training_audit.json`、`report_facts.json` 和报告校验中。
项目独立 Linux 环境在 `.venv-linux`，安装要求为 `requirements-linux.txt`；
实际环境版本记录在各运行 `environment.txt` 或 `environment.json`。

有同样的实验室 SSH 权限时，可用 `coursework.fetch_lab_run` 取回选定文件。
评测入口 `coursework.evaluate_manifest` 支持 `--manifest`、`--data-dir`、
`--hqs-checkpoint`、`--admm-checkpoint` 和 `--extra-checkpoint NAME=PATH`；
完整命令见 `LAB220.md`、`ROUND5000.md` 和 `ROUND10000.md`。
共享文件和旧结果应保留，所有重新运行采用新输出目录。

查看当前续训：

```bash
python3 -m coursework.monitor_training --run runs/lab244_10000_20261009
```

从第 5000 步完整状态续至总计 10000 步，最多增加 5000 步，训练时间上限 150 分钟。
仍采用原训练/验证划分、固定矩阵和相同配方。新固定 52 张原图未参与本项目训练、选模及以前测试。
本次正式续训实际耗时 125.16 分钟，停止于总计 10000 步；日志与审计已取回。新增南京网络场景结果和命令见 `NANJING_WEB30.md`。
GitHub 同步不代表已提交课程平台。旧 ZIP 保留；南京扩展的最终压缩包单独生成，并附文件哈希和三个微调推理权重以及 10000 步完整恢复权重。
