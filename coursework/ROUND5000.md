# 5000 步噪声微调运行指南

用户目标：相对第一轮 100 步最佳 HQS 的算术平均重建 MSE 下降 5–10%。
本次按用户追加授权训练第二轮总计 5000 步，包含前阶段实际完成的 946 步；不是再加 5000 步。
原方案及早期结果保留在 `ROUND2_PROTOCOL.md`、`ROUND2.md` 和 `runs/lab244_round2_20261009`。
预算扩展记录在 `EXTEND5000_PROTOCOL.md`，扩展入口与原入口实际恢复一步的模型、Adam、scheduler、RNG、损失和验证值逐位一致。

实际服务器 **192.168.1.244**，容器 **luozc_mlvc**，项目 **/workspace/ProxUnroll-main**。
本次独立目录 **runs/lab244_5000_20261009**。220 的容器仍是 luozc_nvenc_new，两者共享 /workspace 和 /datasets。
复用项目独立 `.venv-linux`：Python 3.11.11、PyTorch 2.6.0+cu124；不修改其他项目的环境。
GPU5 RTX 3090，UUID GPU-16f6ac58-4d8c-8845-0886-1dc146a0327e，启动前实际核对空闲。
每进程 CUDA 分配上限为该卡的 65%，训练峰值以完成日志为准。

本地入口会校验 SSH 实际 peer 身份；通过 upload 和现有控制连接访问 244，不把 220 的连接当作 244：

```bash
python3 -m coursework.remote244 'COMMAND'
python3 -m coursework.remote220 sync
```

同步只新增或更新已核验的项目文件；遇到未知远端更改会停止覆盖，不删除远端文件。Mac .venv、缓存、Git 元数据不上传。
共享盘仍需在 244 上独立核对哈希。

## 实际训练

所有命令在容器 /workspace/ProxUnroll-main 内执行；复现时输出必须改成新的未存在目录。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-16f6ac58-4d8c-8845-0886-1dc146a0327e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv-linux/bin/python -m coursework.finetune_5000 \
  --manifest runs/lab244_5000_20261009/data/train_val.json \
  --resume runs/lab244_round2_20261009/finetune/last.pth \
  --output runs/NEW_RUN/finetune \
  --max-steps 5000 --val-every 250 --max-minutes 120 --patience 5000
```

实际任务通过 launch_lab_job 的 nohup 后台运行，外层 timeout 7500 秒，断开 SSH 继续执行。
日志 `logs/finetune_5000.log`、`.pid` 和 `.exit`；结束后有 `finetune/COMPLETE`、`summary.json`、`training_audit.json`。
启动不等于完成，正式比较还会检查实际 completed_steps=5000、退出码 0 和冻结矩阵审计。
前阶段从第一轮 best 初始化**新 Adam**，续训则恢复第二轮完整 Adam、scheduler、RNG、步数及最佳状态。
冻结所有六个 H/W 因子，学习率 1e-5；只训练重建部分。
直接灰度真值阶段 RMSE，前五段各 .01，最终段 .95；比例和噪声按锁定随机种子独立抽样。
扩展入口只改变总步数、时间上限、验证间隔及早停上限，梯度更新规则不变。

## 数据与模型选择

800 DIV2K 官方训练原图，16 张单独验证选择原图；按原图源哈希分离。
固定 32 张新测试原图此前未评测，也不参与此次选模型；官方预训练暴露尚未完整审计。
Kodak24、HEVC B 五段 / E 三段视频各三个间隔原始帧为**已观察过的回归测试**。
共 80 张测试图，均中心最大正方形裁剪、area 缩放为 256×256 灰度。
这些是公开及实验室已有数据，不是自采数据；逐帧处理没有利用相邻帧信息。
验证为 16 原图 × 两个比例 × 三档噪声固定测量。
按混合条件平均验证 PSNR 选 best，且干净验证不得比初始化退步超过 .03 dB。
只在训练结束后读取测试结果，不用测试结果挑模型或修改训练。

## 固定测试

```bash
.venv-linux/bin/python -m coursework.audit_round2 runs/lab244_5000_20261009
.venv-linux/bin/python -m coursework.run_experiments \
  --manifest runs/lab244_5000_20261009/data/test.json \
  --output runs/NEW_RUN/comparison --device cuda \
  --extra-checkpoint hqs_round1=runs/lab220_20261009/finetune/best.pth \
  --extra-checkpoint hqs_finetuned=runs/lab244_5000_20261009/finetune/best.pth \
  --methods hqs,hqs_round1,hqs_finetuned
.venv-linux/bin/python -m coursework.verify_lab_run runs/NEW_RUN/comparison --compare
.venv-linux/bin/python -m coursework.analyze_round2 runs/NEW_RUN/comparison
```

三种权重共享每个测量、矩阵、噪声、指标规则和 GPU；保留 2640 条主要指标和 15840 条阶段指标。
五档干净采样率；10% 采样另测 .01/.05 噪声各三种种子。
MSE 百分比为 `100*(1 - 新模型平均 MSE / 参考模型平均 MSE)`，不能转换平均 PSNR 后当作算术平均 MSE。
先按图平均噪声种子，再按视频平均帧，HEVC 类别按视频等权。
官方 HQS 为第二参考；第一轮 100 步 HQS 是这次目标的主要参考。

## 文件索引

- `finetune/best.pth`：验证选中的模型；`last.pth`：第 5000 步完整恢复状态，二者可能不同。
- `finetune/train.csv`、`validation.csv`、`summary.json`、`config.json`：损失、验证、耗时、显存及训练契约。
- `comparison/metrics.csv`、`stages.csv`、`reconstructions/`：原始结果和 PNG/浮点数组。
- `comparison/mse_dataset_summary.csv`、`mse_video_summary.csv`、`mse_paired_deltas.csv`：所有数据集、视频、逐图变化。
- `comparison/round2_negative_cases.csv`：相对第一轮全部退步条件，不只展示提升图片。
- `comparison/verification.json`、`training_audit.json`、`extension_resume_verification.json`：重算、冻结因子、模型选择和恢复审计。
- `data/`、`environment.txt`、`server-preflight.txt`、`continuation_lock.json`：数据来源、分组、哈希、环境和服务器证据。
- `logs/`：后台任务的日志、PID 和退出状态；前阶段日志在原 round2 目录。

```bash
python3 -m coursework.fetch_lab_run --server 244 \
  --run runs/lab244_5000_20261009 \
  --parts data,comparison,finetune,logs,training_audit.json,extension_resume_verification.json,continuation_lock.json \
  --include-weights
```

默认保留远端 .npy 浮点数组不回传；需要时加 --include-arrays。
新增英文报告 / PPT 使用带 5000 和日期的新文件名，保留此前所有版本。
没有课程平台提交、公开发布或不设界限的训练。

## 新测试集的四方法基准

最佳权重固定后，还对同一批 32 张新测试图补齐伴随重建、DCT-FISTA、官方 HQS 和 ADMM。
使用已核验空闲的 GPU6 RTX 3090（GPU-62160106-d537-698b-2e4a-94b4196592a9），不增加训练步骤。
`data/test_fresh_controls.json` 只是固定 80 张清单中的 DIV2K 子集，不另挑选表现好的图片。
`fresh_controls_spec.json` 记录目的、清单和已固定权重的哈希。

```bash
.venv-linux/bin/python -m coursework.run_experiments \
  --manifest runs/lab244_5000_20261009/data/test_fresh_controls.json \
  --output runs/NEW_RUN/fresh_baseline --device cuda \
  --methods adjoint,fista_dct,hqs,admm
.venv-linux/bin/python -m coursework.verify_lab_run runs/NEW_RUN/fresh_baseline
.venv-linux/bin/python -m coursework.verify_control_overlap runs/lab244_5000_20261009
```

完整新基准已产生 1408 条主要指标、4224 条阶段指标；1408 个浮点重建均已独立重算，指标最大误差 0。`fresh_baseline/` 保存四方法的所有结果。
与旧的 48 张四方法基准合起来，80 张图都有四种原始方法的对比。
`control_overlap_verification.json` 核对两张 GPU 上 352 个官方 HQS 重叠条件，测量、矩阵、权重一致，PSNR/SSIM 最大差异均为 0。

## 已完成结果

正式训练完成第二轮总计 **5000 步**：前阶段 946 步，完整状态续训 4054 步。
验证选中 **4500 步 best.pth**；last.pth 保留第 5000 步完整 Adam、scheduler、RNG 等恢复状态。
两段正式训练共 **121.8 分钟**，峰值分配显存 **10.84 GiB**。实际 Adam 步数、连续 5000 条训练记录、冻结的六个测量因子和更新的重建参数已独立审计。
训练、三种 HQS 权重评测及新增四方法基准均退出 0；完成后项目 GPU 进程已退出。

下表是 **10% 采样、相对上一轮 100 步最佳 HQS 的算术平均 MSE 下降**，正数表示误差更小：

| 数据 | 无噪声 | 噪声 .01 | 噪声 .05 |
|---|---:|---:|---:|
| Kodak24 | 1.72% | 1.92% | 16.75% |
| HEVC B | 0.94% | 1.16% | 16.99% |
| HEVC E | 4.86% | 4.88% | 26.38% |
| 新增 DIV2K 32 张 | 1.84% | 1.92% | 11.89% |

主要判断使用未参与选模型的新增 DIV2K：干净 10% 的 1.84% 下降，配对 95% bootstrap 区间为 [0.68%, 3.48%]，**没有达到最低 5%**。
强噪声条件改善更明显，但不能据此宣称普遍提升 5–10%。只有一个训练种子；Kodak/HEVC 是已观察过的回归测试，官方预训练暴露没有完整审计。
相对上轮 **108 个逐图/条件组合**退步，已全部保留；Kimono1 的干净 10% 视频平均 MSE 上升约 0.45%。

三种 HQS 的 80 张测试共有 **2640 条主要 / 15840 条阶段指标**；2640 个浮点重建重算指标最大误差 **0**。
原来的 48 张四方法基准保持原样。新增 32 张四方法基准补齐后，全部 80 张都有伴随、DCT-FISTA、官方 HQS 和 ADMM 控制。

最佳权重 SHA256：`3e85cbd683c10487104f39978abb325f3fb935052e9a0fc669aad1af6f7d95cb`。
恢复权重 SHA256：`497887636e72404b7ba472eac6001b055c3174c10364519d8b011690b5fd426d`。

新版交付文件：`output/pdf/technical_report_5000_20261009.pdf`（英文技术报告六页）、
`output/presentation/course_presentation_5000_20261009.pptx`（十二页、有讲稿和可编辑数据图表）、
`coursework/DEFENSE_5000_ZH.md`、`coursework/PRESENTATION_GUIDE_5000.md`。
`final_artifact_checks.json` 保存页数、图表数值、原件保护及最终文件哈希检查。
