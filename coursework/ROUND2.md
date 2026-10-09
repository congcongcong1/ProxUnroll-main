# 第二轮噪声微调运行指南

目标和预先固定的方案见 `ROUND2_PROTOCOL.md`。提升百分比定义为相对第一轮 100 步最佳 HQS 的平均 MSE 下降；官方 HQS 单独作为第二个参考。
这次使用 244（192.168.1.244），容器 `luozc_mlvc`；220 仍为 `luozc_nvenc_new`。
两者共享 `/gpfs3/area1/luozc -> /workspace` 和 `/gpfs3/area1/shared -> /datasets`，代码、数据和结果在同一份存储上。

项目根目录 `/workspace/ProxUnroll-main`，独立运行目录 `runs/lab244_round2_20261009`。
244 的第 5 张 RTX 3090 启动前为空闲卡；其 UUID 为 `GPU-16f6ac58-4d8c-8845-0886-1dc146a0327e`。
复用项目 `.venv-linux`，验证了 Python 3.11.11、PyTorch 2.6.0+cu124、CUDA 12.4、GPU 实际运算和 pip check。
未修改其他项目环境或停止其他任务。

本地经验证的 244 入口：

```bash
python3 -m coursework.remote244 'COMMAND'
python3 -m coursework.remote220 sync
```

同步经 220 写入共享项目，并比较哈希；244 再独立核对新源码、方案和初始化权重哈希。
现有控制连接为 `/tmp/mlvc-244-sync.sock`，入口通过 `upload`。连接失效时先认证 244，不能拿 220 的连接代替。

容器内先设置：

```bash
cd /workspace/ProxUnroll-main
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-16f6ac58-4d8c-8845-0886-1dc146a0327e
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
```

数据清单固定了 800 张训练原图、16 张模型选择图和 32 张新测试图，均按原图哈希分开。
Kodak24 和 8 段 HEVC 的 24 个原始间隔帧保留作回归测试，不能再说从未看过这些测试成绩。
32 张 DIV2K 新测试图未参与本次训练、模型选择或首轮评测；官方预训练的数据暴露仍未完整审计。

以下命令输出目录需换成未存在的新目录，保护本轮实际结果：

```bash
.venv-linux/bin/python -m coursework.prepare_round2_data --output runs/NEW_ROUND/data
.venv-linux/bin/python -m pytest tests/test_round2.py -q
.venv-linux/bin/python -m coursework.finetune_round2 \
  --manifest runs/NEW_ROUND/data/train_val.json \
  --init runs/lab220_20261009/finetune/best.pth \
  --output runs/NEW_ROUND/train_smoke \
  --max-steps 3 --val-every 3 --max-minutes 3
```

实际训练链为 3 步 -> 完整恢复至 4 步 -> 正式续训；另外连续 4 步仅作为恢复一致性参考。
`resume_verification.json` 检查模型、Adam、scheduler、CPU/CUDA RNG、裁剪、噪声测量哈希、损失与验证值一致。
新一轮 `--init` 创建新 Adam；`--resume` 才恢复本轮完整训练状态，不混用两种概念。

```bash
.venv-linux/bin/python -m coursework.finetune_round2 \
  --manifest runs/lab244_round2_20261009/data/train_val.json \
  --resume runs/lab244_round2_20261009/resume_smoke/last.pth \
  --output runs/NEW_ROUND/finetune \
  --max-steps 1000 --val-every 100 --max-minutes 25
```

脚本设每次最多 1000 步、30 分钟；本次正式续训使用 25 分钟，外层 timeout 1800 秒。
所有测量矩阵冻结，训练直接对灰度真值计算六阶段 RMSE。
训练同时扩大数据、改变损失目标和加入测量噪声，不能把结果提升单独归因给其中一个因素。
按干净/含噪验证平均 PSNR 选 best，干净验证退步不得超过 0.03dB；4 次连续没有符合条件的改善提前停止。

训练完成、权重选择结束后，才开展统一测试：

```bash
.venv-linux/bin/python -m coursework.run_experiments \
  --manifest runs/lab244_round2_20261009/data/test.json \
  --output runs/NEW_ROUND/comparison --device cuda \
  --extra-checkpoint hqs_round1=runs/lab220_20261009/finetune/best.pth \
  --extra-checkpoint hqs_finetuned=runs/lab244_round2_20261009/finetune/best.pth \
  --methods hqs,hqs_round1,hqs_finetuned
.venv-linux/bin/python -m coursework.verify_lab_run runs/NEW_ROUND/comparison --compare
.venv-linux/bin/python -m coursework.analyze_round2 runs/NEW_ROUND/comparison
```

测试覆盖 80 张图、五种干净采样率和 10% 下两档噪声各三种种子，每个条件三种 HQS 权重共享测量。
MSE 先按图、视频平均，再算均值的比值；不能把平均 PSNR 增量直接转换成算术平均 MSE 百分比。

本轮文件索引（均位于远端运行目录，同时可由 fetch 工具回传）：

- `finetune/best.pth`：验证选中的权重；`finetune/last.pth`：完整恢复状态。
- `finetune/train.csv`、`validation.csv`、`summary.json`：训练记录、验证指标和实际资源消耗。
- `logs/pilot.log`、`finetune.log`、`comparison.log` 及对应 `.pid/.exit`：脱离 SSH 的真实运行证据。
- `comparison/metrics.csv`、`stages.csv`、`reconstructions/`：原始指标和重建数组/图片。
- `comparison/mse_dataset_summary.csv`、`mse_video_summary.csv`、`mse_paired_deltas.csv`：数据集、逐视频、逐图 MSE 变化。
- `comparison/round2_negative_cases.csv`：相对第一轮退步的条件；若没有退步则该文件不生成。
- `comparison/verification.json`、`training_audit.json`、`resume_verification.json`：独立复核。
- `environment.txt`、`server-preflight.txt`、`data/protocol.json`：环境、硬件和分组规则。

```bash
python3 -m coursework.fetch_lab_run --server 244 \
  --run runs/lab244_round2_20261009 --parts comparison,finetune,logs,training_audit.json \
  --include-weights
```

此命令默认不回传 `.npy`；所有浮点数组保存在远端，可加 `--include-arrays` 回传。
本轮没有上传课程平台或公开发布。
