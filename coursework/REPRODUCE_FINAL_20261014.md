# 最终材料与实验复现说明（2026-10-14）

代码仓库：https://github.com/congcongcong1/ProxUnroll-main。
当前完整交付入口是仓库根目录的`COURSEWORK_DELIVERY_20261014.md`。
旧日期的报告、PPT、讲稿、协议和审计属于历史证据，不作为20000步最新结果说明。

## 两种复现范围

材料压缩包包含可阅读的报告、可编辑PPT、讲稿与汇总。完整代码和原始CSV在Git仓库。大权重、原图、全部重建数组和实验环境保留在实验室服务器。
只解压材料包即可阅读、汇报；完全重新推理需要实验室数据和权重访问权限。

核验服务器身份使用`coursework.remote244`，经upload跳板到192.168.1.244，在容器luozc_mlvc执行，工作目录`/workspace/ProxUnroll-main`。不能根据以前的连接记录假定服务器/GPU当前仍然空闲。代码同步使用`sync_named_files`先检查全部冲突，保留不同文件，不使用删除远端文件的选项。

独立Linux环境为`.venv-linux`，Python 3.11.11、Torch 2.6.0+cu124。完整环境记录在`runs/lab244_5000_20261009/environment.txt`及后续运行provenance里。原图在容器`/datasets`，只读使用；处理后测试清单和图像留在各运行目录。Mac的`.venv`不上传。

## 只核验既有结果，不重新训练

在完整项目本地使用`.venv/bin/python`，远端使用`.venv-linux/bin/python`：

```bash
.venv/bin/python -m coursework.verify_repository_snapshot
.venv/bin/python -m coursework.audit_20000_eval
.venv/bin/python -m coursework.audit_flow_report
.venv/bin/python -m coursework.verify_final_materials
```

前一轮冻结审计的288个文件、消融262个文件与新评测/技术报告81个文件保持原字节，最新材料另建哈希清单。新版材料核验从原始CSV重新提取期望值，检查PPT原生图表、结果表、12页备注、中文讲稿的600秒计划及PDF页数/来源声明。它也检查逐页预览哈希，不能拿旧预览冒充新PDF/PPT视觉核验。

本轮不追加训练，也不修改指标程序。已完成20000次更新、验证选择19500；`best.pth`用于推理，`last.pth`保留20000终点和优化器等恢复状态。完整恢复与从官方权重初始化不同。

## 权重与来源

| 实验名称 | 实际状态 | 远端项目内路径 |
|---|---|---|
| hqs | 官方发布HQS | weight/hqs_proxunroll.pth |
| admm | 官方发布ADMM | weight/admm_proxunroll.pth |
| hqs_10000 | 验证选中10000 | runs/lab244_10000_20261009/finetune/best.pth |
| hqs_12000 | 12000预算、验证选中11500 | runs/hqs_ablation_20261013/continuation/best.pth |
| hqs_20000 | 20000预算、验证选中19500 | runs/hqs_20000_20261013/finetune/best.pth |
| last_20000 | 固定20000终点、完整恢复 | runs/hqs_20000_20261013/finetune/last.pth |

新best SHA-256：`1464249fc38656689f0ee3baa59e6be0a3d8948ad5400b76160b53049aad8731`。
新last SHA-256：`e921c0b60a00abc794d14f3420a3e6b3efdaa3565e4b19d80c9bae4f089b6f9b`。
其余权重、原图/处理图、六个因子、源码和测量哈希见`runs/hqs_20000_eval_20261014/lock.json`及各阶段`provenance.json`。模型中的测量因子可训练，公平微调时六个因子全部冻结。

## 新目录最小推理示例

下面是复现说明，不会自动执行。先在服务器核验GPU UUID、占用、可写空间和数据/权重，再将占位符改成实际空闲卡。输出目录必须全新，避免覆盖既有证据。

```bash
export CUDA_VISIBLE_DEVICES=GPU_UUID_OF_AN_IDLE_DEVICE
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

timeout 300 .venv-linux/bin/python -m coursework.evaluate_manifest \
  --manifest runs/lab244_10000_20261009/data/test_fresh.json \
  --output runs/reproduce_final_smoke \
  --methods hqs,hqs_10000,hqs_12000,hqs_20000,last_20000 \
  --extra-checkpoint hqs_10000=runs/lab244_10000_20261009/finetune/best.pth \
  --extra-checkpoint hqs_12000=runs/hqs_ablation_20261013/continuation/best.pth \
  --extra-checkpoint hqs_20000=runs/hqs_20000_20261013/finetune/best.pth \
  --extra-checkpoint last_20000=runs/hqs_20000_20261013/finetune/last.pth \
  --smoke

.venv-linux/bin/python -m coursework.verify_lab_run runs/reproduce_final_smoke
```

完整主要52图使用同一清单去掉`--smoke`、采用另一个新输出目录；110图回归清单是`runs/hqs_ablation_20261013/regression_manifest.json`。默认五个干净采样率，10%下噪声.01/.05各三种种子。正式复现需另设有限时间上限并按最小推理实测估算，不直接运行长期训练。

四方法重新推理可用`--methods adjoint,fista_dct,hqs,admm`；同图同条件的测量哈希应完全匹配。原四方法校园18图的CSV仍在`runs/nanjing_web30_20261009/comparison/`。不重新用测试数据选择FISTA参数。

## 实际完成记录在哪里

- 20000主线训练：`runs/hqs_20000_20261013/finetune/{train.csv,validation.csv,config.json,summary.json,audit.json,COMPLETE}`及父目录`logs/finetune_20000.log/.exit`。
- 本次评测：`runs/hqs_20000_eval_20261014/evaluation_primary`与`evaluation_regression`，包括原始CSV、全部负例、每视频表、浮点NPY和展示PNG；日志在`logs/{smoke,primary,regression,finalize,audit}.log/.exit`。
- 800步消融与前次续训：`runs/hqs_ablation_20261013/`，旧超时、补测和跨GPU核验都保留。
- 本次最终材料：`runs/materials_final_20261014/`的构建、数值与视觉记录，旧技术报告视觉记录仍在`runs/report_flow_20261014/`。

## 材料重建

中文讲稿/问答/核对表：`.venv/bin/python -m coursework.build_final_notes`。
文献PDF：`.venv/bin/python -m coursework.build_final_literature`，重建后必须重新渲染并实际逐页检查。
PPT构建器：`coursework/build_final_presentation.mjs`，使用Codex桌面提供的Node与`@oai/artifact-tool`运行时。新PPT重建应设置`PRESENTATION_FILENAME`为全新文件名，finalizer不覆盖旧文件；PPT图表保留可编辑原生对象与工作簿。运行时工具是材料编辑依赖，不是模型训练必须使用的环境。

没有进行PowerPoint原生应用打开、编辑、保存测试；包结构、原生图表数据、重新导入及逐页预览已核验。学生仍需在自己的演示电脑上打开、计时排练、确认身份信息，并自行提交课程平台。
