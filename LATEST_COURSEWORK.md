# 最新课程项目交付：HQS续训与双因素消融（2026-10-13）

旧报告、PPT、实验与证据清单均保持。这里是当前新增实验的入口，原COURSEWORK.md和旧日期的要求核对文件是历史版本。

- [英文技术报告：6页](output/pdf/technical_report_ablation_20261013.pdf)
- [英文PPT：12页](output/presentation/course_presentation_ablation_20261013.pptx)
- [中文逐页讲稿：计划10分钟](coursework/讲稿_续训与消融_20261013.md)
- [中文完整结果、每段视频和复现命令](coursework/HQS_ABLATION_RESULTS_20261013.md)
- [大作业要求核对](coursework/REQUIREMENTS_STATUS_ABLATION_20261013.md)
- [既有三页文献阅读报告](output/pdf/literature_review_layoutfix_20261010.pdf)

## 实际做完的工作

原配方完整恢复10000→12000，新增2000步，验证选择11500。另做训练噪声×阶段监督四组消融，每组两个训练种子、各800步；正式新增8400次更新。所有六个测量因子固定。2+1恢复和连续3步逐位一致，旧/新入口5→6逐位一致，逐步裁剪/采样/随机配对通过。

52张预先锁定、此前项目未评测DIV2K原图主要测试；110张已观察原图/间隔视频帧回归测试。25960条主指标、155760条阶段指标保存，所有最终浮点重建重新计算PSNR/SSIM与保存值最大差0；各数据集/视频等权，所有失败保留。

独立主要集10%采样，选中续训模型对官方HQS的平均MSE下降：干净4.12%，较强噪声12.33%；对之前10000步选中模型仅0.21% / 0.09%。新增收益小，191/572条件退步；强噪声SSIM也有小幅下降。固定800步的加噪消融在强噪声下平均改善8.45%，干净条件退步0.74%；阶段监督影响小且随条件变号。这些实验不支持普遍5–10%提升。

回归原推理90分钟超时，88张完整；只补余下22张，再全量复核。原失败日志、非零exit和CSV归档保留，没有伪造为全程成功。补测采用空闲RTX2080Ti，原推理RTX3090；96条原有重叠记录的最大PSNR差2.42e-6dB、SSIM差5.57e-7。不同GPU的耗时不混成单一卡速度比较。

## 远端保留与核验

SSH经upload到**192.168.1.244**，容器**luozc_mlvc**，根目录`/workspace/ProxUnroll-main/runs/hqs_ablation_20261013`。
`continuation/best.pth`是11500验证选中推理权重，`continuation/last.pth`是12000完整恢复状态；`arms/`是八组消融的best/last与训练记录。全部原图、测量矩阵、checkpoint、大数组、PNG、CSV、环境版本和哈希在远端。Git保留代码、小型审计/CSV和材料，未上传Mac的.venv或临时构建。

本地核验：

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m coursework.verify_repository_snapshot
.venv/bin/python -m coursework.verify_ablation_materials
.venv/bin/python -m coursework.audit_ablation_delivery
```

后两项核验完整新增评测的聚合、来源和材料；第2项核验288个历史冻结文件未变。新版哈希清单见`runs/hqs_ablation_20261013/final_evidence.json`。全部PPT图表为可编辑原生图表，并含数据工作簿；已通过包/几何/字体检查、重新导入与逐页预览，未声称PowerPoint实机编辑保存测试。

## 独立20000步续训

用户另授权12000→20000，并要求不阻塞上面的goal、完成后再查看效果。该训练已退出0并完成训练审计，权重与完整状态保存在`/workspace/ProxUnroll-main/runs/hqs_20000_20261013/finetune`。此次不分析其测试效果、不写入上述报告；[运行说明](coursework/HQS_20000_20261013.md)保留原始启动记录和ETA。它与报告中的12000/800步实验分开。

网络照片属于自行整理的外部作者图像，不是亲自拍摄。视频帧独立重建，不是时序视频模型。AI协助声明保留；学生仍须理解代码、计时排练、完成课堂汇报和提交。未代交课程平台。
