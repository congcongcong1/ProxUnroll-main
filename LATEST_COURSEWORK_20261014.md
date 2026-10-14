# 最新课程项目交付（2026-10-14）

这是20000步评测与技术报告连续排版修订的新增入口。2026-10-13的入口、报告、PPT和证据清单原样保留。

- [英文技术报告：4页，连续双栏](output/pdf/technical_report_flow_20261014.pdf)
- [中文20000步完整结果，含每数据集、每视频](coursework/HQS_20000_RESULTS_20261014.md)
- [技术报告排版修订与核验](coursework/REPORT_FLOW_20261014.md)
- [原12页英文PPT](output/presentation/course_presentation_ablation_20261013.pptx)
- [原中文逐页讲稿](coursework/讲稿_续训与消融_20261013.md)
- [原三页文献报告](output/pdf/literature_review_layoutfix_20261010.pdf)
- [上一轮交付入口](LATEST_COURSEWORK.md)

继续训练实际达到20000次更新，新增8000步，270.32分钟；按原验证规则选择19500步，终点20000另保留。相对上一轮验证选择11500步，在主要52图10%采样下，无噪声MSE增加0.831%，较强噪声增加0.407%，SSIM略升。本次没有进一步取得目标中的5–10%误差下降；不同回归数据集有微小、混合变化。不能把对官方模型的既有改善说成这8000步带来的新增改善。

新的两个状态在162张既有测试图/间隔视频帧上完成3564次最终重建、21384条阶段指标，全部浮点指标重新核验。官方、10000和11500参考结果复用已冻结、已全量核验的CSV，并通过新五状态推理等价检查及逐条件测量哈希配对。两个新状态均在推理前锁定，不依据测试选模型。南京网络图片仍是外部作者拍摄、团队自行整理，HEVC仍是单图重建。

新版技术报告保留四图、六表、原有对比与消融、失败案例及AI声明，补入本次完整结果。原PPT和讲稿描述先前12000/800步实验；汇报本次20000步扩展时请补充上面的结果，不能把原PPT数字解释成20k结果。

远端：192.168.1.244，容器luozc_mlvc；根目录`/workspace/ProxUnroll-main`。
新评测：`runs/hqs_20000_eval_20261014/`，权重：`runs/hqs_20000_20261013/finetune/{best,last}.pth`。
新版材料核验：`runs/report_flow_20261014/`。大权重、原始重建NPY与PNG继续保留远端；本地与Git保存代码、报告、CSV和小型审计，不复制Mac环境。

```bash
.venv/bin/python -m coursework.audit_20000_eval
.venv/bin/python -m coursework.audit_flow_report
.venv/bin/python -m coursework.verify_repository_snapshot
```

完整哈希与远端同步回执见`runs/report_flow_20261014/final_evidence.json`及`final_sync.json`。没有新增训练或代交课程平台。
