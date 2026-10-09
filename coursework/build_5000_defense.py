"""Generate Chinese defense notes only from verified final experiment facts."""
import json
from pathlib import Path
from coursework.run_experiments import ROOT


def main():
    b=ROOT/'runs/lab244_5000_20261009'
    facts=json.loads((b/'report_facts.json').read_text())
    audit=json.loads((b/'comparison/verification.json').read_text())
    train=facts['training'];analysis=facts['analysis']
    assert train['completed_steps']==5000 and audit['float_arrays_recomputed']==2640
    def result(dataset,sigma):
        return next(r for r in analysis['datasets'] if r['dataset']==dataset and float(r['cr'])==.1 and float(r['sigma'])==sigma and r['reference']=='hqs_round1')
    rows=['| 数据 | 干净图 | 噪声 .01 | 噪声 .05 |','|---|---:|---:|---:|']
    for d,label in [('Kodak','Kodak24'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('DIV2K_fresh','新增 DIV2K 32 张')]:
        rows.append('| '+label+' | '+' | '.join(f'{result(d,s)["mse_reduction_percent"]:+.2f}%' for s in [0.,.01,.05])+' |')
    fresh=result('DIV2K_fresh',0.)
    text=f'''# 5000 步实验答辩补充

这次做的是电脑模拟单像素成像。原图经过同一套测量矩阵得到少量测量值，再比较重建方法。
官方结构和预训练权重来自 ProxUnroll；我们新增的是公平评测、可复现的数据划分和固定测量矩阵的 HQS 微调。
没有光学硬件实验，也没有使用相邻帧信息的视频模型。Codex 协助了代码、实验、分析和材料制作。

## 这次实际训练了多少

第二轮共更新 5000 步。前一阶段完成 946 步，新阶段从完整状态恢复，做到第 5000 步。
本轮采用验证选中的第 {train['best_step']} 步权重；最后一份 last.pth 保存第 5000 步恢复状态。
两段正式训练共耗时 {facts['total_training_wall_seconds']/60:.1f} 分钟，峰值分配显存 {train['peak_memory_mib']/1024:.2f} GiB。
服务器是 244，容器 luozc_mlvc，RTX 3090。独立运行目录 runs/lab244_5000_20261009。
所有六个测量因子冻结，只训练重建部分；不能把结果归因于重新学习测量矩阵。

## 数据有没有泄漏

800 张 DIV2K 训练原图、16 张验证选择原图、32 张此前未评测的新测试原图，按原始文件哈希分开。
同一张图的不同裁剪不会跨训练与验证/测试；HEVC 全视频只属于测试。
Kodak24 和 HEVC 的成绩在第一轮已看过，因此这次是回归测试，不能说从未观察过。
新测试的 32 张图未用于这次选模型或先前评测，但官方预训练的数据暴露未完整审计。
数据来自公开数据及实验室已有视频，不能称为自行拍摄。

## 5–10% 是什么，实际到了没有

百分比是相对上轮 100 步最佳 HQS 的算术平均 MSE 下降：
100 × (1 − 新模型平均 MSE / 上轮模型平均 MSE)。
PSNR 是对数量；不能说 PSNR 数字提升 5%，也不能把平均 PSNR 增量换算成算术平均 MSE 的下降。
以下为 10% 采样，正数表示误差更小，负数表示误差更大：

{chr(10).join(rows)}

新增 DIV2K 干净 10% 的点估计为 {fresh['mse_reduction_percent']:+.2f}%，
95% 配对 bootstrap 区间 [{fresh['bootstrap_low']:+.2f}, {fresh['bootstrap_high']:+.2f}]%。
请求的最低 5% 按点估计{'达到' if fresh['target_at_least_5_percent'] else '未达到'}；
这是一次训练种子和有限测试的结果，不代表所有图、采样率或噪声都提高了同样幅度。
相对上轮有 {analysis['negative_image_conditions_vs_round1']} 个逐图条件退步，均保留在 CSV。

## 为什么视频成绩不能简单把所有帧平均

每段视频预先固定抽取三个间隔帧。先平均同图的噪声重复，再平均视频内的帧，最后按视频等权。
HEVC B 五段，HEVC E 三段；每段结果单独保留，避免相邻帧数量给某段视频过多权重。
模型每次只看单张灰度图的测量，不利用前帧或后帧，所以属于单图重建。

## 如何证明真的训练并公平测试

恢复检查实测比较模型、Adam、scheduler、CPU/CUDA RNG、裁剪、噪声哈希和损失，结果逐位相同。
日志、完整 checkpoint、GPU 实际进程及退出码证明启动和完成。
独立审计确认测量因子不变，重建参数更新，best 由验证选择。
三种权重共享矩阵和测量；2640 个浮点重建逐个重算 PSNR/SSIM，误差上限为 {audit['max_metric_error']:.3g}。
15840 条阶段记录的最终阶段也与主要指标一致。
未用测试结果选权重，没有丢掉退步图片，也没有替学生提交课程平台。

完整命令和文件位置见 ROUND5000.md；新版英文技术报告六页，演示十二页，旧版本均保留。
'''
    output=ROOT/'coursework/DEFENSE_5000_ZH.md';output.write_text(text);print(output)


if __name__=='__main__':main()
