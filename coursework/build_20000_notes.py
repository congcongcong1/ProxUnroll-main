"""Plain Chinese interpretation of all completed 20000-update comparisons."""
import csv
import json
from pathlib import Path
import numpy as np

from coursework.run_experiments import ROOT

BASE=ROOT/'runs/hqs_20000_eval_20261014'


def main():
    result=json.loads((BASE/'results.json').read_text())
    audit=json.loads((BASE/'completion_audit.json').read_text())
    assert audit['results_sha256']==__import__('hashlib').sha256((BASE/'results.json').read_bytes()).hexdigest()
    def pair(phase,dataset,sigma,reference,candidate='hqs_20000'):
        return next(r for r in result['phases'][phase]['paired_summary'] if r['dataset']==dataset
                    and r['cr']==.1 and r['sigma']==sigma and r['reference']==reference and r['candidate']==candidate)
    lines=['# 20000步预算模型：完整评测结果（2026-10-14）',
           '训练真的完成20000次更新，新增8000次，用时270.32分钟，峰值10.84 GiB，退出0并通过训练审计。使用原Adam、scheduler、随机状态和固定六个测量因子，配方不变。验证规则选中19500步；“20000预算模型”指这个选中checkpoint，20000步终点单独报告。',
           '主要52图和回归110图都已在上一轮评测过；本次是继续训练后的同集对照，不是新的首次未见测试。不使用这些测试结果选模型或改训练配方。',
           '## 主要52图：验证选中19500步模型',
           '正百分比表示平均MSE下降；负数表示误差增加。先按原图内平均噪声重复，再按原图/整段视频等权。PSNR是dB差值，SSIM是绝对差，不是百分比。',
           '| 测试条件（10%采样） | 对官方MSE下降 | 对10000步MSE下降 | 对上一轮11500步MSE下降 | PSNR变化 vs11500 | SSIM变化 vs11500 |',
           '|---|---:|---:|---:|---:|---:|']
    for sigma,label in [(0.,'干净'),(.01,'噪声.01'),(.05,'噪声.05')]:
        rows=[pair('primary','DIV2K_holdout_10000',sigma,r) for r in ['hqs','hqs_10000','hqs_12000']]
        lines.append(f'| {label} | '+' | '.join(f'{r["mse_reduction_percent"]:+.3f}%' for r in rows)+f' | {rows[-1]["delta_psnr"]:+.4f} dB | {rows[-1]["delta_ssim"]:+.6f} |')
    negative=result['phases']['primary']['negative_selected_vs_12000']
    lines += [f'在572个原图/测量条件中，相对上一轮11500选中模型，19500选中模型有{negative}个条件MSE增加。全部负例保存。',
              '## 终点与验证选中模型分开看',
              '| 状态 | 干净10% MSE下降 vs11500 | 噪声.05 MSE下降 vs11500 |', '|---|---:|---:|']
    for candidate,label in [('hqs_20000','验证选中19500'),('last_20000','固定20000终点')]:
        values=[pair('primary','DIV2K_holdout_10000',s,'hqs_12000',candidate)['mse_reduction_percent'] for s in [0.,.05]]
        lines.append(f'| {label} | {values[0]:+.3f}% | {values[1]:+.3f}% |')
    lines += ['## 每个回归数据集',
              '| 数据集 | 干净10% MSE下降 vs11500 | 噪声.05 MSE下降 vs11500 |', '|---|---:|---:|']
    for dataset,label in [('Kodak','Kodak24'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('DIV2K_fresh','DIV2K32'),('NJU_campus_web','校园18图'),('Nanjing_scenic_web','景区12图')]:
        values=[pair('regression',dataset,s,'hqs_12000')['mse_reduction_percent'] for s in [0.,.05]]
        lines.append(f'| {label} | {values[0]:+.3f}% | {values[1]:+.3f}% |')
    videos=list(csv.DictReader((BASE/'evaluation_regression/video_mse_summary.csv').open()))
    lines += ['## 每段视频', '| 视频 | 间隔帧数 | 干净10% MSE下降 vs11500 | 噪声.05 MSE下降 vs11500 |','|---|---:|---:|---:|']
    for sequence in sorted({r['sequence'] for r in videos if r['dataset'] in ['HEVC_B','HEVC_E']}):
        values=[]
        for sigma in [0.,.05]:
            lookup={r['method']:r for r in videos if r['sequence']==sequence and float(r['cr'])==.1 and float(r['sigma'])==sigma}
            assert int(lookup['hqs_20000']['images'])==3
            values.append(100*(1-float(lookup['hqs_20000']['mse'])/float(lookup['hqs_12000']['mse'])))
        lines.append(f'| {sequence} | 3 | {values[0]:+.3f}% | {values[1]:+.3f}% |')
    lines += ['## 公平性、来源和位置',
              '只重新推理两个新状态，共3564个最终浮点重建和21384条阶段指标；旧官方/10000/11500状态复用原先全量浮点复核后的锁定结果。三种旧参考先做新的五状态最小推理，确认测量哈希和PSNR/SSIM等价；全部图/条件的测量哈希与旧结果逐条匹配。新结果全部从NPY重算指标，比较聚合也独立重算。旧回归22图曾由2080Ti补测，此轮新推理3090，原跨卡96条重叠误差核验保留；不合并这些耗时声称单一卡速度。',
              'HEVC帧仍是单图重建，南京照片仍是外部作者拍摄、我们自行整理。两种子消融解释的是先前800步短期适应，不能据本轮20k结果重新归因。',
              '服务器192.168.1.244，容器luozc_mlvc。评测根目录`/workspace/ProxUnroll-main/runs/hqs_20000_eval_20261014`；模型目录`/workspace/ProxUnroll-main/runs/hqs_20000_20261013/finetune`。',
              '- `best.pth`：19500验证选中推理权重；`last.pth`：20000完整恢复状态。',
              '- 评测`lock.json`：推理前的新权重/旧指标哈希、固定清单和预算。',
              '- `evaluation_primary/`、`evaluation_regression/`：新重建、CSV、每视频/每模型表、配对增减、负例、provenance/verification。',
              '- `logs/{smoke,primary,regression,finalize}.log/.exit`：实际进程命令和完成退出码。',
              '- `completion_audit.json`：覆盖、测量配对、独立等权MSE/PSNR/SSIM计算。',
              '核验命令：`.venv/bin/python -m coursework.audit_20000_eval`。原始推理命令记录在日志和provenance；复现须选空闲GPU和全新输出目录。',
              '新版英文技术报告改为连续双栏流式排版，位于`output/pdf/technical_report_flow_20261014.pdf`，旧六页版原样保留。']
    # Join table rows without inserting blank lines between them.
    text=''
    for i,line in enumerate(lines):
        if i:text+='\n' if line.startswith('|') and lines[i-1].startswith('|') else '\n\n'
        text+=line
    (ROOT/'coursework/HQS_20000_RESULTS_20261014.md').write_text(text+'\n')
    print('Chinese 20000-step results written')


if __name__=='__main__':main()
