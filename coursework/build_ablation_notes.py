"""Chinese results, reproduction commands and speaking notes from completed CSVs."""
import json
import csv
from collections import defaultdict
import numpy as np
from pathlib import Path

from coursework.run_experiments import ROOT

BASE = ROOT / 'runs/hqs_ablation_20261013'
DATASET = 'DIV2K_holdout_10000'


def signed(value):
    return f'{value:+.2f}%'


def markdown_blocks(blocks):
    """Keep consecutive table rows together while separating prose blocks."""
    result = []
    for block in blocks:
        if result:
            result.append('\n' if block.startswith('|') and result[-1].startswith('|') else '\n\n')
        result.append(block)
    return ''.join(result) + '\n'


def main():
    facts = json.loads((BASE / 'report_facts.json').read_text())
    primary = facts['phases']['primary']['analysis']
    regression = facts['phases']['regression']['analysis']
    training = facts['training']
    def pair(candidate, reference, sigma=0., data=primary, dataset=DATASET, cr=.1):
        return next(r for r in data['paired_summary'] if r['candidate'] == candidate and r['reference'] == reference
                    and r['dataset'] == dataset and r['cr'] == cr and r['sigma'] == sigma)
    def factor(comparison, sigma=0., view='last'):
        return next(r for r in primary['seed_stability'] if r['comparison'] == comparison
                    and r['checkpoint_view'] == view and r['dataset'] == DATASET and r['cr'] == .1 and r['sigma'] == sigma)
    def describe_effect(comparison, sigma):
        row = factor(comparison, sigma)
        values = [row['minimum_mse_reduction_percent'], row['maximum_mse_reduction_percent']]
        direction = '两个种子都改善' if min(values) > 0 else '两个种子都退步' if max(values) < 0 else '两个种子的方向不一致或有零变化'
        return f'{direction}，平均 MSE 下降 {signed(row["mean_mse_reduction_percent"])}，两种子范围 {signed(min(values))} 至 {signed(max(values))}'
    clean = pair('hqs_12000', 'hqs_10000')
    noisy = pair('hqs_12000', 'hqs_10000', .05)
    official = pair('hqs_12000', 'hqs')
    selected = training['continuation']['selected_step']
    summary = [
        '# 12000 步 HQS 续训与双因素消融结果',
        '本页由已完成的训练审计、独立浮点复核和原始 CSV 生成。正百分比表示误差下降，负百分比表示误差增加。',
        '## 实际完成状态',
        f'原配方从 10000 步完整状态继续到 12000 步，新增 2000 步，验证选择第 {selected} 步。另有四种配方各两个种子、每组 800 步，共 6400 步。正式新增更新总计 8400 次；预检另有 14 次更新。全部训练任务退出 0，六个测量因子固定，完整恢复状态、损失/梯度、原图/裁剪/采样配对和验证选模均通过审计。',
        f'续训耗时 {training["continuation"]["wall_seconds"]/60:.2f} 分钟；全部新正式训练合计 {sum(r["wall_seconds"] for r in training.values())/60:.2f} GPU 分钟。峰值分配显存 {max(r["peak_memory_mib"] for r in training.values())/1024:.2f} GiB。GPU 分钟是各任务耗时之和，不是三张卡并行后的墙钟时间。',
        '## 主要52图：续训有多少收益',
        '| 10%采样测试条件 | 相比官方HQS的MSE下降 | 相比10000步选中模型的MSE下降 | PSNR变化 vs10000 | SSIM变化 vs10000 |',
        '|---|---:|---:|---:|---:|',
    ]
    for sigma, label in [(0., '干净'), (.01, '相对测量噪声.01'), (.05, '相对测量噪声.05')]:
        row = pair('hqs_12000', 'hqs_10000', sigma)
        base = pair('hqs_12000', 'hqs', sigma)
        summary.append(f'| {label} | {signed(base["mse_reduction_percent"])} | {signed(row["mse_reduction_percent"])} | {row["delta_psnr"]:+.4f} dB | {row["delta_ssim"]:+.5f} |')
    summary += [
        f'干净10%采样下，相比之前10000步选中模型，MSE下降 {signed(clean["mse_reduction_percent"])}；较强噪声下为 {signed(noisy["mse_reduction_percent"])}。是否改善按符号判断，不能把新增训练步数直接等同于更好的测试效果。',
        f'52张图的572个配对条件中，选中的续训模型相对10000步模型有 {facts["continuation_negative_conditions_vs_10000"]} 个条件退步，全部保留。',
        '## 固定800步消融：两个因素的作用',
        '四种配方从同一官方HQS权重和新Adam开始。每个种子共享原图顺序、裁剪、翻转、采样率和潜在噪声；关闭噪声只改变幅度。中间监督为前五阶段各.01/最终.95，最终监督为最终权重1，总权重相同。',
        '| 配对对照 | 测试噪声 | 两种子平均MSE下降 | 最小 | 最大 | 方向一致 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    comparisons = [('noise_effect_at_weighted','加训练噪声，固定阶段监督'),
                   ('noise_effect_at_final','加训练噪声，固定最终监督'),
                   ('stage_effect_at_noise_on','加阶段监督，固定有噪声训练'),
                   ('stage_effect_at_noise_off','加阶段监督，固定干净训练')]
    for comparison, label in comparisons:
        for sigma in [0., .01, .05]:
            row = factor(comparison, sigma)
            summary.append(f'| {label} | {sigma:g} | {signed(row["mean_mse_reduction_percent"])} | {signed(row["minimum_mse_reduction_percent"])} | {signed(row["maximum_mse_reduction_percent"])} | {"是" if row["same_direction_both_seeds"] else "否"} |')
    summary += [
        '以上是固定800步终点的配对差异。验证选中best的结果在同目录单独保存，不与终点混合。两个种子平均是两个配对改善百分比的平均，每个种子的原始值都可查。',
        '因素主效应另用原始MSE差计算，再按同一官方MSE归一化为百分点；交互项是“有阶段监督时噪声收益 − 最终监督时噪声收益”。它与上表不同分母的相对百分比不能混用。',
        '两种训练种子不够证明稳健显著性。800步的消融解释的是这次短期配方作用，不是此前10000步全部收益的原因；没有消融的数据覆盖、学习率与结构不能强行归因。',
        '## 完整双因素主效应和交互项',
        '以下为两个训练种子的平均，固定800步终点；单位是按同一官方MSE归一化的百分点，正数代表误差改善。',
        '| 测试噪声 | 噪声主效应 | 阶段监督主效应 | 交互项 |', '|---|---:|---:|---:|',
    ]
    for sigma in [0., .01, .05]:
        values = {r['effect']: r['official_normalized_percent_points'] for r in primary['factorial_effects']
                  if r['dataset'] == DATASET and r['cr'] == .1 and r['sigma'] == sigma
                  and r['checkpoint_view'] == 'last' and r['training_seed'] == 'mean_of_two'}
        numbers = [values[key] for key in ['noise_main_mse_benefit', 'stage_main_mse_benefit', 'noise_stage_interaction_mse_benefit']]
        summary.append(f'| {sigma:g} | ' + ' | '.join(f'{value:+.3f}' for value in numbers) + ' |')
    summary += [
        '## 回归集：分别保留每个数据集和视频',
        '| 已观察数据集 | 干净10% MSE下降 vs10000 | 噪声.05 MSE下降 vs10000 |',
        '|---|---:|---:|',
    ]
    for dataset, label in [('Kodak','Kodak24'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('DIV2K_fresh','DIV2K32'),('NJU_campus_web','校园18图'),('Nanjing_scenic_web','景区12图')]:
        summary.append(f'| {label} | {signed(pair("hqs_12000","hqs_10000",0.,regression,dataset)["mse_reduction_percent"])} | {signed(pair("hqs_12000","hqs_10000",.05,regression,dataset)["mse_reduction_percent"])} |')
    summary += ['## 每段HEVC视频：续训相对10000步模型',
                '| 视频 | 固定间隔帧数 | 干净10% MSE下降 | 噪声.05 MSE下降 |', '|---|---:|---:|---:|']
    video_values = defaultdict(lambda: defaultdict(list))
    for row in csv.DictReader((BASE / 'evaluation_regression/metrics.csv').open()):
        if row['dataset'] not in ['HEVC_B', 'HEVC_E'] or float(row['cr']) != .1:
            continue
        if row['method'] not in ['hqs_10000', 'hqs_12000'] or float(row['sigma']) not in [0., .05]:
            continue
        video_values[(row['sequence'], float(row['sigma']), row['method'])][row['image']].append(10 ** (-float(row['psnr']) / 10))
    for video in sorted({key[0] for key in video_values}):
        reductions = []
        for sigma in [0., .05]:
            means = {method: float(np.mean([np.mean(repeats) for repeats in video_values[video, sigma, method].values()]))
                     for method in ['hqs_10000', 'hqs_12000']}
            reductions.append(100 * (1 - means['hqs_12000'] / means['hqs_10000']))
        count = len(video_values[video, 0., 'hqs_10000'])
        assert count == 3
        summary.append(f'| {video} | {count} | {signed(reductions[0])} | {signed(reductions[1])} |')
    if facts.get('completion', {}).get('evaluations', {}).get('regression', {}).get('recovered'):
        summary.append('回归原推理在90分钟上限退出，88张完整原图复用，剩余22张以独立45分钟预算在空闲RTX 2080 Ti补测；原失败exit与部分CSV保留。合并后全部浮点指标重新核验。每张原图内所有模型仍使用同一硬件和输入，不能将混合硬件耗时平均声称为同一卡的速度。')
    case = facts['failure_case']
    summary += [
        '每段视频分别有video_summary.csv。逐帧重建仍属于单图任务，没有利用前后帧；按整段视频等权聚合。网络南京照片来自外部作者，属于自行整理的网络图像集，不是亲自拍摄。',
        '## 失败案例与局限',
        f'主要测试诊断图 `{case["image"]}`，采样{float(case["cr"]):.0%}、噪声{case["sigma"]}、测量种子{case["measurement_seed"]}，候选`{case["candidate"]}`相对`{case["reference"]}`的MSE下降为 {signed(float(case["mse_reduction_percent"]))}，PSNR变化{float(case["delta_psnr"]):+.4f} dB。这是测试后选出的诊断案例，不参与选模型；若选中模型无退步则明确使用终点案例。',
        '误差图只能定位重建偏差，不能独立证明训练退步原因。仍有灰度模拟、中心裁剪、训练种子少、样本有限和官方预训练暴露未知等限制。',
        '## 目录和重跑命令',
        '远端：192.168.1.244 / luozc_mlvc / `/workspace/ProxUnroll-main/runs/hqs_ablation_20261013`。独立环境`.venv-linux`；版本在evaluation目录的provenance.json。',
        '- `lock.json`：启动前的训练/验证/测试清单、源码、原权重和测量矩阵哈希。',
        f'- `continuation/`：第{selected}步验证选中best.pth、12000终点完整last.pth、train.csv、validation.csv、config.json、summary.json、audit.json。',
        '- `arms/`：八组配方/种子的权重、完整恢复状态、训练与验证记录。',
        '- `pairing_*.json`：同种子逐步输入和随机配对的完整核验。',
        '- `evaluation_lock.json`：推理前固定best/last权重哈希。',
        '- `evaluation_primary/`：主要52图、20模型状态，含所有浮点/显示重建、逐图/逐条件指标、因素效应和负例。',
        '- `evaluation_regression/`：回归110图、12模型状态，含每段视频结果。',
        '- `logs/`：命令、PID、日志与退出码；checkpoint完成与日志均已实际核验。',
        '- `illustration/`、`figures/`：失败原图/重建/误差图和报告图；大数组保持远端。',
        '重跑必须使用全新输出目录。不要再次执行prepare或原launch以覆盖历史；锁定评测入口本身禁止同目录重复生成。示例在远端项目根目录执行：',
        '```bash\nexport CUDA_VISIBLE_DEVICES=GPU-5c29e362-93ff-356b-1b27-82ef64ea1327\nexport CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4\n.venv-linux/bin/python -m coursework.evaluate_manifest \\\n  --manifest runs/lab244_10000_20261009/data/test_fresh.json \\\n  --output runs/NEW_ABLATION/primary \\\n  --methods hqs,hqs_10000,hqs_12000 \\\n  --extra-checkpoint hqs_10000=runs/lab244_10000_20261009/finetune/best.pth \\\n  --extra-checkpoint hqs_12000=runs/hqs_ablation_20261013/continuation/best.pth\n.venv-linux/bin/python -m coursework.verify_lab_run runs/NEW_ABLATION/primary\n```',
        '完整20状态命令与权重哈希在evaluation_primary/provenance.json及logs/evaluation_primary.log。独立浮点复核成功后才做分析。运行命令前应重新确认GPU空闲。',
        '英文新报告：`output/pdf/technical_report_ablation_20261013.pdf`。新十二页PPT：`output/presentation/course_presentation_ablation_20261013.pptx`。旧材料保持。AI协助声明保留；未提交课程平台。',
    ]
    (ROOT / 'coursework/HQS_ABLATION_RESULTS_20261013.md').write_text(markdown_blocks(summary))
    speeches = [
        '老师、同学们好，我们研究怎样用少量测量值恢复灰度图。我们复现官方模型，增加传统基线和多种数据测试，再开展固定测量矩阵的微调。这次还增加了有控制的消融，检验训练噪声和中间阶段损失是否有用。',
        '单像素成像每次测得的是经过空间编码后的整体亮度，不是简单地留下几个像素。我们在电脑上把原图变成少量测量，再用算法恢复整图。主要输入为256乘256灰度图；10%采样时，测量数量大约只有原像素数量的十分之一。本次没有搭建光学相机。',
        '我们比较伴随重建、DCT-FISTA，以及官方HQS和ADMM。伴随直接反投影，DCT-FISTA利用变换域稀疏性迭代求解，两个网络版本把优化与学习到的图像规律结合。为了公平，所有方法拿到相同矩阵、测量值和噪声，使用相同指标。',
        'ProxUnroll把迭代过程展开为六个阶段：先修正测量不一致，再用网络恢复结构。卷积、注意力和阶段特征帮助重建同一张图。架构和官方权重由论文作者提供。我们的贡献是公平评测、数据扩展、可控微调和消融，而不是把原架构说成自己的发明。',
        '微调用800张DIV2K训练原图和16张独立验证原图；主要测试是此前锁定、项目尚未使用的52张原图。另有110张已观察数据作回归，包括Kodak、HEVC帧、DIV2K32和南京网络照片。南京照片是外部作者拍摄、我们自行整理的，保留了来源。视频帧逐张重建，没有使用前后帧。',
        '原有校园18图在干净10%采样时，伴随、DCT-FISTA、官方HQS、ADMM的平均PSNR分别约23.92、24.60、28.35、28.38 dB。网络明显优于两个传统基线，HQS和ADMM接近。这些旧结果保留，新实验不改写旧成绩。PSNR越高通常误差越小，SSIM衡量结构相似。',
        '这轮做了两类实验。第一类从原10000步完整状态续训2000步，保持配方不变。第二类从同一官方权重开始，组合有无训练噪声、有无中间阶段监督，四组各跑两个种子、每组800步。同一种子共享原图、裁剪和采样，阶段损失总权重相同，所有测量矩阵固定。新增正式更新共8400步。',
        f'权重只由16张验证图选择。续训达到12000步，验证选中第{selected}步。在独立52图、干净10%采样时，相比10000步模型平均MSE下降{signed(clean["mse_reduction_percent"])}，相比官方下降{signed(official["mse_reduction_percent"])}；较强测量噪声下，相比10000步下降{signed(noisy["mse_reduction_percent"])}。正数才表示改善，负数表示退步；这里不是PSNR数值的百分比。',
        '先看训练噪声消融。固定阶段监督、起始权重、训练步数和所有采样规则，只把训练噪声幅度关闭或打开。干净测试下，'+describe_effect('noise_effect_at_weighted',0.)+'；较强噪声测试下，'+describe_effect('noise_effect_at_weighted',.05)+'。这是短期800步配方的因素对照，不是重述原模型与微调模型的总差异。',
        '再看中间阶段监督。固定有噪声训练，比较六阶段加权真值损失与仅最终输出损失。干净测试下，'+describe_effect('stage_effect_at_noise_on',0.)+'；较强噪声测试下，'+describe_effect('stage_effect_at_noise_on',.05)+'。我们同时报告两个种子，不能只选一个好的种子。其他因素水平和交互项在完整表里，两种子不足以证明普遍显著提升。',
        f'我们保留了所有失败。52张图共572个测量条件，续训选中模型相对10000步模型有{facts["continuation_negative_conditions_vs_10000"]}个条件退步。图中案例是{case["image"]}，采样率{float(case["cr"]):.0%}、噪声{case["sigma"]}，候选{case["candidate"]}相对原模型MSE下降为{signed(float(case["mse_reduction_percent"]))}。误差图帮助定位偏差，但不能单凭它证明训练原因。这张图仅供诊断，没有参与选模型。',
        '我们完成了有边界的续训、双因素双种子消融和独立测试，并保存代码、权重、CSV、来源、日志与核验结果。结论要按具体条件和比较对象判断，更多步骤不保证更好，某个因素也可能存在代价。仍有灰度模拟、种子少和官方预训练暴露未知等限制。Codex协助代码、实验、分析和写作，学生需要核查并理解。谢谢大家。',
    ]
    times = [25,45,45,50,55,60,60,60,60,60,45,35]
    guide = ['# 续训与消融版中文讲稿', '对应新的十二页PPT，计划600秒，实际以计时排练为准。百分比来自已核验CSV，旧讲稿保留。']
    for i, (speech, seconds) in enumerate(zip(speeches, times), 1):
        guide += [f'## 第{i}页（约{seconds}秒）', speech]
    guide += ['## 答辩关键边界',
        '“有没有提升？”先说比较对象、数据和噪声条件，再报MSE与PSNR，不能把验证提升当测试收益。',
        '“消融证明什么？”证明两个受控因素在这次官方起点、800步、两个种子条件下的配对差异；不能解释未控制因素或完整历史训练收益。',
        '“为什么还看last？”best反映验证选模后的实际使用模型，last保证四组更新预算相同，分别回答两个问题。',
        '“52图完全没被见过？”与本项目微调/验证原图分开且此前未评测；官方预训练暴露没有全面审计。',
        '“图片自己拍的吗？”南京照片是外部作者网络照片，我们自行整理，不能说亲自拍摄。',
        '完整结果：`coursework/HQS_ABLATION_RESULTS_20261013.md`，原始依据：`runs/hqs_ablation_20261013`下的CSV、audit和provenance。']
    (ROOT / 'coursework/讲稿_续训与消融_20261013.md').write_text('\n\n'.join(guide) + '\n')
    print('Chinese results and twelve-slide speaking notes written')


if __name__ == '__main__':
    main()
