"""Versioned six-page English report from the completed continuation/ablations."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from coursework import build_reports as report
from coursework.run_experiments import ROOT, sha

BASE = ROOT / 'runs/hqs_ablation_20261013'
FIG = BASE / 'figures'
SEEDS = [20261013, 20261014]
DATASET = 'DIV2K_holdout_10000'


def main():
    facts = json.loads((BASE / 'report_facts.json').read_text())
    primary = facts['phases']['primary']['analysis']
    regression = facts['phases']['regression']['analysis']
    training = facts['training']
    lock = facts['lock']
    if not all(r['completed_steps'] == 800 for n, r in training.items() if n != 'continuation'):
        raise ValueError('Matched-budget ablations not complete')

    def pair(data, candidate, reference, sigma=0., dataset=DATASET, cr=.1):
        return next(r for r in data['paired_summary'] if r['candidate'] == candidate and r['reference'] == reference
                    and r['dataset'] == dataset and r['cr'] == cr and r['sigma'] == sigma)

    def factor(comparison, sigma, seed, view='last'):
        return next(r for r in primary['paired_summary'] if r['comparison'] == comparison
                    and r['checkpoint_view'] == view and r['training_seed'] == seed
                    and r['dataset'] == DATASET and r['cr'] == .1 and r['sigma'] == sigma)

    def summary(method, sigma=0., dataset=DATASET):
        return next(r for r in primary['summaries'] if r['method'] == method and r['dataset'] == dataset and r['cr'] == .1 and r['sigma'] == sigma)

    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    validation = list(csv.DictReader((BASE / 'continuation/validation.csv').open()))
    steps = sorted({int(r['step']) for r in validation})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), layout='constrained')
    for label, levels, color in [('Clean', [0.], '#147D78'), ('Mixed', [0., .01, .05], '#D78924'), ('Noise .05', [.05], '#B74760')]:
        values = [np.mean([float(r['psnr']) for r in validation if int(r['step']) == s and float(r['sigma']) in levels]) for s in steps]
        axes[0].plot(steps, values, label=label, color=color)
    axes[0].axvline(10000, color='gray', linestyle='--', linewidth=1)
    axes[0].set(xlabel='Total round-2 update', ylabel='Validation PSNR (dB)')
    axes[0].legend(fontsize=9)
    values = [pair(primary, 'hqs_12000', 'hqs_10000', sigma=s)['mse_reduction_percent'] for s in [0., .01, .05]]
    axes[1].bar(['Clean', 'Noise .01', 'Noise .05'], values, color=['#147D78', '#D78924', '#B74760'])
    axes[1].axhline(0, color='gray', linewidth=.7)
    axes[1].set(ylabel='Primary52 MSE reduction vs 10000 (%)')
    fig.savefig(FIG / 'continuation.png', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), layout='constrained')
    for ax, comparison, title in zip(axes, ['noise_effect_at_weighted', 'stage_effect_at_noise_on'],
                                    ['Noise augmentation, weighted loss', 'Stage supervision, noisy training']):
        for i, seed in enumerate(SEEDS):
            x = np.arange(3) + (i - .5) * .32
            ax.bar(x, [factor(comparison, sigma, seed)['mse_reduction_percent'] for sigma in [0., .01, .05]],
                   width=.32, label=f'Seed {seed}', color=['#147D78', '#D78924'][i])
        ax.axhline(0, color='gray', linewidth=.7)
        ax.set_xticks(np.arange(3), ['Clean', 'Noise .01', 'Noise .05'])
        ax.set(title=title, ylabel='Paired MSE reduction (%)')
        ax.legend(fontsize=9)
    fig.savefig(FIG / 'factorial.png', dpi=220)
    plt.close(fig)
    dataset_labels = [('Kodak', 'Kodak24'), ('HEVC_B', 'HEVC B'), ('HEVC_E', 'HEVC E'),
                      ('DIV2K_fresh', 'DIV2K32'), ('NJU_campus_web', 'Campus18'), ('Nanjing_scenic_web', 'Scenic12')]
    fig, ax = plt.subplots(figsize=(10.5, 3.4), layout='constrained')
    x = np.arange(len(dataset_labels))
    for i, sigma in enumerate([0., .05]):
        ax.bar(x + (i - .5) * .34, [pair(regression, 'hqs_12000', 'hqs_10000', sigma, name)['mse_reduction_percent']
                                   for name, _ in dataset_labels], width=.34, label='Clean' if sigma == 0 else 'Noise .05',
               color=['#147D78', '#D78924'][i])
    ax.set_xticks(x, [label for _, label in dataset_labels])
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set(ylabel='MSE reduction vs 10000 (%)')
    ax.legend()
    fig.savefig(FIG / 'regression.png', dpi=220)
    plt.close(fig)
    case = facts['failure_case']
    gt = np.asarray(Image.open(BASE / 'illustration/reference.png'), dtype=np.float32) / 255
    old, new = [np.clip(np.load(BASE / ('illustration/' + name + '.npy')), 0, 1) for name in ['old', 'new']]
    error = np.abs(new - gt)
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 3.0), layout='constrained')
    titles = ['Reference', '10000-step HQS', 'Selected continuation' if case['candidate'] == 'hqs_12000' else '12000-step endpoint', 'Absolute error']
    for ax, image, title in zip(axes, [gt, old, new, error], titles):
        ax.imshow(image, cmap='magma' if title == 'Absolute error' else 'gray', vmin=0,
                  vmax=.3 if title == 'Absolute error' else 1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.savefig(FIG / 'failure.png', dpi=220)
    plt.close(fig)

    continuation_table = [['Condition', 'vs official', 'vs 10000', 'PSNR / SSIM']]
    for sigma, label in [(0., 'Clean'), (.01, 'Noise .01'), (.05, 'Noise .05')]:
        row = summary('hqs_12000', sigma)
        continuation_table.append([label,
            f'{pair(primary,"hqs_12000","hqs",sigma)["mse_reduction_percent"]:+.2f}%',
            f'{pair(primary,"hqs_12000","hqs_10000",sigma)["mse_reduction_percent"]:+.2f}%',
            f'{row["psnr"]:.2f} / {row["ssim"]:.3f}'])
    factorial_table = [['Seed / test', 'Noise effect', 'Stage effect']]
    for seed in SEEDS:
        for sigma, label in [(0., 'clean'), (.05, 'noise .05')]:
            factorial_table.append([f'{seed} / {label}',
                f'{factor("noise_effect_at_weighted",sigma,seed)["mse_reduction_percent"]:+.2f}%',
                f'{factor("stage_effect_at_noise_on",sigma,seed)["mse_reduction_percent"]:+.2f}%'])
    regression_table = [['Dataset', 'Clean', 'Noise .05']]
    for name, label in dataset_labels:
        regression_table.append([label] + [f'{pair(regression,"hqs_12000","hqs_10000",s,name)["mse_reduction_percent"]:+.2f}%' for s in [0., .05]])
    clean_gain = pair(primary, 'hqs_12000', 'hqs_10000')['mse_reduction_percent']
    noisy_gain = pair(primary, 'hqs_12000', 'hqs_10000', .05)['mse_reduction_percent']
    selected = training['continuation']['selected_step']
    total_gpu_minutes = sum(r['wall_seconds'] for r in training.values()) / 60
    max_memory = max(r['peak_memory_mib'] for r in training.values()) / 1024
    baseline_path = ROOT / 'runs/nanjing_web30_20261009/comparison/dataset_summary.csv'
    baseline_rows = list(csv.DictReader(baseline_path.open()))
    baseline = {r['method']: r for r in baseline_rows if r['dataset'] == 'NJU_campus_web'
                and float(r['cr']) == .1 and float(r['sigma']) == 0}
    baseline_psnr = [float(baseline[m]['psnr']) for m in ['adjoint', 'fista_dct', 'hqs', 'admm']]
    experiment_table = [['Arm', 'Noise', 'Stage loss', 'Updates']]
    for noise, supervision in [('on', 'weighted'), ('off', 'weighted'), ('on', 'final'), ('off', 'final')]:
        experiment_table.append([f'{noise}/{supervision}', noise, '.01 x5 / .95' if supervision == 'weighted' else 'final 1', '800 x2'])
    total_rows = facts['phases']['primary']['verification']['rows'] + facts['phases']['regression']['verification']['rows']
    def effect_description(comparison, sigma):
        row = next(r for r in primary['seed_stability'] if r['comparison'] == comparison
                   and r['dataset'] == DATASET and r['cr'] == .1 and r['sigma'] == sigma and r['checkpoint_view'] == 'last')
        return (f'{row["mean_mse_reduction_percent"]:+.2f}% mean, '
                f'{row["minimum_mse_reduction_percent"]:+.2f}% to {row["maximum_mse_reduction_percent"]:+.2f}% across seeds')
    def factorial_description(sigma):
        values = {row['effect']: row['official_normalized_percent_points'] for row in primary['factorial_effects']
                  if row['dataset'] == DATASET and row['cr'] == .1 and row['sigma'] == sigma
                  and row['checkpoint_view'] == 'last' and row['training_seed'] == 'mean_of_two'}
        return ', '.join(f'{values[key]:+.2f}' for key in ['noise_main_mse_benefit',
                         'stage_main_mse_benefit', 'noise_stage_interaction_mse_benefit'])
    source_name = escape(case['source'].get('source', 'DIV2K original').split('/')[-1])
    recovery_note = ''
    if facts.get('completion', {}).get('evaluations', {}).get('regression', {}).get('recovered'):
        recovery_note = (' The original regression inference reached its 90-minute cap after 88 complete originals. '
                         'A separate bounded run completed the remaining 22 on an idle RTX 2080 Ti; original failures and partial CSVs were retained. '
                         'All merged floating metrics were rechecked. Within every original all states share hardware and inputs; '
                         'mixed-hardware timings are not treated as one-GPU speed results.')
    report.FIG = FIG
    report.STYLE.fontSize = 10.35
    report.STYLE.leading = 13
    report.STYLE.spaceAfter = 8
    report.SMALL.fontSize = 8.5
    report.SMALL.leading = 10.2
    report.HEAD.fontSize = 11.5
    report.HEAD.leading = 13.8
    pages = [
        dict(blocks=[
            ('heading', 'Abstract'),
            ('text', f'We study image reconstruction from matched simulated single-pixel measurements. Preserved experiments compare adjoint reconstruction, DCT-FISTA and released HQS/ADMM ProxUnroll models. This extension continues fixed-operator HQS from 10000 to 12000 updates and runs a two-by-two ablation of training measurement noise and intermediate-stage supervision, with two training seeds and 800 updates per arm. A previously locked 52-image DIV2K project holdout evaluates all selected and fixed-budget endpoint states. Validation selects step {selected} for the continuation. At clean 10% sampling, its mean MSE reduction is {clean_gain:+.2f}% versus the prior selected 10000-step state; at relative noise .05 the reduction is {noisy_gain:+.2f}%. Negative reductions mean higher error. All regressions and seed-specific effects are retained.'),
            ('heading', '1. Problem and Prior Work'),
            ('text', 'Single-pixel imaging records spatially encoded integrated responses [1]. We simulate this acquisition from existing 256 x 256 grayscale images. The task is underdetermined at low sampling rates, so reconstruction combines measurement consistency and image priors. ProxUnroll [2] provides the architecture and official pretrained weights. Our contributions are a common measurement/metric contract, reproducible external evaluations, bounded fixed-matrix fine-tuning and controlled training ablations; we do not claim a new architecture.'),
            ('heading', '2. Common Reconstruction Contract'),
            ('text', 'The separable forward model is Y = H X W<super>T</super> + E. Adjoint reconstruction uses H<super>T</super> Y W. DCT-FISTA [3] uses gradient descent, DCT soft thresholding and acceleration for 200 iterations; lambda .001 was chosen by independent original validation before the external tests. HQS alternates measurement correction with a learned restorer; ADMM adds a dual variable. The supplied six-stage restorer uses convolution, window attention and memory within one image, without previous/future video frames.'),
            ('text', 'All methods receive identical common-MAT factors, measured values and noise realizations. Nominal clean sampling rates are 1%, 4%, 10%, 25% and 50%; actual row counts are recorded. At 10%, zero-mean Gaussian noise has standard deviation .01 or .05 times clean measurement RMS. Three fixed measurement seeds repeat noisy conditions. Predictions are clipped to [0,1] for PSNR/SSIM with data_range 1, without border cropping. Primary metrics precede PNG quantization.'),
            ('text', 'Percent improvement means 100(1 - mean candidate MSE / mean reference MSE). It is not a percentage change in PSNR. Noise repeats average within each image; images average within a video; whole videos have equal weight within a dataset. Still-image units are originals. Training seeds and measurement-noise seeds are separate experimental units.'),
            ('heading', '2.1. Preserved Four-Method Comparison'),
            ('text', f'On the 18 Nanjing campus web photographs at clean 10% sampling, mean PSNR is {baseline_psnr[0]:.2f} dB for adjoint, {baseline_psnr[1]:.2f} for DCT-FISTA, {baseline_psnr[2]:.2f} for official HQS and {baseline_psnr[3]:.2f} for official ADMM. The neural models outperform these traditional baselines under the common contract; HQS and ADMM are close. These are previously observed regression results, separate from the new primary52 continuation and factorial tests.'),
        ]),
        dict(table_widths=[80,32,86,53], blocks=[
            ('heading', '3. Data and Test Roles'),
            ('text', 'Training uses all 800 official DIV2K training originals [4]; 16 separate originals select checkpoints. Split assignment precedes random square crops, area resizing and horizontal/vertical flips. Training/validation/holdout original-source hashes are disjoint. Still photographs use the largest centered square, OpenCV RGB2YCrCb Y and INTER_AREA resize to 256 x 256, then uint8/255.'),
            ('text', f'The primary set contains the 52 remaining DIV2K validation originals locked before the previous continuation. Before this experiment, completed project inference provenance was scanned for overlap; its role is recorded as {escape(lock["fresh_test_role"])}. No test inference is used for checkpoint selection. Official pretraining exposure remains unknown, so project-disjoint does not imply unseen by all upstream training.'),
            ('text', 'Previously observed regression inputs comprise Kodak24, eight HEVC B/E videos with three spaced raw Y frames each, DIV2K32 and 30 Nanjing web photographs, totaling 110 images. The 18 campus and 12 scenic photographs have external authors and documented licenses; they are team-curated, not team-captured. HEVC is 8-bit planar YUV420 with checked dimensions/byte counts and fixed indices 0, floor(N/3), floor(2N/3). Stored Y/255 is retained because trustworthy range metadata are absent.'),
            ('heading', '4. Matched Factorial Ablations'), ('table', experiment_table),
            ('text', 'All four arms start from the same official HQS weights with fresh Adam at 1e-5, batch one and FP32. Seeds 20261013 and 20261014 each run every arm for 800 updates. Within each seed, original order, crops, flips, sampling rates and proposed noise draws are identical. Disabling noise sets only its amplitude to zero. All six H/W sensing parameters remain frozen.'),
            ('text', 'Weighted supervision is direct-ground-truth RMSE with .01 on each of five intermediate outputs and .95 on the final output. Final-only supervision uses weight 1 on the final output, preserving total loss weight. This tests short adaptation from official weights; it does not isolate every change responsible for the earlier 10000-step result.'),
            ('text', 'Validation uses 16 originals, rates 10%/25% and noise 0/.01/.05, every 200 ablation updates. Mixed PSNR selects the highest eligible checkpoint, with clean validation at least initialization minus .03 dB. Initialization is a candidate. Selected best and equal-budget last are evaluated separately.'),
        ]),
        dict(figures=[('continuation.png', 'Figure 1. Preserved validation trajectory and primary52 selected-continuation MSE changes. Test scores never select a checkpoint.')], table_widths=[55,62,62,72], blocks=[
            ('heading', '5. Continuation with the Same Update Rule'),
            ('text', 'Continuation restores the actual step-10000 model, Adam, scheduler, RNG, selection history and best candidate. It adds 2000 updates under the unchanged noisy weighted-RMSE recipe, with validation every 250 steps. This is a training-length comparison, distinct from the official-weight factorial adaptation.'),
            ('text', f'Actual audits confirm 12000 optimizer updates in the inherited trajectory and 800 in every ablation arm; six sensing factors are unchanged. Validation selects continuation step {selected}. CUDA prechecks show 2+1 updates equal uninterrupted 3, and old/new drivers give identical 5-to-6 updates for parameters, Adam, scheduler, RNG, crops, losses and validation.'),
            ('heading', '5.1. Primary Holdout Results'), ('table', continuation_table),
            ('text', f'Table 1 reports MSE reductions at 10% sampling, with positive values indicating lower error. The continuation uses {training["continuation"]["wall_seconds"]/60:.1f} minutes; all new formal training arms total {total_gpu_minutes:.1f} GPU-minutes. Maximum allocated memory is {max_memory:.2f} GiB. Resources are verified RTX 3090 GPUs in server244 container luozc_mlvc, with project-local Python 3.11.11 and Torch 2.6.0+cu124.'),
        ]),
        dict(figures=[('factorial.png', 'Figure 2. Paired fixed-800-update effects for each training seed. Left: noise augmentation with weighted supervision. Right: stage supervision with noisy training.')], table_widths=[105,73,73], blocks=[
            ('heading', '6. Effects of Noise and Stage Supervision'), ('table', factorial_table),
            ('text', 'Table 2 compares fixed-budget endpoints on primary52. Noise effect compares noise-on with noise-off under weighted loss. Stage effect compares weighted with final-only loss under noisy training. Each comparison holds seed, original order, crops, sampling and budget fixed. Positive values indicate lower mean MSE when adding that factor.'),
            ('text', f'At clean 10%, adding training noise with weighted loss gives {effect_description("noise_effect_at_weighted",0.)}; at test noise .05 it gives {effect_description("noise_effect_at_weighted",.05)}. Adding stage supervision with noisy training gives {effect_description("stage_effect_at_noise_on",0.)} on clean measurements and {effect_description("stage_effect_at_noise_on",.05)} at noise .05.'),
            ('text', 'The complete factorial analysis also compares noise under final-only loss and stage supervision under clean training. Main effects average raw-MSE benefits over the other factor; interaction is noise benefit with weighted loss minus noise benefit with final-only loss. These raw effects share official-MSE normalization in percent points, rather than mixing reductions with different denominators. '
                     f'Two-seed mean noise main / stage main / interaction effects are {factorial_description(0.)} percent points on clean data and {factorial_description(.05)} at noise .05.'),
            ('text', 'Selected-best effects and endpoint effects remain separate in CSV. Every seed is reported alongside the two-seed mean and range; two training seeds are too few for strong significance claims. A negative result or an initialization-selected arm is retained. We do not reinterpret related measurement repetitions as independent training replications.'),
        ]),
        dict(figures=[('regression.png', 'Figure 3. Previously observed regression datasets, selected continuation versus selected step-10000 HQS. Each video contributes equally within its dataset.')], table_widths=[105,73,73], blocks=[
            ('heading', '7. Regression and Complete Evidence'), ('table', regression_table),
            ('text', 'For each regression dataset, Table 3 preserves clean and stronger-noise results at 10% sampling. Kodak, HEVC, DIV2K32 and Nanjing photographs were already inspected in previous work; they are regression evidence, not newly unseen confirmation. Each video has its own table, and no long sequence dominates the dataset mean.' + recovery_note),
            ('text', f'The primary set saves {facts["phases"]["primary"]["verification"]["rows"]} metric rows for selected and endpoint states; regressions save {facts["phases"]["regression"]["verification"]["rows"]} rows for selected ablation states and continuation views. All {total_rows} floating reconstructions are independently checked against stored metrics, with data/weight/matrix/source and regenerated measurement hashes. Raw CSV, stage metrics, configurations, versions, arrays and display images remain available.'),
            ('text', f'On primary52, selected continuation increases MSE versus the prior selected state in {facts["continuation_negative_conditions_vs_10000"]}/{facts["continuation_total_conditions"]} measurement conditions. All per-image deltas and every negative comparison are retained, across references and factorial contrasts. Sample illustration selection occurs after testing for diagnosis, not model selection.'),
        ]),
        dict(figures=[('failure.png', f'Figure 4. Primary-set diagnostic: {escape(case["image"])}, rate {float(case["cr"]):g}, relative noise {float(case["sigma"]):g}, measurement seed {case["measurement_seed"]}; {float(case["mse_reduction_percent"]):+.2f}% MSE reduction versus 10000-step HQS. Absolute-error scale: 0 to .3.')], blocks=[
            ('heading', '8. Failure Analysis and Limits'),
            ('text', f'The diagnostic uses DIV2K source {source_name} with fixed center-square Y preprocessing. Candidate {escape(case["candidate"])} has per-condition PSNR change {float(case["delta_psnr"]):+.3f} dB and SSIM change {float(case["delta_ssim"]):+.4f}. The error map locates spatial mismatch; it does not identify a unique training cause. The selected continuation is illustrated unless it never regresses, in which case the endpoint is explicitly labeled. All headline values use floating outputs.'),
            ('text', 'The factorial design controls two recipe factors during 800-step adaptation. It does not separately test expanded data coverage, all learning-rate choices, architecture, sensing co-learning or the complete earlier training history. The unchanged-recipe continuation answers a different question. Two seeds and a modest image set limit generalization; upstream public-image exposure is unaudited.'),
            ('text', 'Cropped grayscale software simulation omits physical masks, detector calibration, photon statistics and full-color acquisition. Video frames are reconstructed independently without temporal inputs. Network-collected photos do not satisfy a claim of team capture. The recorded evidence supports condition-specific benefits and regressions rather than universal superiority.'),
            ('heading', '9. Reproducibility and AI Disclosure'),
            ('text', 'Code, locks and small frozen evidence are versioned. The experiment root is /workspace/ProxUnroll-main/runs/hqs_ablation_20261013. Official weights and MAT factors, selected best.pth, recovery last.pth, logs/exit codes, pairing audits and all test outputs have SHA-256 provenance. Original reports and experiments remain unchanged. Codex assisted code, execution, analysis and materials; students must verify results, understand the algorithms and present them. No course-platform submission is claimed.'),
            ('heading', 'References'),
            ('text', '[1] M. F. Duarte et al. Single-Pixel Imaging via Compressive Sampling. IEEE Signal Processing Magazine, 2008. DOI: 10.1109/MSP.2007.914730.<br/>[2] P. Wang et al. Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging. CVPR, 2025. arXiv:2505.23180. Official code: github.com/pwangcs/ProxUnroll.<br/>[3] A. Beck and M. Teboulle. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM J. Imaging Sciences, 2009. DOI:10.1137/080716542.<br/>[4] E. Agustsson and R. Timofte. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. CVPR Workshops, 2017.'),
        ]),
    ]
    report.draw_report('technical_report_ablation_20261013.pdf',
        'Single-Pixel Reconstruction: Controlled HQS Fine-Tuning and Training Ablations', pages)
    markdown_path = report.OUT / 'technical_report_ablation_20261013.md'
    markdown = markdown_path.read_text().replace('../../coursework/figures/', '../../runs/hqs_ablation_20261013/figures/')
    for page in pages:
        for kind, rows in page['blocks']:
            if kind != 'table':
                continue
            original = '\n\n'.join(' | '.join(str(c) for c in row) for row in rows)
            rendered = ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
            rendered.insert(1, '| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
            markdown = markdown.replace(original, '\n'.join(rendered))
    markdown_path.write_text(markdown)
    (BASE / 'presentation_facts.json').write_text(json.dumps(dict(
        primary_continuation=continuation_table, factorial_endpoint=factorial_table,
        regression_continuation=regression_table, selected_step=selected,
        formal_updates=8400, total_gpu_minutes=total_gpu_minutes,
        maximum_allocated_memory_gib=max_memory, clean_mse_reduction_vs_10000=clean_gain,
        strong_noise_mse_reduction_vs_10000=noisy_gain, primary_rows=primary['rows'],
        regression_rows=regression['rows'], failure_case=case), indent=2) + '\n')
    (BASE / 'preserved_baseline_source.json').write_text(json.dumps(dict(
        path=str(baseline_path.relative_to(ROOT)), sha256=sha(baseline_path),
        dataset='NJU_campus_web', cr=.1, sigma=0,
        methods={m: baseline[m] for m in ['adjoint', 'fista_dct', 'hqs', 'admm']}), indent=2) + '\n')


if __name__ == '__main__':
    main()
