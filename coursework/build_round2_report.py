"""Six-page revision sourced only from completed, verified round-2 results."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from coursework import build_reports as report
from coursework.run_experiments import ROOT

BASE=ROOT/'runs/lab244_5000_20261009'
FIG=BASE/'figures'
DATASETS=['Kodak','HEVC_B','HEVC_E','DIV2K_fresh']
LABELS=['Kodak24','HEVC B','HEVC E','Fresh DIV2K']


def main():
    analysis=json.loads((BASE/'comparison/mse_analysis.json').read_text())
    audit=json.loads((BASE/'comparison/verification.json').read_text())
    train=json.loads((BASE/'finetune/summary.json').read_text())
    first_phase=json.loads((ROOT/'runs/lab244_round2_20261009/finetune/summary.json').read_text())
    total_wall=train['wall_seconds']+first_phase['wall_seconds']
    training_audit=json.loads((BASE/'training_audit.json').read_text())
    legacy=json.loads((ROOT/'runs/lab220_20261009/baseline/summary.json').read_text())
    fresh_controls=json.loads((BASE/'fresh_baseline/summary.json').read_text())
    fresh_control_audit=json.loads((BASE/'fresh_baseline/verification.json').read_text())
    assert fresh_control_audit['rows']==fresh_control_audit['float_arrays_recomputed']==1408
    overlap=json.loads((BASE/'control_overlap_verification.json').read_text());assert overlap['conditions']==352
    legacy['datasets'].extend(fresh_controls['datasets']);legacy['videos'].extend(fresh_controls['videos'])
    legacy['sources']=['runs/lab220_20261009/baseline','runs/lab244_5000_20261009/fresh_baseline']
    def value(d,cr=.1,sigma=0.,reference='hqs_round1'):
        return next(r for r in analysis['datasets'] if r['dataset']==d and float(r['cr'])==cr and float(r['sigma'])==sigma and r['reference']==reference)
    def old(d,m):return next(r['psnr'] for r in legacy['datasets'] if r['dataset']==d and r['method']==m and r['cr']==.1 and r['sigma']==0)
    FIG.mkdir(exist_ok=True);plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.2),layout='constrained')
    for d,label in zip(DATASETS,LABELS):
        axes[0].plot([1,4,10,25,50],[value(d,r)['mse_reduction_percent'] for r in [.01,.04,.1,.25,.5]],'-o',label=label)
    axes[0].axhline(5,color='gray',linestyle='--');axes[0].set(xlabel='Clean nominal sampling (%)',ylabel='MSE reduction\nvs round 1 (%)')
    axes[0].legend(fontsize=11,ncol=2)
    x=np.arange(4)
    for i,s in enumerate([0.,.01,.05]):axes[1].bar(x+(i-1)*.24,[value(d,sigma=s)['mse_reduction_percent'] for d in DATASETS],width=.24,label=f'noise {s:g}')
    axes[1].set_xticks(x,LABELS,rotation=15);axes[1].axhline(0,color='gray',linewidth=.7)
    axes[1].set(ylabel='10% MSE\nreduction (%)');axes[1].legend(fontsize=11)
    fig.savefig(FIG/'mse_results.png',dpi=220,bbox_inches='tight');plt.close(fig)
    validation=list(csv.DictReader((BASE/'finetune/validation.csv').open()));steps=sorted({int(r['step']) for r in validation})
    curves={}
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.0),layout='constrained')
    for name,sigmas in [('Clean',[0.]),('Mixed',[0.,.01,.05]),('Strong noise',[.05])]:
        curve=[float(np.mean([float(r['psnr']) for r in validation if int(r['step'])==s and float(r['sigma']) in sigmas])) for s in steps]
        curves[name]=curve;axes[0].plot(steps,curve,'-o',label=name)
    axes[0].set(xlabel='Round-2 optimization step',ylabel='Selection PSNR (dB)');axes[0].legend(fontsize=11)
    for i,ref in enumerate(['hqs_round1','hqs']):
        axes[1].bar(x+(i-.5)*.32,[value(d,reference=ref)['mse_reduction_percent'] for d in DATASETS],width=.32,label='vs round 1' if i==0 else 'vs official')
    axes[1].set_xticks(x,LABELS,rotation=15);axes[1].axhline(5,color='gray',linestyle='--');axes[1].set(ylabel='Clean 10% MSE\nreduction (%)');axes[1].legend(fontsize=11)
    fig.savefig(FIG/'validation_and_references.png',dpi=220,bbox_inches='tight');plt.close(fig)
    pairs=list(csv.DictReader((BASE/'comparison/mse_paired_deltas.csv').open()))
    worst=min([r for r in pairs if r['reference']=='hqs_round1' and int(r['seed'])==2026],key=lambda r:float(r['mse_reduction_percent']))
    items=json.loads((BASE/'data/test.json').read_text());item=next(r for r in items if r['name']==worst['image'])
    source=Path(item['path']);source=ROOT/str(source).removeprefix('/workspace/ProxUnroll-main/') if str(source).startswith('/workspace/ProxUnroll-main/') else source
    cr=float(worst['cr']);sigma=float(worst['sigma']);name=worst['image']
    paths=[source]+[BASE/'comparison/reconstructions'/f'{name}_cr{cr:g}_noise{sigma:g}_seed2026_{m}.png' for m in ['hqs_round1','hqs_finetuned']]
    fig,axes=plt.subplots(1,3,figsize=(10.5,3),layout='constrained')
    for ax,path,title in zip(axes,paths,['Reference','Round 1 HQS','Round 2 HQS']):
        ax.imshow(np.asarray(Image.open(path)),cmap='gray',vmin=0,vmax=255);ax.set_title(title);ax.axis('off')
    fig.savefig(FIG/'worst_case.png',dpi=220);plt.close(fig)
    baseline_table=[['Dataset','Adjoint','FISTA','HQS','ADMM']]+[[label]+[f'{old(d,m):.2f}' for m in ['adjoint','fista_dct','hqs','admm']] for d,label in zip(DATASETS,LABELS)]
    clean_table=[['Dataset','MSE drop','Delta dB','95% MSE interval']]
    noise_table=[['Dataset','Clean','Noise .01','Noise .05']]
    for d,label in zip(DATASETS,LABELS):
        r=value(d);clean_table.append([label,f'{r["mse_reduction_percent"]:+.2f}%',f'{r["delta_psnr"]:+.3f}',f'[{r["bootstrap_low"]:+.1f}, {r["bootstrap_high"]:+.1f}]%'])
        noise_table.append([label]+[f'{value(d,sigma=s)["mse_reduction_percent"]:+.2f}%' for s in [0.,.01,.05]])
    video_table=[['Video','Clean drop','Noise .05 drop']]
    sequences=sorted({r['sequence'] for r in analysis['videos'] if r['dataset'].startswith('HEVC')})
    for seq in sequences:
        vals=[next(r for r in analysis['videos'] if r['sequence']==seq and float(r['cr'])==.1 and float(r['sigma'])==s and r['reference']=='hqs_round1')['mse_reduction_percent'] for s in [0.,.05]]
        video_table.append([seq]+[f'{v:+.2f}%' for v in vals])
    fresh=value('DIV2K_fresh');target=f'The fresh DIV2K clean 10% MSE reduction is {fresh["mse_reduction_percent"]:+.2f}% versus round 1, with a paired 95% interval [{fresh["bootstrap_low"]:.2f}, {fresh["bootstrap_high"]:.2f}]%; the requested minimum 5% is '+('met' if fresh['target_at_least_5_percent'] else 'not met')+' by this point estimate.'
    report.FIG=FIG
    original_table=report.table
    report.table=lambda rows,widths:original_table(rows,[121,65,65] if rows[0][0]=='Video' else widths)
    pages=[dict(blocks=[
        ('heading','Abstract'),
        ('text',f'We compare four solvers for simulated single-pixel imaging and extend a short HQS fine-tuning pilot with a bounded noise-aware experiment. All sensing factors stay fixed. The second round uses 800 DIV2K training originals and 16 selection originals; 32 additional originals form a previously unevaluated test set. Kodak24 and spaced HEVC frames provide regression comparisons because their earlier scores were already observed. The selected state comes solely from validation. {target} All negative changes and recovery checkpoints are retained.'),
        ('heading','1. Problem and Prior Work'),
        ('text','Single-pixel imaging collects integrated responses under known spatial patterns [1]. This coursework simulates acquisition from existing grayscale images. ProxUnroll [2] shares a convolution/attention restorer across optimization stages. We use released HQS and ADMM weights and attribute the architecture to its authors. DCT-FISTA [3] and the adjoint provide classical comparisons.'),
        ('heading','2. Common Measurement Contract'),
        ('text','Y = H X W<super>T</super> + E, with X a 256 x 256 grayscale image. Every solver receives identical factor rows and measured values. The common MAT operator is fixed for training and testing, although it differs from checkpoint-native factors. HQS performs residual correction and learned restoration; ADMM adds a dual update. Six-stage memory links iterations within one image, not video frames.'),
        ('text','DCT-FISTA uses 200 iterations and lambda 0.001 from the original separate validation study. We do not choose its regularization on Kodak, HEVC or the fresh DIV2K tests. Nominal clean sampling ratios are 1%, 4%, 10%, 25% and 50%; stored row counts determine the actual ratio. At 50%, 181 x 181 available measurements give 49.9893%.'),
        ('text','At 10%, Gaussian measurement noise has relative RMS 0.01 or 0.05, each with three predeclared seeds. Predictions are clipped to [0,1]. PSNR and skimage SSIM use data_range 1 and no border removal. Floating-point metrics precede display PNG quantization. Reconstruction arrays, measurements hashes and source hashes support independent recalculation.'),
    ]),dict(blocks=[
        ('heading','3. Data and Leakage Controls'),
        ('text','Public Kodak24 and laboratory-held HEVC B/E files are existing data, not self-captured photographs. The eight HEVC videos are BQTerrace, BasketballDrive, Cactus, Kimono1, ParkScene, FourPeople, Johnny and KristenAndSara. Each supplies frames 0, floor(N/3), floor(2N/3). All videos belong entirely to test; no adjacent training frames are used.'),
        ('text','The HEVC files match 8-bit planar YUV420 packing, canonical dimensions and integral byte counts. BQTerrace stores 601 frames and BasketballDrive 501, each one beyond its nominal version. We use the actual lengths and raw Y planes. Raw range metadata are absent, so stored Y/255 is preserved without guessed TV/full expansion. Manifests record source hashes, frame hashes, dimensions, frame numbers and observed ranges.'),
        ('text','Every validation/test view uses the largest centered square, then OpenCV INTER_AREA resize to 256 x 256. PNG RGB uses OpenCV RGB2YCrCb Y. The centered crop changes the task and discards side content. DIV2K [4] training originals receive valid step-seeded square crops and flips only after source-image split assignment.'),
        ('text','Round 1 trained on 128 originals for 100 updates. Round 2 uses all 800 official DIV2K training originals and keeps the same 16 selection originals. Thirty-two test originals are selected by fixed evenly spaced indices among the remaining 84 official validation originals. Source hashes are disjoint between all current groups. These 32 were not evaluated in round 1 or used for this selection. Official pretraining exposure is not fully audited.'),
        ('text','Kodak/HEVC already informed the observation that strong-noise results were weak in round 1, so their round-2 scores are explicitly regression tests. The fresh DIV2K split adds evidence against repeated test adaptation. No round-2 test result is used to choose checkpoints, change the training recipe or stop training.'),
        ('heading','3.1. Experimental Units'),
        ('text','Noise seeds average within image, frames within video, then videos equally within each HEVC class. Still images are equal original-image units. MSE percentages use the ratio of equally weighted mean MSE values. They are not percentages of PSNR numbers or a conversion of mean PSNR gains. Bootstrap intervals resample paired originals or entire videos.'),
    ]),dict(table_widths=[59,45,49,49,49],blocks=[
        ('heading','4. Four-Method Controls'),('text','Table 1 reports four-method controls at clean 10% sampling, in PSNR dB. Kodak/HEVC use the preserved first evaluation; fresh DIV2K controls run after checkpoint selection is frozen. All share the common operator. HEVC averages videos equally.'),('table',baseline_table),
        ('text','The first full sweep retained 2,112 main rows and 6,336 stage rows. The 100-update check used eight conditions and three states (official, selected, last): 1,152 main rows and 6,912 stage rows. Fresh four-method controls add 1,408 main and 4,224 stage rows, independently recomputed. The 352-row original study and all prior reports/decks remain preserved.'),
        ('heading','4.1. Round-2 Changes by Video'),
        ('text','Table 2 reports MSE reductions of round 2 versus round 1 at 10%. Each row averages all three preselected frames and every specified noise repeat.'),('table',video_table),
        ('heading','5. Predeclared Round-2 Training'),
        ('text','Round 2 initializes strictly from the first validation-selected HQS weights and creates a new Adam optimizer at learning rate 1e-5. This is new fine-tuning initialization, not restoration of the first optimizer. Subsequent resumptions load the complete round-2 state. All six H/W parameter tensors remain frozen. Only reconstruction tensors update.'),
        ('text','The loss changes to direct grayscale ground-truth weighted RMSE: 0.01 for each of five intermediate outputs and 0.95 for the final output. Training independently draws a ratio from .01,.04,.1,.1,.1,.25,.25,.5 and relative measurement RMS from 0,0,.01,.05. Half the draws are clean. The larger data coverage, objective and noise augmentation change together, so this experiment does not isolate their causal contributions.'),
        ('text','Selection averages PSNR over 16 originals, 10%/25% ratios and noise 0/.01/.05 with fixed validation draws. A candidate is eligible only if its clean validation PSNR stays within 0.03 dB of initialization. Initialization itself is an eligible candidate. Selection and stopping never access test images.'),
        ('text','The user extended the initial 1,000-update pilot to 5,000 total updates in this round before any test inference. The continuation is capped at 120 minutes, validates every 250 updates, and preserves the update rule and complete state. The original patience stop is disabled for this explicit step request; non-finite values and resource bounds still stop training. Full model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, step, crop and noise records are saved atomically and reopened. A separate best file stores the selected state.'),
    ]),dict(figures=[('validation_and_references.png','Figure 1. Validation determines the selected state. Clean 10% MSE changes use separate round-1 and official references.')],table_widths=[71,49,50,81],blocks=[
        ('heading','5.1. Runtime and Resume Verification'),
        ('text',f'The verified peer is 192.168.1.244 and container luozc_mlvc. Its /workspace and /datasets mounts share storage with server 220. The project-local Python 3.11 / PyTorch 2.6.0 CUDA 12.4 environment passes dependency checks. A previously idle 24 GiB RTX 3090 is fixed by GPU UUID. No other project environment or running job is modified.'),
        ('text',f'An actual 3+1 update resume matches uninterrupted four-update training bitwise for parameters, optimizer, scheduler and RNG, including crop, measurement and loss records. A separate one-update check confirms that the budget-extension driver restores the same full state and produces the same update. Round 2 totals {train["completed_steps"]} updates: {first_phase["completed_steps"]} in the first phase plus {train["completed_steps"]-first_phase["completed_steps"]} continued updates, stopping by {train["stop_reason"]}. The two formal phases take {total_wall:.1f} s in total, with peak allocated memory {train["peak_memory_mib"]/1024:.2f} GiB. Validation selects round-2 step {train["best_step"]}. Independent audit confirms unchanged sensing factors and {training_audit["changed_reconstruction_tensors"]} updated reconstruction tensors in the last state.'),
        ('heading','6. Paired Test Outcomes'),('text','Table 3 gives clean 10% changes against the first 100-step HQS. Positive MSE reduction means less error. Intervals are paired 95% bootstrap intervals with one training seed; they do not capture training-seed variation.'),('table',clean_table),
        ('text',target),
    ]),dict(figures=[('mse_results.png','Figure 2. All predeclared clean ratios and noisy 10% conditions. Positive values indicate less MSE than round 1.')],table_widths=[73,56,61,61],blocks=[
        ('heading','6.1. Noise and Per-Video Results'),('text','Table 4 reports 10% MSE reductions against round 1 under all noise levels. The report includes weak or negative groups as well as gains.'),('table',noise_table),
        ('text',f'Round 2 saves {audit["rows"]:,} main metric rows and {audit["stages"]:,} stage rows. Independent verification recalculates every one of {audit["float_arrays_recomputed"]:,} floating-point reconstructions and regenerates measurement hashes. All methods share each measurement. Per-image differences and every negative case remain in CSV.'),
        ('text','The full MSE video summary retains all eight sequences rather than selecting favorable frames. Each video has only three spaced views. These small class sample sizes limit generalization. Neural inference is per image and has no prior/future-frame input.'),
        ('text','The comparison includes official HQS, first-round HQS and the second-round validation-selected state. Test inference runs after training, and all three use the same GPU and metric implementation. The last recovery state is preserved even if validation selected an earlier checkpoint.'),
    ]),dict(figures=[('worst_case.png',f'Figure 3. Worst first-seed MSE change versus round 1: {name}, sampling {cr:g}, noise {sigma:g}, MSE reduction {float(worst["mse_reduction_percent"]):+.2f}%. Post-test diagnostic only.')],blocks=[
        ('heading','7. Conclusions and Limits'),('text',target),
        ('text','One training seed and one bounded recipe cannot establish a universal improvement. The larger training set, direct reconstruction objective and noise mixture were introduced together. Their separate effects remain unmeasured. Kodak/HEVC are repeated regression comparisons; the fresh DIV2K result is separate. All results apply to centered, resized grayscale views and a common separable operator.'),
        ('text','The simulation omits optical calibration, detector hardware, physical masks, photon statistics and a complete color acquisition system. HEVC frames are independent single-image inverse problems. These results do not establish temporal video reconstruction or optical-camera performance.'),
        ('heading','8. Reproducibility'),
        ('text','Remote root /workspace/ProxUnroll-main; new run runs/lab244_5000_20261009. data/ contains immutable manifests. finetune/ contains selected best.pth and full-state last.pth. comparison/ retains CSV, stages, arrays, PNGs, provenance and MSE analysis. logs/ records detached-job PID, output and exit status. The project guide and predeclared protocol supply exact commands, environment versions and checkpoint hashes.'),
        ('heading','AI Disclosure'),('text','Codex assisted with code, remote execution, analysis, English writing and presentation. Architecture and released weights belong to the cited authors. Students must review and understand the work. No course submission or public release is claimed.'),
        ('heading','References'),
        ('text','[1] Duarte et al. Single-Pixel Imaging via Compressive Sampling. IEEE Signal Processing Magazine, 2008. doi:10.1109/MSP.2007.914730.'),
        ('text','[2] Wang et al. Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging. CVPR 2025. arXiv:2505.23180; github.com/pwangcs/ProxUnroll.'),
        ('text','[3] Beck and Teboulle. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM J. Imaging Sciences, 2009. doi:10.1137/080716542.'),
        ('text','[4] Agustsson and Timofte. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. CVPR Workshops 2017; data.vision.ee.ethz.ch/cvl/DIV2K/.'),
    ])]
    report.draw_report('technical_report_5000_20261009.pdf','Single-Pixel Imaging under Limited Measurements:<br/>Noise-Aware Fine-Tuning and Paired Error Evaluation',pages)
    md=ROOT/'output/pdf/technical_report_5000_20261009.md';md.write_text(md.read_text().replace('../../coursework/figures/','../../runs/lab244_5000_20261009/figures/'))
    facts=dict(analysis=analysis,training=train,first_phase=first_phase,total_training_wall_seconds=total_wall,training_audit=training_audit,validation_curve=dict(steps=steps,**curves),
               worst_case=worst,clean_table=clean_table,noise_table=noise_table,video_table=video_table,legacy=legacy,fresh_control_audit=fresh_control_audit,control_overlap=overlap)
    (BASE/'report_facts.json').write_text(json.dumps(facts,indent=2)+'\n')


if __name__=='__main__':main()
