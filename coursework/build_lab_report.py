"""Six-page revision from completed external evaluation and bounded training only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coursework import build_reports as report
from coursework.run_experiments import ROOT

BASE=ROOT/'runs/lab220_20261009'
FIG=BASE/'figures'


def main():
    for run in ['baseline','finetune','comparison']:
        if not (BASE/run/'COMPLETE').exists(): raise RuntimeError('Incomplete '+run)
    baseline=json.loads((BASE/'baseline/summary.json').read_text())
    comparison=json.loads((BASE/'comparison/paired_comparison.json').read_text())
    training=json.loads((BASE/'finetune/summary.json').read_text()); config=json.loads((BASE/'finetune/config.json').read_text())
    protocol=json.loads((BASE/'data/protocol.json').read_text()); old=json.loads((ROOT/'coursework/results/report_facts.json').read_text())
    smoke=json.loads((BASE/'train_smoke/summary.json').read_text()); resumed=json.loads((BASE/'resume_smoke/summary.json').read_text()); audit=json.loads((BASE/'baseline/verification.json').read_text())
    def row(dataset,method,cr=.1,sigma=0.):
        return next(r for r in baseline['datasets'] if r['dataset']==dataset and r['method']==method and r['cr']==cr and r['sigma']==sigma)
    def delta(dataset,cr=.1,name='hqs_finetuned'):
        return next(r for r in comparison['summary'] if r['dataset']==dataset and r['method']==name and float(r['cr'])==cr and float(r['sigma'])==0.)
    FIG.mkdir(exist_ok=True); plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.2),layout='constrained')
    colors={'adjoint':'#777777','fista_dct':'#D78924','hqs':'#147D78','admm':'#B74760'}
    for method,color in colors.items():
        axs[0].plot([1,4,10,25,50],[row('Kodak',method,r)['psnr'] for r in [.01,.04,.1,.25,.5]],'-o',label=method,color=color)
        axs[1].plot([0,.01,.05],[row('Kodak',method,.1,s)['psnr'] for s in [0,.01,.05]],'-o',color=color)
    axs[0].set(xlabel='Nominal sampling (%)',ylabel='Kodak mean PSNR (dB)',title='Clean measurements');axs[1].set(xlabel='Relative noise RMS',ylabel='Kodak mean PSNR (dB)',title='Noise at 10%');axs[0].legend(ncol=2,fontsize=9)
    fig.savefig(FIG/'external_quality.png',dpi=220);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.1),layout='constrained')
    val=__import__('csv').DictReader((BASE/'finetune/validation.csv').open()); steps={}
    for r in val:steps.setdefault(int(r['step']),[]).append(float(r['psnr']))
    xs=sorted(steps);ys=[np.mean(steps[x]) for x in xs];axs[0].plot(xs,ys,'-o',color=colors['hqs']);axs[0].set(xlabel='Optimization steps',ylabel='Validation mean PSNR (dB)',title='DIV2K selection only')
    datasets=['Kodak','HEVC_B','HEVC_E']; vals=[delta(d)['delta_psnr'] for d in datasets]
    ci=np.array([[delta(d)['bootstrap_low'],delta(d)['bootstrap_high']] for d in datasets]);axs[1].bar(datasets,vals,color=colors['hqs']);axs[1].errorbar(range(3),vals,yerr=np.stack([np.maximum(0,np.array(vals)-ci[:,0]),np.maximum(0,ci[:,1]-np.array(vals))]),fmt='none',ecolor='black',capsize=4)
    axs[1].axhline(0,color='gray',linewidth=.8);axs[1].set(ylabel='Fine-tuned minus original PSNR (dB)',title='Untouched test sets at 10%')
    fig.savefig(FIG/'training_test.png',dpi=220);plt.close(fig)
    table=[['Dataset','Adjoint','FISTA','HQS','ADMM']]
    for dataset in datasets:table.append([dataset]+[f'{row(dataset,m)["psnr"]:.2f}' for m in colors])
    paired=[['Dataset','Delta dB','Delta SSIM','95% interval']]
    for d in datasets:
        r=delta(d);paired.append([d,f'{r["delta_psnr"]:+.3f}',f'{r["delta_ssim"]:+.4f}',f'[{r["bootstrap_low"]:+.3f}, {r["bootstrap_high"]:+.3f}]'])
    video_rows=[['Sequence','HQS dB','ADMM dB']]
    for dataset in ['HEVC_B','HEVC_E']:
        sequences=sorted({r['sequence'] for r in baseline['videos'] if r['dataset']==dataset})
        for seq in sequences:
            video_rows.append([seq]+[f'{next(r["psnr"] for r in baseline["videos"] if r["sequence"]==seq and r["method"]==m and r["cr"]==.1 and r["sigma"]==0):.2f}' for m in ['hqs','admm']])
    gain_text="; ".join(f'{d}: {delta(d)["delta_psnr"]:+.3f} dB' for d in datasets)
    worst=min([r for r in comparison['image_deltas'] if r['method']=='hqs_finetuned' and int(r['seed'])==2026],key=lambda r:r['delta_psnr'])
    cr=float(worst['cr']);sigma=float(worst['sigma']);name=worst['image']
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.0),layout='constrained')
    from PIL import Image
    files=[BASE/'data'/(name+'.png')]+[BASE/'comparison/reconstructions'/f'{name}_cr{cr:g}_noise{sigma:g}_seed2026_{m}.png' for m in ['hqs','hqs_finetuned']]
    for ax,path,title in zip(axs,files,['Reference','Original HQS','Fine-tuned HQS']):
        ax.imshow(np.asarray(Image.open(path)),cmap='gray',vmin=0,vmax=255);ax.set_title(title);ax.axis('off')
    fig.savefig(FIG/'regression_case.png',dpi=220);plt.close(fig)
    report.FIG=FIG
    pages=[dict(blocks=[
      ('heading','Abstract'),
      ('text',f'We extend a reproducible single-pixel imaging simulation to 24 public Kodak images and 24 raw frames from eight laboratory-held HEVC sequences. Four reconstruction methods receive identical measurements. We then fine-tune the released HQS reconstruction network for {training["completed_steps"]} steps with fixed sensing factors, using 128 DIV2K training images and 16 disjoint validation images. Kodak and all HEVC videos remain outside training and model selection. We retain raw metrics, floating-point reconstructions, provenance and complete recovery state. The measured test changes are reported with negative cases and their limits.'),
      ('heading','1. Scope and Related Work'),
      ('text','Single-pixel imaging obtains integrated responses under known masks. Duarte et al. [1] connect this acquisition to compressed sensing. Our software starts with existing images, creates measurements numerically, and compares reconstruction methods. It does not demonstrate optical acquisition or fabricate a self-captured dataset.'),
      ('text','Wang et al. [2] release HQS and ADMM ProxUnroll variants. A shared image restorer combines convolution, window attention and stage memory. Their proximal trajectory training motivates the bounded fine-tuning experiment here. Beck and Teboulle [3] motivate our DCT-FISTA baseline. We attribute the architecture and official weights to their authors.'),
      ('heading','2. Measurement and Reconstruction'),
      ('text','The separable forward model is Y = H X W<super>T</super> + E. X is a 256 x 256 grayscale image. H and W are cropped rows of the released MAT-file factors. All solvers use the same Y, H and W. The adjoint is H<super>T</super> Y W. DCT-FISTA minimizes 0.5 ||H X W<super>T</super> - Y||<sub>F</sub><super>2</super> + lambda ||DCT(X)||<sub>1</sub> with 200 iterations and lambda = 0.001, frozen from the original disjoint validation study.'),
      ('text','HQS alternates residual correction with learned restoration. ADMM also updates a dual variable. The measurement-only reconstruct API accepts no reference image, and regression tests match the original clean forward computation. The final prediction follows six restorer stages. Clean-image targets appear only in training labels and metric computation.'),
      ('text','The common factors differ from factors embedded in the released checkpoints. Comparisons therefore include transfer to a common sensing operator. This is documented rather than attributed entirely to architectural superiority. The original experiment also retained native-operator diagnostics.'),
    ]),dict(blocks=[
      ('heading','3. Data and Evaluation Protocol'),
      ('text','Kodak24 is a public still-image benchmark. HEVC B contributes BQTerrace, BasketballDrive, Cactus, Kimono1 and ParkScene. HEVC E contributes FourPeople, Johnny and KristenAndSara. Each entire sequence belongs to the test split. We select zero-based frames 0, floor(N/3) and floor(2N/3) before reconstruction, giving 24 spaced frames.'),
      ('text','Raw HEVC files match the canonical resolutions, integer byte lengths for 8-bit planar 4:2:0. BQTerrace contains 601 frames and BasketballDrive 501, each one more than its nominal version. We record and use the actual count. We read the original Y plane directly. Raw files carry no self-describing color-range metadata. We preserve stored uint8 Y codes divided by 255, record observed extrema and percentiles, and apply no inferred limited/full-range expansion. RGB PNG images use OpenCV RGB2YCrCb Y.'),
      ('text','Every test and validation image receives the same spatial rule: maximal centered square crop, then OpenCV INTER_AREA resize to 256 x 256. This preserves aspect ratio but excludes side content. Manifests retain source paths, original dimensions, source hashes, raw frame hashes, frame indices, crop coordinates and processed PNG hashes. Results apply to these processed views.'),
      ('text','Clean nominal ratios are 1%, 4%, 10%, 25% and 50%. Ceiling-rounded factor row counts determine actual ratios. At nominal 50%, stored factors cap the measurement count at 181 x 181, or 49.9893%. At 10%, Gaussian noise has RMS scale 0.01 or 0.05 relative to clean Y, with seeds 2026, 2027 and 2028.'),
      ('text','Predictions are clipped to [0,1] before PSNR and skimage SSIM, with data_range = 1 and no border removal. Metrics precede PNG quantization. Noise repeats average within image, frames average within video, and video means average equally within each HEVC class. Kodak averages equally across original images. We report classes and videos separately.'),
      ('heading','3.1. Preserved Initial Study'),
      ('text','The original six regular test images, two disjoint validation images and two procedural stress targets remain intact. That study retained 352 main metric rows and 1,056 neural stage rows, including sampling, noise, stage interventions and failure cases. Its 6-page technical report, 3-page reading report and 12-slide presentation remain as prior versions. This revision uses fresh output directories.'),
      ('heading','3.2. Remote Reproducibility'),
      ('text','We verified the SSH peer as 192.168.1.220, inspected luozc_nvenc_new and its /workspace and /datasets bind mounts, and matched synchronized file hashes. A project-local Linux virtual environment contains PyTorch 2.6.0 with CUDA 12.4. The selected GPU is a 24 GiB RTX 3090, identified by UUID. No container-wide package installation or destructive synchronization is used.'),
    ]),dict(figures=[('external_quality.png','Figure 1. Public Kodak24 quality under the common operator. Noise repeats average within each image.')],table_widths=[59,44,49,49,50],blocks=[
      ('heading','4. External Test Results'),('text','Table 1 reports clean 10% PSNR in dB. HEVC class results give equal weight to each video. Full sampling/noise results, SSIM and timings remain in CSV rather than a selected-image score.'),('table',table),
      ('text',f'The complete four-method extension contains {audit["rows"]:,} metric rows and {audit["stages"]:,} stage rows. Every condition has one shared measurement hash across all solvers. Verification recomputed metrics from {audit["float_arrays_recomputed"]:,} retained float arrays. This establishes consistency with stored reconstruction evidence.'),
      ('text','The sweep shows how information loss and measurement noise affect each solver. A larger test set improves coverage, but it cannot establish universal recovery or deployment latency. CPU DCT-FISTA and GPU neural timings represent different execution backends. They must not be presented as a hardware-matched speed comparison.'),
      ('text','The finite six-stage trajectories and memory interventions from the original study remain inference experiments. This extension does not retrain an architecture ablation or prove asymptotic convergence. Video frames are reconstructed independently, with no temporal input or reference propagation.'),
    ]),dict(table_widths=[124,64,63],blocks=[
      ('heading','4.1. Results by Video'),('text','Table 2 reports 10% clean results for every selected video. Each row averages its three preselected, spaced frames. Five B videos and three E videos are separate experimental units.'),('table',video_rows),
      ('heading','5. Fixed-Operator Fine-Tuning'),
      ('text','A fixed seeded selection takes 128 original images from DIV2K train [4]. Sixteen evenly indexed DIV2K validation originals provide selection. We separate source images before any crop and check source hashes across splits. Kodak and all eight HEVC videos are absent from fine-tuning and selection. Original pretraining exposure has not been fully audited, so untouched here refers to this new training procedure.'),
      ('text','Each training step samples a valid random square whose side lies between 256 and the smaller source dimension, then applies area resizing and seeded flips. Step-keyed image order and crop randomness survive resume. This avoids oversized random crops in the upstream training loader. We use batch size one and clean ratios cycling over 1%, 4%, 10%, 25% and 50%.'),
      ('text',f'Training starts from the official HQS state. Its 256 x 256 factors are set to the common MAT operator, and all six sensing-factor parameters are frozen. The optimizer learns reconstruction parameters only. We retain the upstream weighted proximal trajectory RMSE, with weights 0.01 for five intermediate stages and 0.95 for the final stage. Adam uses a fixed {config["lr"]:g} learning rate and gradient-norm clipping at 1.'),
      ('text','Initialization loads model weights strictly and creates a new optimizer. Full resume additionally restores Adam, scheduler, RNG states and the optimization step, with matching data/operator/configuration contracts. Atomic checkpoint writes are reopened and checked. The selected best state and a complete last recovery state are separate artifacts.'),
    ]),dict(figures=[('training_test.png','Figure 2. Validation selects the checkpoint. Test error bars bootstrap original images or whole videos, with one training seed.')],table_widths=[57,50,56,88],blocks=[
      ('heading','5.1. Cost and Held-Out Comparison'),
      ('text',f'The {smoke["completed_steps"]}-step smoke verified finite loss, unchanged sensing factors and loadable optimizer checkpoints. Its mean step cost was {smoke["mean_step_seconds"]:.2f} s with peak allocated GPU memory {smoke["peak_memory_mib"]/1024:.2f} GiB. The continuation updated {training["completed_steps"]-training["start_step"]} more steps in {training["wall_seconds"]:.1f} s, reaching {training["completed_steps"]} total steps in {training["wall_seconds"]+smoke["wall_seconds"]+resumed["wall_seconds"]:.1f} s across the training chain. It selected step {training["best_step"]}, and reached {training["peak_memory_mib"]/1024:.2f} GiB peak allocated memory.'),
      ('text','Table 3 gives the selected HQS checkpoint minus original HQS at 10% sampling. The same test images, operator, noise and metric code apply to both. The last training checkpoint is also evaluated, even when validation selects an earlier state.'),('table',paired),
      ('text',f'At relative noise RMS 0.05, selected-model PSNR changes are Kodak +0.003 dB, HEVC B -0.031 dB and HEVC E -0.041 dB. SSIM increases by about 0.003-0.004. Thus clean-measurement gains coexist with weak or negative PSNR changes under stronger noise. The fine-tuning data include only clean measurements.'),
      ('text','Intervals come from 5,000 paired bootstrap samples at the dataset unit level. They describe this one run and small dataset, with only five B and three E sequences. They cannot quantify variability across independent training seeds. We report 25% and noisy 10% comparisons separately in the raw comparison tables.'),
      ('text',f'The measured clean 10% changes are {gain_text}. A validation gain does not guarantee a test gain. Every image delta remains in paired_comparison.json, including regressions. We identify the worst changes for inspection and preserve their reconstructions. No additional ADMM training or repeated tuning against Kodak/HEVC results is performed in this first bounded extension.'),
    ]),dict(figures=[('regression_case.png',f'Figure 3. Smallest first-seed test change: {name}, {cr*100:g}% sampling, noise RMS {sigma:g}, delta {worst["delta_psnr"]:+.3f} dB. Diagnostic chosen after evaluation, never used for selection.')],blocks=[
      ('heading','6. Limitations and Conclusions'),
      ('text','This project compares four methods on shared synthetic measurements and extends coverage to public Kodak images and laboratory-held HEVC content. It provides a bounded, fixed-operator fine-tuning experiment whose benefit is determined by the measured untouched-test comparison, rather than training loss or selected photographs.'),
      ('text','Important limits remain: centered crop/resizing changes the benchmark task, the raw-video range has no embedded provenance, the original pretrained-data exposure is not fully audited, and the common operator differs from checkpoint-native factors. We use one fine-tuning seed and a small image subset. The test comparison is conditional on these preprocessing and measurement rules.'),
      ('text','The simulation omits optical calibration, mask constraints, detector noise models, quantization and photon statistics. No physical single-pixel camera or full-color recovery is demonstrated. Each HEVC frame is an independent grayscale inverse problem. The model does not use previous or future frames, so these results do not establish a video reconstruction network.'),
      ('heading','7. Evidence and Reproduction'),
      ('text','Remote root: /workspace/ProxUnroll-main. Run root: runs/lab220_20261009. data/ holds manifests and preprocessing records. baseline/ and comparison/ hold raw CSV, stage CSV, float arrays, PNGs and provenance. finetune/ holds train/validation CSV, best.pth, last.pth and the bounded-run summary. logs/ holds detached-job logs, PID records and exit codes. environment.txt freezes Linux package versions.'),
      ('text','Reproduce data with python -m coursework.prepare_lab_data --output NEW_DIR. Evaluate with python -m coursework.run_experiments --manifest NEW_DIR/test.json --output NEW_RESULTS --device cuda. Train with python -m coursework.finetune --manifest NEW_DIR/train_val.json --output NEW_TRAIN --max-steps 100 --max-minutes 10. Use the CUDA/BLAS exports in the guide. Use --init only for new weight initialization, or --resume for full state recovery. The project guide records exact measured-run commands and checkpoint hashes.'),
      ('heading','AI Disclosure'),
      ('text','Codex assisted with implementation, remote execution, analysis, English writing and presentation preparation. The team must review and understand the evidence. The architecture and official weights belong to the cited authors. No student-specific contribution or course-platform submission is claimed.'),
      ('heading','References'),
      ('text','[1] M. F. Duarte et al. Single-Pixel Imaging via Compressive Sampling. IEEE Signal Processing Magazine 25(2), 83-91, 2008. doi:10.1109/MSP.2007.914730.'),
      ('text','[2] P. Wang et al. Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging. CVPR, 2025. arXiv:2505.23180. github.com/pwangcs/ProxUnroll.'),
      ('text','[3] A. Beck and M. Teboulle. A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. SIAM J. Imaging Sciences 2(1), 183-202, 2009. doi:10.1137/080716542.'),
      ('text','[4] E. Agustsson and R. Timofte. NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study. CVPR Workshops, 2017. DIV2K: data.vision.ee.ethz.ch/cvl/DIV2K/.'),
    ])]
    report.draw_report('technical_report_lab220_20261009.pdf','Single-Pixel Imaging under Limited Measurements:<br/>External Evaluation and Fixed-Operator Fine-Tuning',pages)
    (BASE/'report_facts.json').write_text(json.dumps(dict(baseline=baseline,comparison=comparison,training=training,smoke=smoke,validation_curve={'steps':xs,'psnr':ys}),indent=2)+'\n')

if __name__=='__main__': main()
