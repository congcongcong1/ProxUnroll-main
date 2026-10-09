"""Six-page English report from verified web tests and completed training."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from coursework import build_reports as report
from coursework.run_experiments import ROOT

BASE = ROOT/'runs/nanjing_web30_20261009'
TRAIN = ROOT/'runs/lab244_10000_20261009'
FIG = BASE/'figures'
DATASETS = ['NJU_campus_web', 'Nanjing_scenic_web']
LABELS = ['NJU campus (18)', 'Nanjing scenic (12)']
LOCATION_LABELS = {'NJU_Gulou':'NJU Gulou', 'NJU_Xianlin':'NJU Xianlin',
    'Xuanwu_Lake':'Xuanwu Lake', 'Sun_Yat_sen_Mausoleum':'Sun Yat-sen',
    'Ming_Xiaoling':'Ming Xiaoling', 'Qinhuai_Fuzimiao':'Qinhuai/Fuzimiao'}


def main():
    audit = json.loads((BASE/'comparison/verification.json').read_text())
    assert audit['rows'] == audit['float_arrays_recomputed'] == 2310
    assert audit['stages'] == 9900
    analysis = json.loads((BASE/'comparison/mse_analysis.json').read_text())
    summary = json.loads((BASE/'comparison/summary.json').read_text())
    training = json.loads((TRAIN/'finetune/summary.json').read_text())
    train_audit = json.loads((TRAIN/'training_audit.json').read_text())
    assert train_audit['total_steps'] == 10000 and train_audit['validation_selection_and_clean_guard_verified']
    phase1 = json.loads((ROOT/'runs/lab244_5000_20261009/finetune/summary.json').read_text())
    pilot = json.loads((ROOT/'runs/lab244_round2_20261009/finetune/summary.json').read_text())
    total_wall = sum(x['wall_seconds'] for x in [training, phase1, pilot])
    sources = json.loads((BASE/'data/test.json').read_text())
    def value(dataset, sigma=0., ref='hqs_round1', candidate='hqs_10000', cr=.1):
        return next(r for r in analysis['datasets'] if r['dataset']==dataset and r['cr']==cr and r['sigma']==sigma and r['reference']==ref and r['candidate']==candidate)
    def baseline(dataset, method, cr=.1, sigma=0.):
        return next(r for r in summary['datasets'] if r['dataset']==dataset and r['cr']==cr and r['sigma']==sigma and r['method']==method)
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':12, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(10.5,3.3),layout='constrained')
    for ax,d,label in zip(axes,DATASETS,LABELS):
        for m,n,c in [('adjoint','Adjoint','#777777'),('fista_dct','DCT-FISTA','#D78924'),('hqs','Official HQS','#147D78'),('admm','Official ADMM','#B74760')]:
            ax.plot([1,4,10,25,50],[baseline(d,m,cr=r)['psnr'] for r in [.01,.04,.1,.25,.5]],'-o',label=n,color=c)
        ax.set(title=label,xlabel='Nominal clean sampling (%)',ylabel='Mean PSNR (dB)')
        ax.legend(fontsize=10)
    fig.savefig(FIG/'sampling.png',dpi=220);plt.close(fig)
    validation = list(csv.DictReader((TRAIN/'finetune/validation.csv').open()))
    steps = sorted({int(r['step']) for r in validation})
    curves = {}
    fig, axes = plt.subplots(1,2,figsize=(10.5,3.2),layout='constrained')
    for name,levels,color in [('Clean',[0.],'#147D78'),('Mixed',[0.,.01,.05],'#D78924'),('Strong noise',[.05],'#B74760')]:
        curve=[float(np.mean([float(r['psnr']) for r in validation if int(r['step'])==s and float(r['sigma']) in levels])) for s in steps]
        curves[name]=curve;axes[0].plot(steps,curve,label=name,color=color)
    axes[0].axvline(5000,color='#888888',linestyle='--');axes[0].legend(fontsize=10)
    axes[0].set(xlabel='Round-2 total training step',ylabel='Selection PSNR (dB)')
    x=np.arange(2)
    for i,ref in enumerate(['hqs','hqs_round1','hqs_5000']):
        axes[1].bar(x+(i-1)*.24,[value(d,ref=ref)['mse_reduction_percent'] for d in DATASETS],width=.24,label={'hqs':'vs official','hqs_round1':'vs 100-step','hqs_5000':'vs 5000-budget'}[ref])
    axes[1].set_xticks(x,['Campus','Scenic']);axes[1].axhline(5,color='gray',linestyle='--')
    axes[1].set(ylabel='Clean 10% MSE reduction (%)');axes[1].legend(fontsize=10)
    fig.savefig(FIG/'validation_clean.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.0),layout='constrained')
    for d,label in zip(DATASETS,LABELS):
        axes[0].plot([1,4,10,25,50],[value(d,cr=r)['mse_reduction_percent'] for r in [.01,.04,.1,.25,.5]],'-o',label=label)
    axes[0].axhline(0,color='gray',linewidth=.7);axes[0].axhline(5,color='gray',linestyle='--')
    axes[0].set(xlabel='Nominal clean sampling (%)',ylabel='MSE reduction\nvs 100-step (%)');axes[0].legend(fontsize=10)
    for i,sigma in enumerate([0.,.01,.05]):
        axes[1].bar(x+(i-1)*.24,[value(d,sigma)['mse_reduction_percent'] for d in DATASETS],width=.24,label=f'noise {sigma:g}')
    axes[1].set_xticks(x,['Campus','Scenic']);axes[1].axhline(0,color='gray',linewidth=.7)
    axes[1].set(ylabel='10% MSE reduction\nvs 100-step (%)');axes[1].legend(fontsize=10)
    fig.savefig(FIG/'mse_noise.png',dpi=220);plt.close(fig)
    pairs=list(csv.DictReader((BASE/'comparison/mse_paired_deltas.csv').open()))
    worst=min([r for r in pairs if r['candidate']=='hqs_10000' and r['reference']=='hqs_5000' and int(r['seed'])==2026],key=lambda r:float(r['mse_reduction_percent']))
    item=next(r for r in sources if r['name']==worst['image'])
    name=item['name'];cr=float(worst['cr']);sigma=float(worst['sigma'])
    gt=np.asarray(Image.open(BASE/'data'/item['path']),dtype=np.float32)/255
    images=[gt]+[np.asarray(Image.open(BASE/'comparison/reconstructions'/f'{name}_cr{cr:g}_noise{sigma:g}_seed2026_{m}.png'),dtype=np.float32)/255 for m in ['hqs_5000','hqs_10000']]
    error=np.abs(images[-1]-gt)
    # These error-map quantities concern displayed 8-bit images; headline metrics
    # were independently recomputed from unquantized retained floating arrays.
    display_stats=dict(mean_absolute_error=float(error.mean()),maximum_absolute_error=float(error.max()),
                       edge_density=float(np.mean(np.abs(np.diff(gt,axis=0))>.1)),basis='8-bit display, diagnostic only')
    old_float,new_float=[np.clip(np.load(BASE/'comparison/reconstructions'/f'{name}_cr{cr:g}_noise{sigma:g}_seed2026_{m}.npy'),0,1) for m in ['hqs_5000','hqs_10000']]
    regions={}
    for label,begin,end in [('sky',0,106),('skyline',106,162),('water',162,256)]:
        mse=[float(np.mean((gt[begin:end]-p[begin:end])**2,dtype=np.float64)) for p in [old_float,new_float]]
        regions[label]=dict(rows=[begin,end],old_mse=mse[0],new_mse=mse[1],mse_increase_percent=100*(mse[1]/mse[0]-1))
    fig,axes=plt.subplots(1,4,figsize=(10.5,2.9),layout='constrained')
    for ax,im,title in zip(axes,[*images,error],['Reference','5000-budget HQS','10000-step HQS','Absolute error']):
        ax.imshow(im,cmap='gray' if title!='Absolute error' else 'magma',vmin=0,vmax=1 if title!='Absolute error' else .3)
        ax.set_title(title,fontsize=11);ax.axis('off')
    fig.savefig(FIG/'worst_case.png',dpi=220);plt.close(fig)
    controls=[['Dataset','Adjoint','FISTA','HQS','ADMM']]+[[label]+[f'{baseline(d,m)["psnr"]:.2f}' for m in ['adjoint','fista_dct','hqs','admm']] for d,label in zip(DATASETS,['Campus18','Scenic12'])]
    clean=[['Dataset','vs official','vs 100-step','vs 5000']]+[[label]+[f'{value(d,ref=r)["mse_reduction_percent"]:+.2f}%' for r in ['hqs','hqs_round1','hqs_5000']] for d,label in zip(DATASETS,['Campus18','Scenic12'])]
    noisy=[['Dataset','Clean','Noise .01','Noise .05']]+[[label]+[f'{value(d,s)["mse_reduction_percent"]:+.2f}%' for s in [0.,.01,.05]] for d,label in zip(DATASETS,['Campus18','Scenic12'])]
    locs=[['Location (photos)','Clean','Noise .05']]
    for location,label in LOCATION_LABELS.items():
        vals=[next(r for r in analysis['locations'] if r['location']==location and r['cr']==.1 and r['sigma']==s and r['candidate']=='hqs_10000' and r['reference']=='hqs_round1') for s in [0.,.05]]
        locs.append([f'{label} ({vals[0]["images"]})']+[f'{r["mse_reduction_percent"]:+.2f}%' for r in vals])
    result_sentence='At clean 10% sampling, the 10000-step state reduces mean MSE by '+', '.join(f'{value(d)["mse_reduction_percent"]:+.2f}% on {label}' for d,label in zip(DATASETS,['campus','scenic']))+' versus the 100-step state. '
    result_sentence+='The 5% minimum is '+('met on both subsets.' if all(value(d)['target_at_least_5_percent'] for d in DATASETS) else 'not met on both subsets.')
    negative=analysis['negative_10000_conditions_by_reference']
    credit=f"{escape(item['title'])}; {escape(item['author'])}; {item['license']}. Source and license: <link href='{item['page_url']}'>Commons file</link>, <link href='{item['license_url']}'>license</link>. Gray crop, resize, reconstruction and error-map adaptations."
    if any(ord(c)>127 for c in item['title']+item['author']):
        # Retain exact author names without missing-glyph squares in PDF.
        pdfmetrics.registerFont(TTFont('ImageCredits','/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
        credit="<font name='ImageCredits'>"+credit+'</font>'
    report.FIG=FIG
    pages=[dict(blocks=[
        ('heading','Abstract'),('text','We compare adjoint reconstruction, DCT-FISTA and released HQS/ADMM networks under matched simulated single-pixel measurements. The original study is extended with public Kodak24, raw laboratory-held HEVC frames, controlled HQS fine-tuning and 30 externally authored photographs of Nanjing University and Nanjing sights. All sensing factors remain fixed. A validation-selected 10000-step state is compared with official, 100-step and 5000-budget checkpoints on the same photographs. '+result_sentence+' Every negative condition is retained. The web collection is prepared by the team, not photographed by the team.'),
        ('heading','1. Problem and Prior Work'),('text','Single-pixel imaging uses known spatial masks and integrated detector responses [1]. We simulate acquisition from existing 256 x 256 grayscale images. ProxUnroll [2] supplies a shared convolution/attention restorer within a short optimization trajectory. Architecture and released weights are attributed to the authors. DCT-FISTA [3] is a classical sparse comparison, not the original 2008 hardware system.'),
        ('heading','2. Measurement and Solver Contract'),('text','Y = H X W<super>T</super> + E. The adjoint H<super>T</super> Y W maps measurements to image space. DCT-FISTA applies gradient descent, DCT soft thresholding and momentum for 200 iterations. Lambda 0.001 comes from the original separate validation images, before external tests. HQS alternates residual correction with learned restoration. ADMM adds a dual variable. Six-stage feature memory links iterations of one image; it never links video frames.'),
        ('text','Every method receives identical common-MAT factor rows and measurement values. The external MAT overrides checkpoint-native sensing factors, including ADMM. Nominal clean sampling ratios are 1%, 4%, 10%, 25% and 50%. Available row counts determine actual ratios; 181 x 181 measurements realize 49.9893% at nominal 50%.'),
        ('text','At 10%, zero-mean Gaussian measurement noise has relative RMS 0.01 or 0.05. Three fixed seeds repeat each noisy case. Predictions are clipped to [0,1] only for metrics and display. PSNR and skimage SSIM use data_range 1 with no border crop. Headline metrics precede PNG quantization. Percent improvement means 100(1 - mean new MSE / mean reference MSE), not percent PSNR change.'),
    ]),dict(blocks=[
        ('heading','3. Data, Provenance and Splits'),('text','Preserved evaluations include Kodak24 and eight HEVC B/E videos with three spaced raw Y frames each. Manifests retain video names, dimensions, 8-bit planar YUV420 layout, byte counts and frame indices 0, floor(N/3), floor(2N/3). Stored Y/255 is preserved because reliable range metadata are absent. Entire videos are test-only. These are public or laboratory-held data, not our own captures.'),
        ('text','The new web collection contains 18 NJU campus photographs: nine Gulou and nine Xianlin. Twelve scenic photographs cover Xuanwu Lake, Sun Yat-sen Mausoleum, Ming Xiaoling and Qinhuai/Fuzimiao, three each. Sources were chosen by location and visual coverage before reconstruction, with no performance filtering. It includes buildings, greenery, water, text, aerial views and night scenes.'),
        ('text','Wikimedia Commons file pages document external authors and CC BY, CC BY-SA or CC0 licenses. ATTRIBUTION.md provides all 30 titles, authors, source and license links. We chose an available CC license where files are dual-licensed. Source captions and a pre-inference contact-sheet review establish scene identity; this is a convenience collection, not random campus sampling.'),
        ('text','Direct image retrieval timed out. The downloader therefore requests full-resolution PNG transcodes through wsrv.nl, without requested resizing, cropping or filtering. Dimensions and downloaded PNG hashes are recorded. These are not original JPEG byte files; codec and color-profile handling can differ from decoding originals. The source PNGs remain locally retained, while the exact evaluated grayscale inputs accompany the Git evidence bundle.'),
        ('text','For all external still photographs we apply EXIF orientation, the largest centered square, OpenCV RGB2YCrCb Y and INTER_AREA resize to 256 x 256, then uint8/255. Every manifest records crop coordinates and dimensions. Center cropping discards lateral content, and resizing changes high-frequency detail. Source and processed duplicates are checked before inference. Related viewpoints and repeated photographers remain disclosed.'),
        ('text','The new 30 photographs are used only for test. Training uses 800 official DIV2K [4] training originals and 16 separate validation originals. Source-image split assignment precedes random crops or flips. Earlier Kodak/HEVC and 32 DIV2K test results are explicitly preserved regression evidence. Fifty-two additional DIV2K originals were locked before continuation but have not yet been evaluated. Public-image exposure during official pretraining is not fully audited.'),
        ('heading','3.1. Equal Photographic Units'),('text','Noise repeats average within each photograph, then all originals contribute equal weight. Campus/scenic subsets and all six locations are reported separately. Earlier video means average frames within each video before equal video weighting. Related photographs are not asserted statistically independent; web-test percentages are descriptive, without a population confidence claim.'),
    ]),dict(figures=[('sampling.png','Figure 1. Four methods on both complete web subsets; all five predeclared clean sampling ratios.')],table_widths=[59,45,49,49,49],blocks=[
        ('heading','4. Four-Method Web Controls'),('text','Table 1 shows clean 10% PSNR in dB. The same collection, operator and measurement realizations are used for all methods. Neural models use CUDA; FISTA and adjoint use CPU. Timings are synchronized after warmup but are not a same-device algorithm-speed benchmark.'),('table',controls),
        ('heading','5. Controlled HQS Fine-Tuning'),('text','The 100-step pilot initializes official HQS. Round 2 initializes its validation-selected weights with a new Adam optimizer at 1e-5; this is fine-tuning initialization. Subsequent continuation restores full model, optimizer, scheduler, step and Python/NumPy/Torch/CUDA RNG state. All six H/W sensing tensors stay frozen; 427 reconstruction tensors change.'),
        ('text','Batch-one FP32 training draws sampling ratios .01,.04,.1,.1,.1,.25,.25,.5 and relative noise 0,0,.01,.05 independently. Direct-ground-truth RMSE weights five intermediate outputs at .01 each and the final at .95. Training coverage, objective and noise augmentation changed together relative to the pilot; their separate causal effects are unmeasured.'),
        ('text','Validation averages 16 originals at 10%/25% sampling and noise 0/.01/.05. Mixed PSNR selects the best eligible checkpoint, subject to clean validation within 0.03 dB of initialization. Initialization is eligible. No web-test result chooses a checkpoint, changes the recipe or triggers more training.'),
    ]),dict(figures=[('validation_clean.png','Figure 2. Complete validation trace and clean web-test MSE reductions. The 5000-budget model selected step 4500; continued training selected step 10000.')],table_widths=[59,64,64,64],blocks=[
        ('heading','5.1. Actual Completion and Recovery'),('text',f'Verified peer 192.168.1.244, container luozc_mlvc, project /workspace/ProxUnroll-main. Project-local Python 3.11.11 and PyTorch 2.6.0+cu124 keep other environments intact. A UUID-selected RTX 3090 runs bounded detached training. Continuation adds 5000 updates to reach 10000, with a 150-minute bound. It completes in {training["wall_seconds"]/60:.1f} min; all round-2 phases total {total_wall/60:.1f} min. Peak allocated memory is {training["peak_memory_mib"]/1024:.2f} GiB.'),
        ('text','Training exits with code 0 and step_limit. Independent audit checks 10000 recorded optimizer updates, finite losses/gradients, unchanged sensing factors, selected-step consistency and recomputed measurement hashes. Selected best.pth and full-state last.pth are reopened and hashed. Earlier 3+1 and 6-to-7 checks match uninterrupted updates bitwise for parameters, optimizer and RNG. The complete recovery state is distinct from the inference-only selected weights.'),
        ('heading','6. Web-Test Error Changes'),('text','Table 2 gives clean 10% MSE reductions for the 10000-step checkpoint. Official and 100-step references test overall recipe changes; the 5000-budget reference isolates continuation under the same round-2 update rule, while validation still selects weights.'),('table',clean),('text',result_sentence),
        ('text','Earlier fresh DIV2K32 showed only 1.84% clean 10% MSE reduction for the 5000-budget model versus the 100-step pilot. That independent result remains in its original report. New web photographs cannot retrospectively turn that result into a successful 5% claim.'),
    ]),dict(figures=[('mse_noise.png','Figure 3. Every clean sampling ratio and all 10% noise levels, relative to the 100-step HQS reference.')],table_widths=[125,63,63],blocks=[
        ('heading','6.1. Noise and Location Results'),('text','Table 3 gives 10000-step MSE reductions versus the 100-step pilot for every location. Campus sites have nine photographs each; scenic sites have three. Noise .05 averages all three fixed seeds, not a favorable realization.'),('table',locs),
        ('text','At 10% sampling, subset reductions under clean/.01/.05 noise are '+ '; '.join(f'{label}: '+ '/'.join(f'{value(d,s)["mse_reduction_percent"]:+.2f}%' for s in [0.,.01,.05]) for d,label in zip(DATASETS,['campus','scenic']))+'. Strong-noise gains and clean gains are separate outcomes.'),
        ('heading','6.2. Complete Evidence'),('text',f'Full inference saves {audit["rows"]} main rows, {audit["stages"]} stage rows and {audit["paired_conditions"]} matched conditions. Independent verification recalculates all {audit["float_arrays_recomputed"]} retained floating-point reconstructions and regenerates each measurement hash. Data, checkpoint, source and matrix hashes match. Every method shares each measurement. All location and image deltas remain in CSV.'),
        ('text',f'Among 330 conditions, the 10000-step state increases MSE in {negative["hqs"]} versus official HQS, {negative["hqs_round1"]} versus the pilot and {negative["hqs_5000"]} versus the 5000-budget model. Negative-case CSV retains every affected image, ratio, noise seed and reference. We do not select only improved photographs for the main tables.'),
    ]),dict(figures=[('worst_case.png',f'Figure 4. Smallest first-seed reduction versus the 5000-budget model: {name}, sampling {cr:g}, noise {sigma:g}, {float(worst["mse_reduction_percent"]):+.2f}% MSE. Absolute-error scale 0 to 0.3. Diagnostic after test.')],blocks=[
        ('heading','7. Failure Analysis and Limits'),('text',f'This clean 1% Xuanwu Lake view contains small boats, building outlines and water ripples. Both states blur these details; continued training raises overall MSE by {-float(worst["mse_reduction_percent"]):.2f}%. A post-test diagnostic partitions the image into sky rows 0-105, skyline 106-161 and water 162-255. Skyline MSE rises {regions["skyline"]["mse_increase_percent"]:.2f}% (0.001351 to 0.001541), compared with {regions["sky"]["mse_increase_percent"]:.2f}% in sky and {regions["water"]["mse_increase_percent"]:.2f}% in water. The error map supports localized boundary/texture mismatch. Smoothing unsupported high frequencies is a plausible learned-prior effect, but its training cause is not isolated by an ablation. These regions were chosen after testing for diagnosis, not model selection.'),
        ('text',credit),('text','One training seed and one bounded recipe do not establish universal improvement. Public web photographs may overlap unknown official pretraining exposure. Related views and photographers limit independence. Cropped grayscale simulation omits detector calibration, physical masks, photon statistics and full color acquisition. Independent HEVC frames do not demonstrate a temporal video model.'),
        ('heading','8. Reproducibility and Course Status'),('text','runs/nanjing_web30_20261009 contains locked manifests, attribution, protocol, smoke and comparison. CSV, float arrays, PNGs, versions and hashes are retained. runs/lab244_10000_20261009/finetune retains selected and recovery weights. Logs preserve a repaired aggregation-save failure as well as inference/verification completion. Exact commands are in coursework/NANJING_WEB30.md. Linux measurement hashes are audited; other CPU matrix-product implementations can differ at float32 roundoff. Earlier reports remain preserved.'),
        ('text','The six-page technical report, four-method experiments, failure analysis, three-page literature review and 12-slide presentation meet the corresponding artifact requirements. Team-curated web images do not satisfy optional self-capture. Timed rehearsal, student understanding, course registration and course submission remain human course activities. No additional training is launched from these test outcomes.'),
        ('heading','AI Disclosure'),('text','Codex assisted with code, remote execution, analysis, English writing and slides. Students should review and understand the work. Architecture and official weights belong to the cited authors; image authors retain attribution and licenses.'),
        ('heading','References'),('text','[1] Duarte et al. Single-Pixel Imaging via Compressive Sampling. IEEE SPM, 2008. doi:10.1109/MSP.2007.914730.'),('text','[2] Wang et al. Proximal Algorithm Unrolling. CVPR 2025. arXiv:2505.23180; github.com/pwangcs/ProxUnroll.'),('text','[3] Beck and Teboulle. FISTA. SIAM J. Imaging Sciences, 2009. doi:10.1137/080716542.'),('text','[4] Agustsson and Timofte. DIV2K / NTIRE. CVPR Workshops 2017; data.vision.ee.ethz.ch/cvl/DIV2K/.'),
    ])]
    report.draw_report('technical_report_nanjing_20261009.pdf','Single-Pixel Reconstruction:<br/>Controlled Fine-Tuning and Nanjing Web-Scene Tests',pages)
    md=ROOT/'output/pdf/technical_report_nanjing_20261009.md'
    md.write_text(md.read_text().replace('../../coursework/figures/','../../runs/nanjing_web30_20261009/figures/'))
    facts=dict(analysis=analysis, baseline=summary, training=training, training_audit=train_audit,
        total_training_wall_seconds=total_wall, validation_curve=dict(steps=steps, **curves),
        worst_case=worst, display_failure_stats=display_stats, float_failure_regions=regions, controls=controls, clean_table=clean,
        noise_table=noisy, location_table=locs, conclusion=result_sentence, verification=audit)
    (BASE/'report_facts.json').write_text(json.dumps(facts,indent=2)+'\n')


if __name__ == '__main__':main()
