"""Build a results-led report using the final training budget and factorial study."""
import copy
import csv
import hashlib
import json
import re
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, FrameSplitter, Image, PageTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from coursework.build_flow_report import FlowDocument, FigureBlock, COL, MARGIN, GAP, footer

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / 'runs/hqs_20000_eval_20261014'
ABL = ROOT / 'runs/hqs_ablation_20261013/evaluation_primary'
OUT = ROOT / 'output/pdf'
BASE = ROOT / 'runs/report_focused'
FIG = OUT / 'technical_report_figures'
PDF = OUT / 'technical_report_20000_ablation.pdf'
MD = PDF.with_suffix('.md')
SEEDS = [20261013, 20261014]
DATASET = 'DIV2K_holdout_10000'
INPUTS = []

class ReportDocument(FlowDocument):
    def beforePage(self):
        pass

    def handle_pageBegin(self):
        if self.page + 1 == self.balance_page:
            self.pageTemplate.frames = [Frame(x,750-self.balance_height,COL,self.balance_height,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for x,name in [(MARGIN,'left'),(MARGIN+COL+GAP,'right')]]
        super().handle_pageBegin()

    def handle_flowable(self, flowables):
        # Keep each figure in reading order; never float it across a heading.
        BaseDocTemplate.handle_flowable(self, flowables)

def read(path):
    INPUTS.append(path)
    return list(csv.DictReader(path.open()))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {ph: read(EVAL / f'evaluation_{ph}/all_model_summary.csv') for ph in ['primary','regression']}
paired = read(ABL / 'paired_summary.csv')
abl_summary = read(ABL / 'all_model_summary.csv')
factors = read(ABL / 'factorial_effects.csv')
deltas = read(EVAL / 'evaluation_primary/paired_image_deltas.csv')


def score(method, cr=.1, sigma=0., dataset=DATASET, phase='primary'):
    return next(r for r in summary[phase] if r['method']==method and float(r['cr'])==cr and float(r['sigma'])==sigma and r['dataset']==dataset)

def reduction(new, old):
    return 100*(1-float(new['mse'])/float(old['mse']))

def effect(comparison, seed, sigma):
    return float(next(r['mse_reduction_percent'] for r in paired if r['comparison']==comparison and int(r['training_seed'])==seed and r['checkpoint_view']=='last' and float(r['cr'])==.1 and float(r['sigma'])==sigma))


def make_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,1,figsize=(3.5,3.4),layout='constrained')
    for method,label,color in [('hqs','Official HQS','#667085'),('hqs_20000','20k-budget HQS','#147D78')]:
        axes[0].plot([1,4,10,25,50],[float(score(method,cr=r)['psnr']) for r in [.01,.04,.1,.25,.5]],'o-',label=label,color=color,markersize=3)
    axes[0].set(xscale='log',xticks=[1,4,10,25,50],xticklabels=['1','4','10','25','50'],xlabel='Nominal sampling rate (%)',ylabel='Mean PSNR (dB)')
    axes[0].legend()
    vals=[reduction(score('hqs_20000',sigma=s),score('hqs',sigma=s)) for s in [0,.01,.05]]
    axes[1].bar(['Clean','Noise 0.01','Noise 0.05'],vals,color=['#147D78','#529D97','#D78924'])
    axes[1].set(ylabel='MSE reduction (%)',ylim=(0,14),xlabel='Measurement noise at 10% sampling')
    for i,v in enumerate(vals):axes[1].text(i,v+.35,f'{v:.2f}',ha='center',fontsize=8)
    fig.savefig(FIG/'final_performance.png',dpi=260);plt.close(fig)
    fig, axes=plt.subplots(2,1,figsize=(3.5,3.3),layout='constrained')
    for ax,comp,title in zip(axes,['noise_effect_at_weighted','stage_effect_at_noise_on'],['Adding training noise (weighted loss)','Adding stage supervision (noisy training)']):
        for i,seed in enumerate(SEEDS):
            ax.bar(np.arange(3)+(i-.5)*.34,[effect(comp,seed,s) for s in [0,.01,.05]],width=.34,color=['#147D78','#D78924'][i],label=f'Seed {"AB"[i]}')
        ax.axhline(0,color='#666666',lw=.7)
        ax.set(xticks=range(3),xticklabels=['Clean','Noise 0.01','Noise 0.05'],ylabel='MSE reduction (%)',title=title)
        ax.legend(loc='upper left' if comp=='noise_effect_at_weighted' else 'upper right')
    fig.savefig(FIG/'factorial_effects.png',dpi=260);plt.close(fig)
    rows=[r for r in deltas if r['candidate']=='hqs_20000' and r['reference']=='hqs']
    vals=np.sort([float(r['mse_reduction_percent']) for r in rows])
    fig, ax=plt.subplots(figsize=(3.5,2.25),layout='constrained')
    ax.plot(np.arange(1,len(vals)+1),vals,color='#147D78',lw=1.3)
    ax.fill_between(np.arange(1,len(vals)+1),vals,0,where=vals<0,color='#BC4B51',alpha=.6)
    ax.axhline(0,color='#555555',lw=.7)
    ax.set(xlabel='Image / measurement condition (sorted)',ylabel='MSE reduction vs official (%)')
    ax.text(.03,.93,f'{sum(v<0 for v in vals)} of {len(vals)} conditions regress',transform=ax.transAxes,va='top',fontsize=8)
    fig.savefig(FIG/'condition_spread.png',dpi=260);plt.close(fig)

blocks=[]
checks=[]
def p(s):blocks.append(('p',s))
def h(s):blocks.append(('h',s))
def eq(s):blocks.append(('eq',s))
def table(caption,headers,rows,widths):
    blocks.append(('table',(caption,headers,rows,widths)))
    checks.extend(str(c) for row in rows for c in row if re.fullmatch(r'[+-]?\d+\.\d+%?',str(c)))
def figure(name,caption):blocks.append(('fig',(name,caption)))


def write_content():
    h('Abstract')
    p('This report evaluates fixed-operator adaptation of the HQS variant of ProxUnroll for simulated single-pixel imaging. A 20,000-update fine-tuning budget is assessed against the official pretrained model, while a separate two-factor experiment tests measurement-noise augmentation and intermediate-stage supervision. On 52 DIV2K test images at 10% sampling, the validation-selected model reduces mean squared error by 3.32% for clean measurements and 11.98% for stronger measurement noise. The corresponding PSNR gains are 0.205 and 0.780 dB. A matched four-arm ablation, with two training seeds and 800 updates per arm, shows that noise augmentation improves noisy reconstruction but slightly compromises clean reconstruction; intermediate supervision produces much smaller, condition-dependent changes. These results support targeted noise robustness rather than uniformly improved reconstruction. The study uses grayscale simulation and makes no claim of physical-camera validation or a new network architecture.')
    h('1. Introduction')
    p('Single-pixel imaging reconstructs a spatial image from integrated responses to structured illumination or detection patterns [1]. Using fewer measurements than pixels reduces acquisition requirements but makes reconstruction underdetermined. Measurement consistency alone cannot determine the image, so the reconstruction algorithm must also use prior knowledge about natural images. This project studies that balance in a controlled software setting.')
    p('ProxUnroll combines a measurement-aware iterative update with a learned image restorer [2]. Starting from its official HQS weights, we investigate whether adaptation with fixed sensing operators improves reconstruction across sampling rates and noise levels. Keeping the operator fixed makes the comparison about the reconstruction model, without conflating it with a change in the measurement process.')
    p('The experiments address two questions. First, how does the model produced by a 20,000-update budget compare with the official model under identical measurements? Second, what are the separate effects of training noise and intermediate supervision during matched short-budget adaptation? The first comparison evaluates the final model; the second tests specific training choices. Their different initial states and budgets prevent treating the ablation as a decomposition of the final model\'s entire gain.')
    h('2. Reconstruction Model')
    h('2.1. Separable measurements')
    p('For a normalized grayscale image X with n rows and columns, the simulated measurement is')
    eq('Y = H X W<super>T</super> + E.  (1)')
    p('H and W are row-selected sensing factors, and E is measurement noise. For nominal sampling ratio r, each factor uses m = ceil(n sqrt(r)) rows, so the realized ratio is m<super>2</super>/n<super>2</super>. At n = 256 and nominal 10% sampling, m = 81 and the realized ratio is approximately 10.011%. Reporting nominal ratios is convenient, but the same actual row counts are used for both models.')
    p('Noise entries are independent zero-mean Gaussian samples with standard deviation sigma times the RMS of the clean measurement. Thus sigma describes noise relative to measurement energy, not pixel-domain noise. The reconstruction receives Y and the sensing factors; the ground-truth image is used only to generate simulated data, train the model, and compute evaluation metrics.')
    h('2.2. Six-stage HQS reconstruction')
    p('The implementation initializes X<sub>0</sub> = H<super>T</super>YW and performs six measurement-correction and restoration stages:')
    eq('Z<sub>k</sub> = X<sub>k</sub> + rho<sub>k</sub> H<super>T</super>(Y - H X<sub>k</sub> W<super>T</super>)W,<br/>X<sub>k+1</sub> = R<sub>theta</sub>(Z<sub>k</sub>; memory<sub>k</sub>).  (2)')
    p('The correction moves the estimate toward measurement consistency. The learned restorer combines convolution and window attention, passing feature memory between reconstruction stages within the same image. These are finite learned updates; their presence alone does not establish convergence or monotonic improvement of every intermediate reconstruction. All video frames in this study are processed independently.')
    h('2.3. Fine-tuning objective')
    p('All sensing factors remain frozen. The restorer and trainable reconstruction parameters are adapted using Adam, learning rate 10<super>-5</super>, batch size one, and single-precision arithmetic. Training samples a nominal ratio from [0.01, 0.04, 0.10, 0.10, 0.10, 0.25, 0.25, 0.50] and a relative noise level from [0, 0, 0.01, 0.05]. The repeated entries specify their sampling probabilities.')
    eq('L = 0.01 sum<sub>k=1,...,5</sub> RMSE(X<sub>k</sub>, X)<br/>+ 0.95 RMSE(X<sub>6</sub>, X).  (3)')
    p('This direct-ground-truth objective emphasizes the final reconstruction while adding a small loss at each intermediate stage. It is not the original paper\'s proximal-trajectory training objective. The project therefore evaluates an adaptation recipe for the released model, rather than a full reproduction of the original training method.')
    h('3. Experimental Protocol')
    h('3.1. Data and preprocessing')
    p('Training uses the 800 official DIV2K training images [3], with 16 separate images for validation and checkpoint selection. Training uses random square crops, area resizing, and horizontal and vertical flips. Split membership is defined by original image before augmentation, and source hashes are disjoint between the training, validation, and primary test splits.')
    p('The primary test set contains 52 DIV2K images. Additional evaluation uses 110 images: Kodak24, 32 DIV2K images, 18 campus photographs, 12 scenic photographs, and three spaced frames from each of eight HEVC sequences. The photographs are curated from external authors rather than captured by the team. Still images use the largest centered square, the OpenCV Y luminance channel, area resizing to 256 x 256, and normalization by 255. HEVC inputs use stored 8-bit Y values without an assumed studio-range conversion.')
    p('These test sets had already been examined during project development. Their results are comparisons on fixed inputs, not a newly blinded assessment. They do not select the final checkpoint or change the training recipe. Disjoint project splits also cannot establish whether public images appeared in the official model\'s upstream training.')
    h('3.2. Model selection and paired evaluation')
    p('The fine-tuning trajectory completes 20,000 optimizer updates. Validation selects the highest mean PSNR across 16 images, two sampling ratios (10% and 25%), and three noise levels (0, 0.01, and 0.05), subject to clean validation PSNR remaining within 0.03 dB of initialization. Under this fixed rule, step 19,500 is selected. We call it the 20k-budget model, and report the exact step-20,000 endpoint separately. The budget is not a claim that every additional update improves test quality.')
    p('The official model, selected model, and endpoint use identical images, sensing factors, and realized measurements. Clean evaluation covers nominal ratios of 1%, 4%, 10%, 25%, and 50%. Noisy evaluation uses 10% sampling, sigma = 0.01 or 0.05, and three matched measurement-noise realizations. This gives eleven measurement conditions per original image.')
    h('3.3. Metrics and aggregation')
    p('Predictions are clipped to [0, 1], and MSE, PSNR, and SSIM are computed before display-image quantization, with data range one and no border cropping. Noise repetitions are averaged within each image. Still images receive equal weight; video frames are averaged within each sequence and sequences receive equal weight. The relative MSE reduction is')
    eq('G = 100 [1 - mean(MSE<sub>candidate</sub>) /<br/>mean(MSE<sub>reference</sub>)].  (4)')
    p('Positive G indicates lower error. It is a ratio of aggregated errors, not an average of per-image percentage gains and not a percentage change in PSNR. Mean PSNR averages logarithmic image scores, so its ordering can differ slightly from the ordering of mean MSE. Stored floating-output metrics were independently verified, and the report tables are recomputed from per-image metric records.')
    h('4. Results of the 20,000-Update Budget')
    h('4.1. Reconstruction at 10% sampling')
    rows=[]
    for s,label in [(0.,'Clean'),(.01,'0.01'),(.05,'0.05')]:
        for m,name in [('hqs','Official'),('hqs_20000','20k budget')]:
            r=score(m,sigma=s)
            rows.append([label,name,f"{float(r['psnr']):.3f}",f"{float(r['ssim']):.4f}",'-' if m=='hqs' else f"{reduction(r,score('hqs',sigma=s)):+.2f}%"])
    table('Table 1. Primary-set reconstruction at 10% sampling. MSE reduction uses official HQS as the reference. The 20k-budget model is the validation-selected checkpoint.', ['Noise','Model','PSNR','SSIM','MSE gain'],rows,[36,61,48,49,59])
    p('Table 1 shows a modest clean-measurement gain and a larger gain under stronger noise. At sigma = 0.05, mean PSNR increases from 25.135 to 25.915 dB and SSIM from 0.7252 to 0.7838. Clean PSNR increases from 27.809 to 28.014 dB. The noise-dependent difference is central: a single percentage would conceal how strongly the benefit depends on the measurement condition.')
    p('The strong-noise result is consistent with adapting the reconstruction prior to corrupted measurements. The ablation below tests that interpretation directly for short adaptation. It does not justify assigning all final-model improvement to noise augmentation, since the full training budget and its optimization trajectory differ from those of the ablation.')
    figure('final_performance.png','Figure 1. Primary-set performance of official and 20k-budget HQS. Top: clean PSNR across sampling ratios. Bottom: mean-MSE reduction at 10% sampling. All points use matched measurements.')
    h('4.2. Sampling ratio and endpoint sensitivity')
    rows=[]
    for cr in [.01,.04,.1,.25,.5]:
        a,b=score('hqs',cr=cr),score('hqs_20000',cr=cr)
        rows.append([f'{100*cr:.0f}%',f"{float(a['psnr']):.3f}",f"{float(b['psnr']):.3f}",f'{reduction(b,a):+.2f}%'])
    table('Table 2. Clean primary-set results across nominal sampling ratios. PSNR is in dB; MSE gain is relative to official HQS.', ['Ratio','Official','20k budget','MSE gain'],rows,[38,68,77,70])
    p('The selected model improves aggregate clean reconstruction at every tested ratio, but the size of the gain is not constant. At 1% sampling, both models remain near 21 dB, indicating the difficulty of recovering image detail from extremely sparse measurements. The small gain there does not remove the information limitation. Higher sampling ratios improve reconstruction much more than fine-tuning alone.')
    endpoint=[reduction(score('last_20000',sigma=s),score('hqs',sigma=s)) for s in [0,.01,.05]]
    p(f'The exact 20,000-update endpoint gives mean-MSE reductions of {endpoint[0]:.2f}%, {endpoint[1]:.2f}%, and {endpoint[2]:.2f}% for clean, mild-noise, and stronger-noise measurements at 10% sampling. Its slightly lower aggregate MSE does not replace the validation-selected model after inspecting test results. Reporting both states makes the selection rule explicit and shows that the headline conclusion is similar for the two final states.')
    h('4.3. Additional image collections')
    rows=[]
    for dataset,label in [('Kodak','Kodak24'),('DIV2K_fresh','DIV2K32'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('NJU_campus_web','Campus18'),('Nanjing_scenic_web','Scenic12')]:
        gains=[reduction(score('hqs_20000',sigma=s,dataset=dataset,phase='regression'),score('hqs',sigma=s,dataset=dataset,phase='regression')) for s in [0,.05]]
        rows.append([label,*[f'{v:+.2f}%' for v in gains]])
    table('Table 3. MSE reduction against official HQS on the additional collections at 10% sampling. Each video sequence has equal weight within its HEVC class.', ['Collection','Clean','Noise 0.05'],rows,[99,76,78])
    p('The additional collections show the same qualitative pattern: larger relative improvements under stronger measurement noise than under clean measurements. These collection-level results complement the DIV2K primary set without being pooled into a single headline score. The collections differ in image content and sample count, and the HEVC results describe independent frame reconstruction rather than temporal video reconstruction.')
    h('5. Controlled Training Ablations')
    h('5.1. Two-factor design')
    p('A two-by-two factorial experiment varies measurement-noise augmentation and intermediate-stage supervision. Every arm starts from the same official HQS weights with a fresh Adam optimizer and runs for exactly 800 updates. Each arm is repeated with two training seeds, labeled A and B. Within a seed, the image order, crops, flips, sampling ratios, and proposed noise draws are matched. Disabling noise changes only its amplitude.')
    table('Table 4. Matched ablation design. Every row uses 800 updates for each of two training seeds. The sensing factors, optimizer, and data distribution are fixed.', ['Arm','Training noise','Supervision'],[['A','Off','Final only'],['B','Off','Weighted stages'],['C','On','Final only'],['D','On','Weighted stages']],[32,91,130])
    p('Final-only supervision gives the final-stage RMSE weight one. Weighted supervision uses Equation (3), keeping total loss weight equal to one. Comparisons use fixed-budget endpoints, not whichever checkpoint happens to score best on the test set. This isolates recipe effects at the same optimization budget. Training seeds measure optimization variability; the three measurement-noise realizations are a separate source of test variability.')
    h('5.2. Noise augmentation versus stage loss')
    rows=[]
    for comp,name in [('noise_effect_at_weighted','Noise / weighted'),('noise_effect_at_final','Noise / final'),('stage_effect_at_noise_on','Stages / noise on'),('stage_effect_at_noise_off','Stages / noise off')]:
        vals=[np.mean([effect(comp,seed,s) for seed in SEEDS]) for s in [0,.01,.05]]
        rows.append([name,*[f'{v:+.2f}%' for v in vals]])
    table('Table 5. Mean of the two seed-specific MSE reductions on primary52 at 10% sampling. The first factor in each row is added while the second is held fixed.', ['Added / fixed','Clean','0.01','0.05'],rows,[104,49,50,50])
    p('With weighted supervision, adding noise reduces strong-noise MSE by 8.45% on average, with individual-seed reductions of 8.34% and 8.56%. The clean-measurement effect is negative for both seeds, averaging -0.74%. Mild-noise performance also drops slightly. Thus the benefit is not a general improvement across all noise levels: the training distribution trades a small amount of clean accuracy for better robustness to stronger corruption.')
    p('The same direction holds under final-only supervision: noise augmentation improves the stronger-noise result by 8.55% on average while worsening the clean result by 0.91%. Agreement under both supervision choices makes the noise effect more persuasive within this particular design. Two seeds still do not establish a precise population effect or a general guarantee.')
    figure('factorial_effects.png','Figure 2. Fixed-800-update effects for both training seeds. Positive values mean lower mean MSE after adding the indicated factor. The panels use different vertical scales because stage-supervision effects are much smaller.')
    p('With noisy training, adding intermediate supervision improves clean MSE by only 0.18% on average; the two seed effects are 0.03% and 0.32%. Under stronger test noise, both seeds instead regress slightly, by 0.20% and 0.07%. With clean training, the clean-test stage effect changes sign across seeds. These values provide little evidence that the small intermediate loss is the main source of reconstruction improvement.')
    h('5.3. Main effects and interaction')
    p('Conditional percentage reductions in Table 5 use different references. To compare factorial main effects, we instead average raw-MSE benefits across the other factor and normalize by the same official-model MSE. The interaction is the noise benefit with weighted supervision minus the noise benefit with final-only supervision, using that same normalization. Positive interaction indicates a larger noise benefit with weighted supervision.')
    rows=[]
    for s,label in [(0.,'Clean'),(.01,'0.01'),(.05,'0.05')]:
        vals=[]
        for key in ['noise_main_mse_benefit','stage_main_mse_benefit','noise_stage_interaction_mse_benefit']:
            v=[float(r['official_normalized_percent_points']) for r in factors if r['checkpoint_view']=='last' and float(r['cr'])==.1 and float(r['sigma'])==s and r['effect']==key and r['training_seed'] in {str(seed) for seed in SEEDS}]
            assert len(v)==2
            vals.append(np.mean(v))
        rows.append([label,*[f'{v:+.3f}' for v in vals]])
    table('Table 6. Two-seed mean factorial effects in percentage points of official-model MSE. These are common-reference effects, not the relative percentages of Table 5.', ['Test noise','Noise main','Stage main','Interaction'],rows,[58,65,65,65])
    p('The stronger-noise main effect is much larger than either the stage main effect or the interaction. The small interaction suggests that changing stage supervision does not substantially change the noise benefit in this short-budget study. It does not demonstrate that the two factors are universally independent, nor does it predict their interaction after 20,000 updates.')
    h('6. Failure Analysis and Limitations')
    rows=[r for r in deltas if r['candidate']=='hqs_20000' and r['reference']=='hqs']
    negative=[r for r in rows if float(r['mse_reduction_percent'])<0]
    worst=min(rows,key=lambda r:float(r['mse_reduction_percent']))
    assert len(rows)==572 and len(negative)==28
    p('Aggregate gains conceal individual failures. Across the 52 primary images and eleven measurement conditions per image, the selected model has higher MSE than official HQS in 28 of 572 conditions. These conditions share images and, for noisy tests, repeated measurements; the count is descriptive and is not an estimate from 572 independent images.')
    figure('condition_spread.png','Figure 3. All 572 primary image/measurement comparisons, sorted by relative MSE reduction. The negative tail is retained; the curve describes condition-level variation rather than a confidence interval.')
    p(f'The largest relative regression occurs on DIV2K image 0843 at clean 50% sampling: MSE rises by {-float(worst["mse_reduction_percent"]):.2f}%, PSNR drops by {-float(worst["delta_psnr"]):.3f} dB, and SSIM changes by {float(worst["delta_ssim"]):.6f}. This is a measured failure despite an aggregate improvement at the same ratio. A small SSIM change also shows why structural similarity should not replace the pixel-error metric when interpreting this case.')
    p('The failure distribution supports a narrower conclusion than uniform superiority: the adapted prior benefits most measured conditions but can harm particular images. The scalar metrics alone do not identify the damaged structures or establish a unique training cause. We therefore do not infer a visual failure mechanism without corresponding image-level inspection.')
    p('Several limits constrain interpretation. First, Gaussian measurement noise and separable operators simplify physical imaging; detector response, mask error, photon statistics, and calibration are not tested. Second, 256 x 256 luminance crops do not establish performance for full-resolution color imaging. Third, the fixed test collections have been observed during development, and official pretraining exposure is unknown. Finally, two short-budget seeds provide limited evidence about optimization variability. No runtime or physical acquisition advantage is inferred from image-quality scores.')
    h('7. Conclusion')
    p('Under identical simulated measurements, the model selected from a 20,000-update budget improves average reconstruction over official HQS across the tested clean sampling ratios, with substantially larger gains under stronger measurement noise. The controlled ablation identifies noise augmentation as the dominant tested factor for short-budget noisy reconstruction, while exposing a small clean-accuracy trade-off. Intermediate-stage supervision contributes much smaller and inconsistent benefits. The resulting evidence supports noise-aware adaptation of a fixed reconstruction model, with explicit limits on per-image reliability and transfer to physical imaging.')
    h('AI Assistance Disclosure')
    p('Codex assisted with implementation, experiment execution, result analysis, and report preparation. The authors remain responsible for verifying the evidence, understanding the algorithms, and explaining the findings.')
    h('References')
    p('[1] M. F. Duarte et al. <i>Single-Pixel Imaging via Compressive Sampling.</i> IEEE Signal Processing Magazine. <link href="https://doi.org/10.1109/MSP.2007.914730" color="#147D78">Publisher article</link>.')
    p('[2] P. Wang, L. Wang, G. Qu, X. Wang, Y. Zhang, and X. Yuan. <i>Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging.</i> IEEE/CVF Conference on Computer Vision and Pattern Recognition. <link href="https://arxiv.org/abs/2505.23180" color="#147D78">Paper</link>; <link href="https://github.com/pwangcs/ProxUnroll" color="#147D78">official implementation</link>.')
    p('[3] E. Agustsson and R. Timofte. <i>NTIRE Challenge on Single Image Super-Resolution: Dataset and Study.</i> IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. <link href="https://data.vision.ee.ethz.ch/cvl/DIV2K/" color="#147D78">DIV2K dataset</link>.')


def build():
    BASE.mkdir(parents=True,exist_ok=True)
    make_figures()
    write_content()
    body=ParagraphStyle('body',fontName='Times-Roman',fontSize=10.2,leading=12.5,alignment=4,spaceAfter=6)
    heading=ParagraphStyle('heading',parent=body,fontName='Times-Bold',fontSize=11.7,leading=14.1,alignment=0,spaceBefore=7,spaceAfter=5,keepWithNext=True)
    small=ParagraphStyle('caption',fontName='Times-Roman',fontSize=8.7,leading=10.4,spaceAfter=7)
    cell=ParagraphStyle('cell',parent=small,fontSize=8.3,leading=10,spaceAfter=0)
    equation=ParagraphStyle('equation',parent=body,alignment=1,fontSize=9.7,leading=18,spaceBefore=3,spaceAfter=7)
    title='Noise-Aware Single-Pixel Reconstruction:<br/>20,000-Update HQS Adaptation and Training Ablations'
    authors='Zicong Luo (522026230043), Shuhao Yang (502026230083), Yubowen Li (502026230092)'
    story=[Paragraph(title,ParagraphStyle('title',fontName='Times-Bold',fontSize=16.5,leading=19.5,alignment=1,spaceAfter=9)),Paragraph(authors,ParagraphStyle('author',fontName='Times-Roman',fontSize=9.4,leading=12,alignment=1,spaceAfter=9)),FrameSplitter('columns',['left','right'],gap=8,required=140)]
    markdown=['# '+title.replace('<br/>',' '),authors]
    for kind,data in blocks:
        if kind=='h':story.append(Paragraph(data,heading));markdown.append('## '+data)
        elif kind in ['p','eq']:
            story.append(Paragraph(data,equation if kind=='eq' else body));markdown.append(data)
        elif kind=='table':
            caption,headers,rows,widths=data
            assert sum(widths)==COL
            content=[[Paragraph(escape(str(c)),cell) for c in r] for r in [headers,*rows]]
            t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF4F3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3.5),('BOTTOMPADDING',(0,0),(-1,-1),3.5)]))
            prefix=[]
            if isinstance(story[-1],Paragraph) and story[-1].style.name=='heading':prefix=[story.pop()]
            story.append(KeepTogether(prefix+[Paragraph(caption,small),t,Spacer(1,8)]))
            markdown.append(caption)
            markdown.append('\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]]))
        else:
            name,caption=data
            iw,ih=PILImage.open(FIG/name).size
            story.append(FigureBlock(Image(str(FIG/name),width=COL,height=COL*ih/iw),Paragraph(caption,small)))
            markdown.append(f'![{caption}](technical_report_figures/{name})')
    def document(balance_page=None,balance_height=707):
        doc=ReportDocument(str(PDF),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=44,bottomMargin=43,title=title.replace('<br/>',' '),author='Zicong Luo; Shuhao Yang; Yubowen Li',balance_page=balance_page,balance_height=balance_height)
        doc.addPageTemplates([PageTemplate(id='title',frames=[Frame(42,43,528,707,id='title',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer),PageTemplate(id='columns',frames=[Frame(x,43,COL,707,id=n,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0) for x,n in [(42,'left'),(42+COL+22,'right')]],onPage=footer)])
        return doc
    doc=document();doc.build(copy.deepcopy(story))
    pages=len(PdfReader(PDF).pages);unbalanced=PDF.read_bytes();usage=doc.used.get(pages,{})
    height=min(707,max(280,sum(usage.values())/2+30))
    if height<707:
        best_bytes=unbalanced
        best_difference=abs(usage.get('left',0)-usage.get('right',0))
        for trial_height in np.arange(height,min(707,height+110),12):
            bd=document(pages,float(trial_height));bd.build(copy.deepcopy(story))
            newuse=bd.used.get(pages,{})
            if len(PdfReader(PDF).pages)==pages and len(newuse)==2:
                difference=abs(newuse['left']-newuse['right'])
                if difference<best_difference:
                    best_difference=difference;best_bytes=PDF.read_bytes();doc=bd
                if difference<30:break
        PDF.write_bytes(best_bytes)
    # Remove automatically generated timestamps from the delivered PDF metadata.
    reader=PdfReader(PDF);writer=PdfWriter();writer.clone_document_from_reader(reader)
    writer.metadata={k:v for k,v in (reader.metadata or {}).items() if k not in ['/CreationDate','/ModDate']}
    with PDF.open('wb') as stream:writer.write(stream)
    MD.write_text('\n\n'.join(markdown)+'\n')
    text=' '.join(p.extract_text() for p in PdfReader(PDF).pages)
    for token in checks:assert token in text,token
    assert 4<=pages<=6,pages
    for pattern in [r'/workspace/',r'/Users/',r'2026[-_]?10',r'202610',r'server244',r'luozc',r'11500',r'12000',r'GPU.minutes',r'\bminutes\b']:
        assert not re.search(pattern,text,re.I),pattern
    assert len(re.findall(r'Figure [123]\.',text))==3
    assert all(f'Table {i}.' in text for i in range(1,7))
    record={'pdf_sha256':digest(PDF),'markdown_sha256':digest(MD),'pages':pages,'figures':3,'tables':6,'numeric_cells_checked':len(checks),'inputs':{str(x.relative_to(ROOT)):digest(x) for x in INPUTS},'column_used_height_pt':doc.used,'visual_review':'pending'}
    (BASE/'build_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':build()
