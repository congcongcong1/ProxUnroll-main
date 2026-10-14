"""Continuous two-column report, preserving the frozen experimental text."""
import csv
import argparse
import copy
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, FrameSplitter,
                               Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, CondPageBreak)
from pypdf import PdfReader

from coursework.run_experiments import ROOT, sha

BASE = ROOT / 'runs/report_flow_20261014'
EXPERIMENT = ROOT / 'runs/hqs_ablation_20261013'
SOURCE = ROOT / 'output/pdf/technical_report_ablation_20261013.md'
OUTPUT = ROOT / 'output/pdf/technical_report_flow_20261014.pdf'
WIDTH, HEIGHT, MARGIN, GAP = 612, 792, 42, 22
COL = (WIDTH - 2 * MARGIN - GAP) / 2


class FigureBlock(KeepTogether):
    def __init__(self, image, caption):
        super().__init__([image, caption])
        self.required_height=image.drawHeight+caption.wrap(COL,10000)[1]+caption.getSpaceAfter()
        self.deferrals=0


class FlowDocument(BaseDocTemplate):
    def __init__(self, *args, balance_page=None, balance_height=707, **kwargs):
        super().__init__(*args, **kwargs)
        self.balance_page=balance_page
        self.balance_height=balance_height
        self.used={}

    def beforePage(self):
        if self.page==self.balance_page:
            self.pageTemplate.frames=[Frame(x,750-self.balance_height,COL,self.balance_height,id=name,
                leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
                for x,name in [(MARGIN,'left'),(MARGIN+COL+GAP,'right')]]

    def afterFlowable(self, flowable):
        if self.frame is not None and self.frame.id in ['left','right']:
            self.used.setdefault(self.page,{})[self.frame.id]=self.frame._y2-self.frame._y

    def handle_flowable(self, flowables):
        if flowables and isinstance(flowables[0],FigureBlock) and self.frame is not None:
            figure=flowables[0]
            if (not self.frame._atTop and self.frame._y-self.frame._y1p<figure.required_height
                    and figure.deferrals<2):
                # Float past at most two prose blocks, with their intervening
                # heading/table. Stop before another figure; preserve figure,
                # prose and table order, and keep each caption with its image.
                for index,next_flow in enumerate(flowables[1:9],1):
                    if isinstance(next_flow,FigureBlock):break
                    if isinstance(next_flow,Paragraph) and next_flow.style.name=='body':
                        figure.deferrals+=1
                        flowables.pop(0);flowables.insert(index,figure)
                        break
        super().handle_flowable(flowables)


def figures():
    figdir = BASE / 'figures'
    figdir.mkdir(parents=True, exist_ok=True)
    facts = json.loads((EXPERIMENT / 'report_facts.json').read_text())
    primary = facts['phases']['primary']['analysis']
    regression = facts['phases']['regression']['analysis']
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 8.5, 'legend.fontsize': 7})

    def pair(data, sigma, dataset='DIV2K_holdout_10000'):
        return next(r for r in data['paired_summary'] if r['dataset'] == dataset and r['cr'] == .1
                    and r['sigma'] == sigma and r['reference'] == 'hqs_10000' and r['candidate'] == 'hqs_12000')

    validation = list(csv.DictReader((EXPERIMENT / 'continuation/validation.csv').open()))
    steps = sorted({int(r['step']) for r in validation})
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 2.65), layout='constrained')
    for label, levels, color in [('Clean',[0.],'#147D78'),('Mixed',[0.,.01,.05],'#D78924'),('Noise .05',[.05],'#B74760')]:
        ax[0].plot(steps, [np.mean([float(r['psnr']) for r in validation if int(r['step']) == s and float(r['sigma']) in levels]) for s in steps], label=label, color=color)
    ax[0].axvline(10000, color='gray', linestyle='--', linewidth=.7)
    ax[0].set(xlabel='Total round-2 update', ylabel='Validation\nPSNR (dB)')
    ax[0].legend(loc='center right')
    ax[1].bar(['Clean','.01 noise','.05 noise'], [pair(primary,s)['mse_reduction_percent'] for s in [0.,.01,.05]], color=['#147D78','#D78924','#B74760'])
    ax[1].set(ylabel='MSE reduction\nvs 10000 (%)')
    fig.savefig(figdir / 'continuation.png', dpi=250); plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(3.5, 3.05), layout='constrained')
    for a, comparison, title in zip(ax,['noise_effect_at_weighted','stage_effect_at_noise_on'],['Training noise, weighted loss','Stage loss, noisy training']):
        for i,seed in enumerate([20261013,20261014]):
            vals=[next(r['mse_reduction_percent'] for r in primary['paired_summary'] if r['comparison']==comparison and r['training_seed']==seed and r['checkpoint_view']=='last' and r['cr']==.1 and r['sigma']==s) for s in [0.,.01,.05]]
            a.bar(np.arange(3)+(i-.5)*.34,vals,width=.34,label=str(seed),color=['#147D78','#D78924'][i])
        a.axhline(0,color='gray',linewidth=.7); a.set_xticks(range(3),['Clean','.01 noise','.05 noise'])
        a.set(title=title,ylabel='MSE reduction (%)'); a.legend(loc='best')
    fig.savefig(figdir/'factorial.png',dpi=250);plt.close(fig)
    datasets=[('Kodak','Kodak'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('DIV2K_fresh','DIV2K32'),('NJU_campus_web','Campus'),('Nanjing_scenic_web','Scenic')]
    fig,ax=plt.subplots(figsize=(3.5,2.2),layout='constrained')
    for i,s in enumerate([0.,.05]):
        ax.bar(np.arange(6)+(i-.5)*.34,[pair(regression,s,d)['mse_reduction_percent'] for d,_ in datasets],width=.34,label='Clean' if s==0 else 'Noise .05',color=['#147D78','#D78924'][i])
    ax.set_xticks(range(6),[label for _,label in datasets],rotation=25,ha='right');ax.axhline(0,color='gray',linewidth=.7)
    ax.set(ylabel='MSE reduction vs 10000 (%)');ax.legend()
    fig.savefig(figdir/'regression.png',dpi=250);plt.close(fig)
    gt=np.asarray(PILImage.open(EXPERIMENT/'illustration/reference.png'))/255
    old,new=[np.clip(np.load(EXPERIMENT/f'illustration/{name}.npy'),0,1) for name in ['old','new']]
    fig,axes=plt.subplots(2,2,figsize=(3.5,3.15),layout='constrained')
    for a,img,title in zip(axes.flat,[gt,old,new,np.abs(new-gt)],['Reference','10000-step HQS','11500-step HQS','Absolute error']):
        a.imshow(img,cmap='magma' if title=='Absolute error' else 'gray',vmin=0,vmax=.3 if title=='Absolute error' else 1)
        a.set_title(title);a.axis('off')
    fig.savefig(figdir/'failure.png',dpi=250);plt.close(fig)
    return figdir


def footer(canvas, doc):
    canvas.setFont('Times-Italic',8)
    canvas.drawString(MARGIN,767,'Computational Imaging Course Project | Topic 7')
    canvas.drawRightString(570,767,'Reproducible software study')
    canvas.setStrokeColor(colors.HexColor('#888888'));canvas.line(MARGIN,760,570,760)
    canvas.setFont('Times-Roman',9);canvas.drawCentredString(306,27,str(doc.page))


def extended_blocks():
    evaluation = ROOT / 'runs/hqs_20000_eval_20261014'
    assert (evaluation / 'COMPLETE').exists()
    assert int((evaluation / 'logs/finalize.exit').read_text()) == 0
    facts = json.loads((evaluation / 'results.json').read_text())
    def pair(phase, dataset, sigma, reference, candidate='hqs_20000'):
        return next(r for r in facts['phases'][phase]['paired_summary'] if r['dataset']==dataset
                    and r['cr']==.1 and r['sigma']==sigma and r['candidate']==candidate and r['reference']==reference)
    header='| Test noise | vs 10000 | vs 11500 | PSNR / SSIM delta |\n|---|---:|---:|---:|'
    rows=[]
    for sigma,label in [(0.,'Clean'),(.01,'Noise .01'),(.05,'Noise .05')]:
        old=pair('primary','DIV2K_holdout_10000',sigma,'hqs_10000')
        recent=pair('primary','DIV2K_holdout_10000',sigma,'hqs_12000')
        rows.append(f'| {label} | {old["mse_reduction_percent"]:+.2f}% | {recent["mse_reduction_percent"]:+.2f}% | {recent["delta_psnr"]:+.3f} / {recent["delta_ssim"]:+.4f} |')
    selected=pair('primary','DIV2K_holdout_10000',0.,'hqs_12000')
    endpoint=pair('primary','DIV2K_holdout_10000',0.,'hqs_12000','last_20000')
    first=[
        '## 5.2. Extension to 20000 Updates',
        'A subsequent separately authorized run restores the complete 12000-update state and adds 8000 updates with the same Adam, scheduler, random trajectory, learning rate, fixed factors and loss. It completes in 270.3 minutes at 10.84 GiB peak allocation. The unchanged validation rule selects step 19500; the 20000 endpoint is retained separately. No test result changes the recipe or selects the checkpoint.',
        header+'\n'+'\n'.join(rows),
        f'Table 2 reports selected-19500 MSE reductions and PSNR/SSIM deltas versus the prior validation-selected step 11500 on the same 52 originals. These images were already evaluated in the preceding experiment, so this is a continuation regression check, not a newly unseen test. The selected state has {selected["mse_reduction_percent"]:+.3f}% clean MSE reduction versus step 11500; the fixed-20000 endpoint has {endpoint["mse_reduction_percent"]:+.3f}%. There are {facts["phases"]["primary"]["negative_selected_vs_12000"]}/572 conditions with higher MSE for the selected state.',
    ]
    second=['## 7.1. Extended Checkpoint Regression',
            '| Dataset | Clean vs 11500 | Noise .05 vs 11500 |\n|---|---:|---:|']
    for dataset,label in [('Kodak','Kodak24'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('DIV2K_fresh','DIV2K32'),('NJU_campus_web','Campus18'),('Nanjing_scenic_web','Scenic12')]:
        values=[pair('regression',dataset,s,'hqs_12000')['mse_reduction_percent'] for s in [0.,.05]]
        second[-1]+=f'\n| {label} | {values[0]:+.2f}% | {values[1]:+.2f}% |'
    second += ['Table 5 reports the additional selected-checkpoint regression by dataset. The two new states save 3564 final floating reconstructions and 21384 stage metrics on all 162 originals. All new final metrics are independently recomputed. The three reference states reuse the previously verified, frozen metric files; image, source, factor and measurement hashes are matched for every condition. A fresh five-state smoke test confirms reference equivalence. All individual negatives and per-video scores are retained in runs/hqs_20000_eval_20261014; noise repetitions average within an original and videos receive equal weights.']
    return first,second


def build(include_20000=False):
    BASE.mkdir(parents=True,exist_ok=True)
    figdir=figures()
    body=ParagraphStyle('body',fontName='Times-Roman',fontSize=10.2,leading=12.5,alignment=4,spaceAfter=6)
    heading=ParagraphStyle('heading',parent=body,fontName='Times-Bold',fontSize=11.8,leading=14.2,alignment=0,spaceBefore=7,spaceAfter=5)
    table_heading=ParagraphStyle('table_heading',parent=heading,keepWithNext=True)
    small=ParagraphStyle('caption',fontName='Times-Roman',fontSize=8.8,leading=10.5,spaceAfter=7)
    title_style=ParagraphStyle('title',fontName='Times-Bold',fontSize=17,leading=20,alignment=1,spaceAfter=9)
    author=ParagraphStyle('author',fontName='Times-Roman',fontSize=9.5,leading=12,alignment=1,spaceAfter=9)
    doc=FlowDocument(str(OUTPUT),pagesize=(WIDTH,HEIGHT),leftMargin=MARGIN,rightMargin=MARGIN,
                        topMargin=44,bottomMargin=43,title='Single-Pixel Reconstruction: Controlled HQS Fine-Tuning and Training Ablations',
                        author='Zicong Luo; Shuhao Yang; Yubowen Li')
    frames=[Frame(MARGIN,43,COL,707,id='left',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0),
            Frame(MARGIN+COL+GAP,43,COL,707,id='right',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)]
    doc.addPageTemplates([PageTemplate(id='title',frames=[Frame(MARGIN,43,528,707,id='title',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer),
                          PageTemplate(id='columns',frames=frames,onPage=footer)])
    blocks=SOURCE.read_text().split('\n\n')
    blocks=[block.replace('Left: noise augmentation with weighted supervision. Right: stage supervision with noisy training.',
                          'Top: noise augmentation with weighted supervision. Bottom: stage supervision with noisy training.')
            for block in blocks]
    if include_20000:
        first,second=extended_blocks()
        blocks=[block.replace('Table 3','Table 4').replace('Table 2','Table 3') for block in blocks]
        abstract=blocks.index('## Abstract')+1
        blocks[abstract]+=' A further unchanged-recipe 20000-update extension selects step 19500 by validation. On the same previously observed inputs, clean/strong-noise MSE rises by 0.83%/0.41% versus step 11500, while SSIM slightly improves. Selected and endpoint states are retained without test-based checkpoint selection.'
        failure_index=next(i for i,b in enumerate(blocks) if b.startswith('![') and b.endswith('/failure.png)'))
        blocks[failure_index:failure_index]=second
        factorial_index=next(i for i,b in enumerate(blocks) if b.startswith('![') and b.endswith('/factorial.png)'))
        blocks[factorial_index:factorial_index]=first
    story=[Paragraph(blocks[0].removeprefix('# '),title_style),Paragraph(blocks[1],author),
           FrameSplitter('columns',['left','right'],gap=8,required=140)]
    pending_fig=None
    texts=[];table_count=figure_count=0
    for block_index,block in enumerate(blocks[2:]):
        block=block.strip()
        if not block:continue
        if block.startswith('!['):
            match=re.fullmatch(r'!\[(.*)\]\((.*)\)',block)
            caption,path=match.groups();image_path=figdir/Path(path).name
            iw,ih=PILImage.open(image_path).size
            pending_fig=FigureBlock(Image(str(image_path),width=COL,height=COL*ih/iw),Paragraph(caption,small))
            figure_count+=1
            continue
        if block.startswith('## '):
            # First attach the heading to actual prose/table. A tall figure
            # immediately after a heading can push away its content and leave
            # an orphan heading at the bottom of the previous column.
            next_block=blocks[block_index+3] if block_index+3<len(blocks) else ''
            style=table_heading if next_block.lstrip().startswith('|') else heading
            story.extend([CondPageBreak(60),Paragraph(block[3:],style)]);texts.append(block[3:])
        elif block.startswith('|'):
            rows=[[c.strip() for c in line.strip('|').split('|')] for line in block.splitlines()]
            rows=[r for r in rows if not all(re.fullmatch(r'[-: ]+',c) for c in r)]
            if rows[0][0]=='Test noise':widths=[48,48,48,107]
            elif table_count==0: widths=[80,32,86,53]
            elif table_count==1:widths=[55,62,62,72]
            else:widths=[105,73,73]
            t=Table([[Paragraph(c,small) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=0)
            t.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.6,colors.black),('LINEBELOW',(0,0),(-1,0),.4,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#F1F3F5')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
            story.extend([t,Spacer(1,7)]);table_count+=1
            if pending_fig is not None:
                story.append(pending_fig);pending_fig=None
        else:
            story.append(Paragraph(block,body));texts.append(block)
            if pending_fig is not None:
                story.append(pending_fig);pending_fig=None
    assert figure_count==4 and table_count==(6 if include_20000 else 4) and pending_fig is None
    doc.build(copy.deepcopy(story))
    pages=len(PdfReader(OUTPUT).pages)
    # Balance only the final page, after observing the natural page count.
    # Earlier pages still fill normally; no section is assigned to a page.
    unbalanced=OUTPUT.read_bytes()
    usage=doc.used.get(pages,{})
    last_height=min(707,max(280,sum(usage.values())/2+28))
    balanced=False
    if last_height<707:
        balanced_doc=FlowDocument(str(OUTPUT),pagesize=(WIDTH,HEIGHT),leftMargin=MARGIN,rightMargin=MARGIN,
            topMargin=44,bottomMargin=43,title=doc.title,author=doc.author,
            balance_page=pages,balance_height=last_height)
        balanced_doc.addPageTemplates([PageTemplate(id='title',frames=[Frame(MARGIN,43,528,707,id='title',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer),
            PageTemplate(id='columns',frames=[Frame(x,43,COL,707,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
                for x,name in [(MARGIN,'left'),(MARGIN+COL+GAP,'right')]],onPage=footer)])
        balanced_doc.build(copy.deepcopy(story))
        balanced_pages=len(PdfReader(OUTPUT).pages)
        new_usage=balanced_doc.used.get(pages,{})
        if balanced_pages==pages and len(new_usage)==2 and abs(new_usage['left']-new_usage['right'])<abs(usage.get('left',0)-usage.get('right',0)):
            balanced=True;doc=balanced_doc
        else:OUTPUT.write_bytes(unbalanced)
    assert 4<=pages<=6, f'Expected 4-6 pages; natural continuous layout yielded {pages}'
    source_copy=OUTPUT.with_suffix('.md');source_copy.write_text('\n\n'.join(blocks).replace('../../runs/hqs_ablation_20261013/figures/','../../runs/report_flow_20261014/figures/'))
    write=dict(source_markdown_sha256=sha(SOURCE),preserved_original_report_sha256=sha(ROOT/'output/pdf/technical_report_ablation_20261013.pdf'),
               report_sha256=sha(OUTPUT),pages=pages,figures=figure_count,tables=table_count,paragraphs=len(texts),
               pagination='continuous two-column flow; no per-section PageBreak or fixed page content; figures stay with captions',
               body_font_pt=body.fontSize,leading_pt=body.leading,includes_20000_evaluation=include_20000,
               last_page_balanced=balanced,column_used_height_pt=doc.used)
    if include_20000:
        write['extended_results_sha256']=sha(ROOT/'runs/hqs_20000_eval_20261014/results.json')
    (BASE/'layout_receipt.json').write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps(write))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-20000',action='store_true')
    build(parser.parse_args().include_20000)
