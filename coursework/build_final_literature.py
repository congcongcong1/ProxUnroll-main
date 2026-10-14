"""Update project connections while preserving the comparative reading source."""
import copy
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Frame, FrameSplitter, PageTemplate, Paragraph, CondPageBreak

from coursework.build_flow_report import FlowDocument

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/materials_final_20261014'
SOURCE=ROOT/'output/pdf/literature_review_layoutfix_20261010.md'
OUTPUT=ROOT/'output/pdf/literature_review_final_20261014.pdf'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def revised_blocks():
    blocks=SOURCE.read_text().split('\n\n')
    results=json.loads((ROOT/'runs/hqs_20000_eval_20261014/results.json').read_text())
    def pair(sigma):
        return next(r for r in results['phases']['primary']['paired_summary']
                    if r['dataset']=='DIV2K_holdout_10000' and r['cr']==.1 and r['sigma']==sigma
                    and r['candidate']=='hqs_20000' and r['reference']=='hqs_12000')
    for i,block in enumerate(blocks):
        if block.startswith('Our experimental response is to retain every intermediate reconstruction.'):
            blocks[i]='Our initial response retains every intermediate reconstruction and inspects the released six-stage computation. Later paired experiments compare weighted intermediate direct-ground-truth RMSE with final-only RMSE, using two training seeds and 800 updates per arm. This controls that loss factor under a fixed recipe. It does not reproduce or isolate the paper\'s ideal proximal-trajectory targets, prove feature-memory necessity, or establish asymptotic convergence.'
    start=blocks.index('## 6. Connection to the Implemented Project')+1
    end=blocks.index('## 7. Questions for the Presentation')
    blocks[start:end]=[
        'The original small-suite evaluation preserves 352 image-level metrics and 1056 stage metrics. On its six regular images at clean nominal 10% sampling, released HQS achieves 34.31 dB and 0.916 SSIM, while DCT-FISTA gives 29.10 dB and 0.694 SSIM under identical measurements. These historical software results are ours, not values copied from either paper. They cannot be compared directly with scores on a different image collection.',
        'The updated project extends this evidence to Kodak, fixed spaced HEVC raw Y frames, DIV2K, and 30 attributed Nanjing web photographs. Fine-tuning uses 800 DIV2K training originals and 16 disjoint validation originals. The 52-image project holdout was locked before the preceding continuation; the later 20000-update extension reuses it and 110 previously tested regression originals. This later comparison therefore concerns observed inputs. Official pretraining exposure remains unknown, and the web images belong to external photographers.',
        f'Full-state HQS continuation reaches 20000 updates with all six sensing factors frozen. The unchanged validation rule selects step 19500, and the 20000 endpoint is retained separately. On the same primary52 inputs at 10% sampling, selected-19500 MSE increases by {-pair(0.)["mse_reduction_percent"]:.2f}% for clean measurements and {-pair(.05)["mse_reduction_percent"]:.2f}% at relative noise .05 versus selected step 11500, while SSIM increases slightly. More updates do not give a further universal 5-10% error reduction. The extension saves 3564 new final floating reconstructions and 21384 stage metrics, with independent metric verification and matched measurement hashes.',
        'The separate two-by-two noise/stage-loss experiment holds initialization, data order, crops, sensing factors and 800-update budgets fixed within each training seed. Adding training noise under weighted loss reduces strong-noise MSE by 8.45% on average across the two seeds, while clean MSE increases by 0.74%. Intermediate supervision has much smaller, condition-dependent effects. These bounded controls support a limited recipe comparison rather than an explanation of all historical training gains. All negative cases remain in the technical report and raw tables.',
        'The current four-page technical report presents these results separately from the papers\' claims. Further work could test stronger classical priors, additional training replications and real optical calibration. We have completed software reconstruction and controlled adaptation, without claiming the official paper benchmark, physical acquisition or a temporal video model.'
    ]
    return blocks


def footer(canvas,doc):
    canvas.setFont('Times-Italic',8)
    canvas.drawString(42,767,'Computational Imaging Course Project | Topic 7')
    canvas.drawRightString(570,767,'Comparative reading | Updated 2026-10-14')
    canvas.line(42,760,570,760)
    canvas.setFont('Times-Roman',9);canvas.drawCentredString(306,27,str(doc.page))


def document(balance_page=None,balance_height=707):
    doc=FlowDocument(str(OUTPUT),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=44,bottomMargin=43,
                     title='Single-Pixel Imaging: A Comparative Reading',author='Zicong Luo; Shuhao Yang; Yubowen Li',
                     balance_page=balance_page,balance_height=balance_height)
    frames=[Frame(x,43,253,707,id=name,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
            for x,name in [(42,'left'),(317,'right')]]
    doc.addPageTemplates([
        PageTemplate(id='title',frames=[Frame(42,43,528,707,id='title',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer),
        PageTemplate(id='columns',frames=frames,onPage=footer)])
    return doc


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    blocks=revised_blocks()
    body=ParagraphStyle('body',fontName='Times-Roman',fontSize=10.6,leading=13.1,alignment=4,spaceAfter=6)
    heading=ParagraphStyle('heading',fontName='Times-Bold',fontSize=11.7,leading=14.1,spaceBefore=7,spaceAfter=5)
    title=ParagraphStyle('title',fontName='Times-Bold',fontSize=17,leading=20,alignment=1,spaceAfter=9)
    author=ParagraphStyle('author',fontName='Times-Roman',fontSize=9.5,leading=12,alignment=1,spaceAfter=9)
    story=[Paragraph(blocks[0].removeprefix('# '),title),Paragraph(blocks[1],author),FrameSplitter('columns',['left','right'],gap=8,required=140)]
    for block in blocks[2:]:
        if block.startswith('## '):story.extend([CondPageBreak(60),Paragraph(block[3:],heading)])
        elif block.strip():story.append(Paragraph(block,body))
    doc=document();doc.build(copy.deepcopy(story))
    pages=len(PdfReader(OUTPUT).pages);original=OUTPUT.read_bytes()
    usage=doc.used.get(pages,{})
    height=min(707,max(280,sum(usage.values())/2+28));balanced=False
    if height<707:
        new_doc=document(pages,height);new_doc.build(copy.deepcopy(story))
        new_usage=new_doc.used.get(pages,{})
        if len(PdfReader(OUTPUT).pages)==pages and len(new_usage)==2 and abs(new_usage['left']-new_usage['right'])<abs(usage.get('left',0)-usage.get('right',0)):
            doc=new_doc;balanced=True
        else:OUTPUT.write_bytes(original)
    assert 2<=pages<=3, f'Reading report must have 2-3 pages; got {pages}'
    OUTPUT.with_suffix('.md').write_text('\n\n'.join(blocks).rstrip()+'\n')
    record={'pages':pages,'report_sha256':sha(OUTPUT),'source_sha256':sha(SOURCE),
            'preserved_original_pdf_sha256':sha(SOURCE.with_suffix('.pdf')),
            'body_font_pt':body.fontSize,'leading_pt':body.leading,'last_page_balanced':balanced,
            'column_used_height_pt':doc.used,'results_sha256':sha(ROOT/'runs/hqs_20000_eval_20261014/results.json')}
    (BASE/'literature_layout.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':main()
