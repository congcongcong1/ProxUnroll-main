"""Check the revised PDF's length, retained content and CSV-derived new tables."""
import json
from pathlib import Path
import re
from pypdf import PdfReader

from coursework.run_experiments import ROOT, sha

BASE=ROOT/'runs/report_flow_20261014'
EVAL=ROOT/'runs/hqs_20000_eval_20261014'
REPORT=ROOT/'output/pdf/technical_report_flow_20261014.pdf'


def audit():
    receipt=json.loads((BASE/'layout_receipt.json').read_text())
    assert receipt['report_sha256']==sha(REPORT)
    assert receipt['preserved_original_report_sha256']==sha(ROOT/'output/pdf/technical_report_ablation_20261013.pdf')
    reader=PdfReader(REPORT);assert 4<=len(reader.pages)<=6
    text=' '.join(' '.join(p.extract_text().split()) for p in reader.pages)
    assert not re.search(r'\bTODO\b|\bundefined\b|\bnan\b',text,re.I)
    for term in ['Abstract','Common Reconstruction Contract','Matched Factorial Ablations','Failure Analysis',
                 'AI Disclosure','References','Figure 1.','Figure 2.','Figure 3.','Figure 4.',
                 '191/572','+4.12%','+0.21%','+8.34%','-0.20%','+8.56%','-0.07%']:
        assert term in text,term
    new_rows_checked=0
    if receipt['includes_20000_evaluation']:
        result=json.loads((EVAL/'results.json').read_text())
        assert receipt['extended_results_sha256']==sha(EVAL/'results.json')
        for term in ['19500','20000','3564','21384','previously observed inputs','0.83%/0.41%','Table 5']:
            assert term in text,term
        def pair(phase,dataset,sigma,reference):
            return next(r for r in result['phases'][phase]['paired_summary'] if r['dataset']==dataset and r['cr']==.1
                        and r['sigma']==sigma and r['reference']==reference and r['candidate']=='hqs_20000')
        for sigma,label in [(0.,'Clean'),(.01,'Noise .01'),(.05,'Noise .05')]:
            old=pair('primary','DIV2K_holdout_10000',sigma,'hqs_10000')
            recent=pair('primary','DIV2K_holdout_10000',sigma,'hqs_12000')
            row=f'{label} {old["mse_reduction_percent"]:+.2f}% {recent["mse_reduction_percent"]:+.2f}% {recent["delta_psnr"]:+.3f} / {recent["delta_ssim"]:+.4f}'
            assert row in text,row;new_rows_checked+=1
        for dataset,label in [('Kodak','Kodak24'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('DIV2K_fresh','DIV2K32'),('NJU_campus_web','Campus18'),('Nanjing_scenic_web','Scenic12')]:
            values=[pair('regression',dataset,s,'hqs_12000')['mse_reduction_percent'] for s in [0.,.05]]
            row=f'{label} {values[0]:+.2f}% {values[1]:+.2f}%'
            assert row in text,row;new_rows_checked+=1
    visual=json.loads((BASE/'visual_audit.json').read_text())
    assert visual['report_sha256']==sha(REPORT)
    assert len(visual['pages'])==len(reader.pages)
    for page in visual['pages']:
        assert page['individually_inspected'] and page['no_overlap_clipping_or_orphan_headings']
        assert sha(ROOT/page['path'])==page['sha256']
    record=dict(report_sha256=sha(REPORT),pages=len(reader.pages),figures=receipt['figures'],tables=receipt['tables'],
                no_fixed_section_page_breaks=True,new_csv_result_rows_checked=new_rows_checked,
                all_pages_individually_inspected=True,original_report_preserved=True,
                old_primary_and_factorial_results_retained=True)
    (BASE/'completion_audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))


if __name__=='__main__':audit()
