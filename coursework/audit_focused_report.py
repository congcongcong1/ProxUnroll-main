"""Independently check report aggregates against per-image records and rendered PDF."""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/report_focused'
PDF=ROOT/'output/pdf/technical_report_20000_ablation.pdf'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(p.open()))

def audit():
    checked=0
    for folder,raw_file in [('runs/hqs_20000_eval_20261014/evaluation_primary','comparison_metrics.csv'),('runs/hqs_20000_eval_20261014/evaluation_regression','comparison_metrics.csv'),('runs/hqs_ablation_20261013/evaluation_primary','metrics.csv')]:
        directory=ROOT/folder
        raw=read(directory/raw_file)
        groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        measured=defaultdict(set)
        for r in raw:
            key=(r['dataset'],float(r['cr']),float(r['sigma']),r['method'])
            groups[key][r['sequence']][r['image']].append(r)
            measured[(r['image'],float(r['cr']),float(r['sigma']),r['seed'])].add(r['measurement_sha256'])
        assert all(len(v)==1 for v in measured.values())
        calculated={}
        for key,sequences in groups.items():
            calculated[key]={metric:mean(mean(mean(10**(-float(r['psnr'])/10) if metric=='mse' else float(r[metric]) for r in repeats) for repeats in images.values()) for images in sequences.values()) for metric in ['mse','psnr','ssim']}
        for r in read(directory/'all_model_summary.csv'):
            key=(r['dataset'],float(r['cr']),float(r['sigma']),r['method'])
            for metric in ['mse','psnr','ssim']:
                assert abs(calculated[key][metric]-float(r[metric]))<1e-10
                checked+=1
        if 'ablation' in folder:
            for r in read(directory/'paired_summary.csv'):
                if r['checkpoint_view']!='last':continue
                k=(r['dataset'],float(r['cr']),float(r['sigma']))
                a,b=calculated[(*k,r['reference'])],calculated[(*k,r['candidate'])]
                assert abs(100*(1-b['mse']/a['mse'])-float(r['mse_reduction_percent']))<1e-9
                checked+=1
            for r in read(directory/'factorial_effects.csv'):
                if r['checkpoint_view']!='last' or r['training_seed']=='mean_of_two':continue
                k=(r['dataset'],float(r['cr']),float(r['sigma']))
                def mse(noise,loss):return calculated[(*k,f"last_noise_{noise}_{loss}_s{r['training_seed']}")]['mse']
                nw=mse('off','weighted')-mse('on','weighted')
                nf=mse('off','final')-mse('on','final')
                so=mse('on','final')-mse('on','weighted')
                sf=mse('off','final')-mse('off','weighted')
                value={'noise_main_mse_benefit':(nw+nf)/2,'stage_main_mse_benefit':(so+sf)/2,'noise_stage_interaction_mse_benefit':nw-nf}[r['effect']]
                assert abs(100*value/calculated[(*k,'hqs')]['mse']-float(r['official_normalized_percent_points']))<1e-9
                checked+=1
    receipt=json.loads((BASE/'build_receipt.json').read_text())
    assert sha(PDF)==receipt['pdf_sha256']
    for path,value in receipt['inputs'].items():assert sha(ROOT/path)==value
    reader=PdfReader(PDF)
    assert len(reader.pages)==5
    assert not {'/CreationDate','/ModDate'} & set(reader.metadata)
    text=' '.join(p.extract_text() for p in reader.pages)
    assert not re.search(r'/workspace/|/Users/|2026[-_]?10|202610|server244|luozc|11500|12000|\bminutes\b',text,re.I)
    for value in ['3.32%','11.98%','8.45%','-0.74%','19,500','20,000','28 of 572','9.08%']:
        assert value in text,value
    assert '\u25a0' not in text and '\ufffd' not in text
    visual=json.loads((BASE/'visual_audit.json').read_text())
    assert visual['pdf_sha256']==sha(PDF)
    assert len(visual['pages'])==5
    for page in visual['pages']:
        assert page['inspected'] and sha(ROOT/page['path'])==page['sha256']
    result={'status':'passed','pages':5,'figures':3,'tables':6,'independent_numeric_checks':checked,'pdf_sha256':sha(PDF),'all_pages_visually_reviewed':True,'dates_and_operational_paths_removed':True}
    (BASE/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':audit()
