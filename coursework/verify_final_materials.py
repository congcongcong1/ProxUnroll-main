"""Independently compare final editable slides and notes with frozen CSV evidence."""
import csv
import hashlib
import json
from pathlib import Path
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader
from coursework.verify_ablation_materials import chart_series, NS

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/materials_final_20261014'
PPT=ROOT/'output/presentation/course_presentation_final_20261014_v2.pptx'
LIT=ROOT/'output/pdf/literature_review_final_20261014.pdf'
TECH=ROOT/'output/pdf/technical_report_flow_20261014.pdf'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return list(csv.DictReader(path.open()))
def xml_text(element):return ' '.join(n.text or '' for n in element.findall('.//a:t',NS))


def main():
    oldroot=ROOT/'runs/hqs_ablation_20261013'
    newroot=ROOT/'runs/hqs_20000_eval_20261014'
    baseline_path=ROOT/'runs/nanjing_web30_20261009/comparison/dataset_summary.csv'
    source_paths=[baseline_path,oldroot/'evaluation_primary/paired_summary.csv',
                  newroot/'evaluation_primary/paired_summary.csv',newroot/'evaluation_regression/paired_summary.csv']
    baseline,old,new,regression=[rows(p) for p in source_paths]
    def pair(data,candidate,reference,sigma,dataset='DIV2K_holdout_10000'):
        r=next(r for r in data if r['dataset']==dataset and float(r['cr'])==.1 and float(r['sigma'])==sigma
               and r['candidate']==candidate and r['reference']==reference)
        return float(r['mse_reduction_percent'])/100
    def factor(comparison,seed,sigma):
        r=next(r for r in old if r['dataset']=='DIV2K_holdout_10000' and float(r['cr'])==.1
               and r['comparison']==comparison and r['checkpoint_view']=='last'
               and int(r['training_seed'])==seed and float(r['sigma'])==sigma)
        return float(r['mse_reduction_percent'])/100
    expected={
        6:[[float(next(r for r in baseline if r['dataset']=='NJU_campus_web' and r['method']==method
                      and float(r['cr'])==rate and float(r['sigma'])==0)['psnr'])
            for rate in [.01,.04,.1,.25,.5]] for method in ['adjoint','fista_dct','hqs','admm']],
        8:[[pair(old,'hqs_12000','hqs_10000',s) for s in [0.,.01,.05]],
           [pair(new,'hqs_20000','hqs_10000',s) for s in [0.,.01,.05]],
           [pair(new,'last_20000','hqs_10000',s) for s in [0.,.01,.05]]],
        9:[[factor('noise_effect_at_weighted',seed,s) for s in [0.,.01,.05]] for seed in [20261013,20261014]],
        10:[[factor('stage_effect_at_noise_on',seed,s) for s in [0.,.01,.05]] for seed in [20261013,20261014]]}
    native={};slide_text={};note_text={}
    with zipfile.ZipFile(PPT) as z:
        assert z.testzip() is None
        slides=[name for name in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',name)]
        assert len(slides)==12
        for n in range(1,13):
            slide_text[n]=xml_text(ET.fromstring(z.read(f'ppt/slides/slide{n}.xml')))
            note_text[n]=xml_text(ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{n}.xml')))
        for n,values in expected.items():
            target,actual=chart_series(z,n)
            assert len(actual)==len(values) and all(len(a)==len(b) for a,b in zip(actual,values))
            error=max(abs(a-b) for xs,ys in zip(actual,values) for a,b in zip(xs,ys))
            assert error<(6e-6 if n==6 else 1e-7),(n,error)
            tree=ET.fromstring(z.read(target))
            labels=[]
            for ser in tree.findall('.//c:ser',NS):
                tx=ser.find('c:tx',NS);labels.append(' '.join(v.text or '' for v in tx.findall('.//c:v',NS)))
                if n!=6:
                    cats=[p.find('c:v',NS).text for p in sorted(ser.find('c:cat',NS).findall('.//c:pt',NS),key=lambda p:int(p.attrib['idx']))]
                    assert cats==['Clean','Noise .01','Noise .05'],(n,cats)
            if n==8:
                assert labels==['Selected 11500','Selected 19500','Endpoint 20000'],labels
                assert 'vs 10000-step HQS' in xml_text(tree)
            native[str(n)]={'chart_part':target,'max_absolute_error':error,'series_checked':len(values)}
        table=ET.fromstring(z.read('ppt/slides/slide12.xml')).find('.//a:tbl',NS)
        actual_table=[[xml_text(c) for c in r.findall('a:tc',NS)] for r in table.findall('a:tr',NS)]
        table_expected=[['Dataset','Clean','Noise .05']]
        for dataset,label in [('Kodak','Kodak24'),('HEVC_B','HEVC B'),('HEVC_E','HEVC E'),('DIV2K_fresh','DIV2K32'),('NJU_campus_web','Campus18'),('Nanjing_scenic_web','Scenic12')]:
            table_expected.append([label,*[f'{100*pair(regression,"hqs_20000","hqs_12000",s,dataset):+.2f}%' for s in [0.,.05]]])
        assert actual_table==table_expected,(actual_table,table_expected)
        for value in ['On / weighted','Off / weighted','On / final','Off / final','800 x2 seeds']:
            assert value in slide_text[7],value
        for value in ['11500','191/572','19500','258/572','-4.97%']:
            assert value in slide_text[11],value
        for value in ['MSE rises 0.83%','0.41%','SSIM rises slightly','AI disclosure']:
            assert value in slide_text[12],value
        for value in ['previously tested','52 + 110','External-author']:
            assert value in slide_text[5],value
        for value in ['19500','20000','3564','21384','Codex','5–10%']:
            assert value in ' '.join(note_text.values()),value
        source_ppt=ROOT/'output/presentation/course_presentation_ablation_20261013.pptx'
        with zipfile.ZipFile(source_ppt) as original:
            for value in ['creativecommons.org/licenses/by-sa/3.0','creativecommons.org/licenses/by/4.0','Zhongshan','KrUDSO4']:
                assert value in note_text[5],value
            for n in [1,2,3,4]:
                assert slide_text[n]==xml_text(ET.fromstring(original.read(f'ppt/slides/slide{n}.xml'))),n
    zh=ROOT/'coursework/讲稿_最终交付_20261014.md';text=zh.read_text()
    durations=[int(t) for t in re.findall(r'^## 第\d+页[^\n]*（约(\d+)秒）',text,re.M)]
    note_durations=[int(re.search(r'Planned speaking time:\s*(\d+)\s*seconds',note_text[n]).group(1)) for n in range(1,13)]
    assert len(durations)==12 and durations==note_durations and sum(durations)==600
    assert len(re.findall(r'^## Slide \d+', (ROOT/'coursework/PRESENTATION_GUIDE_FINAL_20261014.md').read_text(),re.M))==12
    for term in ['19500','20000','11500','误差','SSIM','0.83%','0.41%','258','Codex','外部作者']:
        assert term in text,term
    lit_reader=PdfReader(LIT);tech_reader=PdfReader(TECH)
    assert len(lit_reader.pages)==3 and len(tech_reader.pages)==4
    lit_text=' '.join(' '.join(p.extract_text().split()) for p in lit_reader.pages)
    for term in ['Duarte','Wang','20000','19500','0.83%','0.41%','3564','21384','direct-ground-truth','Codex']:
        assert term in lit_text,term
    assert 'none is presented here as completed work' not in lit_text
    for text in [lit_text,' '.join(slide_text.values()),' '.join(note_text.values())]:
        assert not re.search(r'\bTODO\b|\bundefined\b|\bnan\b',text,re.I)
    validation=json.loads((BASE/'presentation_validation.json').read_text())
    assert validation['finalSha256']==sha(PPT)
    assert validation['packageIntegrity']['exitCode']==validation['presentationLayout']['exitCode']==0
    assert not validation['presentationLayout']['warnings']
    assert validation['firstPartyImport']['passed']
    visual=json.loads((BASE/'visual_audit.json').read_text())
    assert visual['ppt_sha256']==sha(PPT) and visual['literature_sha256']==sha(LIT)
    for key,count in [('slides',12),('literature_pages',3)]:
        assert len(visual[key])==count
        for p in visual[key]:
            assert p['individually_inspected'] and p['no_clipping_or_overlap']
            assert p['sha256']==sha(ROOT/p['path'])
    preserved={}
    for name in ['coursework/repository_evidence.json','runs/hqs_ablation_20261013/final_evidence.json','runs/report_flow_20261014/final_evidence.json']:
        record=json.loads((ROOT/name).read_text())
        for path,digest in record['files'].items():assert sha(ROOT/path)==digest,path
        preserved[name]=len(record['files'])
    result={'ppt_sha256':sha(PPT),'technical_sha256':sha(TECH),'literature_sha256':sha(LIT),
            'slides':12,'technical_pages':4,'literature_pages':3,'planned_speaking_seconds':600,
            'native_charts':native,'regression_table_rows_checked':6,
            'source_csv_hashes':{str(p.relative_to(ROOT)):sha(p) for p in source_paths},
            'source_notes_credits_retained':True,'checkpoint_names_references_and_negative_cases_consistent':True,
            'historical_evidence_unchanged':preserved,'all_final_slides_and_literature_pages_inspected':True,
            'native_powerpoint_app_verification':False}
    (BASE/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
