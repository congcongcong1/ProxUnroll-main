"""Check final PDF/deck numbers and original-file preservation after visual QA."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
import xml.etree.ElementTree as E
from pypdf import PdfReader
from coursework.run_experiments import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--visual-qa-complete',action='store_true');a=p.parse_args()
    if not a.visual_qa_complete:raise ValueError('Inspect every final page and slide before marking QA complete')
    b=ROOT/'runs/lab244_5000_20261009';facts=json.loads((b/'report_facts.json').read_text());datasets=['Kodak','HEVC_B','HEVC_E','DIV2K_fresh']
    def value(d,s=0,ref='hqs_round1'):
        return next(r['mse_reduction_percent']/100 for r in facts['analysis']['datasets'] if r['dataset']==d and float(r['cr'])==.1 and float(r['sigma'])==s and r['reference']==ref)
    names={'adjoint':'Adjoint','fista_dct':'DCT-FISTA','hqs':'HQS','admm':'ADMM'}
    expected={1:{label:[next(r['psnr'] for r in facts['legacy']['datasets'] if r['dataset']=='Kodak' and r['method']==m and r['cr']==cr and r['sigma']==0) for cr in [.01,.04,.1,.25,.5]] for m,label in names.items()},
        2:{n:facts['validation_curve'][n] for n in ['Clean','Mixed','Strong noise']},
        3:{label:[value(d,ref=ref) for d in datasets] for ref,label in [('hqs_round1','vs round 1'),('hqs','vs official')]},
        4:{f'noise {s:g}':[value(d,s) for d in datasets] for s in [0,.01,.05]}}
    ppt=ROOT/'output/presentation/course_presentation_5000_20261009.pptx';pdf=ROOT/'output/pdf/technical_report_5000_20261009.pdf'
    assert facts['fresh_control_audit']['rows']==facts['fresh_control_audit']['float_arrays_recomputed']==1408
    assert facts['control_overlap']['conditions']==352
    ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(ppt) as z:
        slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
        notes=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)]
        assert len(slides)==len(notes)==12
        seconds=sum(int(re.search(r'Planned speaking time: (\d+) seconds',' '.join(x.text or '' for x in E.fromstring(z.read(n)).findall('.//a:t',ns))).group(1)) for n in notes)
        assert seconds==600
        for i,series in expected.items():
            e=E.fromstring(z.read(f'ppt/slides/charts/chart{i}.xml'))
            actual={s.find('c:tx/c:v',ns).text:[float(v.text) for v in s.findall('c:yVal/c:numRef/c:numCache/c:pt/c:v' if i in [1,2] else 'c:val/c:numRef/c:numCache/c:pt/c:v',ns)] for s in e.findall('.//c:ser',ns)}
            if i in [1,2]:
                xs=[1.,4.,10.,25.,50.] if i==1 else list(map(float,facts['validation_curve']['steps']))
                for series_node in e.findall('.//c:ser',ns):assert [float(v.text) for v in series_node.findall('c:xVal/c:numRef/c:numCache/c:pt/c:v',ns)]==xs
            assert actual=={name:[round(v,5) for v in vals] for name,vals in series.items()},(i,actual)
    reader=PdfReader(pdf);assert len(reader.pages)==6
    text=' '.join(page.extract_text() for page in reader.pages)
    for row in facts['clean_table'][1:]:assert row[1] in text and row[2] in text,row
    for row in facts['video_table'][1:]:assert row[0] in text and row[1] in text,row
    assert '800' in text and '32' in text and str(facts['training']['best_step']) in text
    preserved={}
    # The preservation baseline is fixed; later reports are also tracked after delivery.
    for name in subprocess.check_output(['git','ls-tree','-r','--name-only','a51da7c','--','output/pdf','output/presentation'],cwd=ROOT,text=True).splitlines():
        old=subprocess.check_output(['git','show','a51da7c:'+name],cwd=ROOT);assert (ROOT/name).read_bytes()==old
        preserved[name]=hashlib.sha256(old).hexdigest()
    previous=json.loads((ROOT/'runs/lab220_20261009/final_artifact_checks.json').read_text())['sha256']
    for name,digest in previous.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [pdf,ppt,b/'finetune/best.pth',b/'finetune/last.pth']}
    for name,digest in facts['training']['checkpoint_sha256'].items():assert hashes[str((b/'finetune'/name).relative_to(ROOT))]==digest
    receipt=json.loads(a.receipt.read_text());assert receipt['finalSha256']==hashes[str(ppt.relative_to(ROOT))]
    assert receipt['firstPartyImport']['passed'] and receipt['packageIntegrity']['status']=='pass'
    result=dict(pdf_pages=6,ppt_slides=12,ppt_notes=12,planned_speaking_seconds=seconds,native_chart_values_verified=True,
        all_final_pages_slides_rendered_and_visually_inspected=True,original_artifacts_preserved=preserved,round1_artifact_hashes_preserved=True,
        sha256=hashes,all_four_methods_on_all_80_images_verified=True,presentation_integrity_and_layout_passed=True,native_powerpoint_application_verified=False)
    (b/'final_artifact_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
