"""Verify web-data transformations, complete conditions, arithmetic and deliverables."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as E
import zipfile
import cv2
import numpy as np
from PIL import Image, ImageOps
from pypdf import PdfReader
from coursework.run_experiments import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--presentation-receipt',type=Path,required=True)
    p.add_argument('--visual-qa-complete',action='store_true');a=p.parse_args()
    if not a.visual_qa_complete:raise ValueError('Inspect every final page and slide first')
    b=ROOT/'runs/nanjing_web30_20261009';facts=json.loads((b/'report_facts.json').read_text())
    audit=json.loads((b/'comparison/verification.json').read_text());assert audit['rows']==2310 and audit['stages']==9900 and audit['max_metric_error']==0
    local=json.loads((b/'comparison/local_float_verification.json').read_text())
    assert local['float_arrays_recomputed']==2310 and local['max_metric_error']<2e-5
    assert local['recorded_measurement_hashes_verified']
    assert local['measurement_archive_sha256']==json.loads((b/'audit/measurements.json').read_text())['archive_sha256']
    manifest=json.loads((b/'data/test.json').read_text());lock=json.loads((b/'data/selection_lock.json').read_text())
    assert sha(b/'data/test.json')==lock['manifest_sha256']
    assert sha(ROOT/'coursework/nanjing_web30_sources.json')==lock['catalog_sha256']
    assert len(manifest)==30 and len({r['sha256'] for r in manifest})==30
    for r in manifest:
        source=ROOT/r['downloaded_source_path']
        assert sha(source)==r['downloaded_source_sha256']
        with Image.open(source) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        x,y,w,h=r['crop_xywh'];gray=cv2.cvtColor(rgb[y:y+h,x:x+w],cv2.COLOR_RGB2YCrCb)[:,:,0]
        expected=cv2.resize(gray,(256,256),interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(expected,np.asarray(Image.open(b/'data'/r['path'])))
        assert sha(b/'data'/r['path'])==r['sha256'] and r['split']=='test'
        assert r['author'] and r['page_url'].startswith('https://commons.wikimedia.org/wiki/File:')
    meta=json.loads((b/'comparison/provenance.json').read_text())
    assert set(meta['checkpoints'])==set(lock['checkpoints'])
    for name, entry in meta['checkpoints'].items():
        path=entry['path'].removeprefix('/workspace/ProxUnroll-main/')
        assert path==lock['checkpoints'][name]['path'] and entry['sha256']==lock['checkpoints'][name]['sha256']
    for r in lock['checkpoints'].values():assert sha(ROOT/r['path'])==r['sha256']
    assert meta['matrix_sha256']==lock['matrix_sha256']
    metrics=list(csv.DictReader((b/'comparison/metrics.csv').open()))
    stages=list(csv.DictReader((b/'comparison/stages.csv').open()))
    stage_groups=defaultdict(set)
    for r in stages:stage_groups[tuple(r[k] for k in ['image','cr','sigma','seed','method'])].add(int(r['stage']))
    assert len(stage_groups)==1650 and all(s==set(range(1,7)) for s in stage_groups.values())
    by_photo=defaultdict(list)
    for r in metrics:by_photo[(r['dataset'],float(r['cr']),float(r['sigma']),r['method'],r['image'])].append(10**(-float(r['psnr'])/10))
    aggregate=defaultdict(list)
    for (d,cr,s,m,photo),values in by_photo.items():aggregate[(d,cr,s,m)].append(sum(values)/len(values))
    for r in facts['analysis']['datasets']:
        old=aggregate[(r['dataset'],r['cr'],r['sigma'],r['reference'])]
        new=aggregate[(r['dataset'],r['cr'],r['sigma'],r['candidate'])]
        percent=100*(1-(sum(new)/len(new))/(sum(old)/len(old)))
        assert abs(percent-r['mse_reduction_percent'])<1e-10
    datasets=['NJU_campus_web','Nanjing_scenic_web']
    def value(d,s=0.,ref='hqs_round1'):
        return next(r['mse_reduction_percent']/100 for r in facts['analysis']['datasets'] if r['dataset']==d and r['cr']==.1 and r['sigma']==s and r['reference']==ref and r['candidate']=='hqs_10000')
    names={'adjoint':'Adjoint','fista_dct':'DCT-FISTA','hqs':'Official HQS','admm':'Official ADMM'}
    expected={1:{label:[next(r['psnr'] for r in facts['baseline']['datasets'] if r['dataset']==datasets[0] and r['method']==m and r['cr']==cr and r['sigma']==0) for cr in [.01,.04,.1,.25,.5]] for m,label in names.items()},
        2:{n:facts['validation_curve'][n] for n in ['Clean','Mixed','Strong noise']},
        3:{label:[value(d,ref=ref) for d in datasets] for ref,label in [('hqs','vs official'),('hqs_round1','vs 100-step'),('hqs_5000','vs 5000-budget')]},
        4:{f'noise {s:g}':[value(d,s) for d in datasets] for s in [0.,.01,.05]}}
    ppt=ROOT/'output/presentation/course_presentation_nanjing_20261009.pptx'
    pdf=ROOT/'output/pdf/technical_report_nanjing_20261009.pdf'
    ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(ppt) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)])==12
        notes=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)]
        assert len(notes)==12
        note_text=[' '.join(x.text or '' for x in E.fromstring(z.read(n)).findall('.//a:t',ns)) for n in notes]
        assert sum(int(re.search(r'Planned speaking time: (\d+) seconds',n).group(1)) for n in note_text)==600
        assert all('undefined' not in n for n in note_text)
        for i,values in expected.items():
            chart=E.fromstring(z.read(f'ppt/slides/charts/chart{i}.xml'))
            actual={s.find('c:tx/c:v',ns).text:[float(v.text) for v in s.findall('c:yVal/c:numRef/c:numCache/c:pt/c:v' if i<3 else 'c:val/c:numRef/c:numCache/c:pt/c:v',ns)] for s in chart.findall('.//c:ser',ns)}
            assert actual=={name:[round(v,5) for v in numbers] for name,numbers in values.items()},(i,actual)
    reader=PdfReader(pdf);assert len(reader.pages)==6
    text=' '.join(' '.join(page.extract_text() for page in reader.pages).split())
    assert all(s in text for s in ['10000','2310','9900','AI Disclosure','not photographed by the team'])
    for row in facts['clean_table'][1:]:assert all(v in text for v in row[1:])
    for row in facts['location_table'][1:]:assert all(v in text for v in row)
    preserved={}
    for name,digest in json.loads((ROOT/'coursework/repository_evidence.json').read_text())['files'].items():
        assert sha(ROOT/name)==digest,name
        if (name.startswith('output/') and Path(name).suffix in {'.pdf','.pptx'}
                and name not in {str(pdf.relative_to(ROOT)),str(ppt.relative_to(ROOT))}):preserved[name]=digest
    for name in subprocess.check_output(['git','ls-tree','-r','--name-only','a51da7c','--','output/pdf','output/presentation'],cwd=ROOT,text=True).splitlines():
        assert (ROOT/name).read_bytes()==subprocess.check_output(['git','show','a51da7c:'+name],cwd=ROOT)
    receipt=json.loads(a.presentation_receipt.read_text())
    assert receipt['finalSha256']==sha(ppt) and receipt['firstPartyImport']['passed'] and receipt['packageIntegrity']['status']=='pass'
    result=dict(processed_images=30,source_hashes_and_pixel_preprocessing_recomputed=True,
        all_2310_float_metrics_recomputed_remotely=True,all_9900_stages_complete=True,
        all_2310_float_metrics_recomputed_locally=True,local_metric_max_error=local['max_metric_error'],
        selection_lock_and_checkpoint_hashes_verified=True,paired_mse_arithmetic_verified=True,
        pdf_pages=6,ppt_slides=12,planned_speaking_seconds=600,native_chart_values_verified=True,
        all_final_pages_and_slides_visually_inspected=True,native_powerpoint_application_verified=False,
        previous_artifacts_preserved=preserved,sha256={str(f.relative_to(ROOT)):sha(f) for f in [pdf,ppt]})
    (b/'final_artifact_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
