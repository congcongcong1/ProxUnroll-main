"""Independent coverage and aggregation audit for the 20000-update comparison."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

from coursework.run_experiments import ROOT, sha

BASE=ROOT/'runs/hqs_20000_eval_20261014'
OLD=ROOT/'runs/hqs_ablation_20261013'


def mean_metric(rows, field):
    sequences=defaultdict(lambda:defaultdict(list))
    for r in rows:
        value=10**(-float(r['psnr'])/10) if field=='mse' else float(r[field])
        sequences[r['sequence']][r['image']].append(value)
    return float(np.mean([np.mean([np.mean(v) for v in images.values()]) for images in sequences.values()]))


def audit():
    lock=json.loads((BASE/'lock.json').read_text())
    result=json.loads((BASE/'results.json').read_text())
    assert (BASE/'COMPLETE').exists() and result['lock']==lock
    assert sha(ROOT/'coursework/evaluate_20000.py')==lock['driver_sha256']
    assert sha(ROOT/'coursework/evaluate_manifest.py')==lock['metric_source_sha256']
    for name in ['smoke','primary','regression','finalize']:
        assert int((BASE/'logs'/f'{name}.exit').read_text())==0
    phases={}
    for phase,count in [('primary',52),('regression',110)]:
        d=BASE/('evaluation_'+phase)
        rows=list(csv.DictReader((d/'metrics.csv').open()))
        verification=json.loads((d/'verification.json').read_text())
        meta=json.loads((d/'provenance.json').read_text())
        assert len(rows)==verification['rows']==verification['float_arrays_recomputed']==count*11*2
        stages=list(csv.DictReader((d/'stages.csv').open()))
        assert len(stages)==verification['stages']==len(rows)*6
        assert len({tuple(r[k] for k in ['image','cr','sigma','seed','method']) for r in rows})==len(rows)
        assert verification['measurement_hashes_recomputed'] and verification['data_checkpoint_matrix_source_hashes_verified']
        assert verification['max_metric_error']<2e-5
        for name,e in meta['checkpoints'].items():assert e['sha256']==lock['models'][name]['sha256']
        for name,digest in lock['reference_metric_hashes'][phase].items():assert sha(OLD/('evaluation_'+phase)/name)==digest
        comparisons=list(csv.DictReader((d/'comparison_metrics.csv').open()))
        assert len(comparisons)==count*11*5
        measurements=defaultdict(set);groups=defaultdict(list)
        for r in comparisons:
            measurements[(r['image'],float(r['cr']),float(r['sigma']),int(r['seed']))].add(r['measurement_sha256'])
            groups[(r['dataset'],float(r['cr']),float(r['sigma']),r['method'])].append(r)
        assert len(measurements)==count*11 and all(len(v)==1 for v in measurements.values())
        computed={k:{m:mean_metric(v,m) for m in ['mse','psnr','ssim']} for k,v in groups.items()}
        for r in csv.DictReader((d/'all_model_summary.csv').open()):
            key=(r['dataset'],float(r['cr']),float(r['sigma']),r['method'])
            for field in ['mse','psnr','ssim']:assert abs(float(r[field])-computed[key][field])<1e-10
        paired=list(csv.DictReader((d/'paired_summary.csv').open()))
        for r in paired:
            key=(r['dataset'],float(r['cr']),float(r['sigma']))
            old,new=computed[(*key,r['reference'])],computed[(*key,r['candidate'])]
            assert abs(float(r['mse_reduction_percent'])-100*(1-new['mse']/old['mse']))<1e-9
            for field in ['psnr','ssim']:assert abs(float(r['delta_'+field])-(new[field]-old[field]))<1e-10
        delta=list(csv.DictReader((d/'paired_image_deltas.csv').open()))
        assert len(delta)==count*11*6
        negatives=list(csv.DictReader((d/'negative_cases.csv').open()))
        assert len(negatives)==sum(float(r['mse_reduction_percent'])<0 for r in delta)
        phases[phase]=dict(images=count,new_rows=len(rows),new_stages=len(stages),comparison_rows=len(comparisons),
                          comparisons=len(delta),negative_comparisons=len(negatives),
                          metrics_sha256=sha(d/'metrics.csv'),all_measurements_matched=True,
                          all_equal_original_video_mse_psnr_ssim_aggregates_recomputed=True)
    receipt=dict(phases=phases,selected_step=19500,total_updates=20000,
                 checkpoint_selection='validation only; selected and endpoint both retained',
                 old_scores_reused_after_locked_reference_smoke_equivalence=True,
                 results_sha256=sha(BASE/'results.json'),all_exit_codes_zero=True)
    (BASE/'completion_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':audit()
