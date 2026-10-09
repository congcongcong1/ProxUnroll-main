"""Recompute metrics from retained float reconstructions and check paired provenance."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from coursework.run_experiments import ROOT,matrices,metrics,sha,write_csv
from coursework.baselines import measure


def local_path(name):
    p=Path(name)
    if str(p).startswith('/workspace/ProxUnroll-main/'): p=ROOT/str(p).removeprefix('/workspace/ProxUnroll-main/')
    return p


def verify(output):
    if not (output/'COMPLETE').exists(): raise RuntimeError('Run has no completion marker')
    meta=json.loads((output/'provenance.json').read_text()); manifest=local_path(meta['arguments']['manifest'])
    assert sha(manifest)==meta['manifest_sha256']
    for rel,digest in meta['source_hashes'].items(): assert sha(ROOT/rel)==digest,rel
    assert sha(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat')==meta['matrix_sha256']
    for name,entry in meta['checkpoints'].items(): assert sha(local_path(entry['path']))==entry['sha256'],name
    rows=list(csv.DictReader((output/'metrics.csv').open())); methods=meta['arguments']['methods'].split(',')
    assert len(rows)==len(meta['data'])*len(meta['conditions'])*len(methods)
    assert len({tuple(r[k] for k in ['image','cr','sigma','seed','method']) for r in rows})==len(rows)
    paired=defaultdict(list); image_map={r['name']:r for r in meta['data']}; data_dir=local_path(meta['arguments']['data_dir']) if meta['arguments']['data_dir'] else manifest.parent
    for r in rows: paired[(r['image'],float(r['cr']),float(r['sigma']),int(r['seed']))].append(r)
    max_err=0.; verified_arrays=0
    for (image,cr,sigma,seed),group in paired.items():
        assert {r['method'] for r in group}==set(methods)
        item=image_map[image]; path=Path(item['path']); path=path if path.is_absolute() else data_dir/path
        assert sha(path)==item['sha256']; gt=np.asarray(Image.open(path),dtype=np.float32)/255
        h,w=matrices(cr); y=measure(gt,h,w); rng=np.random.default_rng(seed+int(cr*10000)+sum(image.encode()))
        y=(y+sigma*np.sqrt(np.mean(y**2))*rng.standard_normal(y.shape)).astype(np.float32)
        digest=hashlib.sha256(y.tobytes()).hexdigest()
        for r in group:
            assert r['measurement_sha256']==digest
            assert abs(float(r['actual_cr'])-y.size/gt.size)<1e-12
            stem=f'{image}_cr{cr:g}_noise{sigma:g}_seed{seed}_{r["method"]}'
            array=np.load(output/'reconstructions'/(stem+'.npy'))
            score=metrics(gt,array)
            errors=[abs(float(r[m])-score[m]) for m in ['psnr','ssim']]; max_err=max(max_err,*errors)
            assert max(errors)<2e-5,(stem,errors)
            verified_arrays+=1
    stages=list(csv.DictReader((output/'stages.csv').open())) if (output/'stages.csv').exists() else []
    final={tuple(r[k] for k in ['image','cr','sigma','seed','method']):r for r in rows}
    for r in stages:
        if int(r['stage'])==6:
            score=final[tuple(r[k] for k in ['image','cr','sigma','seed','method'])]
            assert abs(float(score['psnr'])-float(r['psnr']))<1e-9
    audit=dict(rows=len(rows),stages=len(stages),paired_conditions=len(paired),float_arrays_recomputed=verified_arrays,max_metric_error=max_err,
               data_checkpoint_matrix_source_hashes_verified=True,measurement_hashes_recomputed=True)
    (output/'verification.json').write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit))
    return rows


def compare(rows,output):
    groups=defaultdict(dict)
    for r in rows: groups[tuple(r[k] for k in ['image','dataset','sequence','cr','sigma','seed'])][r['method']]=r
    deltas=[]
    for key,methods in groups.items():
        for name in methods.keys()-{'hqs'}:
            if 'hqs' not in methods: continue
            base=methods['hqs']; new=methods[name]
            deltas.append(dict(zip(['image','dataset','sequence','cr','sigma','seed'],key),method=name,
               delta_psnr=float(new['psnr'])-float(base['psnr']),delta_ssim=float(new['ssim'])-float(base['ssim'])))
    summary=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for r in deltas: summary[(r['dataset'],r['cr'],r['sigma'],r['method'])][r['sequence']][r['image']].append(r)
    paired=[]; rng=np.random.default_rng(20261009)
    for key,sequences in sorted(summary.items()):
        dp=np.array([np.mean([np.mean([r['delta_psnr'] for r in repeat]) for repeat in imgs.values()]) for imgs in sequences.values()])
        ds=[np.mean([np.mean([r['delta_ssim'] for r in repeat]) for repeat in imgs.values()]) for imgs in sequences.values()]
        boot=np.mean(rng.choice(dp,size=(5000,len(dp)),replace=True),axis=1)
        paired.append(dict(zip(['dataset','cr','sigma','method'],key),units=len(dp),delta_psnr=float(dp.mean()),delta_ssim=float(np.mean(ds)),
                           bootstrap_low=float(np.percentile(boot,2.5)),bootstrap_high=float(np.percentile(boot,97.5)),positive_units=int(np.sum(dp>0))))
    if deltas: write_csv(output/'paired_deltas.csv',deltas)
    if paired: write_csv(output/'paired_summary.csv',paired)
    negative=sorted([r for r in deltas if r['method']=='hqs_finetuned' and r['delta_psnr']<0],key=lambda r:r['delta_psnr'])
    if negative: write_csv(output/'negative_cases.csv',negative)
    (output/'paired_comparison.json').write_text(json.dumps(dict(bootstrap_unit='original image for Kodak, whole sequence for HEVC; exploratory one training seed',summary=paired,image_deltas=deltas),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--compare',action='store_true'); a=p.parse_args(); rows=verify(a.output)
    if a.compare: compare(rows,a.output)
