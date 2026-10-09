"""Independent manifest/checkpoint evaluation using the original metric contract."""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from PIL import Image
import torch
from coursework.baselines import measure, adjoint, fista_dct
from coursework.run_experiments import ROOT, load_model, matrices, torch_matrices, metrics, sync, sha, write_csv


def read_manifest(path,data_dir,split='test'):
    rows=json.loads(Path(path).read_text()); names=set(); selected=[]
    for item in rows:
        if item['name'] in names: raise ValueError('Duplicate manifest name')
        names.add(item['name'])
        if item['split']!=split: continue
        source=Path(item['path']); source=source if source.is_absolute() else Path(data_dir)/source
        if sha(source)!=item['sha256']: raise ValueError(f'Data hash mismatch: {source}')
        image=Image.open(source)
        if image.mode!='L' or image.size!=(256,256): raise ValueError('Expected 256x256 grayscale preprocessing')
        selected.append((item,np.asarray(image,dtype=np.float32)/255))
    if not selected: raise ValueError('Empty selected split')
    return selected


def summarize(rows,output):
    def aggregate(keys):
        groups=defaultdict(list)
        for r in rows: groups[tuple(r[k] for k in keys)].append(r)
        result=[]
        for key,group in sorted(groups.items()):
            # Average noise repeats per image, then images per video, then videos.
            units=defaultdict(lambda:defaultdict(list))
            for r in group: units[r['sequence']][r['image']].append(r)
            means={metric:[np.mean([np.mean([x[metric] for x in samples]) for samples in images.values()])
                           for images in units.values()] for metric in ['psnr','ssim','seconds']}
            result.append(dict(zip(keys,key),units=len(units),images=len({r['image'] for r in group}),
                               **{m:float(np.mean(v)) for m,v in means.items()}))
        return result
    datasets=aggregate(['dataset','cr','sigma','method']); videos=aggregate(['dataset','sequence','cr','sigma','method'])
    write_csv(output/'dataset_summary.csv',datasets); write_csv(output/'video_summary.csv',videos)
    failures=sorted([r for r in rows if r['method']=='hqs' and r['sigma']==0 and r['cr']==.1],key=lambda r:r['psnr'])
    if failures: write_csv(output/'failure_cases.csv',failures)
    (output/'summary.json').write_text(json.dumps(dict(aggregation='noise repeats within image, then images within sequence, then equal sequence weights within dataset; Kodak units are individual images',datasets=datasets,videos=videos),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--data-dir',type=Path)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--device',choices=['cuda','cpu','mps'],default='cuda')
    p.add_argument('--hqs-checkpoint',type=Path,default=ROOT/'weight/hqs_proxunroll.pth'); p.add_argument('--admm-checkpoint',type=Path,default=ROOT/'weight/admm_proxunroll.pth')
    p.add_argument('--extra-checkpoint',action='append',default=[],help='HQS variant NAME=PATH')
    p.add_argument('--methods',default='adjoint,fista_dct,hqs,admm'); p.add_argument('--rates',default='0.01,0.04,0.1,0.25,0.5')
    p.add_argument('--noise',default='0.01,0.05'); p.add_argument('--seeds',default='2026,2027,2028'); p.add_argument('--smoke',action='store_true'); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); device=torch.device(a.device); torch.set_num_threads(a.threads)
    torch.manual_seed(2026); torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    data=read_manifest(a.manifest,a.data_dir or a.manifest.parent)
    paths={'hqs':('hqs',a.hqs_checkpoint),'admm':('admm',a.admm_checkpoint)}
    for spec in a.extra_checkpoint:
        name,path=spec.split('=',1)
        if name in paths or name in ['adjoint','fista_dct']: raise ValueError('Duplicate method')
        paths[name]=('hqs',Path(path))
    methods=a.methods.split(',')
    if any(m not in paths and m not in ['adjoint','fista_dct'] for m in methods): raise ValueError('Unknown method')
    models={n:load_model(s,device,path) for n,(s,path) in paths.items() if n in methods}
    rates=[float(x) for x in a.rates.split(',')]; seeds=[int(x) for x in a.seeds.split(',')]
    conditions=[(r,0.,seeds[0]) for r in rates]+[(.1,float(s),seed) for s in a.noise.split(',') if s for seed in seeds]
    if a.smoke: data=data[:1]; conditions=[(.1,0.,seeds[0])]
    lam=json.loads((ROOT/'coursework/results/provenance.json').read_text())['selected_lambda']
    with torch.inference_mode():
        for model in models.values(): model.reconstruct(torch.zeros(1,matrices(.1)[0].shape[0],matrices(.1)[1].shape[0],device=device),(256,256),.1,sensing_matrices=torch_matrices(.1,device))
    sync(device)
    provenance=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},platform=platform.platform(),device=str(device),
       torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
       packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},manifest_sha256=sha(a.manifest),data=[r for r,_ in data],
       checkpoints={n:dict(path=str(paths[n][1]),sha256=sha(paths[n][1])) for n in models},conditions=conditions,
       source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'coursework/evaluate_manifest.py',ROOT/'coursework/run_experiments.py',ROOT/'coursework/baselines.py',ROOT/'model/proxunroll.py']},
       matrix_sha256=sha(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat'),fista_lambda=lam,fista_lambda_source='original disjoint skimage validation; frozen before external test',
       metric_contract='clip predictions [0,1]; data_range=1; skimage PSNR/SSIM; no border crop; float metrics before PNG; relative RMS measurement noise',
       model_scope='single-image reconstruction on independent frames; no temporal model',timing='synchronized wall time after warmup; FISTA/adjoint CPU, neural CUDA; one sample per condition')
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n'); rec=a.output/'reconstructions'; rec.mkdir()
    rows=[]; stages=[]; started=time.perf_counter()
    for item,gt in data:
        for cr,sigma,seed in conditions:
            h,w=matrices(cr); y=measure(gt,h,w); rng=np.random.default_rng(seed+int(cr*10000)+sum(item['name'].encode()))
            y=(y+sigma*np.sqrt(np.mean(y**2))*rng.standard_normal(y.shape)).astype(np.float32)
            yt=torch.from_numpy(y).unsqueeze(0).to(device); factors=torch_matrices(cr,device)
            shared=dict(image=item['name'],dataset=item.get('dataset','external'),sequence=item.get('sequence',item['name']),split=item['split'],cr=cr,actual_cr=y.size/gt.size,sigma=sigma,seed=seed,
                        measurement_sha256=hashlib.sha256(y.tobytes()).hexdigest())
            for name in methods:
                sync(device); start=time.perf_counter()
                if name=='adjoint': pred=adjoint(y,h,w)
                elif name=='fista_dct': pred=fista_dct(y,h,w,lam,200)
                else:
                    with torch.inference_mode(): outputs=models[name].reconstruct(yt,(256,256),cr,sensing_matrices=factors)
                    sync(device)
                elapsed=time.perf_counter()-start
                if name in models:
                    outs=outputs[:,0].cpu().numpy(); pred=outs[-1]
                    for stage,out in enumerate(outs,1): stages.append(dict(**shared,method=name,stage=stage,**metrics(gt,out)))
                if not np.isfinite(pred).all(): raise RuntimeError('Non-finite reconstruction')
                score=metrics(gt,pred); rows.append(dict(**shared,method=name,**score,seconds=elapsed,
                    relative_residual=float(np.linalg.norm(measure(np.clip(pred,0,1),h,w)-y)/max(np.linalg.norm(y),1e-12))))
                stem=f'{item["name"]}_cr{cr:g}_noise{sigma:g}_seed{seed}_{name}'
                np.save(rec/(stem+'.npy'),pred.astype(np.float32))
                if seed==seeds[0]: Image.fromarray(np.round(np.clip(pred,0,1)*255).astype(np.uint8)).save(rec/(stem+'.png'))
            write_csv(a.output/'metrics.csv',rows)
            if stages: write_csv(a.output/'stages.csv',stages)
        print(f'{item["name"]}: {len(rows)} rows, {time.perf_counter()-started:.1f}s',flush=True)
    assert len(rows)==len(data)*len(conditions)*len(methods)
    summarize(rows,a.output); (a.output/'COMPLETE').write_text(f'{len(rows)} paired rows; {time.perf_counter()-started:.3f} seconds\n')
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
