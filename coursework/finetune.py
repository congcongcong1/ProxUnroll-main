"""Bounded fixed-operator HQS fine-tuning with exact, explicit state resume."""
import argparse
import json
from pathlib import Path
import random
import time
import cv2
import numpy as np
from PIL import Image
import torch
from coursework.run_experiments import ROOT, load_model, matrices, torch_matrices, metrics, sha, write_csv, sync
from coursework.evaluate_manifest import read_manifest


def crop_original(item,seed,step):
    # Step-keyed randomness gives exactly the same image/crop after resume.
    rng=np.random.default_rng(np.random.SeedSequence([seed,step])); rgb=np.asarray(Image.open(item['path']).convert('RGB'))
    y=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)[:,:,0]; h,w=y.shape
    if min(h,w)<256: raise ValueError('Training original smaller than crop')
    side=int(rng.integers(256,min(h,w)+1)); top=int(rng.integers(0,h-side+1)); left=int(rng.integers(0,w-side+1))
    crop=cv2.resize(y[top:top+side,left:left+side],(256,256),interpolation=cv2.INTER_AREA)
    flip_h=bool(rng.integers(2)); flip_v=bool(rng.integers(2))
    if flip_h: crop=np.fliplr(crop)
    if flip_v: crop=np.flipud(crop)
    return np.ascontiguousarray(crop,dtype=np.float32)/255,dict(source=item['name'],xywh=[left,top,side,side],flip_h=flip_h,flip_v=flip_v)


def training_item(items,seed,step):
    epoch,position=divmod(step,len(items)); permutation=np.random.default_rng(np.random.SeedSequence([seed,epoch,1])).permutation(len(items))
    return items[int(permutation[position])]


def atomic_save(state,path):
    temp=path.with_suffix('.tmp'); torch.save(state,temp); temp.replace(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--data-dir',type=Path)
    p.add_argument('--output',type=Path,required=True); mode=p.add_mutually_exclusive_group(); mode.add_argument('--init',type=Path,default=None); mode.add_argument('--resume',type=Path)
    p.add_argument('--max-steps',type=int,default=100); p.add_argument('--max-minutes',type=float,default=20); p.add_argument('--val-every',type=int,default=25)
    p.add_argument('--lr',type=float,default=5e-6); p.add_argument('--seed',type=int,default=20261009); p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.max_steps<=1000 or not 0<a.max_minutes<=60 or a.val_every<1: raise ValueError('Requires bounded steps/time')
    a.output.mkdir(parents=True,exist_ok=False); device=torch.device('cuda'); torch.set_num_threads(a.threads)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.set_float32_matmul_precision('highest'); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    rows=json.loads(a.manifest.read_text()); train=[r for r in rows if r['split']=='train']; validation=read_manifest(a.manifest,a.data_dir or a.manifest.parent,'validation')
    if not train: raise ValueError('Empty training set')
    train_hashes={r['source_sha256'] for r in train}; val_hashes={r['source_sha256'] for r,_ in validation}
    if train_hashes & val_hashes: raise ValueError('Training/validation source leakage')
    for r in train:
        if sha(Path(r['path']))!=r['sha256']: raise ValueError('Training source modified')
    init=a.init or ROOT/'weight/hqs_proxunroll.pth'; model=load_model('hqs',device,a.resume or init)
    if not a.resume:
        raw=__import__('scipy.io',fromlist=['loadmat']).loadmat(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat')
        with torch.no_grad():
            for key in ['H','W']: getattr(model,key+'_256_256').copy_(torch.as_tensor(raw[key],dtype=torch.float32,device=device))
    frozen={}
    for name,param in model.named_parameters():
        if name.startswith(('H_','W_')): param.requires_grad_(False); frozen[name]=param.detach().clone()
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=a.lr)
    scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.0,total_iters=1)
    config=dict(manifest_sha256=sha(a.manifest),matrix_sha256=sha(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat'),seed=a.seed,lr=a.lr,
      training_rates=[.01,.04,.1,.25,.5],validation_rates=[.1,.25],operator='fixed common MAT; all six sensing factors frozen',
      loss='upstream proximal trajectory weighted RMSE: five weights .01, final .95; sqrt clamp 1e-12',batch_size=1,precision='float32',
      initialization=str(init),initialization_sha256=sha(init),selection='equal-image mean validation PSNR over two fixed clean rates; no test access',
      cropping='step-seeded random square side [256,min(H,W)], valid coordinates, area resize, flips, OpenCV Y',
      source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'coursework/finetune.py',ROOT/'model/proxunroll.py']},
      torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name())
    start_step=0; best=-float('inf'); best_step=0; history=[]; val_rows=[]
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=True)
        for key in ['manifest_sha256','matrix_sha256','seed','lr','training_rates','validation_rates','source_hashes']:
            if state['config'][key]!=config[key]: raise ValueError('Resume contract changed: '+key)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); start_step=state['step']; best=state['best_psnr']; best_step=state['best_step']
        random.setstate(state['rng_python']); ns=state['rng_numpy']; np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda']); history=state['history']; val_rows=state['validation']
        # Keep the selected model when a continuation fails to improve.
        atomic_save(torch.load(a.resume.parent/'best.pth',map_location='cpu',weights_only=True),a.output/'best.pth')
    config.update(mode='full_state_resume' if a.resume else 'fine_tune_from_pretrained',max_steps=a.max_steps,max_minutes=a.max_minutes,val_every=a.val_every,resume=str(a.resume) if a.resume else None)
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    def validate(step):
        model.eval(); scores=[]
        with torch.inference_mode():
            for item,gt in validation:
                tensor=torch.from_numpy(gt).unsqueeze(0).to(device)
                for cr in config['validation_rates']:
                    h,w=torch_matrices(cr,device); pred=model.reconstruct(h@tensor@w.T,(256,256),cr,sensing_matrices=(h,w))[-1,0].cpu().numpy()
                    score=metrics(gt,pred); scores.append(score['psnr']); val_rows.append(dict(step=step,image=item['name'],cr=cr,**score))
        write_csv(a.output/'validation.csv',val_rows); score=float(np.mean(scores)); print(json.dumps(dict(validation_step=step,psnr=score)),flush=True); return score
    def save(step):
        ns=np.random.get_state()
        state=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,best_psnr=best,best_step=best_step,
          rng_python=random.getstate(),rng_numpy=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),
          config=config,history=history,validation=val_rows)
        atomic_save(state,a.output/'last.pth')
        # Reopen and verify a real saved checkpoint, including Adam and frozen factors.
        restored=torch.load(a.output/'last.pth',map_location='cpu',weights_only=True)
        assert restored['step']==step and restored['optimizer']['param_groups'] and all(torch.equal(restored['state_dict'][n],v.cpu()) for n,v in frozen.items())
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    if start_step==0:
        best=validate(0); atomic_save(dict(state_dict=model.state_dict(),step=0,validation_psnr=best,config=config),a.output/'best.pth')
    completed=start_step; stop='step_limit'
    for step in range(start_step,a.max_steps):
        if time.perf_counter()-started>=a.max_minutes*60: stop='time_limit'; break
        model.train(); item=training_item(train,a.seed,step); gt,crop=crop_original(item,a.seed,step)
        tensor=torch.from_numpy(gt).unsqueeze(0).to(device); cr=config['training_rates'][step%len(config['training_rates'])]
        sync(device); began=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        outputs,targets,_=model(tensor,cr)
        weights=[.01]*5+[.95]; loss=sum(w*(outputs[i]-targets[i]).square().mean().clamp_min(1e-12).sqrt() for i,w in enumerate(weights))
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        loss.backward(); grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        if not torch.isfinite(grad): raise RuntimeError('Non-finite gradient')
        optimizer.step(); scheduler.step(); sync(device); completed=step+1
        assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in frozen.items())
        record=dict(step=completed,cr=cr,loss=float(loss.detach()),gradient_norm=float(grad),seconds=time.perf_counter()-began,
                    peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,lr=optimizer.param_groups[0]['lr'],crop=json.dumps(crop))
        history.append(record); print(json.dumps(record),flush=True); write_csv(a.output/'train.csv',history)
        if completed%a.val_every==0 or completed==a.max_steps:
            score=validate(completed)
            if score>best: best=score; best_step=completed; atomic_save(dict(state_dict=model.state_dict(),step=completed,validation_psnr=score,config=config),a.output/'best.pth')
            save(completed)
    if not val_rows or val_rows[-1]['step']!=completed:
        score=validate(completed)
        if score>best: best=score; best_step=completed; atomic_save(dict(state_dict=model.state_dict(),step=completed,validation_psnr=score,config=config),a.output/'best.pth')
    save(completed)
    summary=dict(completed_steps=completed,start_step=start_step,best_step=best_step,best_validation_psnr=best,wall_seconds=time.perf_counter()-started,
                 peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,mean_step_seconds=float(np.mean([r['seconds'] for r in history[-max(1,completed-start_step):]])),stop_reason=stop,
                 checkpoint_sha256={f:sha(a.output/f) for f in ['best.pth','last.pth']},fixed_factors_verified=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (a.output/'COMPLETE').write_text(json.dumps(summary)+'\n'); print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
