"""User-authorized 5000-step continuation, preserving the round-2 update rule."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from coursework.baselines import measure
from coursework.evaluate_manifest import read_manifest
from coursework.finetune import crop_original, training_item, atomic_save
from coursework.run_experiments import ROOT, load_model, matrices, torch_matrices, metrics, sha, write_csv, sync


def noisy_measurement(image,cr,sigma,seed_parts):
    h,w=matrices(cr); clean=measure(image,h,w)
    rng=np.random.default_rng(np.random.SeedSequence(seed_parts))
    return (clean+sigma*np.sqrt(np.mean(clean**2))*rng.standard_normal(clean.shape)).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--init',type=Path);mode.add_argument('--resume',type=Path)
    p.add_argument('--max-steps',type=int,default=5000);p.add_argument('--max-minutes',type=float,default=120)
    p.add_argument('--val-every',type=int,default=250);p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--seed',type=int,default=20261010);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--memory-fraction',type=float,default=.65);p.add_argument('--patience',type=int,default=5000)
    a=p.parse_args()
    if not 1<=a.max_steps<=5000 or not 0<a.max_minutes<=120 or a.val_every<1:raise ValueError('User-authorized continuation requires <=5000 total steps and <=120 minutes per continuation')
    if not 0<a.memory_fraction<=.7 or a.patience<1:raise ValueError('Invalid memory/early-stop bound')
    a.output.mkdir(parents=True,exist_ok=False);device=torch.device('cuda');torch.set_num_threads(a.threads)
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    rows=json.loads(a.manifest.read_text());train=[r for r in rows if r['split']=='train']
    validation=read_manifest(a.manifest,a.manifest.parent,'validation')
    assert train and not {r['source_sha256'] for r in train}&{r['source_sha256'] for r,_ in validation}
    for r in train:
        if sha(Path(r['path']))!=r['sha256']:raise ValueError('Training source modified')
    init=a.init or ROOT/'runs/lab220_20261009/finetune/best.pth';model=load_model('hqs',device,a.resume or init)
    raw=loadmat(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat')
    with torch.no_grad():
        for key in ['H','W']:
            expected=torch.as_tensor(raw[key],dtype=torch.float32,device=device)
            if a.resume:assert torch.equal(getattr(model,key+'_256_256'),expected)
            else:getattr(model,key+'_256_256').copy_(expected)
    frozen={}
    for name,param in model.named_parameters():
        if name.startswith(('H_','W_')):param.requires_grad_(False);frozen[name]=param.detach().clone()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=a.lr);scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1.,total_iters=1)
    config=dict(manifest_sha256=sha(a.manifest),matrix_sha256=sha(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat'),
        seed=a.seed,lr=a.lr,training_rates=[.01,.04,.1,.1,.1,.25,.25,.5],training_noise=[0.,0.,.01,.05],
        validation_rates=[.1,.25],validation_noise=[0.,.01,.05],clean_guard_db=.03,patience=a.patience,
        operator='all six sensing factors frozen; common 256x256 MAT',loss='direct GT weighted RMSE: .01 first five stages and .95 final',
        batch_size=1,precision='float32',initialization=str(init),initialization_sha256=sha(init),
        selection='equal-condition mean validation PSNR over 16 images, two rates and three noise levels, subject to initial clean PSNR -0.03dB guard',
        cropping='step-seeded random square and flips; expanded 800 DIV2K originals; no test input',
        source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'coursework/finetune_round2.py',ROOT/'coursework/finetune_5000.py',ROOT/'coursework/finetune.py',ROOT/'model/proxunroll.py',ROOT/'coursework/baselines.py']},
        torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),memory_fraction=a.memory_fraction)
    start_step=0;best=-float('inf');best_step=0;history=[];val_rows=[];initial_clean=None;stale=0
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=True)
        for key in ['manifest_sha256','matrix_sha256','seed','lr','training_rates','training_noise','validation_rates','validation_noise','clean_guard_db']:
            if state['config'][key]!=config[key]:raise ValueError('Resume contract changed: '+key)
        # Existing source contracts stay mandatory. The new driver adds only a
        # bounded budget/stop-policy extension, not a changed gradient update.
        previous_sources=state['config']['source_hashes']
        for name,digest in previous_sources.items():
            if config['source_hashes'].get(name)!=digest:raise ValueError('Resume source changed: '+name)
        if set(config['source_hashes'])-set(previous_sources) not in [set(),{'coursework/finetune_5000.py'}]:
            raise ValueError('Unexpected source contract extension')
        config['budget_extension_from']=dict(source_hashes=previous_sources,max_steps=state['config']['max_steps'],
            max_minutes=state['config']['max_minutes'],patience=state['config']['patience'],
            reason='User explicitly requested 5000 total steps and authorized in-run changes; update rule unchanged')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start_step=state['step'];best=state['best_psnr'];best_step=state['best_step'];initial_clean=state['initial_clean'];stale=state['stale']
        random.setstate(state['rng_python']);ns=state['rng_numpy'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
        torch.set_rng_state(state['rng_torch']);torch.cuda.set_rng_state_all(state['rng_cuda'])
        history=state['history'];val_rows=state['validation']
        atomic_save(torch.load(a.resume.parent/'best.pth',map_location='cpu',weights_only=True),a.output/'best.pth')
    config.update(mode='full_round2_state_resume' if a.resume else 'new_Adam_from_round1_weights',max_steps=a.max_steps,max_minutes=a.max_minutes,
                  val_every=a.val_every,resume=str(a.resume) if a.resume else None)
    (a.output/'config.json').write_text(json.dumps(config,indent=2)+'\n')

    def validate(step):
        model.eval();current=[]
        with torch.inference_mode():
            for item,gt in validation:
                for cr in config['validation_rates']:
                    for sigma in config['validation_noise']:
                        y=noisy_measurement(gt,cr,sigma,[20261010,int(cr*10000),int(sigma*10000),sum(item['name'].encode())])
                        pred=model.reconstruct(torch.from_numpy(y).unsqueeze(0).to(device),(256,256),cr,sensing_matrices=torch_matrices(cr,device))[-1,0].cpu().numpy()
                        current.append(dict(step=step,image=item['name'],cr=cr,sigma=sigma,**metrics(gt,pred)))
        val_rows.extend(current);write_csv(a.output/'validation.csv',val_rows)
        score=float(np.mean([r['psnr'] for r in current]));clean=float(np.mean([r['psnr'] for r in current if r['sigma']==0]))
        strong=float(np.mean([r['psnr'] for r in current if r['sigma']==.05]))
        print(json.dumps(dict(validation_step=step,psnr=score,clean_psnr=clean,strong_noise_psnr=strong)),flush=True)
        return score,clean

    def save(step):
        ns=np.random.get_state()
        state=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=step,best_psnr=best,best_step=best_step,
            initial_clean=initial_clean,stale=stale,rng_python=random.getstate(),rng_numpy=(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),
            rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),config=config,history=history,validation=val_rows)
        atomic_save(state,a.output/'last.pth');restored=torch.load(a.output/'last.pth',map_location='cpu',weights_only=True)
        assert restored['step']==step and restored['optimizer']['param_groups']
        assert all(torch.equal(restored['state_dict'][n],v.cpu()) for n,v in frozen.items())

    def select(step,score,clean):
        nonlocal best,best_step,stale
        if score>best and clean>=initial_clean-config['clean_guard_db']:
            best=score;best_step=step;stale=0
            atomic_save(dict(state_dict=model.state_dict(),step=step,validation_psnr=best,clean_validation_psnr=clean,config=config),a.output/'best.pth')
        else:stale+=1

    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    if start_step==0:
        best,initial_clean=validate(0)
        atomic_save(dict(state_dict=model.state_dict(),step=0,validation_psnr=best,clean_validation_psnr=initial_clean,config=config),a.output/'best.pth')
    completed=start_step;stop='step_limit'
    for step in range(start_step,a.max_steps):
        if time.perf_counter()-started>=a.max_minutes*60:stop='time_limit';break
        model.train();item=training_item(train,a.seed,step);gt,crop=crop_original(item,a.seed,step)
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,step,7]))
        cr=float(rng.choice(config['training_rates']));sigma=float(rng.choice(config['training_noise']))
        y=noisy_measurement(gt,cr,sigma,[a.seed,step,8]);target=torch.from_numpy(gt).unsqueeze(0).to(device)
        sync(device);began=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        outputs=model.reconstruct(torch.from_numpy(y).unsqueeze(0).to(device),(256,256),cr,sensing_matrices=torch_matrices(cr,device))
        loss=sum(w*(outputs[i]-target).square().mean().clamp_min(1e-12).sqrt() for i,w in enumerate([.01]*5+[.95]))
        if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(grad):raise RuntimeError('Non-finite gradient')
        optimizer.step();scheduler.step();sync(device);completed=step+1
        assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in frozen.items())
        history.append(dict(step=completed,cr=cr,sigma=sigma,measurement_sha256=hashlib.sha256(y.tobytes()).hexdigest(),loss=float(loss.detach()),gradient_norm=float(grad),
            seconds=time.perf_counter()-began,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,lr=optimizer.param_groups[0]['lr'],crop=json.dumps(crop)))
        print(json.dumps(history[-1]),flush=True);write_csv(a.output/'train.csv',history)
        if completed%a.val_every==0 or completed==a.max_steps:
            score,clean=validate(completed);select(completed,score,clean);save(completed)
            if stale>=a.patience:stop='validation_patience';break
    if not val_rows or val_rows[-1]['step']!=completed:
        score,clean=validate(completed);select(completed,score,clean)
    save(completed)
    summary=dict(completed_steps=completed,start_step=start_step,best_step=best_step,best_validation_psnr=best,
        initial_clean_validation_psnr=initial_clean,wall_seconds=time.perf_counter()-started,
        peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        mean_step_seconds=float(np.mean([r['seconds'] for r in history[-max(1,completed-start_step):]])),stop_reason=stop,
        checkpoint_sha256={f:sha(a.output/f) for f in ['best.pth','last.pth']},fixed_factors_verified=True)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'COMPLETE').write_text(json.dumps(summary)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
