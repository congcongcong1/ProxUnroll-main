"""Check trained state, validation selection, and the original reference across runs."""
import csv
import json
from pathlib import Path
import torch
from scipy.io import loadmat
from coursework.run_experiments import ROOT,sha

BASE=ROOT/'runs/lab220_20261009'


def main():
    last=torch.load(BASE/'finetune/last.pth',map_location='cpu',weights_only=True)
    best=torch.load(BASE/'finetune/best.pth',map_location='cpu',weights_only=True)
    initial=torch.load(ROOT/'weight/hqs_proxunroll.pth',map_location='cpu',weights_only=True); initial=initial.get('state_dict',initial)
    raw=loadmat(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat'); frozen=[]
    changed=[]
    for name,tensor in last['state_dict'].items():
        if name.startswith(('H_','W_')):
            expected=torch.from_numpy(raw[name[0]]).float() if name.endswith('256_256') else initial[name]
            assert torch.equal(tensor,expected),name; frozen.append(name)
        elif not torch.equal(tensor,initial[name]):changed.append(name)
    vals={}
    for row in csv.DictReader((BASE/'finetune/validation.csv').open()):vals.setdefault(int(row['step']),[]).append(float(row['psnr']))
    means={k:sum(v)/len(v) for k,v in vals.items()}; selected=max(means,key=means.get)
    assert selected==best['step']==last['best_step']
    assert abs(means[selected]-last['best_psnr'])<1e-9
    summary=json.loads((BASE/'finetune/summary.json').read_text())
    for name,digest in summary['checkpoint_sha256'].items():assert sha(BASE/'finetune'/name)==digest
    audit=dict(total_steps=last['step'],selected_step=selected,validation_gain=means[selected]-means[0],unchanged_sensing_factors=frozen,
               changed_reconstruction_tensors=len(changed),selected_and_last_parameters_equal=all(torch.equal(v,last['state_dict'][k]) for k,v in best['state_dict'].items()))
    if (BASE/'baseline/COMPLETE').exists() and (BASE/'comparison/COMPLETE').exists():
        key=lambda r:tuple(r[k] for k in ['image','cr','sigma','seed'])
        ref={key(r):r for r in csv.DictReader((BASE/'baseline/metrics.csv').open()) if r['method']=='hqs'}
        repeated=[r for r in csv.DictReader((BASE/'comparison/metrics.csv').open()) if r['method']=='hqs']
        max_psnr=max(abs(float(r['psnr'])-float(ref[key(r)]['psnr'])) for r in repeated)
        max_ssim=max(abs(float(r['ssim'])-float(ref[key(r)]['ssim'])) for r in repeated)
        assert all(r['measurement_sha256']==ref[key(r)]['measurement_sha256'] for r in repeated)
        assert max_psnr<1e-4 and max_ssim<1e-6
        audit.update(repeated_reference_conditions=len(repeated),max_reference_psnr_difference=max_psnr,max_reference_ssim_difference=max_ssim)
    (BASE/'training_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
