"""Independently audit fixed factors, validation-only selection and train noise."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from coursework.finetune import training_item, crop_original
from coursework.finetune_round2 import noisy_measurement
from coursework.run_experiments import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();b=a.run
    config=json.loads((b/'finetune/config.json').read_text())
    assert sha(Path(config['initialization']))==config['initialization_sha256']
    initial=torch.load(Path(config['initialization']),map_location='cpu',weights_only=True)['state_dict']
    best=torch.load(b/'finetune/best.pth',map_location='cpu',weights_only=True);last=torch.load(b/'finetune/last.pth',map_location='cpu',weights_only=True)
    summary=json.loads((b/'finetune/summary.json').read_text())
    assert summary['completed_steps']==last['step']==len(last['history'])
    assert [r['step'] for r in last['history']]==list(range(1,last['step']+1))
    assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in last['history'])
    optimizer_steps=sorted({int(state['step']) for state in last['optimizer']['state'].values()})
    assert optimizer_steps and optimizer_steps[0]>0 and optimizer_steps[-1]==last['step']
    for n,digest in summary['checkpoint_sha256'].items():assert sha(b/'finetune'/n)==digest
    for n,digest in config['source_hashes'].items():assert sha(ROOT/n)==digest
    frozen=[n for n in initial if n.startswith(('H_','W_'))]
    assert len(frozen)==6 and all(torch.equal(state['state_dict'][n],initial[n]) for state in [best,last] for n in frozen)
    changed=sum(not torch.equal(v,last['state_dict'][n]) for n,v in initial.items() if n not in frozen)
    assert changed>0
    values={}
    for r in csv.DictReader((b/'finetune/validation.csv').open()):values.setdefault(int(r['step']),[]).append(r)
    means={s:float(np.mean([float(r['psnr']) for r in rows])) for s,rows in values.items()}
    clean={s:float(np.mean([float(r['psnr']) for r in rows if float(r['sigma'])==0])) for s,rows in values.items()}
    eligible=[s for s in means if clean[s]>=clean[0]-config['clean_guard_db']]
    selected=max(eligible,key=lambda s:means[s]);assert best['step']==last['best_step']==selected
    assert abs(best['validation_psnr']-means[selected])<1e-9
    # Full training manifest comes from the locked data directory, never the test manifest.
    data=json.loads((b/'data/train_val.json').read_text());training=[r for r in data if r['split']=='train']
    assert sha(b/'data/train_val.json')==config['manifest_sha256']
    chosen=sorted(set([0,1,2,len(last['history'])//2,len(last['history'])-1]));checked=[]
    for index in chosen:
        record=last['history'][index];step=record['step']-1;item=training_item(training,config['seed'],step)
        gt,crop=crop_original(item,config['seed'],step)
        assert json.dumps(crop)==record['crop']
        y=noisy_measurement(gt,record['cr'],record['sigma'],[config['seed'],step,8])
        assert hashlib.sha256(y.tobytes()).hexdigest()==record['measurement_sha256'];checked.append(step+1)
    audit=dict(total_steps=last['step'],recorded_updates=len(last['history']),optimizer_step_counts=optimizer_steps,all_losses_gradients_finite=True,audit_source_sha256=sha(Path(__file__).resolve()),selected_step=selected,unchanged_sensing_factors=frozen,changed_reconstruction_tensors=changed,
        selected_combined_validation_gain=means[selected]-means[0],selected_clean_validation_gain=clean[selected]-clean[0],
        train_measurement_hashes_recomputed_at_steps=checked,validation_selection_and_clean_guard_verified=True,
        selected_and_last_parameters_equal=all(torch.equal(v,last['state_dict'][n]) for n,v in best['state_dict'].items()))
    (b/'training_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
