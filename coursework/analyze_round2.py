"""Paired, equal-unit MSE reductions against both official and round-1 HQS."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from coursework.run_experiments import write_csv


def analyze(output):
    if not (output/'verification.json').exists():raise ValueError('Verify raw reconstructions before analysis')
    rows=list(csv.DictReader((output/'metrics.csv').open()))
    groups=defaultdict(dict)
    for r in rows:
        key=tuple(r[k] for k in ['image','dataset','sequence','cr','sigma','seed'])
        groups[key][r['method']]=r
    pairs=[]
    for key,methods in groups.items():
        assert set(methods)=={'hqs','hqs_round1','hqs_finetuned'}
        assert len({r['measurement_sha256'] for r in methods.values()})==1
        new=methods['hqs_finetuned'];new_mse=10**(-float(new['psnr'])/10)
        for reference in ['hqs_round1','hqs']:
            old=methods[reference];old_mse=10**(-float(old['psnr'])/10)
            pairs.append(dict(zip(['image','dataset','sequence','cr','sigma','seed'],key),reference=reference,
                              old_mse=old_mse,new_mse=new_mse,mse_reduction_percent=100*(1-new_mse/old_mse),
                              delta_psnr=float(new['psnr'])-float(old['psnr']),delta_ssim=float(new['ssim'])-float(old['ssim'])))
    def aggregate(keys):
        sets=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        for r in pairs:sets[tuple(r[k] for k in keys)][r['sequence']][r['image']].append(r)
        answer=[];rng=np.random.default_rng(20261010)
        for key,videos in sorted(sets.items()):
            units=[]
            for images in videos.values():
                unit={k:float(np.mean([np.mean([r[k] for r in repeated]) for repeated in images.values()]))
                      for k in ['old_mse','new_mse','delta_psnr','delta_ssim']}
                units.append(unit)
            old=np.array([u['old_mse'] for u in units]);new=np.array([u['new_mse'] for u in units])
            indices=rng.integers(len(units),size=(5000,len(units)))
            boot=100*(1-new[indices].mean(axis=1)/old[indices].mean(axis=1))
            reduction=float(100*(1-new.mean()/old.mean()))
            answer.append(dict(zip(keys,key),units=len(units),images=sum(len(v) for v in videos.values()),
                old_mean_mse=float(old.mean()),new_mean_mse=float(new.mean()),mse_reduction_percent=reduction,
                bootstrap_low=float(np.percentile(boot,2.5)),bootstrap_high=float(np.percentile(boot,97.5)),
                delta_psnr=float(np.mean([u['delta_psnr'] for u in units])),delta_ssim=float(np.mean([u['delta_ssim'] for u in units])),
                positive_mse_units=int(np.sum(new<old)),target_at_least_5_percent=reduction>=5))
        return answer
    summary=aggregate(['dataset','cr','sigma','reference'])
    videos=aggregate(['dataset','sequence','cr','sigma','reference'])
    write_csv(output/'mse_paired_deltas.csv',pairs);write_csv(output/'mse_dataset_summary.csv',summary);write_csv(output/'mse_video_summary.csv',videos)
    negative=sorted([r for r in pairs if r['reference']=='hqs_round1' and r['mse_reduction_percent']<0],key=lambda r:r['mse_reduction_percent'])
    if negative:write_csv(output/'round2_negative_cases.csv',negative)
    result=dict(percentage_definition='100*(1 - mean new MSE / mean reference MSE); not percentage change in mean PSNR',
        aggregation='noise repeats within image, images within video, equal videos within dataset; still-image units are originals',
        primary_reference='first-round validation-selected HQS (100 updates)',secondary_reference='official HQS under common MAT operator',
        test_roles={'Kodak':'previously observed regression test','HEVC_B':'previously observed regression test','HEVC_E':'previously observed regression test',
                    'DIV2K_fresh':'32 originals not previously evaluated or used for this fine-tuning selection; original pretrained exposure not fully audited'},
        bootstrap='5000 paired original-image or whole-video resamples; one training seed',datasets=summary,videos=videos,
        total_paired_conditions=len(groups),total_metric_rows=len(rows),negative_image_conditions_vs_round1=len(negative))
    (output/'mse_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([r for r in summary if r['reference']=='hqs_round1' and float(r['cr'])==.1 and float(r['sigma'])==0],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();analyze(a.output)
