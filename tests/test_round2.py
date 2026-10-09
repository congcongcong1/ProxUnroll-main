"""Verify noise calibration independently of the training update code."""
import numpy as np
import csv
import json
from coursework.analyze_round2 import analyze
from coursework.baselines import measure
from coursework.run_experiments import matrices
from coursework.finetune_round2 import noisy_measurement


def test_measurement_noise_is_repeatable_and_has_requested_relative_rms():
    image=np.random.default_rng(7).uniform(size=(256,256)).astype(np.float32)
    h,w=matrices(.1);clean=measure(image,h,w)
    np.testing.assert_array_equal(noisy_measurement(image,.1,0.,[17,4]),clean)
    a=noisy_measurement(image,.1,.05,[17,4]);b=noisy_measurement(image,.1,.05,[17,4])
    np.testing.assert_array_equal(a,b)
    ratio=np.sqrt(np.mean((a-clean)**2)/np.mean(clean**2))
    assert .0475<ratio<.0525
    assert not np.array_equal(a,noisy_measurement(image,.1,.05,[17,5]))


def test_mse_reduction_uses_equal_video_weights_and_ratio_of_means(tmp_path):
    (tmp_path/'verification.json').write_text('{}')
    rows=[]
    for sequence,count,old_mse,new_mse in [('long',100,.01,.009),('short',1,.001,.002)]:
        for i in range(count):
            for method,mse in [('hqs',old_mse),('hqs_round1',old_mse),('hqs_finetuned',new_mse)]:
                rows.append(dict(image=f'{sequence}_{i}',dataset='HEVC_B',sequence=sequence,cr=.1,sigma=0,seed=2026,
                    method=method,psnr=-10*np.log10(mse),ssim=.5,measurement_sha256='same'))
    with (tmp_path/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    analyze(tmp_path)
    result=json.loads((tmp_path/'mse_analysis.json').read_text())['datasets']
    primary=next(r for r in result if r['reference']=='hqs_round1')
    assert primary['units']==2 and abs(primary['mse_reduction_percent'])<1e-10
    assert abs(primary['old_mean_mse']-.0055)<1e-10
    assert not primary['target_at_least_5_percent'] and primary['positive_mse_units']==1
