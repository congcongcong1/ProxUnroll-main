"""Compare actual resumed and uninterrupted CUDA training states."""
import argparse
import json
from pathlib import Path
import torch


def assert_same(a,b,path='state'):
    if torch.is_tensor(a):
        if not torch.equal(a,b): raise AssertionError('Tensor mismatch: '+path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for key in a: assert_same(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)): assert_same(x,y,path+f'[{i}]')
    else: assert a==b,(path,a,b)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('resumed',type=Path);p.add_argument('continuous',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    left=torch.load(a.resumed,map_location='cpu',weights_only=True);right=torch.load(a.continuous,map_location='cpu',weights_only=True)
    for key in ['state_dict','optimizer','scheduler','step','rng_python','rng_numpy','rng_torch','rng_cuda','best_psnr','best_step']:
        assert_same(left[key],right[key],key)
    for x,y in zip(left['history'],right['history']):
        for key in ['step','cr','loss','gradient_norm','crop']:assert_same(x[key],y[key],key)
        for key in ['sigma','measurement_sha256']:
            if key in x or key in y:assert_same(x[key],y[key],key)
    for x,y in zip(left['validation'],right['validation']):assert_same(x,y,'validation')
    result=dict(step=left['step'],model_optimizer_scheduler_rng_bitwise_equal=True,crop_loss_validation_equal=True,
                resumed=str(a.resumed),continuous=str(a.continuous))
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
