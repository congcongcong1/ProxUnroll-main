"""Regression checks for leakage protection, crop bounds, and video weighting."""
import hashlib
import json
import numpy as np
from PIL import Image
import pytest
from coursework.finetune import crop_original
from coursework.evaluate_manifest import read_manifest, summarize


@pytest.mark.parametrize('size',[(256,1024),(1024,256),(257,257)])
def test_training_crop_is_valid_and_repeatable(tmp_path,size):
    path=tmp_path/'source.png'; Image.new('RGB',size,(120,80,200)).save(path)
    for step in range(8):
        a,info=crop_original({'path':str(path),'name':'sample'},20261009,step)
        b,other=crop_original({'path':str(path),'name':'sample'},20261009,step)
        np.testing.assert_array_equal(a,b); assert info==other
        x,y,w,h=info['xywh']; assert 0<=x<=size[0]-w and 0<=y<=size[1]-h
        assert a.shape==(256,256) and a.dtype==np.float32


def test_manifest_rejects_changed_data(tmp_path):
    path=tmp_path/'test.png'; Image.new('L',(256,256),90).save(path)
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps([dict(name='one',path='test.png',split='test',sha256='0'*64)]))
    with pytest.raises(ValueError,match='hash mismatch'): read_manifest(manifest,tmp_path)


def test_video_mean_does_not_reward_dense_sampling(tmp_path):
    rows=[dict(dataset='HEVC_B',sequence=seq,image=f'{seq}_{i}',cr=.1,sigma=0.,method='hqs',psnr=p,ssim=.5,seconds=1.)
          for seq,count,p in [('long',100,40.),('short',1,20.)] for i in range(count)]
    summarize(rows,tmp_path)
    data=json.loads((tmp_path/'summary.json').read_text())
    assert data['datasets'][0]['psnr']==30. and data['datasets'][0]['units']==2
