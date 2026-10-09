"""Predeclare expanded training and a previously unevaluated test split."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from coursework.prepare_lab_data import sha, center_y


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path,default=Path('runs/lab220_20261009/data'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    previous=json.loads((a.previous/'train_val.json').read_text())
    validation=[dict(r,path=str((a.previous/r['path']).resolve())) for r in previous if r['split']=='validation']
    train_sources=sorted(Path('/datasets/DIV2KTrain/DIV2K_train_HR').glob('*.png'))
    if len(train_sources)!=800: raise ValueError('Expected all 800 DIV2K training originals')
    training=[]
    for source in train_sources:
        digest=sha(source)
        training.append(dict(name='div2k_'+source.stem,dataset='DIV2K',sequence=source.stem,split='train',
                             path=str(source),source=str(source),sha256=digest,source_sha256=digest))
    old_test=[dict(r,path=str((a.previous/r['path']).resolve()),role='previously_observed_regression_test')
              for r in json.loads((a.previous/'test.json').read_text())]
    used={r['source'] for r in validation}
    available=[s for s in sorted(Path('/datasets/DIV2K_valid_HR').glob('*.png')) if str(s) not in used]
    if len(available)!=84: raise ValueError('Expected 84 unused validation originals')
    chosen=[available[i] for i in np.linspace(0,83,32,dtype=int)]
    fresh=[]
    for source in chosen:
        rgb=np.asarray(Image.open(source).convert('RGB'))
        y=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)[:,:,0]; out,crop=center_y(y)
        target=a.output/('fresh_div2k_'+source.stem+'.png'); Image.fromarray(out).save(target)
        fresh.append(dict(name=target.stem,dataset='DIV2K_fresh',sequence=target.stem,split='test',path=str(target.resolve()),
                          source=str(source),source_sha256=sha(source),sha256=sha(target),source_dimensions=[y.shape[1],y.shape[0]],
                          crop_xywh=crop,preprocessing='OpenCV RGB2YCrCb Y; center maximal square; INTER_AREA 256x256; uint8 /255',
                          role='not previously evaluated or used in this fine-tuning or selection'))
    groups=[{r['source_sha256'] for r in rows} for rows in [training,validation,old_test,fresh]]
    assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
    (a.output/'train_val.json').write_text(json.dumps(training+validation,indent=2)+'\n')
    (a.output/'test.json').write_text(json.dumps(old_test+fresh,indent=2)+'\n')
    (a.output/'protocol.json').write_text(json.dumps(dict(training_originals=800,selection_originals=16,
        regression_test_images=48,fresh_test_originals=32,fresh_rule='32 evenly indexed originals among the 84 unused DIV2K validation originals',
        test_access='No candidate selection or stopping based on test. Kodak/HEVC were observed in round 1 and are now regression tests.',
        source_split_hashes_disjoint=True,original_pretraining_exposure='not fully audited'),indent=2)+'\n')
    print('Prepared 800 training originals, 16 selection images, 48 regression and 32 fresh test images',flush=True)


if __name__=='__main__':main()
