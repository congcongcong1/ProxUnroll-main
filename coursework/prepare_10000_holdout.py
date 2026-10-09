"""Lock every remaining DIV2K validation original before the next continuation."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from coursework.prepare_lab_data import sha, center_y


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--previous',type=Path,default=Path('runs/lab244_5000_20261009/data'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    train_val=json.loads((a.previous/'train_val.json').read_text())
    observed=json.loads((a.previous/'test.json').read_text())
    used={r['source'] for r in train_val if r['split']=='validation'}|{r['source'] for r in observed}
    sources=sorted(Path('/datasets/DIV2K_valid_HR').glob('*.png'))
    remaining=[s for s in sources if str(s) not in used]
    if len(sources)!=100 or len(remaining)!=52:raise ValueError('Expected exactly 52 previously unused validation originals')
    fresh=[]
    for source in remaining:
        rgb=np.asarray(Image.open(source).convert('RGB'))
        y=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)[:,:,0];out,crop=center_y(y)
        target=a.output/('holdout10000_div2k_'+source.stem+'.png');Image.fromarray(out).save(target)
        fresh.append(dict(name=target.stem,dataset='DIV2K_holdout_10000',sequence=target.stem,split='test',path=str(target.resolve()),
            source=str(source),source_sha256=sha(source),sha256=sha(target),source_dimensions=[y.shape[1],y.shape[0]],
            crop_xywh=crop,preprocessing='OpenCV RGB2YCrCb Y; center maximal square; INTER_AREA 256x256; uint8 /255',
            role='not used in project training, validation selection or any previous project evaluation; released pretraining exposure unknown'))
    groups=[{r['source_sha256'] for r in rows} for rows in [[r for r in train_val if r['split']=='train'],[r for r in train_val if r['split']=='validation'],observed,fresh]]
    # Observed frames share a source hash within each whole video: 24 Kodak,
    # 8 HEVC sources and 32 DIV2K originals, hence 64 original-source units.
    assert [len(g) for g in groups]==[800,16,64,52]
    assert all(not groups[i]&groups[j] for i in range(4) for j in range(i))
    regression=[dict(r,role='previously observed regression test; not used in training or checkpoint selection') for r in observed]
    (a.output/'train_val.json').write_bytes((a.previous/'train_val.json').read_bytes())
    (a.output/'test_fresh.json').write_text(json.dumps(fresh,indent=2)+'\n')
    (a.output/'test_regression.json').write_text(json.dumps(regression,indent=2)+'\n')
    (a.output/'test.json').write_text(json.dumps(regression+fresh,indent=2)+'\n')
    protocol=dict(training_originals=800,selection_originals=16,regression_images=80,new_test_originals=52,
        new_test_rule='all remaining 52 DIV2K official validation originals after excluding the existing 16 selection and 32 observed originals',
        source_split_hashes_disjoint=True,primary_reference='validation-selected 5000-step experiment best.pth at step 4500',
        test_access='No test inference until continuation is complete and the user asks to inspect results',original_pretraining_exposure='not fully audited')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(protocol),flush=True)


if __name__=='__main__':main()
