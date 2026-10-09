"""Fixed, outcome-independent lab-data protocol. Writes a fresh manifest directory."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def center_y(y):
    h,w=y.shape; side=min(h,w); top=(h-side)//2; left=(w-side)//2
    return cv2.resize(y[top:top+side,left:left+side],(256,256),interpolation=cv2.INTER_AREA), [left,top,side,side]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--datasets',type=Path,default=Path('/datasets')); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    rows=[]; inventory=[]
    def save(y,name,dataset,source,**meta):
        out,crop=center_y(y); path=a.output/(name+'.png'); Image.fromarray(out).save(path)
        rows.append(dict(name=name,dataset=dataset,sequence=meta.pop('sequence',name),split='test',path=path.name,
                         source=str(source),source_sha256=sha(source),sha256=sha(path),source_dimensions=[y.shape[1],y.shape[0]],
                         crop_xywh=crop,preprocessing='center maximal square; OpenCV INTER_AREA to 256x256; stored uint8 Y /255; no range expansion',**meta))
    kodak=sorted((a.datasets/'Kodak').glob('kodim*.png'))
    if len(kodak)!=24: raise ValueError('Expected complete Kodak24')
    for source in kodak:
        rgb=np.asarray(Image.open(source).convert('RGB')); y=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)[:,:,0]
        save(y,source.stem,'Kodak',source,color='OpenCV RGB2YCrCb Y',frame=None)
    # Official HEVC B/E names. Expected byte lengths verify 8-bit planar 4:2:0 packing.
    specs=[('B','BQTerrace',1920,1080,60,600),('B','BasketballDrive',1920,1080,50,500),
           ('B','Cactus',1920,1080,50,500),('B','Kimono1',1920,1080,24,240),('B','ParkScene',1920,1080,24,240),
           ('E','FourPeople',1280,720,60,600),('E','Johnny',1280,720,60,600),('E','KristenAndSara',1280,720,60,600)]
    for cls,name,w,h,fps,n in specs:
        source=a.datasets/'HEVC_test_sequences'/('Class'+cls)/f'{name}_{w}x{h}_{fps}.yuv'
        frame_bytes=w*h*3//2
        nominal_frames=n
        n,remainder=divmod(source.stat().st_size,frame_bytes)
        if remainder or n not in [nominal_frames,nominal_frames+1]: raise ValueError(f'Packing/length mismatch: {source}')
        digest=sha(source); sampled=[]
        with source.open('rb') as stream:
            for frame in [0,n//3,2*n//3]:
                stream.seek(frame*frame_bytes); raw=stream.read(frame_bytes)
                y=np.frombuffer(raw[:w*h],dtype=np.uint8).reshape(h,w)
                observed=dict(y_min=int(y.min()),y_max=int(y.max()),y_p01=float(np.percentile(y,1)),y_p99=float(np.percentile(y,99)))
                out,crop=center_y(y); target=a.output/f'hevc_{name}_f{frame:04d}.png'; Image.fromarray(out).save(target)
                rows.append(dict(name=target.stem,dataset='HEVC_'+cls,sequence=name,split='test',path=target.name,source=str(source),
                                 source_sha256=digest,sha256=sha(target),frame=frame,frame_indexing='zero based',fps=fps,
                                 source_dimensions=[w,h],frames=n,bit_depth=8,pixel_format='yuv420p',stored_range=[0,255],
                                 nominal_video_luma_range=[16,235],range_policy='preserve stored Y codes /255; no inferred tv/full conversion',
                                 frame_sha256=hashlib.sha256(raw).hexdigest(),crop_xywh=crop,
                                 preprocessing='raw Y plane; center maximal square; OpenCV INTER_AREA to 256x256; uint8 /255',**observed))
                sampled.append(observed)
        inventory.append(dict(source=str(source),bytes=source.stat().st_size,sha256=digest,frames=n,nominal_frames=nominal_frames,format='8-bit planar yuv420p',
                              metadata_evidence='canonical filename + integral 8-bit frame packing + nominal frame count (accept one recorded trailing frame) + exact byte length; local extraction script also specifies yuv420p',sampled=sampled))
    train=sorted((a.datasets/'DIV2KTrain/DIV2K_train_HR').glob('*.png')); val=sorted((a.datasets/'DIV2K_valid_HR').glob('*.png'))
    if len(train)!=800 or len(val)!=100: raise ValueError('Unexpected DIV2K inventory')
    rng=np.random.default_rng(20261009); selected=[train[i] for i in sorted(rng.choice(len(train),128,replace=False))]
    validation=[val[i] for i in np.linspace(0,len(val)-1,16,dtype=int)]
    training=[]
    for split,files in [('train',selected),('validation',validation)]:
        for source in files:
            name='div2k_'+source.stem
            if split=='validation':
                rgb=np.asarray(Image.open(source).convert('RGB')); y=cv2.cvtColor(rgb,cv2.COLOR_RGB2YCrCb)[:,:,0]
                out,crop=center_y(y); target=a.output/(name+'.png'); Image.fromarray(out).save(target)
                row=dict(path=target.name,sha256=sha(target),crop_xywh=crop)
            else: row=dict(path=str(source),sha256=sha(source))
            training.append(dict(name=name,dataset='DIV2K',sequence=name,split=split,source=str(source),source_sha256=sha(source),**row))
    all_rows=training+rows; by_split={s:{r['source_sha256'] for r in all_rows if r['split']==s} for s in ['train','validation','test']}
    assert not any(by_split[x]&by_split[y] for x,y in [('train','test'),('validation','test'),('train','validation')])
    (a.output/'test.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.output/'train_val.json').write_text(json.dumps(training,indent=2)+'\n')
    (a.output/'protocol.json').write_text(json.dumps(dict(seed=20261009,training_images=128,validation_images=16,test_images=len(rows),
       hevc_rule='three zero-based frames 0, floor(N/3), floor(2N/3) per whole test sequence; none used for training/selection',
       split_unit='original DIV2K image or entire HEVC video; Kodak reserved for final test',
       attribution='Public benchmark images and laboratory-held HEVC files; no self-capture claim',raw_video_inventory=inventory,
       range_limit='Raw files do not carry self-describing range metadata. Numeric uint8 range and sampled extrema are verified; nominal TV range is a convention, not a measured provenance claim.',
       independent_of_released_training='Disjoint from this fine-tuning/selection only; original pretraining exposure is not fully audited'),indent=2)+'\n')
    print(f'Prepared {len(rows)} test images, 128 train originals, 16 validation images',flush=True)

if __name__=='__main__': main()
