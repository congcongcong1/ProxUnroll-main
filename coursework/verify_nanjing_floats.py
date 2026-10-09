"""Recompute float metrics; distinguish recorded measurements from CPU roundoff."""
import argparse
import csv
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
from coursework.evaluate_manifest import read_manifest
from coursework.run_experiments import ROOT, matrices, metrics, sha
from coursework.baselines import measure
from coursework.verify_lab_run import local_path


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--save-measurements',type=Path)
    p.add_argument('--measurement-archive',type=Path)
    a=p.parse_args();out=a.output.resolve()
    meta=json.loads((out/'provenance.json').read_text())
    manifest=local_path(meta['arguments']['manifest']);directory=local_path(meta['arguments']['data_dir'])
    assert sha(manifest)==meta['manifest_sha256']
    data={r['name']:(r,image) for r,image in read_manifest(manifest,directory)}
    rows=list(csv.DictReader((out/'metrics.csv').open()));groups=defaultdict(list)
    for r in rows:groups[(r['image'],float(r['cr']),float(r['sigma']),int(r['seed']))].append(r)
    archive=np.load(a.measurement_archive,allow_pickle=False) if a.measurement_archive else None
    regenerated={};different=0;max_abs=0.;max_relative=0.;metric_error=0.;array_hashes={}
    for (name,cr,sigma,seed),paired in groups.items():
        gt=data[name][1];h,w=matrices(cr);y=measure(gt,h,w)
        rng=np.random.default_rng(seed+int(cr*10000)+sum(name.encode()))
        y=(y+sigma*np.sqrt(np.mean(y**2))*rng.standard_normal(y.shape)).astype(np.float32)
        stem=f'{name}_cr{cr:g}_noise{sigma:g}_seed{seed}'
        expected={r['measurement_sha256'] for r in paired};assert len(expected)==1
        if archive is not None:
            recorded=archive[stem];assert hashlib.sha256(recorded.tobytes()).hexdigest() in expected
            if not np.array_equal(recorded,y):different+=1
            max_abs=max(max_abs,float(np.max(np.abs(recorded-y))))
            max_relative=max(max_relative,float(np.linalg.norm(recorded-y)/max(np.linalg.norm(recorded),1e-12)))
        else:
            assert hashlib.sha256(y.tobytes()).hexdigest() in expected
            regenerated[stem]=y
        if not a.save_measurements:
            for r in paired:
                path=out/'reconstructions'/(stem+'_'+r['method']+'.npy')
                score=metrics(gt,np.load(path,allow_pickle=False))
                error=max(abs(score[m]-float(r[m])) for m in ['psnr','ssim'])
                assert error<2e-5;metric_error=max(metric_error,error)
                array_hashes[str(path.relative_to(ROOT))]=sha(path)
    if a.save_measurements:
        assert not a.save_measurements.exists()
        a.save_measurements.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.save_measurements,**regenerated)
        receipt=dict(conditions=len(regenerated),every_array_matches_recorded_measurement_hash=True,
                     archive_sha256=sha(a.save_measurements),platform=platform.platform(),numpy_version=np.__version__)
        a.save_measurements.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    else:
        for name,expected in meta['source_hashes'].items():assert sha(ROOT/name)==expected
        for r in meta['checkpoints'].values():assert sha(local_path(r['path']))==r['sha256']
        assert sha(ROOT/'measurement_matrix/blind_learned_256_256_matrices.mat')==meta['matrix_sha256']
        receipt=dict(float_arrays_recomputed=len(array_hashes),max_metric_error=metric_error,
            source_data_checkpoint_matrix_hashes_verified=True,paired_conditions=len(groups),
            recorded_measurement_hashes_verified=archive is not None,
            local_regeneration_bitwise_different_conditions=different,
            local_regeneration_max_absolute_difference=max_abs,
            local_regeneration_max_relative_l2_difference=max_relative,
            interpretation='CPU matrix-product implementations can differ at float32 roundoff; recorded Linux measurements remain common to every method',
            platform=platform.platform(),numpy_version=np.__version__,
            packages={name:importlib.metadata.version(name) for name in ['numpy','scikit-image','scipy','Pillow','torch']},
            reconstruction_sha256=array_hashes)
        if archive is not None:receipt['measurement_archive_sha256']=sha(a.measurement_archive)
        (out/'local_float_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='reconstruction_sha256'},indent=2))


if __name__=='__main__':main()
