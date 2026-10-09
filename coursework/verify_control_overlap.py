"""Check official-HQS overlap across the frozen comparison and fresh controls."""
import argparse,csv,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();b=a.run
    metas=[json.loads((b/name/'provenance.json').read_text()) for name in ['comparison','fresh_baseline']]
    assert metas[0]['matrix_sha256']==metas[1]['matrix_sha256']
    assert metas[0]['checkpoints']['hqs']['sha256']==metas[1]['checkpoints']['hqs']['sha256']
    def rows(name):
        return {tuple(r[k] for k in ['image','cr','sigma','seed']):r for r in csv.DictReader((b/name/'metrics.csv').open()) if r['method']=='hqs' and r['dataset']=='DIV2K_fresh'}
    left,right=rows('comparison'),rows('fresh_baseline');assert left.keys()==right.keys() and len(left)==352
    maxima={m:0. for m in ['psnr','ssim']}
    for key,old in left.items():
        new=right[key];assert old['measurement_sha256']==new['measurement_sha256']
        for m in maxima:maxima[m]=max(maxima[m],abs(float(old[m])-float(new[m])))
    assert maxima['psnr']<2e-5 and maxima['ssim']<2e-6,maxima
    result=dict(conditions=352,identical_measurements_matrix_and_official_checkpoint=True,cross_gpu_metric_max_differences=maxima)
    (b/'control_overlap_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
