"""Read compact live state from the verified 244 container; never mutates jobs."""
import shlex
from coursework.remote244 import run
CODE=r'''
import csv,json,math,statistics,re
from pathlib import Path
b=Path('runs/lab244_5000_20261009')
p=b/'logs/finetune_5000.log';lines=p.read_text().splitlines() if p.exists() else []
step=next((json.loads(s)['step'] for s in reversed(lines) if s.startswith('{"step":')),None)
points={}
p=b/'finetune/validation.csv'
if p.exists():
 for r in csv.DictReader(p.open()):
  if r.get('step') and r.get('psnr'):points.setdefault(int(r['step']),[]).append(r)
logged=next((json.loads(s)['validation_step'] for s in reversed(lines) if s.startswith('{"validation_step":')),None)
complete={s:v for s,v in points.items() if len(v)==96 and logged is not None and s<=logged}
latest=max(complete) if complete else None
out={'actual_update_step':step,'latest_complete_validation_step':latest}
if latest is not None:
 initial=complete[0];values=complete[latest]
 out['validation_psnr']={str(sigma):statistics.mean(float(r['psnr']) for r in values if float(r['sigma'])==sigma) for sigma in [0.,.01,.05]}
 out['validation_mean_mse_reduction_at_10pct']={str(sigma):100*(1-statistics.mean(10**(-float(r['psnr'])/10) for r in values if float(r['sigma'])==sigma and float(r['cr'])==.1)/statistics.mean(10**(-float(r['psnr'])/10) for r in initial if float(r['sigma'])==sigma and float(r['cr'])==.1)) for sigma in [0.,.01,.05]}
for job in ['finetune_5000','comparison']:
 p=b/'logs'/f'{job}.exit';out[job+'_exit']=p.read_text().strip() if p.exists() else 'running_or_waiting'
p=b/'finetune/summary.json'
if p.exists():out['completed_training']=json.loads(p.read_text())
p=b/'logs/comparison.log'
if p.exists():
 matches=re.findall(r'([^\s]+): (\d+) rows, ([\d.]+)s',p.read_text())
 if matches:out['comparison_progress']={'last_completed_image':matches[-1][0],'rows':int(matches[-1][1]),'expected_rows':2640,'elapsed_seconds':float(matches[-1][2])}
print(json.dumps(out,ensure_ascii=False))
'''
if __name__=='__main__':run('.venv-linux/bin/python -c '+shlex.quote(CODE))
