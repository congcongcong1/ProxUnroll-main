"""Launch a scoped, logged, detached job inside the verified server-220 container."""
import argparse
import shlex
from coursework.remote220 import run
GPU='GPU-7ad23367-d035-a02e-5d13-8aa855c24fb4'
BASE='runs/lab220_20261009'

def launch(name,command,gpu=GPU,base=BASE,server='220'):
    if not name.replace('_','').isalnum(): raise ValueError('Invalid job name')
    from pathlib import PurePosixPath
    if not base.startswith('runs/') or PurePosixPath(base).is_absolute() or '..' in PurePosixPath(base).parts: raise ValueError('Invalid run directory')
    if server=='244':
        from coursework.remote244 import run as execute
    elif server=='220':execute=run
    else:raise ValueError('Invalid server')
    env=f'export CUDA_VISIBLE_DEVICES={gpu} CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR="$PWD/.cache-linux/matplotlib"; '
    job=env+'set +e; '+command+f'; status=$?; echo "$status" > {base}/logs/{name}.exit; exit "$status"'
    # Separate inner shell keeps ssh from waiting for the detached child.
    script=f'mkdir -p {base}/logs; test ! -e {base}/logs/{name}.pid || exit 1; nohup bash -c '+shlex.quote(job)+f' > {base}/logs/{name}.log 2>&1 < /dev/null & echo $! > {base}/logs/{name}.pid; cat {base}/logs/{name}.pid'
    execute(script)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('command'); p.add_argument('--gpu',default=GPU)
    p.add_argument('--base',default=BASE);p.add_argument('--server',choices=['220','244'],default='220')
    a=p.parse_args(); launch(a.name,a.command,a.gpu,a.base,a.server)
