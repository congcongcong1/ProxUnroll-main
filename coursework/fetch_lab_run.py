"""Fetch only this project's named lab run, preserving matching local files."""
import argparse
import hashlib
import io
import shlex
import tarfile
from pathlib import Path
from coursework.remote220 import ssh,CONTAINER,REMOTE,ROOT


def main():
    p=argparse.ArgumentParser(); p.add_argument('--include-arrays',action='store_true'); p.add_argument('--include-weights',action='store_true'); p.add_argument('--parts',help='Comma-separated completed paths within the run')
    p.add_argument('--server',choices=['220','244'],default='220');p.add_argument('--run',default='runs/lab220_20261009');a=p.parse_args()
    if not a.run.startswith('runs/') or Path(a.run).is_absolute() or '..' in Path(a.run).parts:raise ValueError('Invalid run')
    if a.server=='244':
        from coursework.remote244 import ssh as execute,CONTAINER as container,check_peer
    else:
        from coursework.remote220 import ssh as execute,CONTAINER as container,check_peer
    check_peer()
    excludes=[]
    if not a.include_arrays: excludes+=['--exclude=*.npy']
    if not a.include_weights: excludes+=['--exclude=*.pth']
    parts=a.parts.split(',') if a.parts else ['']
    if any(Path(x).is_absolute() or '..' in Path(x).parts for x in parts): raise ValueError('Invalid selected path')
    selected=[a.run+('/'+x if x else '') for x in parts]
    command=shlex.join(['docker','exec',container,'tar','czf','-',*excludes,'-C',REMOTE,*selected])
    data=execute(command,capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        count=0
        for item in archive.getmembers():
            if not item.isfile(): continue
            if not item.name.startswith(a.run+'/') or Path(item.name).is_absolute() or '..' in Path(item.name).parts: raise RuntimeError('Invalid member')
            content=archive.extractfile(item).read(); path=ROOT/item.name; path.parent.mkdir(parents=True,exist_ok=True)
            # Logs/partial CSV may evolve while an authorized run continues.
            if path.exists() and path.read_bytes()==content: continue
            path.write_bytes(content); count+=1
    print(f'Fetched {count} files ({len(data)} archive bytes)')

if __name__=='__main__': main()
