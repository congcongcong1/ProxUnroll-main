"""Non-deleting upload with an all-file conflict check and SHA-256 receipt."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile

from coursework.run_experiments import ROOT, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', choices=['220', '244'], required=True)
    p.add_argument('--file-list', type=Path, required=True, help='JSON list of repository-relative regular files')
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    if a.server == '244':
        from coursework.remote244 import run
    else:
        from coursework.remote220 import run
    names = json.loads(a.file_list.read_text())
    assert len(names) == len(set(names))
    for name in names:
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert (ROOT/path).is_file() and not (ROOT/path).is_symlink()
        assert not any(x in path.parts for x in ['.git', '.venv', '__pycache__', 'tmp'])
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        for name in names:
            archive.add(ROOT/name, arcname=name, recursive=False)
    remote_code = '''import hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path.cwd()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
    pending=[]
    for member in archive.getmembers():
        rel=pathlib.PurePosixPath(member.name)
        assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
        target=root/member.name
        assert target.resolve().is_relative_to(root.resolve())
        assert not target.is_symlink()
        data=archive.extractfile(member).read()
        if target.exists() and target.read_bytes()!=data:
            raise RuntimeError('Existing remote file differs; preserved: '+member.name)
        pending.append((target,data,member.name))
    for target,data,name in pending:
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
    print(json.dumps({name:hashlib.sha256(target.read_bytes()).hexdigest() for target,data,name in pending}))
'''
    result = run('python3 -c '+shlex.quote(remote_code), input=buf.getvalue(), capture_output=True)
    remote_hashes = json.loads(result.stdout)
    expected = {name:sha(ROOT/name) for name in names}
    assert remote_hashes == expected
    receipt = dict(server=a.server, root='/workspace/ProxUnroll-main', non_deleting=True,
                   existing_different_file_policy='abort before any write', files=expected,
                   compressed_bytes=len(buf.getvalue()), archive_sha256=hashlib.sha256(buf.getvalue()).hexdigest())
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=2)+'\n')
    print(f'Uploaded/verified {len(names)} files; all SHA-256 hashes match')


if __name__ == '__main__':
    main()
