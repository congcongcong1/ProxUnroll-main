"""Authenticated server-220 container commands and non-deleting, conflict-safe sync."""
import argparse, hashlib, io, json, shlex, subprocess, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
TARGET = "LuoZicong@192.168.1.220"
CONTAINER = "luozc_nvenc_new"
REMOTE = "/workspace/ProxUnroll-main"
SSH = ["ssh", "-S", "/tmp/mlvc-220-sync.sock", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", TARGET]

def ssh(command, **kw):
    return subprocess.run(SSH + [command], check=True, **kw)

def check_peer():
    probe = ssh("hostname -I; printf '%s' \"$SSH_CONNECTION\"", capture_output=True, text=True).stdout
    if "192.168.1.220" not in probe.splitlines()[0].split() or probe.splitlines()[-1].split()[-2] != "192.168.1.220":
        raise RuntimeError("Peer identity does not match server 220")


def run(command, **kw):
    check_peer()
    return ssh(shlex.join(["docker", "exec", "-i", CONTAINER, "bash", "-lc", "cd " + REMOTE + " && " + command]), **kw)

def sync():
    # Check the actual peer before writing anything. Never reuse the 244 socket.
    check_peer()
    names = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "coursework").glob("*.py"))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "coursework").glob("*.md"))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "coursework").glob("*.mjs"))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*.py"))
    names.update(str(p.relative_to(ROOT)) for p in ROOT.glob("requirements-linux*.txt"))
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "coursework/archive_original").rglob("*") if p.is_file())
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "output").rglob("*") if p.is_file() and p.suffix in {".pdf", ".pptx", ".md"})
    for run_name in ["lab220_20261009", "lab244_round2_20261009", "lab244_5000_20261009", "lab244_10000_20261009"]:
        run = ROOT / "runs" / run_name
        names.update(str(p.relative_to(ROOT)) for p in (run / "figures").glob("*.png"))
        names.update(str(p.relative_to(ROOT)) for p in [run / x for x in ["report_facts.json", "training_audit.json", "original-verification.json", "server-preflight.txt", "final_artifact_checks.json", "locked_inputs.json", "continuation_lock.json", "startup_verification.json", "resume_verification.json", "delivery_verification.json"]] if p.is_file())
    manifest = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sorted(names) if (ROOT/n).is_file()}
    inventory_code = "import pathlib,json,hashlib,sys; root=pathlib.Path('/workspace/ProxUnroll-main'); print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() if (root/n).is_file() else None for n in json.loads(sys.argv[1])}))"
    actual = json.loads(ssh(shlex.join(["docker", "exec", CONTAINER, "python", "-c", inventory_code, json.dumps(list(manifest))]), capture_output=True, text=True).stdout)
    changed = [n for n in manifest if actual.get(n) != manifest[n]]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name in changed: archive.add(ROOT/name, arcname=name, recursive=False)
    receiver = r"""
import hashlib, io, json, pathlib, sys, tarfile
root=pathlib.Path('/workspace/ProxUnroll-main')
root.mkdir(exist_ok=True)
record=root/'.sync-manifest.json'
previous=json.loads(record.read_text()) if record.exists() else {}
archive=tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz')
desired=json.loads(sys.argv[1])
new={}
for item in archive.getmembers():
    name=item.name
    if not item.isfile() or pathlib.PurePosixPath(name).is_absolute() or '..' in pathlib.PurePosixPath(name).parts: raise RuntimeError('Invalid archive member')
    data=archive.extractfile(item).read(); digest=hashlib.sha256(data).hexdigest()
    dst=root/name
    if dst.exists():
        old=hashlib.sha256(dst.read_bytes()).hexdigest()
        if old != digest and old != previous.get(name): raise RuntimeError('Remote changes preserved; conflict: '+name)
    new[name]=digest
# Validate all conflicts first. Do not remove any remote files.
for name in new:
    dst=root/name; dst.parent.mkdir(parents=True,exist_ok=True)
    data=archive.extractfile(name).read()
    if not dst.exists() or hashlib.sha256(dst.read_bytes()).hexdigest()!=new[name]: dst.write_bytes(data)
record.write_text(json.dumps({**previous,**desired},indent=2)+'\n')
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in desired.items())
print(json.dumps({'matched_files':len(desired),'transferred_files':len(new),'remote':str(root),'bytes':sum((root/n).stat().st_size for n in desired)}))
"""
    ssh(shlex.join(["docker","exec","-i",CONTAINER,"python","-c",receiver,json.dumps(manifest)]),input=buf.getvalue())
    print("All synchronized file hashes verified")

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("action",choices=["sync","run"]); parser.add_argument("command",nargs="?",default="true")
    args=parser.parse_args()
    if args.action=="sync": sync()
    else: run(args.command)
