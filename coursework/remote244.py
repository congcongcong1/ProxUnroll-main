"""Run project commands only after verifying the authorized server-244 peer."""
import argparse
import shlex
import subprocess
from coursework.remote220 import ROOT, REMOTE

TARGET='LuoZicong@192.168.1.244'
CONTAINER='luozc_mlvc'
SSH=['ssh','-J','upload','-S','/tmp/mlvc-244-sync.sock','-o','BatchMode=yes','-o','ConnectTimeout=15',TARGET]

def ssh(command,**kw):return subprocess.run(SSH+[command],check=True,**kw)

def check_peer():
    probe=ssh('hostname -I; printf "%s" "$SSH_CONNECTION"',capture_output=True,text=True).stdout
    if '192.168.1.244' not in probe.splitlines()[0].split() or probe.splitlines()[-1].split()[-2]!='192.168.1.244':
        raise RuntimeError('Peer identity does not match authorized server 244')

def run(command,**kw):
    check_peer()
    return ssh(shlex.join(['docker','exec','-i',CONTAINER,'bash','-lc','cd '+REMOTE+' && '+command]),**kw)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command');a=p.parse_args();run(a.command)
