"""Detached dependency gate: verify training handles, then evaluate locked models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from coursework.ablation_suite import BASE, write_json

JOBS = ['continuation', 'seed_20261013', 'seed_20261014']


def job_state(name):
    exit_path = BASE / 'logs' / (name + '.exit')
    pid_path = BASE / 'logs' / (name + '.pid')
    if exit_path.exists():
        return dict(name=name, terminal=True, exit_code=int(exit_path.read_text()))
    pid = int(pid_path.read_text())
    command_path = Path(f'/proc/{pid}/cmdline')
    if not command_path.exists():
        raise RuntimeError(f'Handle missing without exit record; inspect before recovery: {name}, PID {pid}')
    command = command_path.read_bytes().replace(b'\0', b' ').decode(errors='replace')
    if 'hqs_ablation_20261013' not in command and 'coursework.ablation_suite' not in command:
        raise RuntimeError(f'PID no longer belongs to this job: {name}, PID {pid}')
    return dict(name=name, terminal=False, pid=pid, process_verified=True)


def execute(dataset, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    command = [sys.executable, '-m', 'coursework.evaluate_ablation_suite', '--dataset', dataset]
    stream = (BASE / 'logs' / ('evaluation_' + dataset + '.log')).open('w')
    process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    (BASE / 'logs' / ('evaluation_' + dataset + '.pid')).write_text(str(process.pid) + '\n')
    return process, stream


def require_idle(gpu):
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    memories = {line.split(',')[0].strip(): int(line.split(',')[1].strip()) for line in result.splitlines()}
    if memories[gpu] > 300:
        raise RuntimeError('GPU acquired by another task; evaluation deferred: ' + gpu)


def main():
    started = time.monotonic()
    previous = None
    while True:
        states = [job_state(name) for name in JOBS]
        if any(s.get('exit_code', 0) != 0 for s in states):
            raise RuntimeError('A training/audit job failed; preserved for inspection: ' + json.dumps(states))
        if states != previous:
            print(json.dumps({'waiting_for_training': states}), flush=True)
            previous = states
        write_json(BASE / 'workflow_status.json', dict(phase='waiting_for_training', jobs=states,
                   last_checked_epoch=time.time(), elapsed_seconds=time.monotonic() - started))
        if all(s['terminal'] for s in states):
            break
        if time.monotonic() - started > 10800:
            raise TimeoutError('Training dependency wait budget exceeded; do not restart live jobs')
        time.sleep(20)
    for name in JOBS:
        target = BASE / ('continuation/audit.json' if name == 'continuation' else name + '_complete.json')
        assert target.exists(), target
    gpu_primary = 'GPU-5c29e362-93ff-356b-1b27-82ef64ea1327'
    gpu_regression = 'GPU-16f6ac58-4d8c-8845-0886-1dc146a0327e'
    require_idle(gpu_primary)
    process, stream = execute('smoke', gpu_primary)
    return_code = process.wait(timeout=600)
    stream.close()
    (BASE / 'logs/evaluation_smoke.exit').write_text(str(return_code) + '\n')
    if return_code:
        raise RuntimeError('Minimum matched inference verification failed')
    require_idle(gpu_primary)
    require_idle(gpu_regression)
    jobs = {name: execute(name, gpu) for name, gpu in [('primary', gpu_primary), ('regression', gpu_regression)]}
    deadline = time.monotonic() + 6600
    remaining = set(jobs)
    while remaining:
        for name in sorted(remaining):
            process, stream = jobs[name]
            return_code = process.poll()
            if return_code is not None:
                stream.close()
                (BASE / 'logs' / ('evaluation_' + name + '.exit')).write_text(str(return_code) + '\n')
                remaining.remove(name)
                if return_code:
                    raise RuntimeError('Evaluation failed; independent live evaluator remains untouched: ' + name)
                print(json.dumps({'evaluation_finished': name}), flush=True)
        write_json(BASE / 'workflow_status.json', dict(phase='evaluating', remaining=sorted(remaining),
            evaluation_pids={name: value[0].pid for name, value in jobs.items()}, last_checked_epoch=time.time()))
        if time.monotonic() > deadline:
            # Stop only this workflow's children at its advertised hard budget.
            for name in remaining:
                process, _ = jobs[name]
                process.terminate()
            raise TimeoutError('Evaluation wall-time budget exceeded')
        if remaining:
            time.sleep(20)
    write_json(BASE / 'workflow_status.json', dict(phase='experiments_complete',
        training_jobs=JOBS, evaluation_jobs=list(jobs), last_checked_epoch=time.time()))
    print('TRAINING, AUDITS, MATCHED EVALUATION AND ANALYSIS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
