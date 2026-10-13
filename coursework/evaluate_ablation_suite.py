"""Freeze trained checkpoints, then run matched holdout/regression comparisons."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from coursework.ablation_suite import BASE, check_lock, write_json
from coursework.run_experiments import ROOT, sha


def selection_lock():
    lock = check_lock()
    assert (BASE / 'preflight_complete.json').exists()
    for seed in [20261013, 20261014]:
        assert (BASE / f'seed_{seed}_complete.json').exists()
        assert (BASE / f'pairing_{seed}.json').exists()
        assert int((BASE / 'logs' / f'seed_{seed}.exit').read_text()) == 0
    assert int((BASE / 'logs/continuation.exit').read_text()) == 0
    assert (BASE / 'continuation/audit.json').exists()
    models = {
        'hqs': ROOT / 'weight/hqs_proxunroll.pth',
        'hqs_10000': ROOT / 'runs/lab244_10000_20261009/finetune/best.pth',
        'hqs_12000': BASE / 'continuation/best.pth',
        'last_12000': BASE / 'continuation/last.pth',
    }
    selected_steps = {}
    for arm in lock['arms']:
        directory = BASE / 'arms' / arm['name']
        assert (directory / 'audit.json').exists()
        for view in ['best', 'last']:
            name = view + '_' + arm['name']
            models[name] = directory / (view + '.pth')
        selected_steps[arm['name']] = json.loads((directory / 'audit.json').read_text())['selected_step']
    prior = ROOT / 'runs/lab244_5000_20261009/data/test.json'
    web = ROOT / 'runs/nanjing_web30_20261009/data/test.json'
    regression = []
    for manifest in [prior, web]:
        for row in json.loads(manifest.read_text()):
            path = Path(row['path'])
            path = path if path.is_absolute() else manifest.parent / path
            regression.append(dict(row, path=str(path.resolve()), role='previously observed regression'))
    assert len(regression) == 110 and len({r['name'] for r in regression}) == 110
    regression_manifest = BASE / 'regression_manifest.json'
    content = json.dumps(regression, indent=2) + '\n'
    if regression_manifest.exists():
        assert regression_manifest.read_text() == content
    else:
        regression_manifest.write_text(content)
    result = dict(models={name: {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)} for name, path in models.items()},
        selected_steps=selected_steps, training_lock_sha256=sha(BASE / 'lock.json'),
        fresh_manifest_sha256=lock['fresh_manifest_sha256'],
        regression_manifest_sha256=sha(regression_manifest),
        evaluation_source_hashes={rel: sha(ROOT / rel) for rel in [
            'coursework/evaluate_ablation_suite.py', 'coursework/analyze_ablation.py',
            'coursework/evaluate_manifest.py', 'coursework/verify_lab_run.py']},
        selection='validation only; all selected best and fixed-budget last states retained',
        test_driven_model_selection=False)
    path = BASE / 'evaluation_lock.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        write_json(path, result)
    return lock, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['smoke', 'primary', 'regression'], required=True)
    args = p.parse_args()
    lock, selected = selection_lock()
    models = selected['models']
    if args.dataset == 'regression':
        # Primary evaluation retains both views; regressions use selected bests
        # plus continuation endpoint to avoid duplicating eight endpoint sweeps.
        models = {name: entry for name, entry in models.items() if not name.startswith('last_noise_')}
    manifest = ROOT / lock['fresh_manifest'] if args.dataset == 'primary' else BASE / 'regression_manifest.json'
    output = BASE / ('evaluation_' + args.dataset)
    command = [sys.executable, '-m', 'coursework.evaluate_manifest',
        '--manifest', str(manifest), '--output', str(output), '--methods', ','.join(models)]
    for name, entry in models.items():
        if name != 'hqs':
            command += ['--extra-checkpoint', name + '=' + str(ROOT / entry['path'])]
    if args.dataset == 'smoke':
        command += ['--smoke']
    print(json.dumps({'evaluation_command': command}), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, timeout=5400)
    subprocess.run([sys.executable, '-m', 'coursework.verify_lab_run', str(output)], cwd=ROOT, check=True, timeout=900)
    if args.dataset != 'smoke':
        subprocess.run([sys.executable, '-m', 'coursework.analyze_ablation', str(output)], cwd=ROOT, check=True, timeout=300)
    print(json.dumps({'completed': args.dataset, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
