"""Real CUDA update/recovery checks before the fixed-budget experiment starts."""
import json
from pathlib import Path
import subprocess
import sys

import torch

from coursework.ablation_suite import BASE, write_json
from coursework.check_resume import assert_same
from coursework.run_experiments import ROOT, sha


def execute(module, options, name):
    log = BASE / 'logs' / (name + '.log')
    command = [sys.executable, '-m', module, *options]
    print(json.dumps({'preflight': name, 'command': command}), flush=True)
    with log.open('w') as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
    if result.returncode:
        print(log.read_text()[-6000:], flush=True)
        raise RuntimeError('Preflight failed: ' + name)


def compare(left, right, output):
    execute('coursework.check_resume', [str(left / 'last.pth'), str(right / 'last.pth'),
            '--output', str(output)], output.stem)


def main():
    BASE.mkdir(exist_ok=True)
    (BASE / 'logs').mkdir(exist_ok=True)
    data = json.loads((ROOT / 'runs/lab244_5000_20261009/data/train_val.json').read_text())
    smoke = [r for r in data if r['split'] == 'train'][:8] + [r for r in data if r['split'] == 'validation'][:2]
    manifest = BASE / 'preflight_manifest.json'
    write_json(manifest, smoke)
    outputs = []
    for noise, supervision in [('on', 'weighted'), ('off', 'weighted'), ('on', 'final'), ('off', 'final')]:
        name = f'preflight_{noise}_{supervision}'
        target = BASE / name
        execute('coursework.finetune_ablation', [
            '--manifest', str(manifest), '--output', str(target), '--seed', '20261013',
            '--noise-augmentation', noise, '--supervision', supervision,
            '--max-steps', '2', '--max-minutes', '5', '--val-every', '1'], name)
        state = torch.load(target / 'last.pth', map_location='cpu', weights_only=True)
        assert state['step'] == 2 and all(torch.isfinite(torch.tensor(r['loss'])) for r in state['history'])
        summary = json.loads((target / 'summary.json').read_text())
        outputs.append({'name': name, 'summary': summary})
    # 2+1 and continuous 3, with validation every step, have identical trajectories.
    common = ['--manifest', str(manifest), '--seed', '20261013', '--max-steps', '3',
              '--max-minutes', '5', '--val-every', '1']
    resumed = BASE / 'preflight_resumed3'
    continuous = BASE / 'preflight_continuous3'
    execute('coursework.finetune_ablation', [*common, '--output', str(resumed),
        '--resume', str(BASE / 'preflight_on_weighted/last.pth')], 'preflight_resumed3')
    execute('coursework.finetune_ablation', [*common, '--output', str(continuous)], 'preflight_continuous3')
    compare(resumed, continuous, BASE / 'preflight_resume_equivalence.json')
    # Use the actual retained step-5 legacy fixture, not a fabricated training state.
    fixture = ROOT / 'runs/lab244_5000_20261009/extension_smoke/last.pth'
    assert torch.load(fixture, map_location='cpu', weights_only=True)['step'] == 5
    options = ['--manifest', str(ROOT / 'runs/lab244_5000_20261009/data/train_val.json'),
               '--resume', str(fixture), '--max-steps', '6', '--max-minutes', '8', '--val-every', '1']
    legacy = BASE / 'preflight_legacy6'
    new = BASE / 'preflight_new6'
    execute('coursework.finetune_10000', [*options, '--output', str(legacy)], 'preflight_legacy6')
    execute('coursework.finetune_ablation', [*options, '--output', str(new),
        '--experiment', 'continuation'], 'preflight_new6')
    compare(legacy, new, BASE / 'preflight_legacy_equivalence.json')
    old_state = torch.load(legacy / 'last.pth', map_location='cpu', weights_only=True)
    new_state = torch.load(new / 'last.pth', map_location='cpu', weights_only=True)
    assert_same(old_state['history'][-1]['measurement_sha256'], new_state['history'][-1]['measurement_sha256'])
    result = dict(preflight_updates=14, all_four_arms_have_real_cuda_updates_and_checkpoints=True,
        ablation_resume_2_plus_1_equals_3=True, legacy_resume_5_to_6_bitwise_equal=True,
        retained_legacy_fixture_sha256=sha(fixture), arms=outputs)
    write_json(BASE / 'preflight_complete.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
