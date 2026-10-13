"""Lock, execute and audit a bounded two-seed, two-factor HQS experiment."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from scipy.io import loadmat

from coursework.evaluate_manifest import read_manifest
from coursework.finetune import crop_original, training_item
from coursework.finetune_ablation import noisy_measurement, training_condition, stage_weights
from coursework.run_experiments import ROOT, sha

BASE = ROOT / 'runs/hqs_ablation_20261013'
SOURCES = [
    'coursework/finetune_ablation.py', 'coursework/ablation_suite.py',
    'coursework/finetune.py', 'coursework/finetune_round2.py',
    'coursework/finetune_5000.py', 'coursework/finetune_10000.py',
    'coursework/evaluate_manifest.py', 'coursework/run_experiments.py',
    'coursework/baselines.py', 'model/proxunroll.py',
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def prepare():
    if (BASE / 'lock.json').exists():
        raise ValueError('Existing experiment lock is immutable')
    manifest = ROOT / 'runs/lab244_5000_20261009/data/train_val.json'
    fresh_path = ROOT / 'runs/lab244_10000_20261009/data/test_fresh.json'
    data = json.loads(manifest.read_text())
    fresh = json.loads(fresh_path.read_text())
    train = [r for r in data if r['split'] == 'train']
    val = [r for r in data if r['split'] == 'validation']
    assert (len(train), len(val), len(fresh)) == (800, 16, 52)
    groups = [{r['source_sha256'] for r in rows} for rows in [train, val, fresh]]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    for r in train:
        assert sha(Path(r['path'])) == r['sha256'] == r['source_sha256']
    read_manifest(manifest, manifest.parent, 'validation')
    read_manifest(fresh_path, fresh_path.parent)
    observed = []
    fresh_hashes = groups[-1]
    # Detect any previously completed inference on the nominally fresh originals.
    for p in (ROOT / 'runs').glob('**/provenance.json'):
        if (p.parent / 'COMPLETE').exists():
            previous = json.loads(p.read_text())
            overlap = fresh_hashes & {r.get('source_sha256') for r in previous.get('data', [])}
            if overlap:
                observed.append({'provenance': str(p.relative_to(ROOT)), 'sources': sorted(overlap)})
    arms = []
    for seed in [20261013, 20261014]:
        for noise, supervision in [('on', 'weighted'), ('off', 'weighted'), ('on', 'final'), ('off', 'final')]:
            arms.append(dict(name=f'noise_{noise}_{supervision}_s{seed}', seed=seed,
                             noise=noise, supervision=supervision, steps=800,
                             max_minutes=35, val_every=200))
    weights = ['weight/hqs_proxunroll.pth',
               'runs/lab244_10000_20261009/finetune/best.pth',
               'runs/lab244_10000_20261009/finetune/last.pth']
    last = torch.load(ROOT / weights[-1], map_location='cpu', weights_only=True)
    assert last['step'] == 10000
    for rel, digest in last['config']['source_hashes'].items():
        assert sha(ROOT / rel) == digest, rel
    lock = dict(
        locked_utc=datetime.now(timezone.utc).isoformat(),
        manifest=str(manifest.relative_to(ROOT)), manifest_sha256=sha(manifest),
        fresh_manifest=str(fresh_path.relative_to(ROOT)), fresh_manifest_sha256=sha(fresh_path),
        fresh_test_role='previously observed regression' if observed else 'previously unused project holdout',
        earlier_fresh_exposure=observed, original_pretraining_exposure='not fully audited',
        data_source_units=dict(train=800, validation=16, primary_test=52),
        source_split_hashes_disjoint=True,
        source_hashes={rel: sha(ROOT / rel) for rel in SOURCES},
        checkpoint_hashes={rel: sha(ROOT / rel) for rel in weights},
        matrix_sha256=sha(ROOT / 'measurement_matrix/blind_learned_256_256_matrices.mat'),
        arms=arms, continuation=dict(start=10000, target=12000, additional_updates=2000,
            max_minutes=65, val_every=250, seed=20261010, lr=1e-5, mode='full legacy state resume'),
        formal_additional_updates=8400, preflight_maximum_updates=30,
        evaluation=dict(clean_rates=[.01, .04, .1, .25, .5], noise_at_rate=.1,
            noise=[.01, .05], measurement_seeds=[2026, 2027, 2028],
            selection='validation only; mixed PSNR with clean initialization -0.03dB guard',
            checkpoint_views=['validation_selected_best', 'fixed_budget_endpoint_last']),
        aggregation='noise repeats within original; originals within whole video; equal source units',
        interpretation='two training seeds, 800-step adaptation from official weights; no claim of full 10000-step causal attribution',
    )
    write_json(BASE / 'lock.json', lock)
    print(json.dumps(lock, indent=2), flush=True)


def check_lock():
    lock = json.loads((BASE / 'lock.json').read_text())
    for rel, digest in {**lock['source_hashes'], **lock['checkpoint_hashes']}.items():
        assert sha(ROOT / rel) == digest, rel
    for name in ['manifest', 'fresh_manifest']:
        assert sha(ROOT / lock[name]) == lock[name + '_sha256']
    assert sha(ROOT / 'measurement_matrix/blind_learned_256_256_matrices.mat') == lock['matrix_sha256']
    return lock


def audit_training(directory):
    config = json.loads((directory / 'config.json').read_text())
    summary = json.loads((directory / 'summary.json').read_text())
    assert (directory / 'COMPLETE').exists()
    best = torch.load(directory / 'best.pth', map_location='cpu', weights_only=True)
    last = torch.load(directory / 'last.pth', map_location='cpu', weights_only=True)
    assert summary['completed_steps'] == last['step'] == config['max_steps']
    assert len(last['history']) == last['step']
    assert [r['step'] for r in last['history']] == list(range(1, last['step'] + 1))
    assert summary['stop_reason'] == 'step_limit'
    assert all(np.isfinite(r['loss']) and np.isfinite(r['gradient_norm']) for r in last['history'])
    optimizer_steps = sorted({int(v['step']) for v in last['optimizer']['state'].values()})
    assert optimizer_steps == [last['step']]
    assert last['scheduler']['last_epoch'] == last['step']
    for rel, digest in config['source_hashes'].items():
        assert sha(ROOT / rel) == digest
    for name, digest in summary['checkpoint_sha256'].items():
        assert sha(directory / name) == digest
    initial = torch.load(config['initialization'], map_location='cpu', weights_only=True)
    initial = initial.get('state_dict', initial)
    frozen = [n for n in initial if n.startswith(('H_', 'W_'))]
    assert len(frozen) == 6
    raw = loadmat(ROOT / 'measurement_matrix/blind_learned_256_256_matrices.mat')
    for name in frozen:
        expected = torch.from_numpy(raw[name[0]].astype(np.float32)) if name.endswith('_256_256') else initial[name]
        assert all(torch.equal(state['state_dict'][name], expected) for state in [best, last]), name
    values = {}
    for row in csv.DictReader((directory / 'validation.csv').open()):
        values.setdefault(int(row['step']), []).append(row)
    scores = {s: float(np.mean([float(r['psnr']) for r in rows])) for s, rows in values.items()}
    clean = {s: float(np.mean([float(r['psnr']) for r in rows if float(r['sigma']) == 0])) for s, rows in values.items()}
    for rows in values.values():
        assert len(rows) == 96 and len({(r['image'], r['cr'], r['sigma']) for r in rows}) == 96
    eligible = [s for s in scores if clean[s] >= last['initial_clean'] - config['clean_guard_db']]
    selected = max(eligible, key=lambda s: scores[s])
    assert best['step'] == last['best_step'] == summary['best_step'] == selected
    assert abs(best['validation_psnr'] - scores[selected]) < 1e-9
    assert abs(last['initial_clean'] - clean[0]) < 1e-9
    train = [r for r in json.loads((ROOT / 'runs/lab244_5000_20261009/data/train_val.json').read_text()) if r['split'] == 'train']
    assert config['manifest_sha256'] == sha(ROOT / 'runs/lab244_5000_20261009/data/train_val.json')
    checked = []
    first = summary['start_step']
    for index in sorted({first, min(first + 1, last['step'] - 1), (first + last['step']) // 2, last['step'] - 1}):
        record = last['history'][index]
        gt, crop = crop_original(training_item(train, config['seed'], index), config['seed'], index)
        assert json.dumps(crop) == record['crop']
        cr, sigma, proposed = training_condition(config['seed'], index, config['training_rates'], config['noise_augmentation'] == 'on')
        assert (cr, sigma, proposed) == (record['cr'], record['sigma'], record['proposed_sigma'])
        y = noisy_measurement(gt, cr, sigma, [config['seed'], index, 8])
        assert hashlib.sha256(y.tobytes()).hexdigest() == record['measurement_sha256']
        checked.append(index + 1)
    changed = sum(not torch.equal(value, last['state_dict'][name]) for name, value in initial.items() if name not in frozen)
    assert changed > 0
    result = dict(completed_steps=last['step'], new_updates=last['step'] - first,
        selected_step=selected, optimizer_step_counts=optimizer_steps,
        finite_loss_and_gradient=True, frozen_sensing_factors=frozen,
        changed_reconstruction_tensors=changed, independent_validation_selection=True,
        checked_crop_and_measurement_steps=checked, checkpoint_hashes=summary['checkpoint_sha256'],
        selected_mixed_validation_gain_db=scores[selected] - scores[0],
        selected_clean_validation_gain_db=clean[selected] - clean[0],
        endpoint_mixed_validation_gain_db=scores[last['step']] - scores[0],
        endpoint_clean_validation_gain_db=clean[last['step']] - clean[0],
        wall_seconds=summary['wall_seconds'], peak_memory_mib=summary['peak_memory_mib'])
    write_json(directory / 'audit.json', result)
    print(json.dumps(result), flush=True)
    return result


def train_seed(seed):
    lock = check_lock()
    arms = [r for r in lock['arms'] if r['seed'] == seed]
    assert len(arms) == 4
    if not os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-'):
        raise ValueError('A named GPU UUID is mandatory')
    results = {}
    for arm in arms:
        output = BASE / 'arms' / arm['name']
        command = [sys.executable, '-m', 'coursework.finetune_ablation',
            '--experiment', 'ablation', '--manifest', str(ROOT / lock['manifest']),
            '--output', str(output), '--init', str(ROOT / 'weight/hqs_proxunroll.pth'),
            '--seed', str(seed), '--noise-augmentation', arm['noise'],
            '--supervision', arm['supervision'], '--max-steps', str(arm['steps']),
            '--max-minutes', str(arm['max_minutes']), '--val-every', str(arm['val_every'])]
        print(json.dumps({'starting_arm': arm, 'command': command}), flush=True)
        log = BASE / 'logs' / (arm['name'] + '.log')
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=arm['max_minutes'] * 60 + 600)
        (BASE / 'logs' / (arm['name'] + '.exit')).write_text(str(result.returncode) + '\n')
        if result.returncode:
            raise RuntimeError('Arm failed; no automatic restart: ' + arm['name'])
        results[arm['name']] = audit_training(output)
    audit_pairing(seed)
    write_json(BASE / f'seed_{seed}_complete.json', results)


def audit_pairing(seed):
    arms = [r for r in check_lock()['arms'] if r['seed'] == seed]
    histories = {}
    for arm in arms:
        state = torch.load(BASE / 'arms' / arm['name'] / 'last.pth', map_location='cpu', weights_only=True)
        histories[arm['name']] = state['history']
    base = histories[f'noise_on_weighted_s{seed}']
    for name, history in histories.items():
        assert len(history) == len(base) == 800
        for reference, row in zip(base, history):
            for key in ['step', 'crop', 'cr', 'proposed_sigma', 'lr']:
                assert row[key] == reference[key], (name, key)
            if 'noise_on_' in name:
                assert row['measurement_sha256'] == reference['measurement_sha256']
            else:
                assert row['sigma'] == 0
    for noise in ['on', 'off']:
        left = histories[f'noise_{noise}_weighted_s{seed}']
        right = histories[f'noise_{noise}_final_s{seed}']
        assert all(a['measurement_sha256'] == b['measurement_sha256'] for a, b in zip(left, right))
    result = dict(seed=seed, steps=800, arms=4,
        same_image_order_crops_flips_sampling_and_potential_noise=True,
        same_measurements_between_supervision_arms=True,
        clean_arms_have_zero_training_noise=True,
        loss_total_weight_one=all(abs(sum(stage_weights(r['supervision'])) - 1) < 1e-12 for r in arms))
    write_json(BASE / f'pairing_{seed}.json', result)
    print(json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'train-seed', 'audit', 'pairing'])
    p.add_argument('--seed', type=int)
    p.add_argument('--directory', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'train-seed':
        train_seed(args.seed)
    elif args.action == 'audit':
        audit_training(args.directory)
    else:
        audit_pairing(args.seed)


if __name__ == '__main__':
    main()
