"""Bounded 20000-step evaluation against already verified, locked references."""
import argparse
import csv
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from coursework.run_experiments import ROOT, sha, write_csv
from coursework.analyze_ablation import unit_means

BASE = ROOT / 'runs/hqs_20000_eval_20261014'
OLD = ROOT / 'runs/hqs_ablation_20261013'
TRAIN = ROOT / 'runs/hqs_20000_20261013/finetune'
REFERENCES = ['hqs', 'hqs_10000', 'hqs_12000']
NEW = ['hqs_20000', 'last_20000']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def key(row):
    return row['image'], float(row['cr']), float(row['sigma']), int(row['seed'])


def prepare():
    assert int((TRAIN.parent / 'logs/finetune_20000.exit').read_text()) == 0
    summary = json.loads((TRAIN / 'summary.json').read_text())
    audit = json.loads((TRAIN / 'audit.json').read_text())
    assert summary['completed_steps'] == audit['completed_steps'] == 20000
    assert summary['best_step'] == audit['selected_step'] == 19500
    models = {name: dict(path=str(TRAIN / file), sha256=sha(TRAIN / file))
              for name, file in zip(NEW, ['best.pth', 'last.pth'])}
    for name, file in zip(NEW, ['best.pth', 'last.pth']):
        state = torch.load(TRAIN / file, map_location='cpu', weights_only=True)
        assert state['step'] == (19500 if name == 'hqs_20000' else 20000)
        assert sha(TRAIN / file) == summary['checkpoint_sha256'][file]
    sources = {}
    manifests = {}
    for phase in ['primary', 'regression']:
        directory = OLD / ('evaluation_' + phase)
        verify = json.loads((directory / 'verification.json').read_text())
        meta = json.loads((directory / 'provenance.json').read_text())
        assert verify['float_arrays_recomputed'] == verify['rows'] and verify['max_metric_error'] == 0
        for rel, digest in meta['source_hashes'].items():
            assert sha(ROOT / rel) == digest
        sources[phase] = {name: sha(directory / name) for name in ['metrics.csv', 'verification.json', 'provenance.json']}
        manifests[phase] = meta['arguments']['manifest']
        assert sha(Path(manifests[phase])) == meta['manifest_sha256']
    BASE.mkdir(exist_ok=False)
    result = dict(models=models, reference_metric_hashes=sources, manifests=manifests,
                  selected_step=19500, endpoint_step=20000, selection='unchanged validation rule; both states locked before inference',
                  test_role='same 52-image project holdout and 110-image regression; both now previously observed; no new test-based selection',
                  driver_sha256=sha(Path(__file__)), metric_source_sha256=sha(ROOT / 'coursework/evaluate_manifest.py'),
                  inference_budget_per_phase_seconds=2400, verification_budget_per_phase_seconds=900,
                  created_epoch=time.time())
    write_json(BASE / 'lock.json', result)
    print(json.dumps(result), flush=True)


def evaluate(phase):
    lock = json.loads((BASE / 'lock.json').read_text())
    assert sha(Path(__file__)) == lock['driver_sha256']
    assert sha(ROOT / 'coursework/evaluate_manifest.py') == lock['metric_source_sha256']
    assert os.environ['CUDA_VISIBLE_DEVICES'].startswith('GPU-')
    methods = REFERENCES + NEW if phase == 'smoke' else NEW
    manifest = lock['manifests']['primary' if phase == 'smoke' else phase]
    output = BASE / ('evaluation_' + phase)
    command = [sys.executable, '-m', 'coursework.evaluate_manifest', '--manifest', manifest,
               '--output', str(output), '--methods', ','.join(methods)]
    old = json.loads((OLD / 'evaluation_primary/provenance.json').read_text())
    for name in methods:
        if name == 'hqs':
            continue
        checkpoint = lock['models'][name]['path'] if name in NEW else old['checkpoints'][name]['path']
        command += ['--extra-checkpoint', name + '=' + checkpoint]
    if phase == 'smoke':
        command += ['--smoke']
    print(json.dumps(dict(command=command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, timeout=2400)
    subprocess.run([sys.executable, '-m', 'coursework.verify_lab_run', str(output)], cwd=ROOT, check=True, timeout=900)
    if phase == 'smoke':
        current = list(csv.DictReader((output / 'metrics.csv').open()))
        oldrows = {(*key(r), r['method']): r for r in csv.DictReader((OLD / 'evaluation_primary/metrics.csv').open())}
        errors = []
        for r in current:
            if r['method'] not in REFERENCES:
                continue
            ref = oldrows[(*key(r), r['method'])]
            assert r['measurement_sha256'] == ref['measurement_sha256']
            errors.extend(abs(float(r[m]) - float(ref[m])) for m in ['psnr', 'ssim'])
        assert len(errors) == 6 and max(errors) < 2e-5
        write_json(BASE / 'smoke_reference_equivalence.json', dict(reference_states=3, maximum_metric_difference=max(errors)))


def analyze():
    lock = json.loads((BASE / 'lock.json').read_text())
    results = {}
    for phase, count in [('primary', 52), ('regression', 110)]:
        assert int((BASE / 'logs' / f'{phase}.exit').read_text()) == 0
        directory = BASE / ('evaluation_' + phase)
        olddir = OLD / ('evaluation_' + phase)
        verify = json.loads((directory / 'verification.json').read_text())
        newmeta = json.loads((directory / 'provenance.json').read_text())
        oldmeta = json.loads((olddir / 'provenance.json').read_text())
        for name, digest in lock['reference_metric_hashes'][phase].items():
            assert sha(olddir / name) == digest
        assert newmeta['data'] == oldmeta['data'] and newmeta['conditions'] == oldmeta['conditions']
        assert newmeta['matrix_sha256'] == oldmeta['matrix_sha256']
        assert newmeta['metric_contract'] == oldmeta['metric_contract']
        assert verify['rows'] == verify['float_arrays_recomputed'] == count * 11 * 2
        assert verify['max_metric_error'] == 0
        rows = [r for r in csv.DictReader((olddir / 'metrics.csv').open()) if r['method'] in REFERENCES]
        rows += list(csv.DictReader((directory / 'metrics.csv').open()))
        paired = defaultdict(dict)
        groups = defaultdict(list)
        for r in rows:
            for n in ['cr', 'sigma', 'psnr', 'ssim']:
                r[n] = float(r[n])
            r['mse'] = 10 ** (-r['psnr'] / 10)
            assert r['method'] not in paired[key(r)]
            paired[key(r)][r['method']] = r
            groups[(r['dataset'], r['cr'], r['sigma'], r['method'])].append(r)
        assert len(rows) == count * 11 * 5 and len(paired) == count * 11
        assert all(set(v) == set(REFERENCES + NEW) and len({r['measurement_sha256'] for r in v.values()}) == 1 for v in paired.values())
        write_csv(directory / 'comparison_metrics.csv', rows)
        summaries = []
        video_summaries = []
        for fields, values in sorted(groups.items()):
            units = unit_means(values)
            common = dict(zip(['dataset', 'cr', 'sigma', 'method'], fields))
            summaries.append(dict(**common, units=len(units), images=len({r['image'] for r in values}),
                                  **{m: float(np.mean([u[m] for u in units])) for m in ['mse', 'psnr', 'ssim']}))
            video_summaries.extend(dict(**common, **u) for u in units)
        lookup = {(r['dataset'], r['cr'], r['sigma'], r['method']): r for r in summaries}
        comparisons = []
        for dataset, cr, sigma in sorted({k[:3] for k in lookup}):
            for candidate in NEW:
                for reference in REFERENCES:
                    old, new = lookup[dataset, cr, sigma, reference], lookup[dataset, cr, sigma, candidate]
                    comparisons.append(dict(dataset=dataset, cr=cr, sigma=sigma, candidate=candidate, reference=reference,
                        mse_reduction_percent=100 * (1 - new['mse'] / old['mse']),
                        delta_psnr=new['psnr'] - old['psnr'], delta_ssim=new['ssim'] - old['ssim'], units=old['units']))
        deltas = []
        for condition, methods in paired.items():
            for candidate in NEW:
                for reference in REFERENCES:
                    old, new = methods[reference], methods[candidate]
                    deltas.append(dict(image=condition[0], cr=condition[1], sigma=condition[2], seed=condition[3],
                        dataset=new['dataset'], sequence=new['sequence'], candidate=candidate, reference=reference,
                        mse_reduction_percent=100 * (1 - new['mse'] / old['mse']),
                        delta_psnr=new['psnr'] - old['psnr'], delta_ssim=new['ssim'] - old['ssim']))
        write_csv(directory / 'all_model_summary.csv', summaries)
        write_csv(directory / 'paired_summary.csv', comparisons)
        write_csv(directory / 'video_mse_summary.csv', video_summaries)
        write_csv(directory / 'paired_image_deltas.csv', deltas)
        write_csv(directory / 'negative_cases.csv', [r for r in deltas if r['mse_reduction_percent'] < 0])
        results[phase] = dict(images=count, new_rows=verify['rows'], comparison_rows=len(rows),
                             verification=verify, paired_summary=comparisons,
                             negative_selected_vs_12000=sum(r['candidate'] == 'hqs_20000' and r['reference'] == 'hqs_12000' and r['mse_reduction_percent'] < 0 for r in deltas),
                             all_measurements_matched=True, reference_metrics_reused_from_verified_frozen_run=True)
    write_json(BASE / 'results.json', dict(lock=lock, phases=results,
               interpretation='20000 budget alias uses validation-selected 19500; endpoint reported separately; no checkpoint selected from these tests'))
    (BASE / 'COMPLETE').write_text('Full selected and endpoint evaluation; locked references reused and all measurements checked\n')
    print(json.dumps({k: dict(images=v['images'], new_rows=v['new_rows']) for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'smoke', 'primary', 'regression', 'analyze'])
    action = parser.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'analyze':
        analyze()
    else:
        evaluate(action)
