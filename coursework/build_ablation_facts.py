"""Collect completed evidence and a small, reproducible failure illustration."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from coursework.run_experiments import ROOT, sha
from coursework.ablation_completion import require_workflow


def collect(base):
    lock = json.loads((base / 'lock.json').read_text())
    selection = json.loads((base / 'evaluation_lock.json').read_text())
    workflow = json.loads((base / 'workflow_status.json').read_text())
    if workflow['phase'] != 'experiments_complete':
        raise ValueError('Experiments are not complete')
    completion = require_workflow(base)
    for name in ['continuation', 'seed_20261013', 'seed_20261014', 'evaluation_smoke']:
        assert int((base / 'logs' / (name + '.exit')).read_text()) == 0, name
    phases = {}
    for phase in ['primary', 'regression']:
        directory = base / ('evaluation_' + phase)
        verification = json.loads((directory / 'verification.json').read_text())
        analysis = json.loads((directory / 'ablation_analysis.json').read_text())
        assert verification['rows'] == analysis['rows']
        assert verification['float_arrays_recomputed'] == analysis['rows']
        assert verification['max_metric_error'] < 2e-5
        phases[phase] = dict(verification=verification, analysis=analysis,
            provenance_sha256=sha(directory / 'provenance.json'),
            metrics_sha256=sha(directory / 'metrics.csv'),
            analysis_sha256=sha(directory / 'ablation_analysis.json'))
    training = {'continuation': json.loads((base / 'continuation/audit.json').read_text())}
    for arm in lock['arms']:
        training[arm['name']] = json.loads((base / 'arms' / arm['name'] / 'audit.json').read_text())
        assert training[arm['name']]['completed_steps'] == 800
    assert training['continuation']['completed_steps'] == 12000
    assert sum(r['new_updates'] for r in training.values()) == 8400
    for seed in [20261013, 20261014]:
        pairing = json.loads((base / f'pairing_{seed}.json').read_text())
        assert pairing['same_image_order_crops_flips_sampling_and_potential_noise']
    # Inspect all candidate conditions, including every measurement repeat.
    directory = base / 'evaluation_primary'
    pairs = list(csv.DictReader((directory / 'paired_image_deltas.csv').open()))
    continuation_pairs = [r for r in pairs if r['reference'] == 'hqs_10000' and r['candidate'] == 'hqs_12000']
    assert len(continuation_pairs) == 52 * 11
    worst = min(continuation_pairs, key=lambda r: float(r['mse_reduction_percent']))
    if float(worst['mse_reduction_percent']) >= 0:
        # An endpoint regression is explicitly labeled; never silently substitute
        # it for the validation-selected model's result.
        endpoint_pairs = [r for r in pairs if r['reference'] == 'hqs_10000' and r['candidate'] == 'last_12000']
        worst = min(endpoint_pairs, key=lambda r: float(r['mse_reduction_percent']))
    meta = json.loads((directory / 'provenance.json').read_text())
    item = next(r for r in meta['data'] if r['name'] == worst['image'])
    source = Path(item['path'])
    if not source.is_absolute():
        manifest = Path(meta['arguments']['manifest'])
        source = (Path(meta['arguments']['data_dir']) if meta['arguments']['data_dir'] else manifest.parent) / source
    assert sha(source) == item['sha256']
    gt = np.asarray(Image.open(source), dtype=np.float32) / 255
    cr, sigma, seed = float(worst['cr']), float(worst['sigma']), int(worst['measurement_seed'])
    illustration = base / 'illustration'
    illustration.mkdir(exist_ok=True)
    Image.open(source).save(illustration / 'reference.png')
    hashes = {'reference.png': sha(illustration / 'reference.png')}
    for name, label in [(worst['reference'], 'old'), (worst['candidate'], 'new')]:
        stem = f'{item["name"]}_cr{cr:g}_noise{sigma:g}_seed{seed}_{name}'
        raw = np.load(directory / 'reconstructions' / (stem + '.npy'))
        assert raw.shape == gt.shape and np.isfinite(raw).all()
        np.save(illustration / (label + '.npy'), raw)
        Image.fromarray(np.round(np.clip(raw, 0, 1) * 255).astype(np.uint8)).save(illustration / (label + '.png'))
        hashes[label + '.npy'] = sha(illustration / (label + '.npy'))
        hashes[label + '.png'] = sha(illustration / (label + '.png'))
    negative_count = sum(float(r['mse_reduction_percent']) < 0 for r in continuation_pairs)
    result = dict(lock=lock, evaluation_selection=selection, training=training, phases=phases, completion=completion,
        continuation_negative_conditions_vs_10000=negative_count,
        continuation_total_conditions=52 * 11, failure_case=dict(worst, source=item,
            illustration_hashes=hashes, selection_rule='smallest per-condition MSE reduction on the primary set; selected model first, endpoint only if selected never regresses',
            interpretation='diagnostic selection after test, never checkpoint selection'),
        figures_derived_from='verified floating metrics; no use of rounded display PNGs for headline metrics')
    bridge_path = base / 'hardware_bridge_overlap_audit.json'
    if completion['evaluations']['regression']['recovered']:
        bridge = json.loads(bridge_path.read_text())
        assert bridge['common_rows'] == 96 and bridge['paired_measurement_hashes_equal']
        assert bridge['max_psnr_difference_db'] < 2e-5 and bridge['max_ssim_difference'] < 2e-5
        result['hardware_bridge_overlap'] = dict(bridge, evidence_sha256=sha(bridge_path))
    (base / 'report_facts.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'formal_updates': 8400, 'primary_rows': phases['primary']['verification']['rows'],
                     'regression_rows': phases['regression']['verification']['rows'],
                     'failure_image': worst['image'], 'negative_continuation_conditions': negative_count}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, default=ROOT / 'runs/hqs_ablation_20261013')
    args = p.parse_args()
    collect(args.base)
