"""Equal-source evaluation and paired factorial effects, preserving all regressions."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from coursework.run_experiments import write_csv


SEEDS = [20261013, 20261014]
FACTORS = [('on', 'weighted'), ('off', 'weighted'), ('on', 'final'), ('off', 'final')]


def unit_means(rows):
    sequences = defaultdict(lambda: defaultdict(list))
    for row in rows:
        sequences[row['sequence']][row['image']].append(row)
    units = []
    for sequence, images in sorted(sequences.items()):
        metrics = {}
        for metric in ['mse', 'psnr', 'ssim']:
            metrics[metric] = float(np.mean([np.mean([r[metric] for r in repeats]) for repeats in images.values()]))
        units.append(dict(sequence=sequence, images=len(images), **metrics))
    return units


def analyze(directory):
    verification = json.loads((directory / 'verification.json').read_text())
    assert verification['data_checkpoint_matrix_source_hashes_verified']
    rows = list(csv.DictReader((directory / 'metrics.csv').open()))
    assert len(rows) == verification['rows']
    conditions = defaultdict(dict)
    for r in rows:
        for name in ['cr', 'sigma', 'psnr', 'ssim']:
            r[name] = float(r[name])
        r['mse'] = 10 ** (-r['psnr'] / 10)
        key = (r['image'], r['dataset'], r['sequence'], r['cr'], r['sigma'], r['seed'])
        assert r['method'] not in conditions[key]
        conditions[key][r['method']] = r
    for methods in conditions.values():
        assert len({r['measurement_sha256'] for r in methods.values()}) == 1
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r['cr'], r['sigma'], r['method'])].append(r)
    summaries = []
    grouped = {}
    for key, values in sorted(groups.items()):
        units = unit_means(values)
        row = dict(zip(['dataset', 'cr', 'sigma', 'method'], key), units=len(units),
            images=sum(u['images'] for u in units),
            **{m: float(np.mean([u[m] for u in units])) for m in ['mse', 'psnr', 'ssim']})
        summaries.append(row)
        grouped[key] = row
    write_csv(directory / 'all_model_summary.csv', summaries)

    pairs = []
    for key, methods in conditions.items():
        common = dict(zip(['image', 'dataset', 'sequence', 'cr', 'sigma', 'measurement_seed'], key))
        specs = []
        for reference in ['hqs', 'hqs_10000']:
            for candidate in methods:
                if candidate != reference:
                    specs.append((reference, candidate, 'model_vs_' + reference, None, None))
        for view in ['best', 'last']:
            for seed in SEEDS:
                names = {(noise, loss): f'{view}_noise_{noise}_{loss}_s{seed}' for noise, loss in FACTORS}
                for loss in ['weighted', 'final']:
                    specs.append((names['off', loss], names['on', loss], 'noise_effect_at_' + loss, view, seed))
                for noise in ['on', 'off']:
                    specs.append((names[noise, 'final'], names[noise, 'weighted'], 'stage_effect_at_noise_' + noise, view, seed))
        for reference, candidate, comparison, view, seed in specs:
            if reference not in methods or candidate not in methods:
                continue
            old, new = methods[reference], methods[candidate]
            pairs.append(dict(**common, reference=reference, candidate=candidate,
                comparison=comparison, checkpoint_view=view or '', training_seed=seed or '',
                old_mse=old['mse'], new_mse=new['mse'],
                mse_reduction_percent=100 * (1 - new['mse'] / old['mse']),
                delta_psnr=new['psnr'] - old['psnr'], delta_ssim=new['ssim'] - old['ssim']))
    write_csv(directory / 'paired_image_deltas.csv', pairs)
    negative = sorted((r for r in pairs if r['mse_reduction_percent'] < 0), key=lambda r: r['mse_reduction_percent'])
    write_csv(directory / 'negative_cases.csv', negative) if negative else (directory / 'negative_cases.csv').write_text('comparison,mse_reduction_percent\n')
    pair_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fields = ['dataset', 'cr', 'sigma', 'reference', 'candidate', 'comparison', 'checkpoint_view', 'training_seed']
    for r in pairs:
        pair_groups[tuple(r[k] for k in fields)][r['sequence']][r['image']].append(r)
    pair_summary = []
    for key, sequences in sorted(pair_groups.items(), key=lambda item: str(item[0])):
        units = []
        for images in sequences.values():
            units.append({m: float(np.mean([np.mean([r[m] for r in repeats]) for repeats in images.values()]))
                          for m in ['old_mse', 'new_mse', 'delta_psnr', 'delta_ssim']})
        old = float(np.mean([u['old_mse'] for u in units]))
        new = float(np.mean([u['new_mse'] for u in units]))
        pair_summary.append(dict(zip(fields, key), units=len(units),
            old_mean_mse=old, new_mean_mse=new, mse_reduction_percent=100 * (1 - new / old),
            delta_psnr=float(np.mean([u['delta_psnr'] for u in units])),
            delta_ssim=float(np.mean([u['delta_ssim'] for u in units])),
            improved_units=sum(u['new_mse'] < u['old_mse'] for u in units)))
    write_csv(directory / 'paired_summary.csv', pair_summary)
    effects = []
    dataset_conditions = sorted({(r['dataset'], r['cr'], r['sigma']) for r in summaries})
    for dataset, cr, sigma in dataset_conditions:
        official = grouped[dataset, cr, sigma, 'hqs']['mse']
        for view in ['best', 'last']:
            per_seed = []
            for seed in SEEDS:
                names = {(noise, loss): f'{view}_noise_{noise}_{loss}_s{seed}' for noise, loss in FACTORS}
                if not all((dataset, cr, sigma, name) in grouped for name in names.values()):
                    continue
                mse = {factor: grouped[dataset, cr, sigma, name]['mse'] for factor, name in names.items()}
                noise_w = mse['off', 'weighted'] - mse['on', 'weighted']
                noise_f = mse['off', 'final'] - mse['on', 'final']
                stage_on = mse['on', 'final'] - mse['on', 'weighted']
                stage_off = mse['off', 'final'] - mse['off', 'weighted']
                changes = dict(noise_main_mse_benefit=(noise_w + noise_f) / 2,
                    stage_main_mse_benefit=(stage_on + stage_off) / 2,
                    noise_stage_interaction_mse_benefit=noise_w - noise_f)
                per_seed.append(changes)
                for effect, value in changes.items():
                    effects.append(dict(dataset=dataset, cr=cr, sigma=sigma, checkpoint_view=view,
                        training_seed=seed, effect=effect, mse_benefit=value,
                        official_normalized_percent_points=100 * value / official))
            if len(per_seed) == 2:
                for effect in per_seed[0]:
                    values = [r[effect] for r in per_seed]
                    effects.append(dict(dataset=dataset, cr=cr, sigma=sigma, checkpoint_view=view,
                        training_seed='mean_of_two', effect=effect, mse_benefit=float(np.mean(values)),
                        official_normalized_percent_points=100 * float(np.mean(values)) / official))
    write_csv(directory / 'factorial_effects.csv', effects)
    seed_groups = defaultdict(list)
    for row in pair_summary:
        if row['training_seed']:
            seed_groups[(row['dataset'], row['cr'], row['sigma'], row['checkpoint_view'], row['comparison'])].append(row)
    stability = []
    for key, group in sorted(seed_groups.items()):
        assert len(group) == 2
        values = [r['mse_reduction_percent'] for r in group]
        stability.append(dict(zip(['dataset', 'cr', 'sigma', 'checkpoint_view', 'comparison'], key),
            training_seeds=2, mean_mse_reduction_percent=float(np.mean(values)),
            minimum_mse_reduction_percent=min(values), maximum_mse_reduction_percent=max(values),
            same_direction_both_seeds=all(v > 0 for v in values) or all(v < 0 for v in values)))
    write_csv(directory / 'seed_effect_stability.csv', stability)
    result = dict(rows=len(rows), paired_conditions=len(conditions), comparisons=len(pairs),
        negative_comparisons=len(negative), summaries=summaries, paired_summary=pair_summary,
        factorial_effects=effects, seed_stability=stability,
        metric_definition='MSE=10^(-PSNR/10); percent reduction is ratio of equal-source mean MSE',
        factor_definition='positive MSE benefit means adding the factor reduces error; main effect averages the other factor; interaction=noise benefit with weighted loss minus noise benefit with final loss',
        normalization='factorial raw-MSE benefits normalized by the same official mean MSE, reported in percent points',
        limitations='two training seeds; selected and fixed-budget endpoint views kept separate; 800-step ablations do not identify all effects of prior 10000-step recipe')
    (directory / 'ablation_analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'conditions': len(conditions), 'negative_comparisons': len(negative)}))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    analyze(args.directory)
