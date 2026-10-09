"""Describe every predeclared web-test condition with equal photograph weights."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from coursework.run_experiments import ROOT, write_csv, sha

METHODS = {'adjoint', 'fista_dct', 'hqs', 'admm', 'hqs_round1', 'hqs_5000', 'hqs_10000'}


def analyze(output):
    audit = json.loads((output/'verification.json').read_text())
    assert audit['max_metric_error'] < 2e-5 and audit['data_checkpoint_matrix_source_hashes_verified']
    meta = json.loads((output/'provenance.json').read_text())
    sources = {r['name']:r for r in meta['data']}
    assert len(sources) == 30
    rows = list(csv.DictReader((output/'metrics.csv').open()))
    paired = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ['image', 'cr', 'sigma', 'seed'])
        assert row['method'] not in paired[key]
        paired[key][row['method']] = row
    differences = []
    for (name, cr, sigma, seed), states in paired.items():
        assert set(states) == METHODS
        assert len({r['measurement_sha256'] for r in states.values()}) == 1
        for new_name, old_name in [('hqs_10000', 'hqs'), ('hqs_10000', 'hqs_round1'),
                                   ('hqs_10000', 'hqs_5000'), ('hqs_5000', 'hqs_round1')]:
            old, new = states[old_name], states[new_name]
            old_mse, new_mse = [10**(-float(r['psnr'])/10) for r in [old, new]]
            differences.append(dict(image=name, dataset=sources[name]['dataset'], location=sources[name]['location'],
                cr=float(cr), sigma=float(sigma), seed=int(seed), candidate=new_name, reference=old_name,
                old_mse=old_mse, new_mse=new_mse, mse_reduction_percent=100*(1-new_mse/old_mse),
                delta_psnr=float(new['psnr'])-float(old['psnr']), delta_ssim=float(new['ssim'])-float(old['ssim'])))

    def reduce(records, keys, metrics):
        groups = defaultdict(lambda:defaultdict(list))
        for row in records:
            groups[tuple(row[k] for k in keys)][row['image']].append(row)
        answer = []
        for key, originals in sorted(groups.items()):
            means = {m:float(np.mean([np.mean([float(r[m]) for r in repeat]) for repeat in originals.values()])) for m in metrics}
            entry = dict(zip(keys, key), images=len(originals), **means)
            if 'old_mse' in metrics:
                entry['mse_reduction_percent'] = 100*(1-means['new_mse']/means['old_mse'])
                entry['improved_images'] = sum(np.mean([r['new_mse'] for r in repeat]) < np.mean([r['old_mse'] for r in repeat]) for repeat in originals.values())
                entry['target_at_least_5_percent'] = entry['mse_reduction_percent'] >= 5
            answer.append(entry)
        return answer

    pair_metrics = ['old_mse', 'new_mse', 'delta_psnr', 'delta_ssim']
    datasets = reduce(differences, ['dataset', 'cr', 'sigma', 'candidate', 'reference'], pair_metrics)
    locations = reduce(differences, ['location', 'cr', 'sigma', 'candidate', 'reference'], pair_metrics)
    photographs = reduce(differences, ['image', 'cr', 'sigma', 'candidate', 'reference'], pair_metrics)
    main_records = [dict(row, location=sources[row['image']]['location'],
                         mse=10**(-float(row['psnr'])/10)) for row in rows]
    method_locations = reduce(main_records, ['location', 'cr', 'sigma', 'method'], ['mse', 'psnr', 'ssim', 'seconds'])
    negative = sorted([r for r in differences if r['candidate']=='hqs_10000' and r['mse_reduction_percent']<0],
                      key=lambda r:r['mse_reduction_percent'])
    for filename, records in [('mse_paired_deltas.csv', differences), ('mse_dataset_summary.csv', datasets),
        ('mse_location_summary.csv', locations), ('mse_photo_summary.csv', photographs),
        ('method_location_summary.csv', method_locations), ('negative_cases.csv', negative)]:
        if records: write_csv(output/filename, records)
    result = dict(percentage_definition='100*(1-mean candidate MSE/mean reference MSE); clipped-float MSE = 10**(-PSNR/10)',
        aggregation='average noise repeats within each original photograph, then equal photograph weights; both datasets and every location separately',
        scope='descriptive convenience web collection; related scenes/authors; not independent benchmark sampling or self-capture',
        no_test_selection='catalog, preprocessing and checkpoint hashes locked before inference; no training or checkpoint selection uses these photographs',
        original_pretraining_exposure='not fully audited for publicly available imagery', datasets=datasets, locations=locations,
        method_locations=method_locations, paired_conditions=len(paired), paired_deltas=len(differences),
        negative_10000_conditions_by_reference={ref:sum(r['reference']==ref for r in negative) for ref in ['hqs','hqs_round1','hqs_5000']},
        metrics_csv_sha256=sha(output/'metrics.csv'))
    (output/'mse_analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(conditions=len(paired), negatives=result['negative_10000_conditions_by_reference'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('output', type=Path)
    analyze(p.parse_args().output)
