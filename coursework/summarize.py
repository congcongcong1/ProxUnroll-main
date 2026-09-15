"""Generate report-ready tables and scientific figures from completed runs."""

import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from coursework.run_experiments import ROOT, write_csv


def read_rows(name):
    rows = list(csv.DictReader((ROOT / 'coursework/results' / name).open()))
    for r in rows:
        for k in ['cr', 'actual_cr', 'sigma', 'seed', 'psnr', 'ssim', 'seconds', 'stage']:
            if k in r:
                r[k] = float(r[k])
    return rows


def main():
    result = ROOT / 'coursework/results'
    if not (result / 'COMPLETE').exists():
        raise RuntimeError('A complete experiment run is required')
    output = ROOT / 'coursework/figures'
    output.mkdir(exist_ok=True)
    rows = read_rows('metrics.csv')
    stages = read_rows('stages.csv')
    intervention = read_rows('interventions.csv')
    native = read_rows('native_operators.csv')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['split'], r['cr'], r['sigma'], r['method'])].append(r)
    summary = []
    for (split, cr, sigma, method), rs in sorted(groups.items()):
        by_image = defaultdict(list)
        for r in rs:
            by_image[r['image']].append(r)
        psnr = [np.mean([r['psnr'] for r in rr]) for rr in by_image.values()]
        summary.append(dict(split=split, cr=cr, actual_cr=rs[0]['actual_cr'], sigma=sigma,
                            method=method, images=len(by_image), trials=len(rs),
                            psnr=float(np.mean(psnr)), image_sd=float(np.std(psnr, ddof=1)),
                            ssim=float(np.mean([r['ssim'] for r in rs])),
                            seconds=float(np.median([r['seconds'] for r in rs]))))
    write_csv(result / 'summary.csv', summary)
    colors = {'adjoint': '#777777', 'fista_dct': '#D78924', 'hqs': '#147D78', 'admm': '#B74760'}
    names = {'adjoint': 'Adjoint', 'fista_dct': 'DCT-FISTA', 'hqs': 'HQS', 'admm': 'ADMM'}
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout='constrained')
    for method in colors:
        rs = [r for r in summary if r['split'] == 'test' and r['sigma'] == 0 and r['method'] == method]
        for ax, metric in zip(axes, ['psnr', 'ssim']):
            ax.plot([100*r['actual_cr'] for r in rs], [r[metric] for r in rs], 'o-', label=names[method], color=colors[method])
            ax.set(xlabel='Actual measurements / pixels (%)', ylabel='PSNR (dB)' if metric == 'psnr' else 'SSIM')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(output / 'quality.png', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout='constrained')
    for method in ['fista_dct', 'hqs', 'admm']:
        rs = [r for r in summary if r['split'] == 'test' and r['cr'] == .1 and r['method'] == method]
        axes[0].plot([r['sigma'] for r in rs], [r['psnr'] for r in rs], 'o-', color=colors[method], label=names[method])
    axes[0].set(xlabel='Measurement noise / clean RMS', ylabel='PSNR (dB)')
    axes[0].legend(fontsize=8)
    for method in ['hqs', 'admm']:
        rs = [r for r in stages if r['split'] == 'test' and r['cr'] == .1 and r['sigma'] == 0 and r['method'] == method]
        axes[1].plot(range(1, 7), [np.mean([r['psnr'] for r in rs if r['stage'] == k]) for k in range(1, 7)], 'o-', color=colors[method], label=names[method])
    axes[1].set(xlabel='Restorer stage', ylabel='PSNR (dB)')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(output / 'robustness.png', dpi=220)
    plt.close(fig)
    clean = [r for r in rows if r['split'] == 'test' and r['sigma'] == 0 and r['method'] == 'hqs' and r['cr'] == .1]
    hardest = min(clean, key=lambda r: r['psnr'])['image']
    targets = [('camera', .1), (hardest, .01), ('checkerboard', .1), ('small_text', .01)]
    # Contact sheets are scientific evidence, with fixed range and no retouching.
    for index, (name, cr) in enumerate(targets):
        fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), layout='constrained')
        for ax, method in zip(axes, ['ground_truth', 'fista_dct', 'hqs', 'admm']):
            if method == 'ground_truth':
                path = ROOT / f'coursework/data/{name}.png'
                title = 'Reference'
            else:
                path = result / f'reconstructions/{name}_cr{cr:g}_noise0_{method}.png'
                rr = next(r for r in rows if r['image'] == name and r['cr'] == cr and r['sigma'] == 0 and r['method'] == method)
                title = f'{names[method]} / {rr["psnr"]:.2f} dB'
            ax.imshow(Image.open(path), cmap='gray', vmin=0, vmax=255)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        fig.suptitle(f'{name}: nominal CR = {cr:.0%}', fontsize=11)
        fig.savefig(output / f'comparison_{index+1}.png', dpi=220)
        plt.close(fig)
    no_mem = np.mean([r['psnr'] for r in intervention if r['split'] == 'test'])
    full = np.mean([r['psnr'] for r in clean])
    pairs = []
    for r in clean:
        baseline = next(x for x in rows if x['image'] == r['image'] and x['cr'] == .1 and x['sigma'] == 0 and x['method'] == 'fista_dct')
        pairs.append(r['psnr'] - baseline['psnr'])
    rng = np.random.default_rng(2026)
    boot = rng.choice(pairs, (10000, len(pairs)), replace=True).mean(axis=1)
    facts = dict(summary=summary, hardest_at_10pct=hardest, no_memory_psnr=float(no_mem),
                 hqs_full_psnr=float(full), paired_hqs_fista_mean=float(np.mean(pairs)),
                 paired_hqs_fista_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
                 metric_rows=len(rows), stage_rows=len(stages),
                 stage_means={m: [float(np.mean([r['psnr'] for r in stages if r['split'] == 'test'
                               and r['cr'] == .1 and r['sigma'] == 0 and r['method'] == m and r['stage'] == k]))
                               for k in range(1, 7)] for m in ['hqs', 'admm']},
                 native_means={operator: {method: float(np.mean([r['psnr'] for r in native if r['split'] == 'test'
                               and r['operator'] == operator and r['method'] == method]))
                               for method in [operator, 'fista_dct']} for operator in ['hqs', 'admm']},
                 clean_cases=[r for r in rows if r['sigma'] == 0])
    (result / 'report_facts.json').write_text(json.dumps(facts, indent=2) + '\n')
    print(json.dumps({k:v for k,v in facts.items() if k not in ['summary', 'clean_cases']}, indent=2))


if __name__ == '__main__':
    main()
