"""Paired comparisons with immutable checkpoints, raw results and provenance."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from model.proxunroll import ProxUnroll
from coursework.baselines import measure, adjoint, fista_dct


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(solver, device):
    model = ProxUnroll(solver=solver)
    checkpoint = torch.load(ROOT / f'weight/{solver}_proxunroll.pth', map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


@lru_cache(maxsize=16)
def matrices(cr):
    raw = loadmat(ROOT / 'measurement_matrix/blind_learned_256_256_matrices.mat')
    n = int(np.ceil(256 * np.sqrt(cr)))
    return raw['H'][:n].astype(np.float32), raw['W'][:n].astype(np.float32)


def torch_matrices(cr, device):
    return tuple(torch.from_numpy(x).to(device) for x in matrices(cr))


def metrics(gt, pred):
    pred = np.clip(pred, 0, 1)
    return dict(psnr=float(peak_signal_noise_ratio(gt, pred, data_range=1)),
                ssim=float(structural_similarity(gt, pred, data_range=1)))


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refusing to write an empty result table')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--output', type=Path, default=ROOT / 'coursework/results')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'metrics.csv').exists():
        raise FileExistsError('Choose a fresh --output directory to preserve existing results')
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    models = {s: load_model(s, device) for s in ['hqs', 'admm']}
    differences = {key: float((getattr(models['hqs'], key) - getattr(models['admm'], key)).detach().abs().max())
                   for key in ['H_256_256', 'W_256_256']}
    manifest = json.loads((ROOT / 'coursework/data/manifest.json').read_text())
    images = {x['name']: np.asarray(Image.open(ROOT / x['path']), dtype=np.float32) / 255 for x in manifest}
    validation = [x for x in manifest if x['split'] == 'validation']
    test = [x for x in manifest if x['split'] != 'validation']
    if args.smoke:
        test = test[:1]
    h, w = matrices(.1)
    warm_y = torch.from_numpy(measure(images[test[0]['name']], h, w)).unsqueeze(0).to(device)
    with torch.inference_mode():
        for model in models.values():
            model.reconstruct(warm_y, (256, 256), .1, sensing_matrices=torch_matrices(.1, device))
            sync(device)
    tuning = []
    # One global lambda, selected only on the two validation images at three rates.
    for lam in [.001, .005, .02]:
        for item in validation:
            gt = images[item['name']]
            for cr in [.01, .1, .5]:
                h, w = matrices(cr)
                pred = fista_dct(measure(gt, h, w), h, w, lam, 200)
                tuning.append(dict(image=item['name'], cr=cr, lam=lam, **metrics(gt, pred)))
    lam = max([.001, .005, .02], key=lambda v: np.mean([r['psnr'] for r in tuning if r['lam'] == v]))
    write_csv(args.output / 'validation.csv', tuning)
    conditions = [(r, 0., 2026) for r in [.01, .04, .1, .25, .5]]
    conditions += [(.1, noise, seed) for noise in [.01, .05] for seed in [2026, 2027, 2028]]
    if args.smoke:
        conditions = [(.1, 0., 2026)]
    h, w = matrices(.5)
    provenance = dict(
        platform=platform.platform(), processor=platform.processor(), device=str(device),
        torch_threads=args.threads, seed=2026, fista_iterations=200, selected_lambda=lam,
        packages={p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'timm', 'numpy', 'scipy', 'scikit-image', 'opencv-python-headless', 'Pillow']},
        checkpoints={s: sha(ROOT / f'weight/{s}_proxunroll.pth') for s in models},
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [ROOT / name for name in
                       ['coursework/baselines.py', 'coursework/prepare_data.py',
                        'coursework/run_experiments.py', 'model/proxunroll.py']]},
        data=manifest, conditions=conditions, checkpoint_operator_max_differences=differences,
        common_operator='measurement_matrix/blind_learned_256_256_matrices.mat',
        common_operator_sha256=sha(ROOT / 'measurement_matrix/blind_learned_256_256_matrices.mat'),
        common_operator_note='Main results use the same external MAT-file factors for both networks. Checkpoint factors differ. This tests transfer to a common operator; native-operator results are recorded separately.',
        max_row_orthogonality_error=dict(H=float(np.max(np.abs(h @ h.T - np.eye(len(h))))), W=float(np.max(np.abs(w @ w.T - np.eye(len(w)))))),
        timing='Wall time after one full warmup per neural solver. Excludes measurement generation and device transfer. Neural times synchronize the device. One timed execution per input. FISTA on CPU includes spectral norm estimation.',
        noise='E = sigma * RMS(Y_clean) * N(0,1); no measurement clipping; identical Y for all solvers.',
        smoke=args.smoke)
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    rows, stage_rows = [], []
    rec_dir = args.output / 'reconstructions'
    rec_dir.mkdir(exist_ok=True)
    for item in test:
        gt = images[item['name']]
        for cr, sigma, seed in conditions:
            h, w = matrices(cr)
            y = measure(gt, h, w)
            rng = np.random.default_rng(seed + int(cr * 10000) + sum(item['name'].encode()))
            y = (y + sigma * np.sqrt(np.mean(y ** 2)) * rng.standard_normal(y.shape)).astype(np.float32)
            yt = torch.from_numpy(y).unsqueeze(0).to(device)
            factors = torch_matrices(cr, device)
            common = dict(image=item['name'], split=item['split'], cr=cr, actual_cr=y.size / gt.size,
                          sigma=sigma, seed=seed)
            for name in ['adjoint', 'fista_dct', 'hqs', 'admm']:
                sync(device)
                start = time.perf_counter()
                if name == 'adjoint':
                    pred = adjoint(y, h, w)
                elif name == 'fista_dct':
                    pred = fista_dct(y, h, w, lam, 200)
                else:
                    with torch.inference_mode():
                        outputs = models[name].reconstruct(yt, (256, 256), cr, sensing_matrices=factors)
                    sync(device)
                elapsed = time.perf_counter() - start
                if name in models:
                    outputs = outputs[:, 0].cpu().numpy()
                    pred = outputs[-1]
                    for stage, intermediate in enumerate(outputs, 1):
                        stage_rows.append(dict(**common, method=name, stage=stage, **metrics(gt, intermediate)))
                score = metrics(gt, pred)
                residual = np.linalg.norm(measure(np.clip(pred, 0, 1), h, w) - y) / max(np.linalg.norm(y), 1e-12)
                rows.append(dict(**common, method=name, **score, seconds=elapsed, relative_residual=float(residual)))
                if seed == 2026:
                    filename = f'{item["name"]}_cr{cr:g}_noise{sigma:g}_{name}.png'
                    Image.fromarray(np.round(np.clip(pred, 0, 1) * 255).astype(np.uint8)).save(rec_dir / filename)
                print(f'{item["name"]:12} CR={cr:.2f} noise={sigma:.2f} seed={seed} {name:10} PSNR={score["psnr"]:.2f} time={elapsed:.3f}s', flush=True)
            write_csv(args.output / 'metrics.csv', rows)
            write_csv(args.output / 'stages.csv', stage_rows)
    # A separate memory intervention at one rate, never mislabeled as retraining.
    interventions = []
    for item in test:
        gt = images[item['name']]
        h, w = matrices(.1)
        yt = torch.from_numpy(measure(gt, h, w)).unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = models['hqs'].reconstruct(yt, (256, 256), .1, use_memory=False,
                                            sensing_matrices=torch_matrices(.1, device))[-1, 0].cpu().numpy()
        interventions.append(dict(image=item['name'], split=item['split'], method='hqs_no_memory', cr=.1, **metrics(gt, pred)))
    write_csv(args.output / 'interventions.csv', interventions)
    native = []
    for item in test:
        gt = images[item['name']]
        for solver, model in models.items():
            ht, wt, _, _ = model.measurement_matrices(256, 256, .1)
            h, w = ht[0, 0].detach().cpu().numpy(), wt[0, 0].detach().cpu().numpy()
            y = measure(gt, h, w)
            with torch.inference_mode():
                pred = model.reconstruct(torch.from_numpy(y).unsqueeze(0).to(device), (256, 256), .1)[-1, 0].cpu().numpy()
            native.append(dict(image=item['name'], split=item['split'], operator=solver, method=solver, **metrics(gt, pred)))
            baseline = fista_dct(y, h, w, lam, 200)
            native.append(dict(image=item['name'], split=item['split'], operator=solver, method='fista_dct', **metrics(gt, baseline)))
    write_csv(args.output / 'native_operators.csv', native)
    (args.output / 'COMPLETE').write_text(f'{len(rows)} paired metric rows\n')
    print(f'Complete: {len(rows)} metric rows; lambda={lam}')


if __name__ == '__main__':
    main()
