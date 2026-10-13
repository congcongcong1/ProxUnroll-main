"""Complete only missing originals after a terminal evaluation timeout.

Never modifies the locked evaluators or rewrites a failed exit record. The
partial directory is retained in an archive; the merged directory links to
its complete originals and a separate, bounded recovery run.
"""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from coursework.run_experiments import ROOT, sha, write_csv
from coursework.evaluate_manifest import summarize

BASE = ROOT / 'runs/hqs_ablation_20261013'


def identity(row):
    return (row['image'], float(row['cr']), float(row['sigma']), int(row['seed']), row['method'])


def complete_originals(rows, meta):
    """Require every locked condition and method, not just a row count."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['image']].append(row)
    methods = meta['arguments']['methods'].split(',')
    complete = set()
    for item in meta['data']:
        name = item['name']
        expected = {(name, float(cr), float(sigma), int(seed), method)
                    for cr, sigma, seed in meta['conditions'] for method in methods}
        keys = [identity(r) for r in grouped[name]]
        if len(keys) == len(expected) and set(keys) == expected:
            complete.add(name)
    return complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['primary', 'regression'], required=True)
    parser.add_argument('--max-minutes', type=int, default=45)
    a = parser.parse_args()
    assert 1 <= a.max_minutes <= 60
    phase = a.phase
    original = BASE / ('evaluation_' + phase)
    exit_path = BASE / 'logs' / ('evaluation_' + phase + '.exit')
    assert exit_path.exists() and int(exit_path.read_text()) != 0, 'Recovery requires a terminal failed exit'
    assert not (original / 'COMPLETE').exists(), 'A completed run needs diagnosis, not inference recovery'
    for pid_file in [BASE / 'logs' / ('evaluation_' + phase + '.pid')]:
        pid = int(pid_file.read_text())
        assert not Path(f'/proc/{pid}/cmdline').exists(), 'Original evaluator still live'
    # A timed-out subprocess may still have an orphan; refuse to duplicate it.
    processes = subprocess.check_output(['ps', '-eo', 'args'], text=True).splitlines()
    assert not any('coursework.evaluate_manifest' in cmd and str(original) in cmd for cmd in processes)
    assert not any('coursework.evaluate_manifest' in cmd and str(original.relative_to(ROOT)) in cmd for cmd in processes)
    meta = json.loads((original / 'provenance.json').read_text())
    selected = json.loads((BASE / 'evaluation_lock.json').read_text())
    for relative, digest in selected['evaluation_source_hashes'].items():
        assert sha(ROOT / relative) == digest
    for name, checkpoint in meta['checkpoints'].items():
        assert checkpoint['sha256'] == selected['models'][name]['sha256']
        assert sha(Path(checkpoint['path'])) == checkpoint['sha256']
    rows = list(csv.DictReader((original / 'metrics.csv').open()))
    complete = complete_originals(rows, meta)
    missing = [item for item in meta['data'] if item['name'] not in complete]
    assert missing and len(missing) <= 35, 'Unexpectedly large missing portion; inspect before allocating more time'
    data_root = Path(meta['arguments']['data_dir']) if meta['arguments']['data_dir'] else Path(meta['arguments']['manifest']).parent
    missing = [dict(item, path=str(Path(item['path']) if Path(item['path']).is_absolute()
                                  else data_root / item['path'])) for item in missing]
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    assert gpu.startswith('GPU-'), 'Named GPU required'
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    memory = int(next(line.split(',')[1] for line in usage.splitlines() if line.startswith(gpu)))
    assert memory < 300, 'Recovery GPU is occupied; defer inference'
    recovery = BASE / ('recovery_' + phase + '_1')
    recovery.mkdir(exist_ok=False)
    manifest = recovery / 'missing_originals.json'
    manifest.write_text(json.dumps(missing, indent=2) + '\n')
    hashes = {name: sha(original / name) for name in ['metrics.csv', 'stages.csv', 'provenance.json']}
    protocol = dict(phase=phase, original_exit_code=int(exit_path.read_text()),
        complete_originals_reused=sorted(complete), incomplete_originals_recomputed=[r['name'] for r in missing],
        partial_csv_hashes=hashes, max_minutes=a.max_minutes,
        locked_weights_source_and_measurements=True, training_restarted=False)
    (recovery / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    output = recovery / 'inference'
    command = [sys.executable, '-m', 'coursework.evaluate_manifest', '--manifest', str(manifest),
               '--output', str(output), '--methods', meta['arguments']['methods']]
    for name, checkpoint in meta['checkpoints'].items():
        if name != 'hqs':
            command += ['--extra-checkpoint', name + '=' + checkpoint['path']]
    print(json.dumps(dict(command=command, protocol=protocol)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True, timeout=a.max_minutes * 60)
    subprocess.run([sys.executable, '-m', 'coursework.verify_lab_run', str(output)],
                   cwd=ROOT, check=True, timeout=900)
    newmeta = json.loads((output / 'provenance.json').read_text())
    assert newmeta['checkpoints'] == meta['checkpoints'] and newmeta['conditions'] == meta['conditions']
    for name, digest in hashes.items():
        assert sha(original / name) == digest, 'Terminal partial data changed during recovery'
    newrows = list(csv.DictReader((output / 'metrics.csv').open()))
    assert complete_originals(newrows, newmeta) == {item['name'] for item in missing}
    finalrows = [r for r in rows if r['image'] in complete] + newrows
    assert complete_originals(finalrows, meta) == {item['name'] for item in meta['data']}
    archive = BASE / ('evaluation_' + phase + '_terminal_partial')
    assert not archive.exists()
    original.rename(archive)
    original.mkdir()
    linked = original / 'reconstructions'
    linked.mkdir()
    sources = {item['name']: archive if item['name'] in complete else output for item in meta['data']}
    for row in finalrows:
        image, cr, sigma, seed, method = identity(row)
        stem = f'{image}_cr{cr:g}_noise{sigma:g}_seed{seed}_{method}'
        for extension in ['.npy', '.png']:
            target = sources[image] / 'reconstructions' / (stem + extension)
            if extension == '.npy':
                assert target.exists()
            if target.exists():
                (linked / target.name).symlink_to(os.path.relpath(target, linked))
    write_csv(original / 'metrics.csv', finalrows)
    stages = [r for r in csv.DictReader((archive / 'stages.csv').open()) if r['image'] in complete]
    stages += list(csv.DictReader((output / 'stages.csv').open()))
    assert len(stages) == len(finalrows) * 6
    write_csv(original / 'stages.csv', stages)
    meta['recovery'] = dict(protocol, archived_directory=str(archive), inference_directory=str(output),
                            recovery_driver_sha256=sha(Path(__file__)))
    (original / 'provenance.json').write_text(json.dumps(meta, indent=2) + '\n')
    numericrows = [{**r, **{field: float(r[field]) for field in ['cr', 'sigma', 'psnr', 'ssim', 'seconds']}}
                   for r in finalrows]
    summarize(numericrows, original)
    (original / 'COMPLETE').write_text(f'{len(finalrows)} rows combined from terminal partial and bounded missing-original recovery\n')
    subprocess.run([sys.executable, '-m', 'coursework.verify_lab_run', str(original)], cwd=ROOT, check=True, timeout=900)
    subprocess.run([sys.executable, '-m', 'coursework.analyze_ablation', str(original)], cwd=ROOT, check=True, timeout=300)
    protocol.update(recovered_rows=len(finalrows), archived_directory=str(archive), completed_epoch=time.time(),
                    original_failed_exit_preserved=int(exit_path.read_text()))
    (BASE / ('evaluation_' + phase + '_recovery_complete.json')).write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(protocol), flush=True)


if __name__ == '__main__':
    main()
