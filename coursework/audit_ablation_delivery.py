"""Independent completion audit of training, paired metrics and versioned materials."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from pypdf import PdfReader

from coursework.run_experiments import ROOT, sha
from coursework.ablation_completion import require_workflow

BASE = ROOT / 'runs/hqs_ablation_20261013'


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def means(rows, field):
    videos = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = 10 ** (-float(row['psnr']) / 10) if field == 'mse' else float(row[field])
        videos[row['sequence']][row['image']].append(value)
    values = [float(np.mean([np.mean(v) for v in originals.values()])) for originals in videos.values()]
    return float(np.mean(values)), len(values)


def audit(experiments_only=False):
    lock = json.loads((BASE / 'lock.json').read_text())
    selection = json.loads((BASE / 'evaluation_lock.json').read_text())
    facts = json.loads((BASE / 'report_facts.json').read_text())
    check(sha(BASE / 'lock.json') == selection['training_lock_sha256'], 'Training lock changed')
    for relative, digest in lock['source_hashes'].items():
        check(sha(ROOT / relative) == digest, 'Training source changed: ' + relative)
    for relative, digest in selection['evaluation_source_hashes'].items():
        check(sha(ROOT / relative) == digest, 'Evaluation source changed: ' + relative)
    check(json.loads((BASE / 'workflow_status.json').read_text())['phase'] == 'experiments_complete', 'Workflow incomplete')
    total_updates = 0
    for name, record in facts['training'].items():
        directory = BASE / ('continuation' if name == 'continuation' else 'arms/' + name)
        history = list(csv.DictReader((directory / 'train.csv').open()))
        config = json.loads((directory / 'config.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        audit_record = json.loads((directory / 'audit.json').read_text())
        expected = 12000 if name == 'continuation' else 800
        check(len(history) == expected and [int(r['step']) for r in history] == list(range(1, expected + 1)), 'Update history incomplete: ' + name)
        check(summary['completed_steps'] == audit_record['completed_steps'] == expected, 'Checkpoint update count: ' + name)
        check(audit_record['optimizer_step_counts'] == [expected], 'Adam steps: ' + name)
        check(len(audit_record['frozen_sensing_factors']) == 6 and audit_record['finite_loss_and_gradient'], 'Training invariant: ' + name)
        check(all(np.isfinite(float(r['loss'])) and np.isfinite(float(r['gradient_norm'])) for r in history), 'Nonfinite history: ' + name)
        check(summary['stop_reason'] == 'step_limit', 'Incomplete budget: ' + name)
        check(summary['best_step'] == audit_record['selected_step'] == record['selected_step'], 'Selected step: ' + name)
        check(config['max_steps'] == expected, 'Budget mismatch: ' + name)
        total_updates += audit_record['new_updates']
    check(total_updates == 8400, 'New update total')
    for seed in [20261013, 20261014]:
        record = json.loads((BASE / f'pairing_{seed}.json').read_text())
        check(record['steps'] == 800 and record['arms'] == 4, 'Pairing coverage')
        check(all(record[k] for k in ['same_image_order_crops_flips_sampling_and_potential_noise',
              'same_measurements_between_supervision_arms', 'clean_arms_have_zero_training_noise', 'loss_total_weight_one']), 'Pairing failed')
    completion = require_workflow(BASE)
    if completion['evaluations']['regression']['recovered']:
        bridge = facts['hardware_bridge_overlap']
        check(bridge['evidence_sha256'] == sha(BASE / 'hardware_bridge_overlap_audit.json'), 'Hardware overlap evidence changed')
        check(bridge['max_psnr_difference_db'] < 2e-5 and bridge['max_ssim_difference'] < 2e-5, 'Hardware metric difference exceeds tolerance')
    for name in ['continuation', 'seed_20261013', 'seed_20261014', 'evaluation_smoke']:
        check(int((BASE / 'logs' / (name + '.exit')).read_text()) == 0, 'Nonzero job exit: ' + name)
    results = {}
    for phase, images, models in [('primary', 52, 20), ('regression', 110, 12)]:
        directory = BASE / ('evaluation_' + phase)
        check((directory / 'COMPLETE').exists(), 'Inference incomplete')
        verification = json.loads((directory / 'verification.json').read_text())
        meta = json.loads((directory / 'provenance.json').read_text())
        rows = list(csv.DictReader((directory / 'metrics.csv').open()))
        check(len(rows) == images * 11 * models == verification['rows'] == verification['float_arrays_recomputed'], 'Inference coverage')
        check(verification['stages'] == len(rows) * 6, 'Stage coverage')
        check(verification['max_metric_error'] < 2e-5 and verification['measurement_hashes_recomputed'], 'Float metric verification')
        check(len({r['name'] for r in meta['data']}) == images, 'Original image count')
        for name, entry in meta['checkpoints'].items():
            check(entry['sha256'] == selection['models'][name]['sha256'], 'Weight changed after selection')
        check(len(meta['checkpoints']) == models, 'Model state count')
        groups = defaultdict(list)
        measurements = defaultdict(set)
        for r in rows:
            groups[(r['dataset'], float(r['cr']), float(r['sigma']), r['method'])].append(r)
            measurements[(r['image'], r['cr'], r['sigma'], r['seed'])].add(r['measurement_sha256'])
        check(len(measurements) == images * 11 and all(len(v) == 1 for v in measurements.values()), 'Measurement pairing')
        computed = {key: {field: means(values, field)[0] for field in ['mse', 'psnr', 'ssim']} for key, values in groups.items()}
        for r in csv.DictReader((directory / 'all_model_summary.csv').open()):
            key = (r['dataset'], float(r['cr']), float(r['sigma']), r['method'])
            for field in ['mse', 'psnr', 'ssim']:
                check(abs(float(r[field]) - computed[key][field]) < 1e-10, 'Aggregate mismatch')
        comparisons = list(csv.DictReader((directory / 'paired_summary.csv').open()))
        for r in comparisons:
            key = (r['dataset'], float(r['cr']), float(r['sigma']))
            old, new = computed[(*key, r['reference'])], computed[(*key, r['candidate'])]
            expected = 100 * (1 - new['mse'] / old['mse'])
            check(abs(float(r['mse_reduction_percent']) - expected) < 1e-9, 'MSE ratio mismatch')
            check(abs(float(r['delta_psnr']) - (new['psnr'] - old['psnr'])) < 1e-10, 'PSNR difference mismatch')
            check(abs(float(r['delta_ssim']) - (new['ssim'] - old['ssim'])) < 1e-10, 'SSIM difference mismatch')
        check(facts['phases'][phase]['metrics_sha256'] == sha(directory / 'metrics.csv'), 'Facts use different CSV')
        check(facts['phases'][phase]['analysis_sha256'] == sha(directory / 'ablation_analysis.json'), 'Facts use different analysis')
        results[phase] = dict(images=images, conditions=images * 11, model_states=models, rows=len(rows),
            stages=verification['stages'], all_mse_psnr_ssim_aggregates_independently_recomputed=True,
            metrics_sha256=sha(directory / 'metrics.csv'), complete_float_verification=verification)
    result = dict(formal_updates=total_updates, training_arms=8, continuation_total_updates=12000,
        selected_continuation_step=facts['training']['continuation']['selected_step'], phases=results,
        locked_sources_verified=True, test_model_selection=False, completion=completion)
    if not experiments_only:
        report = ROOT / 'output/pdf/technical_report_ablation_20261013.pdf'
        ppt = ROOT / 'output/presentation/course_presentation_ablation_20261013.pptx'
        reader = PdfReader(report)
        check(4 <= len(reader.pages) <= 6, 'Required technical report length')
        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        check(not re.search(r'\bundefined\b|\bnan\b|\bTODO\b', text, re.I), 'Unresolved report text')
        for term in ['800', '12000', '52', 'AI Disclosure', 'References', '800-step']:
            check(term in text, 'Report missing scope: ' + term)
        with zipfile.ZipFile(ppt) as package:
            check(package.testzip() is None, 'Corrupt PPT package')
            slides = [n for n in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
            check(len(slides) == 12, 'Required slide count')
            note_names = [n for n in package.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml', n)]
            check(len(note_names) == 12, 'Missing speaking notes')
            note_text = '\n'.join(' '.join(x.text or '' for x in ET.fromstring(package.read(n)).iter()) for n in note_names)
            check(sum(map(int, re.findall(r'Planned speaking time:\s*(\d+)\s*seconds', note_text))) == 600, 'Speaking plan')
            slide_text = '\n'.join(' '.join(x.text or '' for x in ET.fromstring(package.read(n)).iter()) for n in slides)
            check('AI disclosure' in slide_text and 'Codex' in note_text, 'Missing visible AI disclosure or detailed tool attribution')
        guide = ROOT / 'coursework/讲稿_续训与消融_20261013.md'
        check(len(re.findall(r'^## 第\d+页', guide.read_text(), re.M)) == 12, 'Chinese notes coverage')
        numeric = json.loads((BASE / 'materials_numeric_audit.json').read_text())
        check(numeric['ppt_sha256'] == sha(ppt), 'Numeric audit refers to a different PPT')
        check(numeric['report_sha256'] == sha(report), 'Numeric audit refers to a different report')
        check(numeric['report_facts_sha256'] == sha(BASE / 'report_facts.json'), 'Numeric facts changed')
        check(set(numeric['native_chart_checks']) == {'6', '8', '9', '10'}, 'Incomplete native chart audit')
        for relative, digest in numeric['numeric_csv_hashes'].items():
            check(sha(ROOT / relative) == digest, 'Numeric source changed: ' + relative)
        visual = json.loads((BASE / 'materials_visual_audit.json').read_text())
        check(visual['report_sha256'] == sha(report) and visual['ppt_sha256'] == sha(ppt), 'Visual audit refers to different materials')
        check(len(visual['pdf_pages']) == len(reader.pages) and len(visual['slides']) == 12, 'Visual inspection coverage')
        for preview in visual['pdf_pages'] + visual['slides']:
            check(preview['individually_inspected'] and preview['no_clipping_or_overlap'], 'Unverified visual layout')
            check(sha(ROOT / preview['path']) == preview['sha256'], 'Rendered preview changed')
        result['materials'] = dict(report_pages=len(reader.pages), slides=12, speaking_seconds=600,
            native_chart_values_independently_checked=True, all_pages_slides_individually_inspected=True,
            files={str(p.relative_to(ROOT)): sha(p) for p in [report, ppt, guide, ROOT / 'coursework/HQS_ABLATION_RESULTS_20261013.md',
                BASE / 'materials_numeric_audit.json', BASE / 'materials_visual_audit.json']})
    destination = BASE / ('experiment_completion_audit.json' if experiments_only else 'delivery_audit.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-only', action='store_true')
    audit(parser.parse_args().experiments_only)
