"""Reopen the delivered PPT and compare editable chart values with frozen CSV."""
import csv
import json
import posixpath
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

from coursework.run_experiments import ROOT, sha

BASE = ROOT / 'runs/hqs_ablation_20261013'
NS = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}


def chart_series(package, slide):
    relationships = ET.fromstring(package.read(f'ppt/slides/_rels/slide{slide}.xml.rels'))
    targets = [r.attrib['Target'] for r in relationships if r.attrib['Type'].endswith('/chart')]
    assert len(targets) == 1, f'Slide {slide} needs one editable native chart'
    target = targets[0]
    name = target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/slides/' + target)
    chart = ET.fromstring(package.read(name))
    result = []
    for series in chart.findall('.//c:ser', NS):
        # Line/scatter charts use yVal, column charts use val.
        values = series.find('c:val', NS)
        if values is None:
            values = series.find('c:yVal', NS)
        points = sorted(values.findall('.//c:pt', NS), key=lambda p: int(p.attrib['idx']))
        result.append([float(p.find('c:v', NS).text) for p in points])
    assert result
    return name, result


def main():
    facts = json.loads((BASE / 'report_facts.json').read_text())
    primary_csv = BASE / 'evaluation_primary/paired_summary.csv'
    rows = list(csv.DictReader(primary_csv.open()))
    rows = [r for r in rows if r['dataset'] == 'DIV2K_holdout_10000' and float(r['cr']) == .1]

    def pair(candidate, reference, sigma):
        row = next(r for r in rows if r['candidate'] == candidate and r['reference'] == reference
                   and float(r['sigma']) == sigma)
        return float(row['mse_reduction_percent']) / 100

    def factor(comparison, seed, sigma):
        row = next(r for r in rows if r['comparison'] == comparison and r['checkpoint_view'] == 'last'
                   and int(r['training_seed']) == seed and float(r['sigma']) == sigma)
        return float(row['mse_reduction_percent']) / 100

    expected = {
        8: [[pair('hqs_12000', reference, s) for s in [0, .01, .05]] for reference in ['hqs', 'hqs_10000']],
        9: [[factor('noise_effect_at_weighted', seed, s) for s in [0, .01, .05]] for seed in [20261013, 20261014]],
        10: [[factor('stage_effect_at_noise_on', seed, s) for s in [0, .01, .05]] for seed in [20261013, 20261014]],
    }
    baseline_csv = ROOT / 'runs/nanjing_web30_20261009/comparison/dataset_summary.csv'
    baseline = list(csv.DictReader(baseline_csv.open()))
    expected[6] = [[float(next(r for r in baseline if r['dataset'] == 'NJU_campus_web'
                              and r['method'] == m and float(r['cr']) == cr and float(r['sigma']) == 0)['psnr'])
                    for cr in [.01, .04, .1, .25, .5]] for m in ['adjoint', 'fista_dct', 'hqs', 'admm']]
    ppt = ROOT / 'output/presentation/course_presentation_ablation_20261013.pptx'
    result = {}
    with zipfile.ZipFile(ppt) as package:
        for slide, values in expected.items():
            chart, actual = chart_series(package, slide)
            assert len(actual) == len(values) and all(len(a) == len(b) for a, b in zip(actual, values))
            error = max(abs(a - b) for xs, ys in zip(actual, values) for a, b in zip(xs, ys))
            assert error < (6e-6 if slide == 6 else 1e-7), (slide, error)
            result[str(slide)] = dict(chart_part=chart, max_absolute_numeric_error=error,
                                     all_series_and_training_seeds_checked=True)
        text = ' '.join(node.text or '' for node in ET.fromstring(package.read('ppt/slides/slide7.xml')).findall('.//a:t', NS))
        for value in ['On / weighted', 'Off / weighted', 'On / final', 'Off / final', '800 x2 seeds']:
            assert value in text, value
        case = facts['failure_case']
        text = ' '.join(node.text or '' for node in ET.fromstring(package.read('ppt/slides/slide11.xml')).findall('.//a:t', NS))
        assert f'{float(case["mse_reduction_percent"]):.2f}%' in text
        assert case['candidate'] in text
    record = dict(ppt_sha256=sha(ppt), report_facts_sha256=sha(BASE / 'report_facts.json'),
                  numeric_csv_hashes={str(p.relative_to(ROOT)): sha(p) for p in [primary_csv, baseline_csv]},
                  native_chart_checks=result, paired_design_table_checked=True,
                  failure_candidate_and_signed_error_checked=True,
                  visual_check='Separate rendered-page inspection required; numeric audit alone is insufficient')
    (BASE / 'materials_numeric_audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
