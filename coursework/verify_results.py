"""Audit result coverage, provenance, aggregation, and submission artifacts."""

import csv
import hashlib
import json
import math
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / 'coursework/results' / name).open() as f:
        return list(csv.DictReader(f))


def verify_presentation(facts):
    ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
          'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart'}
    labels = {'adjoint': 'Adjoint', 'fista_dct': 'DCT-FISTA', 'hqs': 'HQS', 'admm': 'ADMM'}

    def metric(method, cr=.1, sigma=0, field='psnr'):
        return next(r[field] for r in facts['summary'] if r['split'] == 'test'
                    and r['method'] == method and r['cr'] == cr and r['sigma'] == sigma)

    expected = {
        6: (['1%', '4%', '10%', '25%', '50%'],
            {label: [metric(m, cr) for cr in [.01, .04, .1, .25, .5]] for m, label in labels.items()}),
        7: (list(labels.values()), {'Median time': [metric(m, field='seconds') for m in labels]}),
        8: (['0', '0.01', '0.05'],
            {labels[m]: [metric(m, sigma=s) for s in [0, .01, .05]] for m in ['fista_dct', 'hqs', 'admm']}),
        9: ([str(i) for i in range(1, 7)],
            {labels[m]: facts['stage_means'][m] for m in ['hqs', 'admm']}),
    }
    deck = ROOT / 'output/presentation/course_presentation.pptx'
    with ZipFile(deck) as package:
        assert package.testzip() is None
        slides = [n for n in package.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)]
        notes = [n for n in package.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml', n)]
        assert len(slides) == len(notes) == 12
        total_seconds = 0
        for name in notes:
            content = ' '.join(ET.fromstring(package.read(name)).itertext())
            duration = re.search(r'Planned speaking time: (\d+) seconds', content)
            assert duration, name
            total_seconds += int(duration[1])
        assert total_seconds == 600
        final_slide = ET.fromstring(package.read('ppt/slides/slide12.xml'))
        assert 'Codex' in ' '.join(final_slide.itertext())
        for number, (categories, values) in expected.items():
            rels = ET.fromstring(package.read(f'ppt/slides/_rels/slide{number}.xml.rels'))
            targets = [r.attrib['Target'] for r in rels if r.attrib['Type'].endswith('/chart')]
            assert len(targets) == 1, number
            target = targets[0]
            chart_path = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join('ppt/slides', target))
            chart = ET.fromstring(package.read(chart_path))
            series = chart.findall('.//c:ser', ns)
            assert len(series) == len(values), number
            observed = {}
            for item in series:
                label = item.findtext('c:tx/c:v', namespaces=ns)
                assert label not in observed, (number, label)
                cats = [p.text for p in item.findall('c:cat//c:pt/c:v', ns)]
                assert cats == categories, (number, cats)
                observed[label] = [float(p.text) for p in item.findall('c:val//c:pt/c:v', ns)]
            assert observed.keys() == values.keys(), number
            for label, reference in values.items():
                assert len(observed[label]) == len(reference)
                assert all(abs(a - b) < 0.0000051 for a, b in zip(observed[label], reference)), (number, label)
    return dict(slides=12, notes=12, speaking_seconds=total_seconds,
                chart_source_values_verified=True, sha256=hashlib.sha256(deck.read_bytes()).hexdigest())


def main():
    base = ROOT / 'coursework/results'
    assert (base / 'COMPLETE').is_file()
    meta = json.loads((base / 'provenance.json').read_text())
    assert not meta['smoke']
    for rel, expected in meta['source_hashes'].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, rel
    for entry in meta['data']:
        assert hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest() == entry['sha256']
    for method, digest in meta['checkpoints'].items():
        assert hashlib.sha256((ROOT / f'weight/{method}_proxunroll.pth').read_bytes()).hexdigest() == digest
    assert hashlib.sha256((ROOT / meta['common_operator']).read_bytes()).hexdigest() == meta['common_operator_sha256']
    rows = read('metrics.csv')
    stages = read('stages.csv')
    native = read('native_operators.csv')
    assert len(rows) == 8 * 11 * 4, len(rows)
    assert len(stages) == 8 * 11 * 2 * 6, len(stages)
    assert len(read('validation.csv')) == 18
    assert len(read('interventions.csv')) == 8
    assert len(native) == 32
    keys = [(r['image'], r['cr'], r['sigma'], r['seed'], r['method']) for r in rows]
    assert len(set(keys)) == len(keys)
    groups = defaultdict(list)
    for r in rows:
        for metric in ['psnr', 'ssim', 'seconds', 'relative_residual', 'actual_cr']:
            assert math.isfinite(float(r[metric]))
        assert -1 <= float(r['ssim']) <= 1
        assert float(r['seconds']) >= 0
        groups[(r['image'], r['cr'], r['sigma'], r['seed'])].append(r)
    for rs in groups.values():
        assert {r['method'] for r in rs} == {'adjoint', 'fista_dct', 'hqs', 'admm'}
        assert len({r['actual_cr'] for r in rs}) == 1
    for r in stages:
        if int(r['stage']) == 6:
            final = next(x for x in rows if all(x[k] == r[k] for k in ['image', 'cr', 'sigma', 'seed', 'method']))
            assert abs(float(final['psnr']) - float(r['psnr'])) < 1e-10
    expected_counts = Counter(r['split'] for r in rows)
    assert expected_counts == {'test': 264, 'stress': 88}, expected_counts
    facts = json.loads((base / 'report_facts.json').read_text())
    for summary in facts['summary']:
        selected = [r for r in rows if r['split'] == summary['split'] and r['method'] == summary['method']
                    and float(r['cr']) == summary['cr'] and float(r['sigma']) == summary['sigma']]
        assert abs(sum(float(r['psnr']) for r in selected) / len(selected) - summary['psnr']) < 1e-9
    counts = {}
    for filename, expected in [('technical_report.pdf', 6), ('literature_review.pdf', 3)]:
        pdf = PdfReader(ROOT / 'output/pdf' / filename)
        counts[filename] = len(pdf.pages)
        assert len(pdf.pages) == expected, counts
        text = '\n'.join(p.extract_text() for p in pdf.pages)
        for student_id in ['522026230043', '502026230083', '502026230092']:
            assert student_id in text
        assert 'Codex' in text
        assert 'TODO' not in text
        for p in pdf.pages:
            assert len(p.extract_text()) > 700
    audit = dict(metric_rows=len(rows), stage_rows=len(stages), native_rows=len(native),
                 split_rows=dict(expected_counts), pdf_pages=counts,
                 provenance_hashes_verified=True, paired_conditions_verified=True,
                 aggregation_verified=True, presentation=verify_presentation(facts))
    (base / 'verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
