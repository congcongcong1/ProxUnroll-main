"""Verify the frozen GitHub evidence bundle without CUDA or laboratory access."""
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    index = json.loads((ROOT / 'coursework/repository_evidence.json').read_text())
    for name, expected in index['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Invalid evidence path: {name}')
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Evidence hash mismatch: {name}')

    counts = {}
    for name, expected in index['csv_rows'].items():
        with (ROOT / name).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected:
            raise ValueError(f'Row count mismatch: {name}')
        if name.endswith('/metrics.csv'):
            keys = [tuple(r[k] for k in ['image', 'cr', 'sigma', 'seed', 'method']) for r in rows]
            if len(keys) != len(set(keys)):
                raise ValueError(f'Duplicate paired condition: {name}')
        counts[name] = len(rows)

    pages = {}
    for name, expected in index['pdf_pages'].items():
        actual = len(PdfReader(ROOT / name).pages)
        if actual != expected:
            raise ValueError(f'Page count mismatch: {name}')
        pages[name] = actual
    slides = {}
    for name, expected in index['ppt_slides'].items():
        with zipfile.ZipFile(ROOT / name) as package:
            if package.testzip() is not None:
                raise ValueError(f'Corrupt presentation: {name}')
            actual = sum(bool(re.fullmatch(r'ppt/slides/slide\d+\.xml', n))
                         for n in package.namelist())
        if actual != expected:
            raise ValueError(f'Slide count mismatch: {name}')
        slides[name] = actual

    print(json.dumps(dict(verified_files=len(index['files']), csv_rows=counts,
                          pdf_pages=pages, ppt_slides=slides,
                          scope='Frozen evidence hashes and coverage; no new training or evaluation'),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
