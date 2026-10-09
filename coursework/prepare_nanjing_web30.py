"""Download a fixed licensed web collection and preserve preprocessing provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata
from urllib.parse import quote, urlencode

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from coursework.run_experiments import ROOT, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-download', action='store_true')
    a = p.parse_args()
    a.output = a.output.resolve()
    catalog = ROOT / 'coursework/nanjing_web30_sources.json'
    items = json.loads(catalog.read_text())['images']
    assert len(items) == 30 and len({r['name'] for r in items}) == 30
    a.output.mkdir(parents=True, exist_ok=a.resume_download)
    sources = a.output / 'source_png'; sources.mkdir(exist_ok=True)
    data = a.output / 'data'; data.mkdir(exist_ok=True)

    def prepare(item):
        name = item['name']; target = sources / (name + '.png')
        filename = unicodedata.normalize('NFC', item['title'].replace(' ', '_'))
        digest = hashlib.md5(filename.encode()).hexdigest()
        # Wikimedia/relay paths retain literal comma and parentheses. Encoding
        # these path characters caused valid filenames to return relay 404s.
        original_url = 'https://upload.wikimedia.org/wikipedia/commons/' + digest[0] + '/' + digest[:2] + '/' + quote(filename, safe=",()'")
        download_url = 'https://wsrv.nl/?' + urlencode(dict(url=original_url, output='png'))
        if not target.exists():
            partial = target.with_suffix('.part')
            for attempt in range(2):
                result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--location',
                    '--max-time', '120', '--max-filesize', str(64 * 1024 * 1024),
                    download_url, '-o', str(partial)], capture_output=True, text=True)
                if result.returncode == 0:
                    with Image.open(partial) as im:
                        assert im.format == 'PNG'
                        assert list(im.size) == item['source_dimensions'], (name, im.size)
                        im.verify()
                    partial.replace(target)
                    break
            else:
                raise RuntimeError(f'Download failed for {name}: {result.stderr.strip()}')
        with Image.open(target) as im:
            assert im.format == 'PNG' and list(im.size) == item['source_dimensions']
            rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        height, width = rgb.shape[:2]; side = min(height, width)
        x, y = (width - side) // 2, (height - side) // 2
        gray = cv2.cvtColor(rgb[y:y+side, x:x+side], cv2.COLOR_RGB2YCrCb)[:, :, 0]
        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
        processed = data / (name + '.png')
        Image.fromarray(gray).save(processed)
        row = dict(item, sequence=name, path=processed.name, sha256=sha(processed),
            original_image_url=original_url, downloaded_url=download_url,
            downloaded_source_path=str(target.relative_to(ROOT)), downloaded_source_sha256=sha(target),
            download_representation='full-resolution lossless PNG transcode via wsrv.nl; not original JPEG bytes',
            downloaded_dimensions=[width, height], crop_xywh=[x, y, side, side],
            preprocessing='EXIF orientation; center square; OpenCV RGB2YCrCb Y; INTER_AREA to 256x256; uint8 PNG then /255 at evaluation',
            source_role='external author web photo; collected and prepared by team; never used for training or checkpoint selection')
        print(json.dumps(dict(prepared=name, source_bytes=target.stat().st_size), ensure_ascii=False), flush=True)
        return row

    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(prepare, items))
    assert len({r['sha256'] for r in manifest}) == len(manifest), 'Duplicate processed image'
    assert len({r['downloaded_source_sha256'] for r in manifest}) == len(manifest), 'Duplicate source image'
    (data / 'test.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    # A source-only contact sheet for review before any reconstruction is run.
    sheet = Image.new('RGB', (5 * 256, 6 * 288), 'white'); draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(manifest):
        xx, yy = (i % 5) * 256, (i // 5) * 288
        sheet.paste(Image.open(data / item['path']).convert('RGB'), (xx, yy))
        draw.text((xx + 4, yy + 258), item['name'], fill='black')
    sheet.save(a.output / 'contact_sheet.png')
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), catalog_sha256=sha(catalog),
        count=30, campus_images=18, scenic_images=12, training_or_selection_use=False,
        grouping='each original photo is one equal-weight unit; campus/scenic and all six locations reported separately',
        collection_type='team-curated web images, not team-captured photographs',
        transport='direct Wikimedia retrieval timed out; wsrv.nl full-size PNG transcode, no requested resize, crop or filter; codec/color-profile handling can differ from original JPEG decoding',
        manifest_sha256=sha(data / 'test.json'), visual_source_review_required_before_evaluation=True)
    (data / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    credits = ['# Image attribution', '', 'These are externally authored web photographs, collected by the team.',
        'Downloaded files are full-resolution PNG transcodes; test images are center-square grayscale resizes.',
        'Each photograph and its image-derived adaptations retain the indicated source license.',
        'The code license does not replace these image licenses.', '', '| ID | Original title / author | Source and license |', '|---|---|---|']
    for r in manifest:
        credits.append(f"| {r['name']} | {r['title']} / {r['author']} | [Source]({r['page_url']}) · [{r['license']}]({r['license_url']}) |")
    (a.output / 'ATTRIBUTION.md').write_text('\n'.join(credits) + '\n')
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    main()
