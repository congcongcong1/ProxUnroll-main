"""Build a small, explicitly named evaluation suite from bundled skimage images."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from skimage import data
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'coursework/data'
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Validation images are disjoint from evaluation images.
    specs = [('camera', 'test'), ('astronaut', 'test'), ('coins', 'test'),
             ('moon', 'test'), ('page', 'test'), ('clock', 'test'),
             ('coffee', 'validation'), ('chelsea', 'validation')]
    for name, split in specs:
        image = getattr(data, name)()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        path = output / f'{name}.png'
        Image.fromarray(image).save(path)
        manifest.append(dict(name=name, split=split, source=f'skimage.data.{name}',
                             path=str(path.relative_to(ROOT)),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    yy, xx = np.indices((256, 256))
    checker = (((xx // 3 + yy // 3) % 2) * 255).astype(np.uint8)
    text_image = Image.new('L', (256, 256), 245)
    draw = ImageDraw.Draw(text_image)
    font = ImageFont.load_default(size=17)
    for i, line in enumerate(['SPI / CS 2026', '0123456789', 'HQS vs ADMM', 'Fine print: 1% CR', 'ABCD efgh 5678']):
        draw.text((12, 20 + i * 44), line, fill=15, font=font)
    for name, image in [('checkerboard', Image.fromarray(checker)), ('small_text', text_image)]:
        path = output / f'{name}.png'
        image.save(path)
        manifest.append(dict(name=name, split='stress', source='procedural stress target',
                             path=str(path.relative_to(ROOT)),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(manifest)} images in {output}')


if __name__ == '__main__':
    main()
