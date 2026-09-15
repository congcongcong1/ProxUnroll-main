"""Extract the labeled architecture overview panel for readable projection."""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[1]
source = Image.open(root / 'fig/network.png')
# Panel (a) is the complete upper overview, above panels (b) and (c).
source.crop((0, 0, source.width, 2200)).save(root / 'coursework/figures/restorer_overview.png')
