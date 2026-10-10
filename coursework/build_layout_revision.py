"""Build layout-revised copies of the English technical and literature reports.

The measured content is unchanged. This wrapper only changes the ReportLab typography
and output names, preserving the previously delivered PDFs.
"""
from pathlib import Path
import shutil

from coursework import build_reports
from coursework import build_nanjing_report

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf'
TMP = ROOT / 'tmp/layoutfix'
TMP.mkdir(parents=True, exist_ok=True)

# Slightly larger body text and more open leading fill the sparse technical-report
# columns while retaining the required six-page conference-paper format.
def apply_style(font_size, leading, space_after, heading_size, heading_leading, small_size, small_leading):
    s = build_reports.STYLE
    s.fontSize = font_size
    s.leading = leading
    s.spaceAfter = space_after
    h = build_reports.HEAD
    h.fontSize = heading_size
    h.leading = heading_leading
    h.spaceBefore = 7
    h.spaceAfter = 5
    sm = build_reports.SMALL
    sm.fontSize = small_size
    sm.leading = small_leading


def main():
    original_draw = build_reports.draw_report
    apply_style(10.35, 13.0, 8.2, 11.5, 13.8, 8.75, 10.5)

    def wrapped_draw(filename, title, pages):
        mapping = {
            'technical_report.pdf': 'technical_report_layoutfix_20261010.pdf',
            'literature_review.pdf': 'literature_review_layoutfix_20261010.pdf',
            'technical_report_nanjing_20261009.pdf': 'technical_report_nanjing_layoutfix_20261010.pdf',
        }
        target = mapping.get(filename, filename)
        return original_draw(target, title, pages)

    build_reports.draw_report = wrapped_draw
    build_nanjing_report.report.draw_report = wrapped_draw

    # The generic builder is used only for the three-page literature review here;
    # its generic six-page report is written to a temporary discard target.
    build_reports.main()
    # Use a private figure directory so the historical report's figure cache and
    # repository evidence remain byte-for-byte unchanged.
    build_nanjing_report.FIG = TMP
    build_nanjing_report.report.FIG = TMP
    build_nanjing_report.main()

    # build_nanjing_report writes its markdown beside the historical filename after
    # the wrapped PDF call. Copy it beside the revised PDF for a matching source.
    old_md = OUT / 'technical_report_nanjing_20261009.md'
    if old_md.exists():
        shutil.copy2(old_md, OUT / 'technical_report_nanjing_layoutfix_20261010.md')
    lit_md = OUT / 'literature_review.md'
    if lit_md.exists():
        shutil.copy2(lit_md, OUT / 'literature_review_layoutfix_20261010.md')

    # Remove only the generic report created by the wrapper; all historical PDFs
    # and the two revised deliverables remain in output/pdf.
    for name in ['technical_report_layoutfix_20261010.pdf', 'technical_report_layoutfix_20261010.md']:
        p = OUT / name
        if p.exists(): p.unlink()
    print(OUT / 'technical_report_nanjing_layoutfix_20261010.pdf')
    print(OUT / 'literature_review_layoutfix_20261010.pdf')


if __name__ == '__main__':
    main()
