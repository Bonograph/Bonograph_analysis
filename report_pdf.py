"""Export one AnalysisResult as a designed, paginated A4 PDF report:
a title band carrying the program's mark, the headline result, every table (header row repeated on
each page), every graph as a numbered figure, and the notes.

Built with Qt's own QTextDocument + QPdfWriter, so no extra PDF library
is needed."""

from __future__ import annotations

import datetime
import html
import io
import re

from PySide6.QtCore import QMarginsF, Qt, QUrl
from PySide6.QtGui import QFont, QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument

INK = "#1d2530"
MUTED = "#65707d"
ACCENT = "#2f6db5"
BAND = "#1f4d85"
HEAD_BG = "#e6f0fa"
ALT_BG = "#f5f9fd"
RULE = "#d3e0ee"
# Chart width in layout units. print_() adds its own page margins inside
# the PDF page margins, leaving a text column of ~557 units (measured on a
# rendered A4 page) - wider images run off the right edge.
IMAGE_WIDTH_PX = 540
LOGO_NAME = "bonograph-mark.png"    # the name the mark goes by inside the document
LOGO_PX = 46                        # its height and width on the page, in layout units

_NUMBER = re.compile(r"^[\s<>≤≥=±~-]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*%?$")

_CSS = f"""
body {{ color: {INK}; }}
h2 {{ color: {ACCENT}; font-size: 13pt; font-weight: 700; margin-top: 18px; margin-bottom: 6px; }}
p.caption {{ color: {MUTED}; font-size: 9pt; margin-top: 2px; margin-bottom: 14px; }}
p.table-title {{ font-size: 10.5pt; font-weight: 600; margin-top: 12px; margin-bottom: 4px; }}
th {{ background-color: {HEAD_BG}; color: {BAND}; font-weight: 600; font-size: 9pt; }}
td {{ font-size: 9pt; }}
li {{ color: #333a44; font-size: 9.5pt; margin-bottom: 3px; }}
"""


def _chart_png(chart, width_in: float = 6.6, height_in: float = 3.9, dpi: int = 200) -> QImage:
    import graphs  # noqa: F401 - sys.path bootstrap for the graphs package
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from graphs.engine.charts import render_chart

    fig = Figure(figsize=(width_in, height_in))
    FigureCanvasAgg(fig)
    render_chart(fig, chart)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
    return QImage.fromData(buffer.getvalue(), "PNG")


def _column_alignments(table) -> list[str]:
    """Right-align a column (header included) when most of its filled
    cells are numbers, so headers sit over their values."""
    aligns = []
    for c in range(len(table.headers)):
        values = [str(row[c]).strip() for row in table.rows if c < len(row) and str(row[c]).strip()]
        numeric = sum(1 for v in values if _NUMBER.match(v))
        aligns.append("right" if values and numeric * 2 >= len(values) else "left")
    return aligns


def _cell(value, align: str, header: bool = False, alt: bool = False) -> str:
    text = html.escape(str(value))
    if header:
        return f'<th align="{align}">{text}</th>'
    bg = f' bgcolor="{ALT_BG}"' if alt else ""
    return f'<td align="{align}"{bg}>{text}</td>'


def _mark() -> QImage:
    """The program's mark for the title band, or a null image if its file is not there."""
    import app_paths

    image = QImage(app_paths.resource("bonograph.png"))
    if image.isNull():
        return image
    # kept well above the size it is shown at, so that it stays sharp in print
    return image.scaled(LOGO_PX * 6, LOGO_PX * 6, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def build_report_html(result, subtitle: str | None, chart_count: int, generated: datetime.datetime,
                      mark: bool = False) -> str:
    """``mark`` puts the program's mark and name at the right of the title band
    (the caller must have added the image to the document as LOGO_NAME)."""
    esc = html.escape
    meta = " &nbsp;•&nbsp; ".join(
        part for part in (esc(subtitle) if subtitle else "", f"Generated {generated:%d %b %Y, %H:%M}") if part
    )
    out = [f"<html><head><style>{_CSS}</style></head><body>"]
    out.append(
        f'<table width="100%" cellspacing="0" cellpadding="14" bgcolor="{BAND}"><tr><td valign="middle">'
        f'<span style="color:#ffffff; font-size:19pt; font-weight:700;">{esc(result.title)}</span><br/>'
        f'<span style="color:#cfe2f7; font-size:9pt;">{meta}</span>'
        "</td>"
        + (f'<td width="{LOGO_PX + 64}" align="center" valign="middle" style="white-space:nowrap;">'
           f'<img src="{LOGO_NAME}" width="{LOGO_PX}" height="{LOGO_PX}"/><br/>'
           '<span style="color:#cfe2f7; font-size:7.5pt; white-space:nowrap;">Bonograph™</span></td>' if mark else "")
        + "</tr></table>"
    )
    if result.summary:
        out.append("<h2>Result</h2>")
        out.append(
            f'<table width="100%" cellspacing="0" cellpadding="10" bgcolor="{HEAD_BG}"><tr><td>'
            f'<span style="color:{BAND}; font-size:11pt; font-weight:600;">{esc(result.summary)}</span>'
            "</td></tr></table>"
        )

    if result.tables:
        out.append("<h2>Tables</h2>")
        for table in result.tables:
            out.append(f'<p class="table-title">{esc(table.title)}</p>')
            out.append(
                f'<table width="100%" cellspacing="0" cellpadding="5" border="0.5" '
                f'style="border-color:{RULE}; border-style:solid; border-collapse:collapse;">'
            )
            aligns = _column_alignments(table)
            out.append("<thead><tr>" + "".join(_cell(h, aligns[c], header=True) for c, h in enumerate(table.headers))
                       + "</tr></thead><tbody>")
            for i, row in enumerate(table.rows):
                out.append("<tr>" + "".join(_cell(v, aligns[c] if c < len(aligns) else "left", alt=bool(i % 2))
                                            for c, v in enumerate(row)) + "</tr>")
            out.append("</tbody></table>")

    if chart_count:
        out.append("<h2>Graphs</h2>")
        for i, chart in enumerate(result.charts[:chart_count]):
            # image and caption in one block, so a page break can't split them
            out.append(
                f'<p class="caption" align="center"><img src="chart{i}.png" width="{IMAGE_WIDTH_PX}"/><br/>'
                f"Figure {i + 1}. {esc(chart.title)}</p>"
            )

    if result.notes:
        out.append("<h2>Notes</h2><ul>")
        out.extend(f"<li>{esc(note)}</li>" for note in result.notes)
        out.append("</ul>")
    out.append("</body></html>")
    return "".join(out)


def export_result_pdf(result, path: str, subtitle: str | None = None) -> int:
    """Write the report to ``path``; returns the number of pages."""
    document = QTextDocument()
    document.setDocumentMargin(0)
    document.setDefaultFont(QFont("Segoe UI", 10))
    for i, chart in enumerate(result.charts):
        document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(f"chart{i}.png"), _chart_png(chart))
    mark = _mark()
    if not mark.isNull():
        document.addResource(QTextDocument.ResourceType.ImageResource, QUrl(LOGO_NAME), mark)
    document.setHtml(build_report_html(result, subtitle, len(result.charts), datetime.datetime.now(), mark=not mark.isNull()))

    writer = QPdfWriter(path)
    writer.setTitle(result.title)
    writer.setCreator("Bonograph")
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(16, 14, 16, 16), QPageLayout.Unit.Millimeter)
    document.print_(writer)  # paginates (a private copy of) the document and numbers the pages
    del writer  # finishes and closes the file
    return _page_count(path)


def _page_count(path: str) -> int:
    """Pages actually written - print_() lays out its own copy of the
    document, so the original QTextDocument can't tell us."""
    try:
        from PySide6.QtPdf import QPdfDocument
    except ImportError:
        return 0
    pdf = QPdfDocument()
    pdf.load(path)
    count = pdf.pageCount()
    pdf.close()
    return count
