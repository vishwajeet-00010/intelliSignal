"""Render the reviewed evidence in pdf_content.json; no simulation or selection.

Run with the bundled PDF runtime documented in COMMANDS.md.
"""
from pathlib import Path
from html import escape
import json

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable,
    KeepTogether,
)
from pypdf import PdfReader

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
OUT = ROOT / "output/pdf/IntelliSignal_Controlled_Comparison_20261119.pdf"
NAVY = colors.HexColor("#16364B")
INK = colors.HexColor("#293C4A")
MUTED = colors.HexColor("#617786")
PALE = colors.HexColor("#EFF4F7")
BLUE = colors.HexColor("#287BA6")
ORANGE = colors.HexColor("#CB732D")
STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=24,
                            leading=28, textColor=NAVY, spaceAfter=12),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10,
                               leading=14, textColor=MUTED, spaceAfter=12),
    "heading": ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=13,
                              leading=17, textColor=NAVY, spaceBefore=9, spaceAfter=7),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9.5,
                           leading=13.3, textColor=INK, spaceAfter=7),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8,
                            leading=11, textColor=MUTED, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8,
                           leading=10.5, textColor=INK),
    "headcell": ParagraphStyle("headcell", fontName="Helvetica-Bold", fontSize=8,
                               leading=10.5, textColor=colors.white),
}


def paragraph(value, style="body"):
    # Only the content file's explicit lightweight report markup is interpreted.
    return Paragraph(str(value), STYLES[style])


def evidence_table(block):
    widths = block.get("widths")
    if widths:
        widths = [499 * x / sum(widths) for x in widths]
    else:
        widths = [499 / len(block["headers"])] * len(block["headers"])
    rows = [[paragraph(v, "headcell") for v in block["headers"]]]
    rows += [[paragraph(v, "cell") for v in row] for row in block["rows"]]
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.HexColor("#CFDCE4")),
    ]))
    return table


class WaitingBars(Flowable):
    """Zero-based, separately labelled scales for the two traffic scenarios."""

    def __init__(self, panels):
        super().__init__()
        self.panels = panels
        self.width = 499
        self.height = 155

    def draw(self):
        c = self.canv
        for panel_index, panel in enumerate(self.panels):
            x = panel_index * 255
            maximum = panel["maximum"]
            c.setFillColor(NAVY)
            c.setFont("Helvetica-Bold", 10)
            c.drawString(x, 141, panel["title"])
            c.setFont("Helvetica", 7.5)
            c.setFillColor(MUTED)
            c.drawString(x, 127, "Mean waiting, seconds per entered vehicle")
            palette = [MUTED, BLUE, ORANGE, colors.HexColor("#7B5A9B")]
            for i, row in enumerate(panel["rows"]):
                y = 104 - 23 * i
                value = float(row["value"])
                c.setFillColor(INK)
                c.setFont("Helvetica", 7.5)
                c.drawString(x, y + 3, row["label"])
                c.setFillColor(palette[i % len(palette)])
                length = 112 * value / maximum
                c.rect(x + 76, y, length, 13, fill=1, stroke=0)
                c.setFillColor(INK)
                c.drawString(x + 79 + length, y + 3, f"{value:.2f}")
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 7)
            c.drawString(x, 7, f"Scale 0-{maximum:g} s; lower is better.")


def footer(c, doc):
    width, _ = A4
    c.saveState()
    c.setStrokeColor(colors.HexColor("#CFDCE4"))
    c.line(48, 42, width - 48, 42)
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawString(48, 28, "INTELLISIGNAL | CONTROLLED EXPERIMENT | 19-20 NOV 2026")
    c.drawRightString(width - 48, 28, str(doc.page))
    c.restoreState()


def main():
    content = json.loads((RUN / "pdf_content.json").read_text(encoding="utf-8"))
    story = []
    for page_index, page in enumerate(content["pages"]):
        if page_index:
            story.append(PageBreak())
        story.append(paragraph(page["title"], "title"))
        if page.get("subtitle"):
            story.append(paragraph(page["subtitle"], "subtitle"))
        for block in page["blocks"]:
            kind = block.get("type", "body")
            if kind == "table":
                story.extend([evidence_table(block), Spacer(1, 8)])
            elif kind == "chart":
                story.append(WaitingBars(block["panels"]))
            else:
                story.append(paragraph(block["text"], kind))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=48, rightMargin=48,
        topMargin=42, bottomMargin=56,
        title="IntelliSignal: Controlled Experiments - Good and Bad",
        author="IntelliSignal Project",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(str(OUT))
    assert len(reader.pages) == len(content["pages"]), "Unexpected page overflow"
    assert all(len(page.extract_text()) > 150 for page in reader.pages)
    print(json.dumps({"pdf": str(OUT), "pages": len(reader.pages),
                      "bytes": OUT.stat().st_size}))


if __name__ == "__main__":
    main()
