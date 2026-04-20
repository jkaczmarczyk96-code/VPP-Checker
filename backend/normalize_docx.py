from docx import Document
from docx.shared import Pt
from pathlib import Path
import re

INPUT_FILE = "sample.docx"


def format_paragraph(
    paragraph,
    size,
    bold=False,
    italic=False
):
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic


def looks_like_heading(text: str) -> bool:
    if len(text) > 80:
        return False

    if text.endswith(";"):
        return False

    if text.endswith(":"):
        return True

    words = text.split()

    if len(words) <= 8:
        return True

    return False


doc = Document(INPUT_FILE)

for p in doc.paragraphs:
    text = p.text.strip()

    if not text:
        continue

    if re.search(
        r"^(Článok|Článek|Čl\.)\s*\d+",
        text,
        re.IGNORECASE
    ):
        format_paragraph(
            p, 13, True
        )

    elif re.search(
        r"^(Časť|Část|Oddíl|Sekce|Kapitola)",
        text,
        re.IGNORECASE
    ):
        format_paragraph(
            p, 14, True
        )

    elif looks_like_heading(text):
        format_paragraph(
            p, 14, True
        )

    else:
        format_paragraph(
            p, 11, False
        )

output = Path(INPUT_FILE).stem + "_clean.docx"
doc.save(output)

print("HOTOVO:", output)