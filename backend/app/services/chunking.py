from typing import List, Dict, Any
import re


# =========================
# CLEAN
# =========================
def clean_lines(text: str) -> List[str]:
    lines = text.splitlines()

    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if len(line) == 1:
            continue

        cleaned.append(line)

    return cleaned


# =========================
# DETECTORS
# =========================
def is_part(line: str) -> bool:
    patterns = [
        r"^(Část|Časť|Oddíl|Oddiel|Sekce|Kapitola|Hlava)\b",
        r"^\d+\.\s+[A-ZÁČĎÉĚÍĹĽŇÓÔŘŠŤÚŮÝŽ]",
    ]

    return any(
        re.search(p, line, re.IGNORECASE)
        for p in patterns
    )


def is_article(line: str) -> bool:
    patterns = [
        r"^(Článek|Článok|ČLÁNEK|ČLÁNOK)\s+\d+",
        r"^Čl\.\s*\d+",
    ]

    return any(
        re.search(p, line, re.IGNORECASE)
        for p in patterns
    )


def detect_article_number(
    line: str
) -> str:
    patterns = [
        r"(Článek|Článok|ČLÁNEK|ČLÁNOK)\s+(\d+)",
        r"Čl\.\s*(\d+)",
    ]

    for p in patterns:
        m = re.search(
            p,
            line,
            re.IGNORECASE
        )

        if m:
            return m.group(
                len(m.groups())
            )

    return ""


# =========================
# MAIN PARSER
# =========================
def chunk_text(
    text: str,
    chunk_size: int = 8000
) -> List[Dict[str, Any]]:

    if not text:
        return []

    lines = clean_lines(text)

    chunks = []

    current_part = ""
    current_article = ""
    current_title = ""
    current_lines = []

    index = 1

    def flush():
        nonlocal index
        nonlocal current_lines
        nonlocal current_article
        nonlocal current_title

        if not current_lines:
            return

        body = "\n".join(
            current_lines
        ).strip()

        if not body:
            return

        section = current_title

        if not section:
            if current_article:
                section = (
                    f"Článek {current_article}"
                )
            else:
                section = "Neuvedeno"

        chunks.append({
            "index": index,
            "part": current_part,
            "section": section,
            "article": current_article,
            "paragraph": 1,
            "text": body
        })

        index += 1

    i = 0

    while i < len(lines):
        line = lines[i]

        # PART
        if is_part(line):
            current_part = line
            i += 1
            continue

        # ARTICLE START
        if is_article(line):
            flush()

            current_lines = []
            current_article = detect_article_number(
                line
            )
            current_title = line

            current_lines.append(line)

            # next line as title
            if i + 1 < len(lines):
                next_line = lines[i + 1]

                if (
                    not is_article(next_line)
                    and not is_part(next_line)
                    and len(next_line) < 120
                ):
                    current_title = (
                        line + " – " + next_line
                    )
                    current_lines.append(
                        next_line
                    )
                    i += 1

            i += 1
            continue

        # BODY
        current_lines.append(line)
        i += 1

    flush()

    # fallback
    if not chunks:
        chunks.append({
            "index": 1,
            "part": "",
            "section": "Dokument",
            "article": "",
            "paragraph": 1,
            "text": "\n".join(lines)
        })

    return chunks