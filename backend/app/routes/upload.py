from pathlib import Path
from app.services.chunking import chunk_text
import os
import re

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Cookie,
)

from docx import Document
from docx.shared import Pt
from pypdf import PdfReader

from app.services.search import set_document_text
from app.services.storage import (
    add_document,
    get_documents,
    _write_all,
)

router = APIRouter(
    prefix="/api/v1/upload",
    tags=["Upload"]
)

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

UPLOAD_DIR = Path(
    os.path.join(BASE_DIR, "uploads")
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

COOKIE_NAME = "vpp_admin_session"


# =========================
# AUTH
# =========================
def require_admin(
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    )
):
    if session != "logged_in":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


# =========================
# HELPERS
# =========================
def sanitize_filename(
    filename: str
) -> str:
    filename = filename.strip()
    filename = filename.replace(" ", "_")
    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "",
        filename
    )

    return filename


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


def looks_like_part(text: str) -> bool:
    patterns = [
        r"^(Část|Časť|Oddíl|Oddiel|Sekce|Kapitola|Hlava)\b",
        r"^\d+\.\s+[A-ZÁČĎÉĚÍĹĽŇÓÔŘŠŤÚŮÝŽ]",
    ]

    return any(
        re.search(p, text, re.IGNORECASE)
        for p in patterns
    )


def looks_like_article(text: str) -> bool:
    patterns = [
        r"^(Článek|Článok|Čl\.)\s*\d+",
    ]

    return any(
        re.search(p, text, re.IGNORECASE)
        for p in patterns
    )


def looks_like_short_heading(text: str) -> bool:
    if len(text) > 80:
        return False

    if text.endswith(";"):
        return False

    if len(text.split()) <= 8:
        return True

    return False


# =========================
# VISUAL NORMALIZER
# =========================
def normalize_docx(
    file_path: Path
):
    doc = Document(file_path)

    for p in doc.paragraphs:
        text = p.text.strip()

        if not text:
            continue

        # PART / SECTION
        if looks_like_part(text):
            format_paragraph(
                p,
                size=14,
                bold=True
            )

        # ARTICLE
        elif looks_like_article(text):
            format_paragraph(
                p,
                size=13,
                bold=True
            )

        # SHORT TITLES
        elif looks_like_short_heading(text):
            format_paragraph(
                p,
                size=12,
                bold=True
            )

        # BODY TEXT
        else:
            format_paragraph(
                p,
                size=11,
                bold=False
            )

    doc.save(file_path)


# =========================
# DOCX PARSER
# =========================
def parse_docx(
    file_path: Path
) -> str:
    # 1 normalize first
    normalize_docx(file_path)

    # 2 re-open normalized file
    doc = Document(file_path)

    lines = []

    for p in doc.paragraphs:
        text = p.text.strip()

        if not text:
            continue

        lines.append(text)

    return "\n".join(lines)


# =========================
# PDF PARSER
# =========================
def parse_pdf(
    file_path: Path
) -> str:
    reader = PdfReader(str(file_path))

    pages = []

    for i, page in enumerate(
        reader.pages,
        start=1
    ):
        page_text = page.extract_text()

        if page_text:
            pages.append(
                f"[[PAGE:{i}]]\n{page_text.strip()}"
            )

    return "\n\n".join(pages)


def remove_old_record(
    filename: str
):
    docs = get_documents()

    docs = [
        d for d in docs
        if d.get("file_name")
        != filename
    ]

    _write_all(docs)


# =========================
# LIST FILES
# =========================
@router.get("/")
async def list_files():
    files = []

    for file in UPLOAD_DIR.iterdir():
        if (
            file.is_file()
            and file.name
            != "documents.json"
            and file.name
            != "insurers.json"
        ):
            files.append(file.name)

    return {
        "files": sorted(files)
    }


# =========================
# UPLOAD
# =========================
@router.post("/")
async def upload_file(
    insurer: str = Form(...),
    document_title: str = Form(...),
    file: UploadFile = File(...),
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    ),
):
    require_admin(session)

    if not insurer.strip():
        raise HTTPException(
            status_code=400,
            detail="Pojišťovna je povinná."
        )

    if not document_title.strip():
        raise HTTPException(
            status_code=400,
            detail="Název dokumentu je povinný."
        )

    safe_name = sanitize_filename(
        file.filename
    )

    if not safe_name:
        raise HTTPException(
            status_code=400,
            detail="Neplatný název souboru."
        )

    file_path = UPLOAD_DIR / safe_name

    filename_lower = (
        safe_name.lower()
    )

    if not (
        filename_lower.endswith(".pdf")
        or filename_lower.endswith(
            ".docx"
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Podporován je pouze DOCX nebo PDF."
        )

    remove_old_record(safe_name)

    if file_path.exists():
        file_path.unlink()

    content = await file.read()

    with open(
        file_path,
        "wb"
    ) as f:
        f.write(content)

    # =====================
    # PARSE
    # =====================
    if filename_lower.endswith(
        ".docx"
    ):
        text = parse_docx(file_path)
    else:
        text = parse_pdf(file_path)

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Nepodařilo se načíst text dokumentu."
        )

    # =====================
    # CONTEXT + CHUNKS
    # =====================
    set_document_text(text)

    chunks = chunk_text(text)

    add_document({
        "insurer": insurer.strip(),
        "document_title":
            document_title.strip(),
        "file_name": safe_name,
        "text_content": text,
        "chunks": chunks
    })

    return {
        "filename": safe_name,
        "insurer": insurer,
        "document_title":
            document_title,
        "status": "uploaded",
        "chunks": len(chunks),
        "text_preview":
            text[:300]
    }


# =========================
# DELETE
# =========================
@router.delete("/{filename}")
async def delete_file(
    filename: str,
    session: str = Cookie(
        default=None,
        alias=COOKIE_NAME
    ),
):
    require_admin(session)

    safe_name = sanitize_filename(
        filename
    )

    file_path = UPLOAD_DIR / safe_name

    if file_path.exists():
        file_path.unlink()

    remove_old_record(safe_name)

    return {
        "status": "deleted",
        "filename": safe_name
    }