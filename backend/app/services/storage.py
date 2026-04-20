import json
import os
from typing import List, Dict, Any

# jednoduché file-based úložiště
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

DATA_FILE = os.path.join(
    UPLOAD_DIR,
    "documents.json"
)

INSURERS_FILE = os.path.join(
    UPLOAD_DIR,
    "insurers.json"
)


# =========================
# FILE HELPERS
# =========================
def _ensure_files():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    if not os.path.exists(INSURERS_FILE):
        with open(INSURERS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)


def _read_json(path: str):
    _ensure_files()

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_json(path: str, data):
    _ensure_files()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


# =========================
# DOCUMENTS
# =========================
def _read_all():
    return _read_json(DATA_FILE)


def _write_all(data):
    _write_json(DATA_FILE, data)


def add_document(doc):
    data = _read_all()
    data.append(doc)
    _write_all(data)


def get_documents(
    insurer: str = None
) -> List[Dict[str, Any]]:
    data = _read_all()

    if insurer:
        data = [
            d for d in data
            if d.get("insurer") == insurer
        ]

    return data


def get_documents_titles(
    insurer: str
) -> List[str]:
    data = get_documents(insurer)

    titles = list({
        d.get("document_title")
        for d in data
        if d.get("document_title")
    })

    return sorted(titles)


# =========================
# INSURERS
# =========================
def get_insurers() -> List[str]:
    docs = _read_all()
    saved = _read_json(INSURERS_FILE)

    from_docs = {
        d.get("insurer")
        for d in docs
        if d.get("insurer")
    }

    from_saved = {
        item
        for item in saved
        if item
    }

    return sorted(
        list(from_docs | from_saved)
    )


def add_insurer(name: str):
    name = name.strip()

    if not name:
        return

    data = _read_json(INSURERS_FILE)

    if name not in data:
        data.append(name)

    data = sorted(list(set(data)))

    _write_json(
        INSURERS_FILE,
        data
    )


def delete_insurer(name: str):
    data = _read_json(INSURERS_FILE)

    data = [
        item for item in data
        if item != name
    ]

    _write_json(
        INSURERS_FILE,
        data
    )