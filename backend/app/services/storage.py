from typing import List, Dict, Any
from app.supabase_client import supabase


# ==================================================
# BACKWARD-COMPATIBLE STORAGE LAYER (Supabase)
# Zachovává stejné funkce/signatury jako původní file storage
# ==================================================


# =========================
# INTERNAL HELPERS
# =========================
def _safe(value, default=None):
    return value if value is not None else default


def _slugify(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "-")
        .replace("/", "-")
    )


# --------------------------------------------------
# LEGACY HELPERS (kvůli kompatibilitě se starým kódem)
# --------------------------------------------------
def _ensure_files():
    # už není potřeba, necháváme kvůli kompatibilitě
    return


def _read_json(path: str):
    # legacy fallback
    return []


def _write_json(path: str, data):
    # legacy fallback
    return


# =========================
# DOCUMENTS
# =========================
def _read_all():
    """
    Vrací stejné pole dictů jako původní documents.json
    """
    result = (
        supabase
        .table("documents")
        .select("*")
        .order("created_at", desc=False)
        .execute()
    )

    rows = []

    for row in result.data or []:
        rows.append({
            "insurer": _safe(row.get("insurer_name"), ""),
            "document_title": _safe(row.get("title"), ""),
            "file_name": _safe(row.get("file_name"), ""),
            "text_content": _safe(row.get("extracted_text"), ""),
            "chunks": _safe(row.get("chunks"), []),
        })

    return rows


def _write_all(data):
    """
    Kompatibilita pro starý remove_old_record().
    Přepíše documents tabulku podle dodaných dat.
    """
    # smaž vše
    supabase.table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    # vlož znovu
    for doc in data:
        add_document(doc)


def add_document(doc):
    """
    Zachovává starý vstupní formát:
    {
      insurer,
      document_title,
      file_name,
      text_content,
      chunks
    }
    """
    insurer = doc.get("insurer", "").strip()

    payload = {
        "insurer_name": insurer,
        "title": doc.get("document_title", ""),
        "file_name": doc.get("file_name", ""),
        "storage_path": doc.get("file_name", ""),
        "extracted_text": doc.get("text_content", ""),
        "chunks": doc.get("chunks", []),
    }

    # duplicita dle file_name -> smaž starý záznam
    if payload["file_name"]:
        supabase.table("documents").delete().eq(
            "file_name",
            payload["file_name"]
        ).execute()

    supabase.table("documents").insert(payload).execute()

    # pokud insurer existuje, automaticky ho zapiš i do insurers
    if insurer:
        add_insurer(insurer)


def get_documents(
    insurer: str = None
) -> List[Dict[str, Any]]:

    query = supabase.table("documents").select("*")

    if insurer:
        query = query.eq("insurer_name", insurer)

    result = query.order("created_at", desc=False).execute()

    rows = []

    for row in result.data or []:
        rows.append({
            "insurer": _safe(row.get("insurer_name"), ""),
            "document_title": _safe(row.get("title"), ""),
            "file_name": _safe(row.get("file_name"), ""),
            "text_content": _safe(row.get("extracted_text"), ""),
            "chunks": _safe(row.get("chunks"), []),
        })

    return rows


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
    """
    Stejné chování jako dříve:
    union insurers tabulky + insurers z dokumentů
    """
    result = (
        supabase
        .table("insurers")
        .select("name")
        .order("name")
        .execute()
    )

    from_table = {
        row["name"]
        for row in (result.data or [])
        if row.get("name")
    }

    from_docs = {
        d.get("insurer")
        for d in get_documents()
        if d.get("insurer")
    }

    return sorted(list(from_table | from_docs))


def add_insurer(name: str):
    name = name.strip()

    if not name:
        return

    supabase.table("insurers").upsert({
        "name": name,
        "slug": _slugify(name),
    }).execute()


def delete_insurer(name: str):
    supabase.table("insurers").delete().eq(
        "name",
        name
    ).execute()
