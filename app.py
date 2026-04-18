# =========================
# IMPORTS
# =========================
import hashlib
import json
import logging
import math
import re
import statistics
import time
import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import google.generativeai as genai
import gspread
import streamlit as st
from docx import Document as DocxDocument
from oauth2client.service_account import ServiceAccountCredentials
from qdrant_client import QdrantClient
from qdrant_client.models import *
from sentence_transformers import CrossEncoder, SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="VPP Checker", layout="wide")
VECTOR_SIZE = 768
COLLECTION_NAME = "docs_v2"
APP_VERSION = "2026-04-18"
DATA_DIR = Path("data")
MEMORY_FILE = DATA_DIR / "memory.jsonl"
LEARNING_FILE = DATA_DIR / "learning.jsonl"
METRICS_FILE = DATA_DIR / "metrics.jsonl"

INSURERS = [
    "CZ - GČPOJ", "CZ - Direct", "CZ - TravelCare", "CZ - Uniqa",
    "SK - TravelCare", "SK - Generali", "SK - ECP", "SK - Wüstenrot", "SK - Uniqa"
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("vpp-checker")
DATA_DIR.mkdir(exist_ok=True)

# =========================
# LIMITS
# =========================
AI_TIMEOUT_SEC = 45
CIRCUIT_WINDOW_SEC = 180
CIRCUIT_MAX_FAILURES = 5
MEMORY_HALF_LIFE_HOURS = 72
BM25_CORPUS_LIMIT = 5000
BM25_TOP_N = 80
BM25_CACHE_TTL_SEC = 300
LEARNING_ACTIVE_LIMIT = 3000
LEARNING_PRUNE_KEEP = 2000
RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX_QUERIES = 20
RERANK_CANDIDATE_LIMIT = 48
STREAM_RENDER_INTERVAL_SEC = 0.08

# =========================
# SESSION
# =========================
defaults = {
    "messages": [],
    "feedback": [],
    "feedback_done": {},
    "feedback_open": {},
    "feedback_chunk_ids": {},
    "logged": False,
    "upload_key": str(uuid.uuid4()),
    "ai_failures": deque(maxlen=8),
    "topic": "",
    "last_errors": [],
    "debug_mode": False,
    "query_times": deque(maxlen=RATE_LIMIT_MAX_QUERIES * 2),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =========================
# CENTRAL ERROR HANDLING
# =========================
def log_event(event, **fields):
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(record, ensure_ascii=False, default=str))


def append_jsonl(path, row):
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error("jsonl_write_failed path=%s error=%s", path, e)


def read_jsonl(path, limit=500):
    if not path.exists():
        return []
    try:
        rows = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(r) for r in rows if r.strip()]
    except Exception as e:
        logger.error("jsonl_read_failed path=%s error=%s", path, e)
        return []


def prune_jsonl(path, keep=LEARNING_PRUNE_KEEP, active_limit=LEARNING_ACTIVE_LIMIT):
    if not path.exists():
        return
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) <= active_limit:
            return
        archive = path.with_suffix(f".archive-{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl")
        archive.write_text("\n".join(lines[:-keep]) + "\n", encoding="utf-8")
        path.write_text("\n".join(lines[-keep:]) + "\n", encoding="utf-8")
        log_event("jsonl_pruned", path=str(path), archived=str(archive), kept=keep)
    except Exception as e:
        logger.error("jsonl_prune_failed path=%s error=%s", path, e)


def parse_ts(ts):
    try:
        return datetime.fromisoformat((ts or "").replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def handle_error(user_msg, err=None, trace_id=None, show=True):
    if err:
        logger.exception("%s trace=%s", user_msg, trace_id)
    st.session_state.last_errors.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "trace_id": trace_id,
        "message": user_msg,
        "detail": str(err) if err else "",
    })
    st.session_state.last_errors = st.session_state.last_errors[-20:]
    if show:
        st.error(user_msg if not st.session_state.debug_mode else f"{user_msg} ({err})")


# =========================
# GEMINI (PRODUCTION LAYER)
# =========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
MODELS = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]


def circuit_open():
    now = time.time()
    failures = [t for t in st.session_state.ai_failures if now - t < CIRCUIT_WINDOW_SEC]
    return len(failures) >= CIRCUIT_MAX_FAILURES


def rate_limited():
    now = time.time()
    st.session_state.query_times = deque(
        [t for t in st.session_state.query_times if now - t < RATE_LIMIT_WINDOW_SEC],
        maxlen=RATE_LIMIT_MAX_QUERIES * 2,
    )
    if len(st.session_state.query_times) >= RATE_LIMIT_MAX_QUERIES:
        return True
    st.session_state.query_times.append(now)
    return False


def generate_safe(prompt, stream=True, trace_id=None):
    if circuit_open():
        append_jsonl(METRICS_FILE, {
            "trace_id": trace_id,
            "type": "ai",
            "status": "circuit_open",
            "latency_ms": 0,
        })
        return None

    started = time.perf_counter()
    last_err = None
    for m in MODELS:
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(m)
                resp = model.generate_content(
                    prompt,
                    stream=stream,
                    request_options={"timeout": AI_TIMEOUT_SEC},
                )
                append_jsonl(METRICS_FILE, {
                    "trace_id": trace_id,
                    "type": "ai",
                    "status": "ok",
                    "model": m,
                    "attempt": attempt + 1,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                })
                return resp
            except Exception as e:
                last_err = e
                st.session_state.ai_failures.append(time.time())
                log_event("ai_error", trace_id=trace_id, model=m, attempt=attempt + 1, error=str(e))
                time.sleep(1.5 * (attempt + 1))

    try:
        model = genai.GenerativeModel(MODELS[0])
        resp = model.generate_content(
            prompt,
            stream=False,
            request_options={"timeout": AI_TIMEOUT_SEC},
        )
        append_jsonl(METRICS_FILE, {
            "trace_id": trace_id,
            "type": "ai",
            "status": "fallback_ok",
            "latency_ms": round((time.perf_counter() - started) * 1000),
        })
        return resp
    except Exception as e:
        st.session_state.ai_failures.append(time.time())
        append_jsonl(METRICS_FILE, {
            "trace_id": trace_id,
            "type": "ai",
            "status": "failed",
            "error": str(e),
            "last_error": str(last_err),
            "latency_ms": round((time.perf_counter() - started) * 1000),
        })
        return None


# =========================
# GOOGLE SHEETS
# =========================
@st.cache_resource
def get_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        dict(st.secrets["gcp_service_account"]), scope
    )
    return gspread.authorize(creds)


def save_feedback(q, a, r, n, chunk_ids=None):
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "q": q,
        "a": a,
        "r": r,
        "note": n,
        "insurer": selected_insurer,
        "vpp": selected_vpp,
        "query_embedding": cached_query_embedding(q),
        "query_cluster": detect_topic(q),
        "chunk_ids": chunk_ids or [],
    }
    st.session_state.feedback.append(row)
    append_jsonl(LEARNING_FILE, row)
    prune_jsonl(LEARNING_FILE)
    try:
        get_client().open("VPP_Feedback").worksheet("feedback").append_row(
            [q, a, r, n, "", selected_insurer, selected_vpp, ",".join(chunk_ids or [])]
        )
    except Exception as e:
        handle_error("Feedback se nepodařilo uložit do Google Sheets.", e, show=False)


def log_query(q, ins, vpp, conf, trace_id):
    try:
        get_client().open("VPP_Feedback").worksheet("logs").append_row(
            [str(datetime.now()), trace_id, q, ins, vpp, conf]
        )
    except Exception as e:
        handle_error("Log dotazu se nepodařilo uložit do Google Sheets.", e, show=False)


# =========================
# ALERTY
# =========================
st.sidebar.markdown("## Stav systému")
try:
    data = get_client().open("VPP_Feedback").worksheet("feedback").get_all_records()
    bad = [r for r in data[-20:] if (r.get("r") == "dislike" or r.get("rating") == "dislike")]
    st.sidebar.error(f"{len(bad)} negativních") if bad else st.sidebar.success("OK")
except Exception:
    st.sidebar.info("Nedostupné")


# =========================
# QDRANT
# =========================
qdrant = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_API_KEY"])


REQUIRED_PAYLOAD = {"text", "page", "vpp_name", "insurer", "chunk_hash", "schema_version"}


def init_collection(force=False):
    try:
        qdrant.get_collection(COLLECTION_NAME)
        if force:
            qdrant.delete_collection(COLLECTION_NAME)
            raise RuntimeError("recreate")
    except Exception:
        qdrant.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
    for f in ["insurer", "vpp_name", "chunk_hash", "schema_version", "doc_id"]:
        try:
            qdrant.create_payload_index(COLLECTION_NAME, f, PayloadSchemaType.KEYWORD)
        except Exception:
            pass


def validate_payload(payload):
    missing = REQUIRED_PAYLOAD - set(payload)
    if missing:
        raise ValueError(f"Missing payload keys: {', '.join(sorted(missing))}")
    if not payload["text"].strip():
        raise ValueError("Empty text payload")
    return True


def batch_upsert(points, batch_size=64, retries=3):
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        for attempt in range(retries):
            try:
                qdrant.upsert(COLLECTION_NAME, batch)
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                log_event("qdrant_upsert_retry", attempt=attempt + 1, error=str(e))
                time.sleep(1.2 * (attempt + 1))


init_collection()


# =========================
# MODELS
# =========================
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


model = load_model()
reranker = load_reranker()


def embed_query(x):
    return model.encode([f"query: {x}"], normalize_embeddings=True)[0].tolist()


def embed_doc(x):
    return model.encode([f"passage: {x}"], normalize_embeddings=True)[0].tolist()


@st.cache_data(show_spinner=False, max_entries=5000)
def cached_query_embedding(text):
    return embed_query(text)


@st.cache_data(ttl=60, show_spinner=False, max_entries=4)
def cached_learning_rows(limit=1000):
    return read_jsonl(LEARNING_FILE, limit=limit)


# =========================
# HELPERS
# =========================
STOPWORDS = {
    "a", "i", "je", "jsou", "se", "si", "ve", "v", "na", "do", "pro", "pod", "nad",
    "the", "and", "or", "to", "of", "pojištění", "pojistné", "podmínky",
    "sú", "pre", "podmienky", "poistenie", "poistné", "výluky", "plnenie", "krytie",
    "škoda", "škody", "úraz", "úrazy", "zodpovednosť", "alebo", "ako"
}

SYNONYMS = {
    "výluka": ["vylouceni", "nekryje", "nevztahuje", "neplati", "vylúčenie", "nevzťahuje", "neplatí"],
    "výluky": ["vyluky", "výluka", "vylouceni", "nevztahuje", "neplati", "vylúčenie", "nevzťahuje", "neplatí"],
    "krytí": ["plneni", "hradí", "vztahuje", "pojistna udalost", "krytie", "plnenie", "hradí", "vzťahuje"],
    "plnění": ["plnenie", "krytie", "hradí", "vyplatenie"],
    "podmínky": ["podmienky", "podmienka", "podmienkách", "podmienok"],
    "pojištění": ["poistenie", "poistné", "poistenia", "poistení"],
    "pojistné": ["poistné", "pojistenia", "poistenia"],
    "limit": ["maximalni castka", "strop", "omezeni", "limit", "strop"],
    "léčba": ["leceni", "zdravotni pece", "osetreni", "liečenie", "zdravotná starostlivosť"],
    "léčebných": ["liečebných", "léčebných nákladů", "liečebných nákladov", "výloh", "nákladů", "nakladov"],
    "výloh": ["nákladů", "nakladov", "léčebných", "liečebných"],
    "nákladů": ["výloh", "léčebných", "liečebných", "nakladov"],
    "nakladov": ["výloh", "léčebných", "liečebných", "nákladů"],
    "zavazadla": ["bagaz", "osobni veci", "cestovný batožina"],
    "škoda": ["skoda", "škody", "škodová udalosť"],
    "úraz": ["uraz", "úrazy", "úrazu", "úrazov"],
}


def normalize(x):
    return x.strip().upper() if x else ""


def clean_text(t):
    t = re.sub(r"-\n(?=\w)", "", t or "")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\S\r\n]+", " ", t)
    return t.strip()


def tokenise(t):
    return [
        w for w in re.findall(r"[a-zá-ž0-9]+", (t or "").lower())
        if len(w) > 1 and w not in STOPWORDS
    ]


def smart_chunk(t, s=800, o=150):
    t = clean_text(t)
    out, i = [], 0
    while i < len(t):
        end = min(i + s, len(t))
        cut = max(t.rfind(". ", i, end), t.rfind("\n", i, end))
        if cut > i + 250:
            end = cut + 1
        out.append(t[i:end].strip())
        if end == len(t):
            i = end
        else:
            i = max(end - o, i + 1)
    return [c for c in out if c]


def keyword_score(q, t):
    return len(set(tokenise(q)) & set(tokenise(t)))


def section_focus_boost(query, heading, subheading):
    q_norm = clean_text(query).lower()
    hs = clean_text(f"{heading} {subheading}").lower()
    if not hs:
        return 0.0
    boost = 0.0
    if "výluk" in q_norm and "výluk" in hs:
        boost += 1.2
    if "léčeb" in q_norm and ("liečeb" in hs or "léčeb" in hs):
        boost += 1.0
    shared = set(tokenise(q_norm)) & set(tokenise(hs))
    boost += min(len(shared) * 0.2, 0.8)
    return boost


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    na = math.sqrt(sum(x * x for x in a)) + 1e-9
    nb = math.sqrt(sum(x * x for x in b)) + 1e-9
    return dot(a, b) / (na * nb)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def extract_sentences(p):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', p) if s.strip()]


def extract_sentence(p, q):
    s = extract_sentences(p)
    return max(s, key=lambda x: keyword_score(q, x), default=p)


def extract_heading(text):
    match = re.search(r"(^|\s)(\d+(\.\d+)*\.?\s+[A-ZÁ-Ž][^.!?]{5,80})", text)
    return match.group(2).strip() if match else ""


def query_terms(q):
    terms = tokenise(q)
    expanded = list(terms)
    for t in terms:
        expanded.extend(SYNONYMS.get(t, []))
    return expanded


def rewrite_query(q):
    return " ".join(query_terms(
        f"{q} výluky krytí krytie podmínky podmienky limit plnění plnenie úraz škody"
    ))


def dedupe_messages(items):
    seen, out = set(), []
    for item in items:
        key = hashlib.sha256(clean_text(item["content"]).lower().encode("utf-8")).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def detect_topic(q):
    terms = tokenise(q)
    if not terms:
        return ""
    return " ".join([w for w, _ in Counter(terms).most_common(4)])


def persist_memory(role, content):
    cleaned = clean_text(content)
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "content": cleaned,
        "topic": detect_topic(cleaned),
        "embedding": cached_query_embedding(cleaned),
    }
    append_jsonl(MEMORY_FILE, row)


def get_memory(q):
    live = [
        {"role": m["role"], "content": clean_text(m["content"]), "topic": detect_topic(m["content"])}
        for m in st.session_state.messages
    ]
    persisted = read_jsonl(MEMORY_FILE, limit=300)
    msgs = dedupe_messages(live + persisted)
    if not msgs:
        return ""

    qv = cached_query_embedding(q)
    qtopic = detect_topic(q)
    scored = []
    for m in msgs:
        mv = m.get("embedding") or cached_query_embedding(m["content"])
        role_weight = 1.15 if m.get("role") == "user" else 0.9
        topic_weight = 1.25 if qtopic and qtopic == m.get("topic") else 1.0
        ts = parse_ts(m.get("ts"))
        if ts:
            age_hours = max((datetime.now() - ts).total_seconds() / 3600, 0)
            recency_weight = 0.35 + 0.65 * math.exp(-math.log(2) * age_hours / MEMORY_HALF_LIFE_HOURS)
        else:
            recency_weight = 1.0
        scored.append((m, cosine(qv, mv) * role_weight * topic_weight * recency_weight))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return "\n".join([f"{m['role']}: {m['content']}" for m, _ in scored[:5]])


def safe_scroll(limit=500):
    try:
        return qdrant.scroll(COLLECTION_NAME, limit=limit, with_payload=True)[0]
    except Exception as e:
        handle_error("Dokumenty z Qdrantu teď nejdou načíst.", e, show=False)
        return []


def scroll_records(limit=BM25_CORPUS_LIMIT, scroll_filter=None):
    records = []
    offset = None
    while len(records) < limit:
        try:
            batch, offset = qdrant.scroll(
                COLLECTION_NAME,
                limit=min(256, limit - len(records)),
                offset=offset,
                scroll_filter=scroll_filter,
                with_payload=True,
            )
        except Exception as e:
            handle_error("Qdrant scroll selhal.", e, show=False)
            break
        records.extend(batch)
        if offset is None or not batch:
            break
    return records


def filter_expected(scroll_filter):
    if not scroll_filter:
        return {}
    must = getattr(scroll_filter, "must", None) or []
    expected = {}
    for cond in must:
        key = getattr(cond, "key", None)
        match = getattr(cond, "match", None)
        value = getattr(match, "value", None)
        if key and value is not None:
            expected[key] = value
    return expected


def filter_signature(expected):
    return json.dumps(expected or {}, sort_keys=True, ensure_ascii=False)


def build_filter(expected):
    if not expected:
        return None
    return Filter(must=[
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in expected.items()
    ])


@st.cache_data(ttl=BM25_CACHE_TTL_SEC, show_spinner=False, max_entries=32)
def cached_bm25_corpus(filter_key, limit):
    expected = json.loads(filter_key) if filter_key else {}
    records = scroll_records(limit=limit, scroll_filter=build_filter(expected))
    return tuple(
        {
            "id": str(getattr(r, "id", idx)),
            "payload": r.payload or {},
        }
        for idx, r in enumerate(records)
    )


def records_match_filter(records, scroll_filter):
    if not scroll_filter:
        return records
    must = getattr(scroll_filter, "must", None) or []
    expected = {}
    for cond in must:
        key = getattr(cond, "key", None)
        match = getattr(cond, "match", None)
        value = getattr(match, "value", None)
        if key and value is not None:
            expected[key] = value
    if not expected:
        return records
    return [
        r for r in records
        if all((r.payload or {}).get(k) == v for k, v in expected.items())
    ]


def get_vpps(ins):
    ins = normalize(ins)
    return sorted(set(
        r.payload.get("vpp_name")
        for r in safe_scroll()
        if r.payload and r.payload.get("insurer") == ins
    ))


def get_uploaded_docs():
    docs = defaultdict(lambda: {"pages": set(), "chunks": 0, "insurer": "", "vpp": ""})
    for r in safe_scroll(limit=2000):
        p = r.payload or {}
        doc_id = p.get("doc_id") or f"{p.get('insurer')}:{p.get('vpp_name')}"
        docs[doc_id]["pages"].add(p.get("page"))
        docs[doc_id]["chunks"] += 1
        docs[doc_id]["insurer"] = p.get("insurer", "")
        docs[doc_id]["vpp"] = p.get("vpp_name", "")
    return docs


# =========================
# INGEST
# =========================
def extract_docx_sections(uploaded_file):
    uploaded_file.seek(0)
    doc = DocxDocument(uploaded_file)
    sections = []
    current_heading = ""
    current_subheading = ""
    buffer = []
    non_empty_paragraphs = 0
    heading_count = 0
    subheading_count = 0

    def flush_buffer():
        if not buffer:
            return
        text = clean_text(" ".join(buffer))
        if text:
            sections.append({
                "heading": current_heading or "Bez nadpisu",
                "subheading": current_subheading or "",
                "text": text,
            })
        buffer.clear()

    for para in doc.paragraphs:
        raw = (para.text or "").strip()
        if not raw:
            continue
        non_empty_paragraphs += 1
        style_name = ((para.style.name if para.style else "") or "").lower()
        if "heading 1" in style_name or "nadpis 1" in style_name:
            flush_buffer()
            current_heading = raw
            current_subheading = ""
            heading_count += 1
            continue
        if "heading 2" in style_name or "nadpis 2" in style_name:
            flush_buffer()
            current_subheading = raw
            subheading_count += 1
            continue
        buffer.append(raw)

    flush_buffer()
    return {
        "sections": sections,
        "non_empty_paragraphs": non_empty_paragraphs,
        "heading_count": heading_count,
        "subheading_count": subheading_count,
    }


def assess_docx_quality(doc_meta):
    sections = doc_meta["sections"]
    non_empty_paragraphs = doc_meta["non_empty_paragraphs"]
    heading_count = doc_meta["heading_count"]
    subheading_count = doc_meta["subheading_count"]
    lengths = [len(s.get("text", "")) for s in sections if s.get("text")]
    avg_len = int(sum(lengths) / max(len(lengths), 1)) if lengths else 0
    max_len = max(lengths) if lengths else 0
    missing_heading = sum(1 for s in sections if not s.get("heading") or s.get("heading") == "Bez nadpisu")
    issues = []
    score = 100

    if not sections:
        issues.append("Dokument neobsahuje čitelný text.")
        score -= 70
    if non_empty_paragraphs > 0 and heading_count == 0:
        issues.append("Dokument nemá styly Nadpis 1. Doporučuji je doplnit kvůli přesným citacím.")
        score -= 35
    if heading_count > 0 and subheading_count == 0:
        issues.append("Dokument nemá styly Nadpis 2. Citace budou méně detailní.")
        score -= 10
    if avg_len and avg_len < 140:
        issues.append("Sekce jsou velmi krátké. Zkontroluj, zda převod nerozsekal odstavce.")
        score -= 15
    if max_len > 5000:
        issues.append("Některé sekce jsou příliš dlouhé. Doporučuji jemnější členění nadpisy.")
        score -= 10
    if sections and missing_heading / max(len(sections), 1) > 0.6:
        issues.append("Většina obsahu je bez nadpisu. Citace budou méně přesné.")
        score -= 10

    score = max(min(score, 100), 0)
    return {
        "score": score,
        "issues": issues,
        "stats": {
            "odstavce": non_empty_paragraphs,
            "sekce": len(sections),
            "nadpisy": heading_count,
            "podnadpisy": subheading_count,
            "prumerna_delka_sekce": avg_len,
            "max_delka_sekce": max_len,
        },
    }


def ingest_documents(files, vpp, ins):
    vpp = normalize(vpp)
    ins = normalize(ins)
    if not files or not vpp or not ins:
        st.sidebar.warning("Vyber soubor, VPP název a pojišťovnu.")
        return

    progress = st.sidebar.progress(0)
    stats = st.sidebar.empty()
    upload_status = st.sidebar.empty()
    report = []
    upload_status.info("Nahrávání bylo spuštěno...")

    total_units = 0
    file_units = {}
    parsed_docx = {}
    for f in files:
        try:
            suffix = Path(f.name).suffix.lower()
            if suffix != ".docx":
                raise ValueError("Podporovaný formát je pouze DOCX.")
            doc_meta = extract_docx_sections(f)
            quality = assess_docx_quality(doc_meta)
            parsed_docx[f.name] = {"meta": doc_meta, "quality": quality}
            file_units[f.name] = max(len(doc_meta["sections"]), 1)
            total_units += file_units[f.name]
        except Exception as e:
            report.append({"file": f.name, "status": "error", "message": str(e)})

    existing_hashes = {
        (r.payload or {}).get("chunk_hash")
        for r in safe_scroll(limit=5000)
        if r.payload
    }
    seen_hashes = set()
    done, total_chunks, duplicates = 0, 0, 0
    start = time.time()
    chunks = []

    for f in files:
        upload_status.info(f"Zpracovávám: {f.name}")
        doc_id = hashlib.sha256(f"{ins}:{vpp}:{f.name}".encode("utf-8")).hexdigest()[:16]
        suffix = Path(f.name).suffix.lower()
        if suffix != ".docx":
            report.append({"file": f.name, "status": "error", "message": "Podporovaný formát je pouze DOCX."})
            continue

        file_chunk_count = 0
        parsed = parsed_docx.get(f.name) or {}
        doc_meta = parsed.get("meta") or {"sections": []}
        quality = parsed.get("quality") or {"score": 0, "issues": ["Chybí data kvality."], "stats": {}}
        sections = doc_meta["sections"]
        if not sections:
            report.append({
                "file": f.name,
                "status": "error",
                "message": "DOCX neobsahuje čitelný text.",
                "quality": quality,
            })
            continue

        for idx, section in enumerate(sections, start=1):
            for c in smart_chunk(section["text"]):
                if len(c) < 50:
                    continue
                chunk_hash = hashlib.sha256(clean_text(c).lower().encode("utf-8")).hexdigest()
                if chunk_hash in existing_hashes or chunk_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(chunk_hash)
                payload = {
                    "id": str(uuid.uuid4()),
                    "text": c,
                    "page": idx,
                    "vpp_name": vpp,
                    "insurer": ins,
                    "doc_id": doc_id,
                    "file_name": f.name,
                    "heading": section.get("heading", ""),
                    "subheading": section.get("subheading", ""),
                    "chunk_hash": chunk_hash,
                    "schema_version": APP_VERSION,
                    "ingested_at": datetime.now().isoformat(timespec="seconds"),
                }
                validate_payload(payload)
                chunks.append(payload)
                total_chunks += 1
                file_chunk_count += 1
            done += 1
            if total_units:
                progress.progress(min(done / total_units, 1.0))
            elapsed = time.time() - start
            eta = (elapsed / done) * (total_units - done) if done and total_units else 0
            stats.markdown(
                f"Sekce: {done}/{total_units} | Chunks: {total_chunks} | Duplicity: {duplicates} | ETA: {round(eta, 1)} s"
            )

        file_status = "ok" if quality["score"] >= 60 else "warning"
        report.append({
            "file": f.name,
            "status": file_status,
            "chunks": file_chunk_count,
            "quality": quality,
        })
        if quality["score"] < 60:
            st.sidebar.warning(f"{f.name}: nízká kvalita dokumentu ({quality['score']} %). Zkontroluj Ingest report.")

    if not chunks:
        upload_status.warning("Nahrávání skončilo bez nových chunků.")
        st.sidebar.info("Nenahrál se žádný nový chunk.")
        with st.sidebar.expander("Ingest report"):
            st.json(report)
        return

    vec = [embed_doc(c["text"]) for c in chunks]
    pts = [PointStruct(id=c["id"], vector=v, payload=c) for c, v in zip(chunks, vec)]
    try:
        batch_upsert(pts)
        upload_status.success(f"Nahrávání dokončeno. Nahráno {len(chunks)} chunků.")
        st.sidebar.success(f"Nahráno {len(chunks)} chunků.")
    except Exception as e:
        upload_status.error("Nahrávání selhalo při ukládání do Qdrantu.")
        handle_error("Nahrávání do Qdrantu selhalo.", e)

    with st.sidebar.expander("Ingest report"):
        st.json(report)


# =========================
# SEARCH (HYBRID + DEBUG)
# =========================
def bm25_scores(query, records, k1=1.5, b=0.75):
    docs = [tokenise((r.payload or {}).get("text", "")) for r in records]
    if not docs:
        return {}
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    df = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    qterms = query_terms(query)
    scores = {}
    for idx, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d) or 1
        score = 0.0
        for term in qterms:
            if not tf[term]:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * ((tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / avgdl)))
        scores[str(getattr(records[idx], "id", idx))] = score
    return scores


@st.cache_data(ttl=BM25_CACHE_TTL_SEC, show_spinner=False, max_entries=64)
def cached_bm25_scores(query, corpus_rows, k1=1.5, b=0.75):
    docs = [tokenise(text) for _, text in corpus_rows]
    if not docs:
        return {}
    avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
    df = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    qterms = query_terms(query)
    scores = {}
    for idx, d in enumerate(docs):
        tf = Counter(d)
        dl = len(d) or 1
        score = 0.0
        for term in qterms:
            if not tf[term]:
                continue
            idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * ((tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / avgdl)))
        scores[corpus_rows[idx][0]] = score
    return scores


def bm25_candidates(query, scroll_filter=None):
    expected = filter_expected(scroll_filter)
    filter_key = filter_signature(expected)
    corpus_rows_raw = cached_bm25_corpus(filter_key, BM25_CORPUS_LIMIT)
    corpus = [
        SimpleNamespace(id=row["id"], payload=row["payload"])
        for row in corpus_rows_raw
    ]
    corpus_rows = tuple(
        (str(getattr(r, "id", idx)), (r.payload or {}).get("text", ""))
        for idx, r in enumerate(corpus)
    )
    scores = cached_bm25_scores(query, corpus_rows)
    by_id = {str(getattr(r, "id", idx)): r for idx, r in enumerate(corpus)}
    ranked_ids = sorted(scores, key=lambda rid: scores[rid], reverse=True)
    candidates = []
    for rid in ranked_ids[:BM25_TOP_N]:
        record = by_id.get(str(rid))
        if record:
            candidates.append(record)
    return candidates, scores, len(corpus)


def normalize_scores(values):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-9:
        if abs(hi) < 1e-9:
            return [0.0 for _ in values]
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def overlap_ratio(a, b):
    left, right = set(tokenise(a)), set(tokenise(b))
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), 1)


def learning_adjustment(qv, chunk_id, text, qtopic):
    rows = cached_learning_rows(limit=1000)
    direct = 0.0
    cluster = 0.0
    global_tuning = 0.0
    cluster_size = 0
    for fb in rows:
        q_emb = fb.get("query_embedding")
        if not q_emb:
            continue
        similarity = cosine(qv, q_emb)
        direction = 1 if fb.get("r") == "like" else -1
        if chunk_id in fb.get("chunk_ids", []) and similarity > 0.55:
            direct += (0.4 if direction > 0 else 0.75) * direction * similarity
        if qtopic and fb.get("query_cluster") == qtopic:
            cluster_size += 1
            cluster += 0.12 * direction
        if similarity > 0.65:
            signal_text = " ".join([fb.get("q", ""), fb.get("a", ""), fb.get("note", "")])
            semantic_overlap = overlap_ratio(signal_text, text)
            global_tuning += 0.25 * direction * similarity * semantic_overlap
    total = max(min(direct + cluster + global_tuning, 1.25), -1.25)
    return {
        "total": total,
        "direct": direct,
        "cluster": cluster,
        "global": global_tuning,
        "cluster_size": cluster_size,
    }


def confidence_score(ranked, coverage):
    if not ranked:
        return 0
    scores = [x["final"] for x in ranked]
    top = ranked[0]
    gap = scores[0] - scores[1] if len(scores) > 1 else 0.4
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.1
    z_gap = gap / (stdev + 1e-6)
    raw = (
        1.25 * z_gap
        + 1.1 * top["rerank_norm"]
        + 0.9 * top["vector_norm"]
        + 0.65 * coverage
        + 0.4 * top["bm25_norm"]
        - 1.0
    )
    return int(min(max(sigmoid(raw) * 100, 5), 98))


def search(q, trace_id):
    started = time.perf_counter()
    debug = {
        "trace_id": trace_id,
        "query": q,
        "expanded_query": rewrite_query(q),
        "query_cluster": detect_topic(q),
        "pipeline": [],
        "candidates": [],
    }
    q_exp = rewrite_query(q)
    qv = cached_query_embedding(q_exp)

    filt = None
    if selected_insurer != "— vyber —" and selected_vpp != "— vyber —":
        filt = Filter(must=[
            FieldCondition(key="insurer", match=MatchValue(value=normalize(selected_insurer))),
            FieldCondition(key="vpp_name", match=MatchValue(value=normalize(selected_vpp)))
        ])

    try:
        vector_res = qdrant.query_points(COLLECTION_NAME, query=qv, query_filter=filt, limit=80).points
    except Exception as e:
        handle_error("Vector search selhal.", e, trace_id=trace_id, show=False)
        vector_res = []

    if not vector_res:
        try:
            vector_res = qdrant.query_points(COLLECTION_NAME, query=qv, limit=80).points
        except Exception as e:
            handle_error("Fallback search selhal.", e, trace_id=trace_id, show=False)
            vector_res = []

    debug["pipeline"].append({"stage": "vector", "count": len(vector_res)})
    bm25_res, bm25_full_scores, bm25_corpus_count = bm25_candidates(q_exp, filt)
    debug["pipeline"].append({
        "stage": "bm25_full_collection",
        "count": len(bm25_res),
        "corpus_count": bm25_corpus_count,
        "limit": BM25_CORPUS_LIMIT,
    })

    vector_scores = {str(r.id): float(getattr(r, "score", 0.0) or 0.0) for r in vector_res}
    merged = {}
    for r in bm25_res + vector_res:
        merged[str(r.id)] = r
    candidate_res = list(merged.values())

    if not candidate_res:
        append_jsonl(METRICS_FILE, {
            "trace_id": trace_id,
            "type": "search",
            "status": "empty",
            "latency_ms": round((time.perf_counter() - started) * 1000),
        })
        return [], 0, debug

    vector_raw, bm25_raw, keyword_raw, heading_raw = [], [], [], []
    base = []
    for r in candidate_res:
        txt = r.payload["text"]
        heading = r.payload.get("heading", "")
        subheading = r.payload.get("subheading", "")
        heading_text = f"{heading} {subheading}".strip()
        vector = vector_scores.get(str(r.id), 0.0)
        bscore = bm25_full_scores.get(str(r.id), 0.0)
        kw = keyword_score(q, txt)
        heading_kw = keyword_score(q, heading_text)
        focus_boost = section_focus_boost(q, heading, subheading)
        vector_raw.append(vector)
        bm25_raw.append(bscore)
        keyword_raw.append(kw)
        heading_raw.append(heading_kw)
        base.append({
            "record": r,
            "vector": vector,
            "bm25": bscore,
            "keyword": kw,
            "heading_keyword": heading_kw,
            "focus_boost": focus_boost,
        })

    vector_norm = normalize_scores(vector_raw)
    bm25_norm = normalize_scores(bm25_raw)
    keyword_norm = normalize_scores(keyword_raw)
    heading_norm = normalize_scores(heading_raw)
    for idx, item in enumerate(base):
        item["vector_norm"] = vector_norm[idx]
        item["bm25_norm"] = bm25_norm[idx]
        item["keyword_norm"] = keyword_norm[idx]
        item["heading_norm"] = heading_norm[idx]

    debug["pipeline"].append({"stage": "candidate_merge_keyword", "count": len(base)})

    pre_ranked = sorted(
        base,
        key=lambda item: (
            0.42 * item["vector_norm"]
            + 0.24 * item["bm25_norm"]
            + 0.12 * item["keyword_norm"]
            + 0.22 * item["heading_norm"]
            + 0.25 * item["focus_boost"]
        ),
        reverse=True,
    )
    rerank_base = pre_ranked[:RERANK_CANDIDATE_LIMIT]
    debug["pipeline"].append({
        "stage": "pre_rerank_limit",
        "count": len(rerank_base),
        "limit": RERANK_CANDIDATE_LIMIT,
    })

    pairs = [(q, item["record"].payload["text"]) for item in rerank_base]
    try:
        rerank_raw = list(reranker.predict(pairs))
    except Exception as e:
        handle_error("Reranker selhal, pokračuji bez něj.", e, trace_id=trace_id, show=False)
        rerank_raw = [0.0] * len(pairs)
    rerank_norm = normalize_scores(rerank_raw)

    ranked = []
    q_terms = set(tokenise(q))
    qtopic = detect_topic(q)
    for idx, item in enumerate(rerank_base):
        r = item["record"]
        txt = r.payload["text"]
        sentence = extract_sentence(txt, q)
        coverage = len(q_terms & set(tokenise(sentence))) / max(len(q_terms), 1)
        learn = learning_adjustment(qv, str(r.id), txt, qtopic)
        final = (
            0.28 * item["vector_norm"]
            + 0.20 * item["bm25_norm"]
            + 0.10 * item["keyword_norm"]
            + 0.16 * item["heading_norm"]
            + 0.26 * rerank_norm[idx]
            + 0.20 * item["focus_boost"]
            + learn["total"]
        )
        ranked.append({
            "id": str(r.id),
            "record": r,
            "final": final,
            "vector": item["vector"],
            "vector_norm": item["vector_norm"],
            "bm25": item["bm25"],
            "bm25_norm": item["bm25_norm"],
            "keyword": item["keyword"],
            "keyword_norm": item["keyword_norm"],
            "heading_keyword": item["heading_keyword"],
            "heading_norm": item["heading_norm"],
            "focus_boost": item["focus_boost"],
            "rerank": float(rerank_raw[idx]),
            "rerank_norm": rerank_norm[idx],
            "learning": learn["total"],
            "learning_breakdown": learn,
            "coverage": coverage,
            "exact": sentence,
            "token_hits": sorted(q_terms & set(tokenise(sentence))),
        })

    ranked = sorted(ranked, key=lambda x: x["final"], reverse=True)
    debug["pipeline"].append({"stage": "rerank_learning", "count": len(ranked)})

    for pos, item in enumerate(ranked, start=1):
        r = item["record"]
        debug["candidates"].append({
            "rank": pos,
            "id": item["id"],
            "page": r.payload.get("page"),
            "insurer": r.payload.get("insurer"),
            "vpp": r.payload.get("vpp_name"),
            "heading": r.payload.get("heading"),
            "final": round(item["final"], 4),
            "vector": round(item["vector"], 4),
            "vector_norm": round(item["vector_norm"], 4),
            "bm25": round(item["bm25"], 4),
            "bm25_norm": round(item["bm25_norm"], 4),
            "keyword": item["keyword"],
            "keyword_norm": round(item["keyword_norm"], 4),
            "heading_keyword": item["heading_keyword"],
            "heading_norm": round(item["heading_norm"], 4),
            "focus_boost": round(item["focus_boost"], 4),
            "rerank": round(item["rerank"], 4),
            "rerank_norm": round(item["rerank_norm"], 4),
            "learning": round(item["learning"], 4),
            "learning_breakdown": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in item["learning_breakdown"].items()
            },
            "coverage": round(item["coverage"], 4),
            "token_hits": item["token_hits"],
            "exact": item["exact"],
        })

    coverage = ranked[0]["coverage"] if ranked else 0
    conf = confidence_score(ranked, coverage)
    debug["confidence"] = {
        "value": conf,
        "method": "sigmoid(z_gap + rerank + vector + bm25 + coverage)",
    }
    append_jsonl(METRICS_FILE, {
        "trace_id": trace_id,
        "type": "search",
        "status": "ok",
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "candidates": len(ranked),
        "confidence": conf,
    })

    ctx = [{
        "id": item["id"],
        "text": item["record"].payload["text"],
        "exact": item["exact"],
        "page": item["record"].payload["page"],
        "heading": item["record"].payload.get("heading", ""),
        "subheading": item["record"].payload.get("subheading", ""),
    } for item in ranked[:16]]

    return ctx, conf, debug


# =========================
# CITATION VALIDATION
# =========================
def compact_for_match(text):
    text = clean_text(text).lower()
    text = re.sub(r"\[str\.\s*\d+\]", "", text, flags=re.I)
    text = re.sub(r"[^a-z0-9á-ž]+", " ", text)
    return text.strip()


def find_exact_span(needle, haystack):
    if not needle or not haystack:
        return None
    direct = haystack.find(needle)
    if direct >= 0:
        return {"start": direct, "end": direct + len(needle), "text": needle}

    compact_needle = compact_for_match(needle)
    compact_haystack = compact_for_match(haystack)
    if compact_needle and compact_needle in compact_haystack:
        return {"start": None, "end": None, "text": needle, "normalized_match": True}
    return None


def strict_answer_from_context(ctx):
    lines = []
    for c in ctx:
        span = find_exact_span(c.get("exact", ""), c.get("text", ""))
        if span:
            lines.append(f"- {span['text']} [str. {c['page']}]")
    return "\n".join(lines)


def citations_from_context(ctx):
    blocks = []
    for i, c in enumerate(ctx, start=1):
        heading = c.get("heading") or "Bez nadpisu"
        subheading = c.get("subheading") or ""
        text = clean_text(c.get("text", ""))
        if not text:
            continue
        block = [f"[CITACE {i}]", f"Nadpis: {heading}"]
        if subheading:
            block.append(f"Podnadpis: {subheading}")
        block.append("Text:")
        block.append(text)
        blocks.append("\n".join(block))
    return "\n\n" + ("\n\n" + ("-" * 72) + "\n\n").join(blocks) if blocks else ""


def build_summary_from_context(ctx, question):
    if not ctx:
        return "V dostupném textu to není uvedeno."
    focused = [
        c for c in ctx
        if "výluk" in clean_text(f"{c.get('heading', '')} {c.get('subheading', '')}").lower()
    ]
    chosen = focused[:2] if focused else ctx[:2]
    heads = []
    for c in chosen:
        h = c.get("heading") or "Bez nadpisu"
        s = c.get("subheading") or ""
        heads.append(f"{h} / {s}" if s else h)
    sample_text = clean_text(chosen[0].get("text", ""))[:260] if chosen else ""
    sample_text = sample_text + "..." if sample_text and len(sample_text) >= 260 else sample_text
    if heads:
        return (
            f"K otázce „{question}“ jsou relevantní výluky uvedeny v sekcích: "
            + "; ".join(heads)
            + (f". Stručně: {sample_text}" if sample_text else ".")
        )
    return "V dostupném textu to není uvedeno."


def validate_answer(answer, ctx):
    source_texts = [c.get("text", "") for c in ctx]
    source_spans = [c.get("exact", "") for c in ctx]
    compact_sources = [compact_for_match(x) for x in source_texts + source_spans]
    answer_sentences = extract_sentences(answer)
    unsupported = []
    matched_spans = []
    for s in answer_sentences:
        plain = re.sub(r"\[str\.\s*\d+\]", "", s, flags=re.I).strip(" -")
        if not plain:
            continue
        compact_plain = compact_for_match(plain)
        exact_match = any(find_exact_span(plain, src) for src in source_texts + source_spans)
        substring_match = compact_plain and any(compact_plain in src for src in compact_sources)
        token_match = any(overlap_ratio(plain, src) >= 0.8 for src in source_spans)
        if exact_match or substring_match or token_match:
            matched_spans.append(plain)
        else:
            unsupported.append(s)
    return unsupported, matched_spans


def render_chat_history(messages):
    for idx, m in enumerate(messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("role") == "assistant" and m.get("citations"):
                with st.expander("Citace"):
                    st.code(m["citations"])


def render_typing_indicator(ph, label="Pisu odpoved"):
    ph.markdown(f"{label}...")


def render_feedback_panel():
    messages = st.session_state.messages
    if not messages or messages[-1].get("role") != "assistant":
        return

    idx = len(messages) - 1
    if st.session_state.feedback_done.get(idx):
        return

    prompt_text = ""
    for i in range(idx - 1, -1, -1):
        if messages[i].get("role") == "user":
            prompt_text = messages[i].get("content", "")
            break
    answer_text = messages[idx].get("content", "")
    chunk_ids = st.session_state.feedback_chunk_ids.get(idx, [])

    c1, c2 = st.columns(2)
    if c1.button("Odpověď pomohla", key=f"l{idx}"):
        save_feedback(prompt_text, answer_text, "like", "", chunk_ids)
        st.session_state.feedback_done[idx] = True
        st.rerun()

    if c2.button("Nahlásit problém", key=f"d{idx}"):
        st.session_state.feedback_open[idx] = True

    if st.session_state.feedback_open.get(idx):
        note = st.text_input("Co bylo špatně?", key=f"n{idx}")
        if st.button("Odeslat", key=f"s{idx}"):
            save_feedback(prompt_text, answer_text, "dislike", note, chunk_ids)
            st.session_state.feedback_done[idx] = True
            st.session_state.feedback_open[idx] = False
            st.rerun()


def inject_design():
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f7f9fc;
            --surface: #ffffff;
            --surface-muted: #f4f7fc;
            --text: #101828;
            --muted: #667085;
            --border: #e6ebf2;
            --accent: #1d4ed8;
            --accent-strong: #1e40af;
            --accent-press: #1e3a8a;
            --accent-soft: #eff5ff;
            --ok: #0f766e;
            --danger: #b42318;
            --shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
            --app-font: "Inter", "Segoe UI", Arial, Helvetica, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(47, 111, 237, 0.08), transparent 28rem),
                linear-gradient(180deg, #fbfdff 0%, #f4f7fc 100%);
            color: var(--text);
            font-family: var(--app-font);
        }

        html,
        body,
        [class*="css"],
        [class*="st-"],
        button,
        input,
        textarea,
        select {
            font-family: var(--app-font) !important;
        }

        .material-symbols-outlined,
        .material-symbols-rounded,
        .material-symbols-sharp {
            font-family: "Segoe UI Symbol", "Apple Color Emoji", "Noto Color Emoji", sans-serif !important;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f7faff 100%);
            border-right: 1px solid var(--border);
            box-shadow: 8px 0 30px rgba(15, 23, 42, 0.03);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: var(--text);
        }

        [data-testid="stSidebar"]::before {
            content: "";
            display: block;
            height: 4px;
            background: linear-gradient(90deg, var(--accent), #466ea8);
            margin: -1rem -1rem 1.1rem -1rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            color: var(--accent);
            letter-spacing: 0;
            font-weight: 700;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 5.8rem;
        }

        .app-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: var(--shadow);
            padding: 34px 38px;
            margin-bottom: 24px;
        }

        .app-hero::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 5px;
            background: linear-gradient(180deg, var(--accent), #466ea8);
        }

        .app-hero::after {
            content: "";
            position: absolute;
            right: -70px;
            top: -110px;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: rgba(47, 111, 237, 0.11);
        }

        .app-eyebrow {
            position: relative;
            z-index: 1;
            color: var(--accent);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .app-title {
            position: relative;
            z-index: 1;
            color: #0f172a;
            font-size: 2.35rem;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: 0;
            margin: 0 0 12px 0;
        }

        .app-subtitle {
            position: relative;
            z-index: 1;
            color: var(--muted);
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 840px;
            margin: 0;
        }

        .selection-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 14px;
            margin: 0 0 24px 0;
        }

        .selection-item {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.96);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
            padding: 16px 18px;
        }

        .selection-label {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 2px;
        }

        .selection-value {
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
            overflow-wrap: anywhere;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.98);
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }

        div[data-testid="stChatMessage"] * {
            color: var(--text) !important;
        }

        div[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] svg,
        div[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] {
            color: var(--accent) !important;
            fill: var(--accent) !important;
            opacity: 1 !important;
        }

        [data-testid="chatAvatarIcon-user"] span,
        [data-testid="chatAvatarIcon-assistant"] span {
            font-size: 0 !important;
        }

        [data-testid="chatAvatarIcon-user"]::after,
        [data-testid="chatAvatarIcon-assistant"]::after {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
            border-radius: 7px;
            color: #ffffff;
            font-size: 11px;
            font-weight: 700;
            content: "";
        }

        [data-testid="chatAvatarIcon-user"]::after {
            content: "U";
            background: #1d4ed8;
        }

        [data-testid="chatAvatarIcon-assistant"]::after {
            content: "AI";
            background: #0f766e;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: var(--accent-soft);
            border-color: rgba(31, 78, 140, 0.18);
        }

        .stChatInputContainer {
            border-top: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.94);
            backdrop-filter: blur(10px);
        }

        textarea,
        input,
        [data-baseweb="select"] > div,
        [data-testid="stFileUploader"] section {
            border-radius: 12px !important;
            border-color: var(--border) !important;
            background: #ffffff !important;
            color: var(--text) !important;
        }

        [data-baseweb="input"] > div,
        [data-baseweb="textarea"] > div,
        [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
        }

        [data-baseweb="select"] span,
        [data-baseweb="select"] input,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"] textarea {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }

        [data-baseweb="select"] svg,
        [data-baseweb="input"] svg {
            color: var(--accent) !important;
            fill: var(--accent) !important;
        }

        [data-baseweb="select"] .material-symbols-rounded,
        [data-baseweb="select"] .material-symbols-outlined {
            font-size: 0 !important;
            width: 14px;
            height: 14px;
            display: inline-block;
            position: relative;
        }

        [data-baseweb="select"] .material-symbols-rounded::after,
        [data-baseweb="select"] .material-symbols-outlined::after {
            content: "▾";
            font-size: 11px;
            color: var(--accent);
            position: absolute;
            inset: -1px 0 0 0;
            text-align: center;
        }

        textarea:focus,
        input:focus,
        button:focus,
        [data-baseweb="select"] div:focus {
            outline: 3px solid rgba(31, 78, 140, 0.24) !important;
            outline-offset: 2px !important;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid var(--accent);
            background: var(--accent);
            color: white;
            font-weight: 700;
            padding: 0.5rem 1rem;
            box-shadow: 0 10px 22px rgba(29, 78, 216, 0.34);
            min-height: 42px;
            transition: background 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease;
        }

        .stButton > button:hover {
            border-color: var(--accent-strong);
            background: var(--accent-strong);
            color: white;
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(30, 64, 175, 0.36);
        }

        .stButton > button:active {
            border-color: var(--accent-press);
            background: var(--accent-press);
            transform: translateY(0);
        }

        .stButton > button:disabled,
        .stButton > button[disabled] {
            background: #dbe7ff !important;
            border-color: #d1ddf6 !important;
            color: #6b7fa6 !important;
            box-shadow: none !important;
            opacity: 1 !important;
            transform: none !important;
            cursor: not-allowed !important;
        }

        .stButton > button span,
        .stButton > button p {
            color: inherit !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }

        [data-testid="stFileUploader"] section {
            border: 1px dashed #c8d7f7 !important;
            border-radius: 14px !important;
            background: #f8fbff !important;
            padding: 0.9rem !important;
        }

        [data-testid="stFileUploader"] button {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent-strong) !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            min-height: 40px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            white-space: nowrap !important;
            box-shadow: 0 8px 18px rgba(29, 78, 216, 0.28) !important;
        }

        [data-testid="stFileUploader"] button:hover {
            background: var(--accent-strong) !important;
            border-color: var(--accent-press) !important;
        }

        [data-testid="stFileUploader"] button span,
        [data-testid="stFileUploader"] button p {
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }

        [data-testid="stFileUploader"] button p {
            display: none !important;
        }

        [data-testid="stFileUploader"] small {
            color: var(--muted) !important;
        }

        [data-testid="stProgressBar"] + div,
        [data-testid="stProgressBar"] + div * {
            color: var(--text) !important;
        }

        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code {
            background: #0f172a !important;
            color: #e2e8f0 !important;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.86);
        }

        .stAlert {
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid rgba(31, 78, 140, 0.15);
            background: #f7faff;
            color: #1e3a64;
        }

        [data-testid="stBottomBlockContainer"],
        [data-testid="stChatInput"],
        [data-testid="stChatInput"] > div {
            background: rgba(255, 255, 255, 0.96) !important;
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid var(--border);
            box-shadow: 0 -8px 22px rgba(15, 23, 42, 0.05);
            padding: 0.55rem 0.8rem 0.75rem 0.8rem;
            backdrop-filter: blur(8px);
        }

        [data-testid="stChatInput"] textarea,
        [data-testid="stChatInput"] textarea:focus {
            background: #ffffff !important;
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
            border: 1px solid rgba(31, 78, 140, 0.55) !important;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
            border-radius: 14px !important;
            min-height: 52px !important;
            padding: 0.72rem 0.9rem !important;
            caret-color: var(--accent) !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: #667085 !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] button {
            background: var(--accent) !important;
            color: #ffffff !important;
            border-radius: 12px !important;
            border: 1px solid var(--accent-strong) !important;
            box-shadow: 0 8px 18px rgba(29, 78, 216, 0.30) !important;
        }

        [data-testid="stChatInput"] button svg {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        .small-note {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 1.4rem;
            }
            .app-hero {
                padding: 22px;
            }
            .app-title {
                font-size: 1.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <section class="app-hero">
            <div class="app-eyebrow">Interní nástroj pro analýzu dokumentů</div>
            <h1 class="app-title">VPP Checker</h1>
            <p class="app-subtitle">
                Odpovědi se opírají pouze o nahrané pojistné podmínky.
                Vyber pojišťovnu a dokument, polož dotaz a zkontroluj citace ve výsledku.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_selection_status(insurer, vpp):
    insurer_label = insurer if insurer != "— vyber —" else "Není vybráno"
    vpp_label = vpp if vpp != "— vyber —" else "Není vybráno"
    readiness = "Připraveno k dotazu" if insurer != "— vyber —" and vpp != "— vyber —" else "Vyber dokument pro chat"
    st.markdown(
        f"""
        <div class="selection-strip">
            <div class="selection-item">
                <div class="selection-label">Pojistovna</div>
                <div class="selection-value">{insurer_label}</div>
            </div>
            <div class="selection-item">
                <div class="selection-label">VPP dokument</div>
                <div class="selection-value">{vpp_label}</div>
            </div>
            <div class="selection-item">
                <div class="selection-label">Stav</div>
                <div class="selection-value">{readiness}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# UI
# =========================
inject_design()
render_header()
st.session_state.debug_mode = st.sidebar.toggle("Debug režim", value=st.session_state.debug_mode)

st.sidebar.markdown("## Vyhledávání")
selected_insurer = st.sidebar.selectbox("Pojišťovna", ["— vyber —"] + INSURERS)
selected_vpp = "— vyber —"

if selected_insurer != "— vyber —":
    vpps = get_vpps(selected_insurer)
    if vpps:
        selected_vpp = st.sidebar.selectbox("VPP", ["— vyber —"] + vpps)

render_selection_status(selected_insurer, selected_vpp)

# ADMIN
if not st.session_state.logged:
    st.sidebar.markdown("## Správa")
    if st.sidebar.text_input("Admin heslo", type="password") == st.secrets["ADMIN_PASSWORD"]:
        st.session_state.logged = True
        st.rerun()
else:
    st.sidebar.markdown("## Správa dokumentů")
    if st.sidebar.button("Odhlásit se"):
        st.session_state.logged = False
        st.rerun()
    f = st.sidebar.file_uploader(
        "Dokumenty (DOCX)",
        type=["docx"],
        accept_multiple_files=True,
        key=st.session_state.upload_key,
    )
    v = st.sidebar.text_input("VPP název")
    i = st.sidebar.selectbox("Pojišťovna", INSURERS, key="admin_insurer")
    if st.sidebar.button("Nahrát"):
        ingest_documents(f or [], v or "", i)
        st.session_state.upload_key = str(uuid.uuid4())
        st.rerun()

    with st.sidebar.expander("Nahrané VPP"):
        docs = get_uploaded_docs()
        if docs:
            for doc_id, meta in docs.items():
                st.write(f"{meta['insurer']} / {meta['vpp']} | pages: {len(meta['pages'])} | chunks: {meta['chunks']}")
        else:
            st.info("Zatím nejsou nahrané dokumenty.")

    del_vpp = st.sidebar.selectbox("Smazat VPP", ["— vyber —"] + get_vpps(selected_insurer) if selected_insurer != "— vyber —" else ["— vyber —"])
    if st.sidebar.button("Smazat vybranou VPP") and selected_insurer != "— vyber —" and del_vpp != "— vyber —":
        try:
            qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(filter=Filter(must=[
                    FieldCondition(key="insurer", match=MatchValue(value=normalize(selected_insurer))),
                    FieldCondition(key="vpp_name", match=MatchValue(value=normalize(del_vpp))),
                ]))
            )
            st.sidebar.success("VPP smazána.")
            st.rerun()
        except Exception as e:
            handle_error("Mazání VPP selhalo.", e)

    if st.sidebar.button("Rebuild kolekce"):
        try:
            init_collection(force=True)
            st.sidebar.success("Kolekce obnovena.")
            st.rerun()
        except Exception as e:
            handle_error("Rebuild kolekce selhal.", e)

if st.session_state.last_errors and st.session_state.debug_mode:
    with st.sidebar.expander("Poslední chyby"):
        st.json(st.session_state.last_errors[-5:])

# CHAT
render_chat_history(st.session_state.messages)
render_feedback_panel()

if prompt := st.chat_input("Napiš dotaz k dokumentu..."):
    trace_id = str(uuid.uuid4())
    if selected_insurer == "— vyber —" or selected_vpp == "— vyber —":
        st.warning("Vyber pojišťovnu a VPP")
        st.stop()

    if rate_limited():
        st.warning("Prilis mnoho dotazu za kratkou dobu. Zkus to prosim za chvili.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})
    persist_memory("user", prompt)

    with st.chat_message("assistant"):
        ph = st.empty()
        full = ""
        citations = ""
        last_render = 0
        render_typing_indicator(ph)

        with st.spinner("Přemýšlím..."):
            ctx, conf, debug = search(prompt, trace_id)

        if not ctx:
            full = "Nenalezeno. Kontaktuj vedení směny."
            ph.markdown(full)
        else:
            memory = get_memory(prompt)
            combined = "\n\n".join([
                f"[NADPIS] {c.get('heading') or 'Bez nadpisu'}\n"
                f"[PODNADPIS] {c.get('subheading') or 'Bez podnadpisu'}\n"
                f"{c['text']}"
                for c in ctx
            ])

            prompt_ai = f"""
TEXT je relevantní výběr z dokumentu. Může být v češtině nebo ve slovenštině.
Použij pouze věty v části TEXT. Nevymýšlej.
Projdi celý dodaný TEXT a najdi pasáže, které odpovídají na otázku.
Napiš krátké, konkrétní shrnutí v češtině (3-6 vět).
Uveď, ze kterých sekcí nebo nadpisů je odpověď čerpána, pokud to jde.
Nepřidávej informace z jiných zdrojů.
Pokud je v TEXT alespoň jedna relevantní pasáž, nikdy nepiš "V dostupném textu to není uvedeno.".
Použij tuto větu pouze tehdy, pokud v TEXT opravdu není nic relevantního.
Shrnutí musí být konkrétní, ne jen obecné názvy sekcí.

KONTEXT PAMĚTI:
{memory}

TEXT:
{combined}

Otázka: {prompt}
"""

            stream = generate_safe(prompt_ai, True, trace_id=trace_id)

            if stream:
                try:
                    for chunk in stream:
                        text_piece = ""
                        if hasattr(chunk, "text"):
                            text_piece = chunk.text
                        elif isinstance(chunk, dict):
                            text_piece = chunk.get("text", "")
                        else:
                            text_piece = str(chunk)
                        if text_piece:
                            full += text_piece
                            now = time.perf_counter()
                            if now - last_render >= STREAM_RENDER_INTERVAL_SEC:
                                ph.markdown(full + "▌")
                                last_render = now
                except Exception as e:
                    handle_error("Stream odpovědi se přerušil.", e, trace_id=trace_id, show=False)

            if not full.strip():
                full = "V dostupném textu to není uvedeno."

            summary = clean_text(full)
            unsupported, matched_spans = validate_answer(summary, ctx)
            debug["post_validation"] = {
                "status": "ok_with_summary_mode",
                "unsupported_count": len(unsupported),
                "matched_spans": matched_spans,
                "mode": "summary_plus_explicit_citations",
            }

            citations = citations_from_context(ctx).strip()
            if citations and (
                "v dostupném textu to není uvedeno" in summary.lower()
                or len(summary) < 40
            ):
                summary = build_summary_from_context(ctx, prompt)
            if citations:
                full = f"**Shrnutí**\n{summary}\n\nCitace otevřeš přes tlačítko **Citace** níže."
            else:
                full = f"**Shrnutí**\n{summary}"

            ph.markdown(full)
            if citations:
                with st.expander("Citace"):
                    st.code(citations)

        st.caption(f"Spolehlivost: {conf}% | Trace: {trace_id}")
        st.session_state.messages.append({"role": "assistant", "content": full, "citations": citations})
        persist_memory("assistant", full)
        log_query(prompt, selected_insurer, selected_vpp, conf, trace_id)

        with st.expander("Debug"):
            st.json(debug if st.session_state.debug_mode else {
                "trace_id": trace_id,
                "confidence": debug.get("confidence"),
                "top_candidates": debug.get("candidates", [])[:5],
            })

        assistant_idx = len(st.session_state.messages) - 1
        st.session_state.feedback_chunk_ids[assistant_idx] = [c["id"] for c in ctx]
