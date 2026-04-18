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
import pdfplumber
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
from pypdf import PdfReader
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
    "the", "and", "or", "to", "of", "pojištění", "pojistné", "podmínky"
}

SYNONYMS = {
    "výluka": ["vylouceni", "nekryje", "nevztahuje", "neplati"],
    "krytí": ["plneni", "hradí", "vztahuje", "pojistna udalost"],
    "limit": ["maximalni castka", "strop", "omezeni"],
    "léčba": ["leceni", "zdravotni pece", "osetreni"],
    "zavazadla": ["bagaz", "osobni veci"],
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
    return " ".join(query_terms(f"{q} výluky krytí podmínky limit plnění"))


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
def ingest_pdf(files, vpp, ins):
    vpp = normalize(vpp)
    ins = normalize(ins)
    if not files or not vpp or not ins:
        st.sidebar.warning("Vyber PDF, VPP název a pojišťovnu.")
        return

    progress = st.sidebar.progress(0)
    stats = st.sidebar.empty()
    report = []

    total_pages = 0
    file_pages = {}
    for f in files:
        try:
            file_pages[f.name] = len(PdfReader(f).pages)
            total_pages += file_pages[f.name]
            f.seek(0)
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
        try:
            f.seek(0)
            reader = PdfReader(f)
            doc_id = hashlib.sha256(f"{ins}:{vpp}:{f.name}".encode("utf-8")).hexdigest()[:16]
        except Exception as e:
            report.append({"file": f.name, "status": "error", "message": str(e)})
            continue

        file_chunk_count = 0
        for i, p in enumerate(reader.pages):
            t = None
            try:
                t = p.extract_text()
            except Exception:
                t = None

            if not t:
                try:
                    f.seek(0)
                    with pdfplumber.open(f) as pdf:
                        t = pdf.pages[i].extract_text()
                except Exception as e:
                    report.append({"file": f.name, "page": i + 1, "status": "error", "message": str(e)})
                    continue

            for c in smart_chunk(t):
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
                    "page": i + 1,
                    "vpp_name": vpp,
                    "insurer": ins,
                    "doc_id": doc_id,
                    "file_name": f.name,
                    "heading": extract_heading(c),
                    "chunk_hash": chunk_hash,
                    "schema_version": APP_VERSION,
                    "ingested_at": datetime.now().isoformat(timespec="seconds"),
                }
                validate_payload(payload)
                chunks.append(payload)
                total_chunks += 1
                file_chunk_count += 1

            done += 1
            if total_pages:
                progress.progress(min(done / total_pages, 1.0))

            elapsed = time.time() - start
            eta = (elapsed / done) * (total_pages - done) if done and total_pages else 0
            stats.markdown(
                f"Pages: {done}/{total_pages} | Chunks: {total_chunks} | Duplicity: {duplicates} | ETA: {round(eta, 1)}s"
            )
        report.append({"file": f.name, "status": "ok", "chunks": file_chunk_count})

    if not chunks:
        st.sidebar.info("Nenahrál se žádný nový chunk.")
        with st.sidebar.expander("Ingest report"):
            st.json(report)
        return

    vec = [embed_doc(c["text"]) for c in chunks]
    pts = [PointStruct(id=c["id"], vector=v, payload=c) for c, v in zip(chunks, vec)]
    try:
        batch_upsert(pts)
        st.sidebar.success(f"Nahráno {len(chunks)} chunků.")
    except Exception as e:
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

    vector_raw, bm25_raw, keyword_raw = [], [], []
    base = []
    for r in candidate_res:
        txt = r.payload["text"]
        vector = vector_scores.get(str(r.id), 0.0)
        bscore = bm25_full_scores.get(str(r.id), 0.0)
        kw = keyword_score(q, txt)
        vector_raw.append(vector)
        bm25_raw.append(bscore)
        keyword_raw.append(kw)
        base.append({"record": r, "vector": vector, "bm25": bscore, "keyword": kw})

    vector_norm = normalize_scores(vector_raw)
    bm25_norm = normalize_scores(bm25_raw)
    keyword_norm = normalize_scores(keyword_raw)
    for idx, item in enumerate(base):
        item["vector_norm"] = vector_norm[idx]
        item["bm25_norm"] = bm25_norm[idx]
        item["keyword_norm"] = keyword_norm[idx]

    debug["pipeline"].append({"stage": "candidate_merge_keyword", "count": len(base)})

    pre_ranked = sorted(
        base,
        key=lambda item: (
            0.48 * item["vector_norm"]
            + 0.34 * item["bm25_norm"]
            + 0.18 * item["keyword_norm"]
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
            0.34 * item["vector_norm"]
            + 0.26 * item["bm25_norm"]
            + 0.12 * item["keyword_norm"]
            + 0.28 * rerank_norm[idx]
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
    } for item in ranked[:5]]

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


def grouped_messages(messages):
    grouped = []
    for msg in messages:
        if grouped and grouped[-1]["role"] == msg["role"]:
            grouped[-1]["content"] += "\n\n" + msg["content"]
        else:
            grouped.append(dict(msg))
    return grouped


def render_chat_history(messages):
    for m in grouped_messages(messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])


def render_typing_indicator(ph, label="Pisu odpoved"):
    ph.markdown(f"{label}...")


def inject_design():
    st.markdown(
        """
        <style>
        @font-face {
            font-family: 'DaxlinePro';
            src:
                url('https://www.europ-assistance.cz/assets/fonts/DaxlinePro-Regular.woff2?peze2E') format('woff2'),
                url('https://www.europ-assistance.cz/assets/fonts/DaxlinePro-Regular.woff?peze2E') format('woff');
            font-weight: 400;
            font-style: normal;
            font-display: swap;
        }

        @font-face {
            font-family: 'DaxlinePro';
            src:
                url('https://www.europ-assistance.cz/assets/fonts/DaxlinePro-Medium.woff2?peze2E') format('woff2'),
                url('https://www.europ-assistance.cz/assets/fonts/DaxlinePro-Medium.woff?peze2E') format('woff');
            font-weight: 500;
            font-style: normal;
            font-display: swap;
        }

        @font-face {
            font-family: 'DaxlinePro';
            src:
                url('https://www.europ-assistance.cz/assets/fonts/DaxlinePro-Bold.woff2?peze2E') format('woff2'),
                url('https://www.europ-assistance.cz/assets/fonts/DaxlinePro-Bold.woff?peze2E') format('woff');
            font-weight: 700;
            font-style: normal;
            font-display: swap;
        }

        :root {
            --app-bg: #f6f8f7;
            --surface: #ffffff;
            --surface-muted: #eef4f1;
            --text: #17201c;
            --muted: #5d6b63;
            --border: #d9e2dd;
            --accent: #117a65;
            --accent-strong: #0b604f;
            --warning: #b45309;
            --danger: #b42318;
            --shadow: 0 12px 34px rgba(23, 32, 28, 0.08);
            --ea-font: 'DaxlinePro', Arial, Helvetica, sans-serif;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f6f8f7 0%, #edf3f0 100%);
            color: var(--text);
            font-family: var(--ea-font);
        }

        html,
        body,
        [class*="css"],
        [class*="st-"],
        button,
        input,
        textarea,
        select {
            font-family: var(--ea-font) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(246, 248, 247, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(217, 226, 221, 0.65);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            color: var(--text);
            letter-spacing: 0;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 5rem;
        }

        .app-hero {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: var(--surface);
            box-shadow: var(--shadow);
            padding: 22px 24px;
            margin-bottom: 18px;
        }

        .app-eyebrow {
            color: var(--accent-strong);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .app-title {
            color: var(--text);
            font-size: 2rem;
            line-height: 1.15;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0 0 8px 0;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.55;
            max-width: 760px;
            margin: 0;
        }

        .selection-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 10px;
            margin: 0 0 18px 0;
        }

        .selection-item {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.92);
            padding: 12px 14px;
        }

        .selection-label {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 2px;
        }

        .selection-value {
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 8px 22px rgba(23, 32, 28, 0.05);
            padding: 0.85rem 1rem;
            margin-bottom: 0.85rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: #eef7f3;
            border-color: #c9ded6;
        }

        .stChatInputContainer {
            border-top: 1px solid var(--border);
            background: rgba(246, 248, 247, 0.92);
            backdrop-filter: blur(10px);
        }

        textarea,
        input,
        [data-baseweb="select"] > div,
        [data-testid="stFileUploader"] section {
            border-radius: 8px !important;
            border-color: var(--border) !important;
        }

        textarea:focus,
        input:focus,
        button:focus,
        [data-baseweb="select"] div:focus {
            outline: 3px solid rgba(17, 122, 101, 0.22) !important;
            outline-offset: 2px !important;
        }

        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--accent);
            background: var(--accent);
            color: white;
            font-weight: 700;
        }

        .stButton > button:hover {
            border-color: var(--accent-strong);
            background: var(--accent-strong);
            color: white;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.86);
        }

        .stAlert {
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .small-note {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 1.2rem;
            }
            .app-hero {
                padding: 18px;
            }
            .app-title {
                font-size: 1.55rem;
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
            <div class="app-eyebrow">Interni nastroj pro kontrolu VPP</div>
            <h1 class="app-title">VPP Checker</h1>
            <p class="app-subtitle">
                Odpovedi se opiraji pouze o nahrane pojistne podminky.
                Vyber pojistovnu a dokument, poloz dotaz a zkontroluj citace ve vysledku.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_selection_status(insurer, vpp):
    insurer_label = insurer if insurer != "— vyber —" else "Neni vybrano"
    vpp_label = vpp if vpp != "— vyber —" else "Neni vybrano"
    readiness = "Pripraveno k dotazu" if insurer != "— vyber —" and vpp != "— vyber —" else "Vyber dokument pro chat"
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
    f = st.sidebar.file_uploader("PDF", accept_multiple_files=True, key=st.session_state.upload_key)
    v = st.sidebar.text_input("VPP název")
    i = st.sidebar.selectbox("Pojišťovna", INSURERS, key="admin_insurer")
    if st.sidebar.button("Nahrát"):
        ingest_pdf(f or [], v or "", i)
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

if prompt := st.chat_input("Zeptej se..."):
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
        last_render = 0
        render_typing_indicator(ph)

        with st.spinner("Přemýšlím..."):
            ctx, conf, debug = search(prompt, trace_id)

        if not ctx:
            full = "Nenalezeno. Kontaktuj vedení směny."
            ph.markdown(full)
        else:
            memory = get_memory(prompt)
            combined = "\n\n".join([f"[STRANA {c['page']}] {c['exact']}" for c in ctx])

            prompt_ai = f"""
Použij pouze věty v části TEXT. Nevymýšlej.
Odpověz stručně česky a každé tvrzení doplň citací [str. X].
Když odpověď není v TEXT, napiš: "V dostupném textu to není uvedeno."

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
                        if hasattr(chunk, "text") and chunk.text:
                            full += chunk.text
                            now = time.perf_counter()
                            if now - last_render >= STREAM_RENDER_INTERVAL_SEC:
                                ph.markdown(full + "▌")
                                last_render = now
                except Exception as e:
                    handle_error("Stream odpovědi se přerušil.", e, trace_id=trace_id, show=False)

            if not full.strip():
                full = strict_answer_from_context(ctx)

            unsupported, matched_spans = validate_answer(full, ctx)
            if unsupported:
                full = strict_answer_from_context(ctx)
                debug["post_validation"] = {
                    "status": "replaced_by_strict_extract",
                    "unsupported": unsupported,
                }
            else:
                debug["post_validation"] = {
                    "status": "ok",
                    "matched_spans": matched_spans,
                    "mode": "substring_span",
                }

            ph.markdown(full)

        st.caption(f"Spolehlivost: {conf}% | Trace: {trace_id}")
        st.session_state.messages.append({"role": "assistant", "content": full})
        persist_memory("assistant", full)
        log_query(prompt, selected_insurer, selected_vpp, conf, trace_id)

        with st.expander("Debug"):
            st.json(debug if st.session_state.debug_mode else {
                "trace_id": trace_id,
                "confidence": debug.get("confidence"),
                "top_candidates": debug.get("candidates", [])[:5],
            })

        # FEEDBACK
        idx = len(st.session_state.messages)
        chunk_ids = [c["id"] for c in ctx]
        if idx not in st.session_state.feedback_done:
            c1, c2 = st.columns(2)

            if c1.button("Odpověď pomohla", key=f"l{idx}"):
                save_feedback(prompt, full, "like", "", chunk_ids)
                st.session_state.feedback_done[idx] = True
                st.rerun()

            if c2.button("Nahlásit problém", key=f"d{idx}"):
                st.session_state.feedback_open[idx] = True

            if st.session_state.feedback_open.get(idx):
                note = st.text_input("Co bylo špatně?", key=f"n{idx}")
                if st.button("Odeslat", key=f"s{idx}"):
                    save_feedback(prompt, full, "dislike", note, chunk_ids)
                    st.session_state.feedback_done[idx] = True
                    st.session_state.feedback_open[idx] = False
                    st.rerun()
