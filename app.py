import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import pdfplumber
import uuid
import re
import google.generativeai as genai

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="VPP Checker", layout="wide")

VECTOR_SIZE = 768

INSURERS = [
    "CZ - GČPOJ","CZ - Direct","CZ - TravelCare","CZ - Uniqa",
    "SK - TravelCare","SK - Generali","SK - ECP","SK - Wüstenrot","SK - Uniqa"
]

# =========================
# SESSION INIT (FIX)
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "logged" not in st.session_state:
    st.session_state.logged = False

# =========================
# GEMINI
# =========================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# QDRANT (DIM FIX)
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

def init_collection():
    try:
        col = qdrant.get_collection("docs")
        if col.config.params.vectors.size != VECTOR_SIZE:
            qdrant.delete_collection("docs")
            qdrant.create_collection(
                collection_name="docs",
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
    except:
        qdrant.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

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

def embed(x): return model.encode(x).tolist()

# =========================
# HELPERS
# =========================

def normalize(x): return x.strip().upper() if x else ""

def smart_chunk(t, s=800, o=150):
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i+s])
        i += s-o
    return out

def extract_sentence(p, q):
    s = re.split(r'(?<=[.!?]) +', p)
    qw = set(q.lower().split())
    return max(s, key=lambda x: len(qw & set(x.lower().split())), default=p)

def safe_scroll():
    try:
        return qdrant.scroll("docs", limit=1000, with_payload=True)[0]
    except:
        return []

def get_vpps(ins):
    return sorted(set(
        r.payload.get("vpp_name")
        for r in safe_scroll()
        if r.payload.get("insurer") == ins
    ))

# =========================
# INGEST (WITH PROGRESS)
# =========================

def ingest_pdf(files, vpp, ins):
    vpp = normalize(vpp)

    if not vpp:
        st.sidebar.error("Zadej VPP"); return

    if vpp in get_vpps(ins):
        st.sidebar.warning("VPP už existuje"); return

    progress = st.sidebar.progress(0)
    status = st.sidebar.empty()

    total_pages = sum(len(PdfReader(f).pages) for f in files)
    done = 0
    chunks = []

    for f in files:
        reader = PdfReader(f)
        status.text(f"Zpracovávám {f.name}")

        for i, p in enumerate(reader.pages):
            try:
                t = p.extract_text()
            except:
                t = None

            if not t:
                try:
                    with pdfplumber.open(f) as pdf:
                        t = pdf.pages[i].extract_text()
                except:
                    continue

            for c in smart_chunk(t):
                if len(c) < 50:
                    continue
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": c,
                    "page": i+1,
                    "vpp_name": vpp,
                    "insurer": ins
                })

            done += 1
            progress.progress(min(done / total_pages, 1.0))

    if not chunks:
        status.text("❌ PDF chyba")
        return

    status.text("🔄 Embedding...")
    vec = embed([c["text"] for c in chunks])

    pts = [{"id": c["id"], "vector": v, "payload": c} for c, v in zip(chunks, vec)]

    status.text("📦 Ukládám...")

    try:
        qdrant.upsert("docs", pts)
        progress.progress(1.0)
        status.text("✅ Hotovo")
    except:
        status.text("❌ Qdrant chyba")

# =========================
# SEARCH
# =========================

def search(q):
    try:
        vec = embed([q])[0]

        cond = []
        if selected_insurer != "— vyber —":
            cond.append(FieldCondition(key="insurer", match=MatchValue(value=selected_insurer)))
        if selected_vpp != "— vyber —":
            cond.append(FieldCondition(key="vpp_name", match=MatchValue(value=selected_vpp)))

        filt = Filter(must=cond) if cond else None

        res = qdrant.query_points("docs", query=vec, query_filter=filt, limit=10).points

        if not res:
            return [], 0

        pairs = [(q, r.payload["text"]) for r in res]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(res, scores), key=lambda x: x[1], reverse=True)

        ctx = [{
            "text": r.payload["text"],
            "exact": extract_sentence(r.payload["text"], q),
            "page": r.payload["page"]
        } for r, _ in ranked[:5]]

        conf = max(0, min(100, int((float(ranked[0][1]) + 5) * 10)))

        return ctx, conf

    except:
        return [], 0

# =========================
# AI
# =========================

def ask(q, ctx, conf):
    if "zub" in q.lower():
        return "👉 Použij Zuby AI v Copilotu"

    if not ctx:
        return "❌ Nenalezeno → kontaktuj vedení"

    text = "\n\n".join(c["text"] for c in ctx)

    try:
        r = model_gemini.generate_content(f"Odpověz pouze z textu:\n{text}")
        ans = r.text.strip()
    except:
        ans = ctx[0]["exact"]

    cites = "\n".join(f"- \"{c['exact']}\" (str. {c['page']})" for c in ctx)

    return f"{ans}\n\n{cites}"

# =========================
# UI
# =========================

st.title("🛡️ VPP Checker")

st.sidebar.markdown("## 📂 Filtr dokumentů")

selected_insurer = st.sidebar.selectbox(
    "Pojišťovna",
    ["— vyber —"] + INSURERS,
    key="filter_ins"
)

selected_vpp = "— vyber —"
if selected_insurer != "— vyber —":
    vpps = get_vpps(selected_insurer)
    selected_vpp = st.sidebar.selectbox(
        "VPP",
        ["— vyber —"] + vpps,
        key="filter_vpp"
    )

st.sidebar.markdown("## 🔐 Administrace")

if not st.session_state.logged:
    pwd = st.sidebar.text_input("Heslo", type="password")
    if st.sidebar.button("Přihlásit"):
        if pwd == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.logged = True
            st.rerun()
else:
    files = st.sidebar.file_uploader("PDF", accept_multiple_files=True)
    vpp = st.sidebar.text_input("Název VPP")
    ins = st.sidebar.selectbox(
        "Pojišťovna",
        ["— vyber —"] + INSURERS,
        key="admin_ins"
    )

    if st.sidebar.button("Nahrát"):
        if files and vpp and ins != "— vyber —":
            ingest_pdf(files, vpp, ins)

# =========================
# CHAT
# =========================

q = st.chat_input("Zeptej se...")

if q:
    if selected_vpp == "— vyber —":
        st.warning("Vyber VPP")
    else:
        ctx, conf = search(q)
        ans = ask(q, ctx, conf)
        st.session_state.history.insert(0, {"q": q, "a": ans})

for h in st.session_state.history:
    st.markdown(f"<div class='chat-user'>{h['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'>{h['a']}</div>", unsafe_allow_html=True)
