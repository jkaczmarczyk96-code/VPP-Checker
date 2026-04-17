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

VECTOR_SIZE = 768
DEBUG = st.secrets.get("DEBUG", False)

st.set_page_config(page_title="VPP Checker", layout="wide")

# =========================
# DEBUG PANEL
# =========================

if DEBUG:
    st.markdown("### 🛠️ Debug panel")
    st.json(st.session_state)

def log(msg):
    if DEBUG:
        st.write(f"🪵 {msg}")

# =========================
# SESSION
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "files" not in st.session_state:
    st.session_state.files = {}

if "logged" not in st.session_state:
    st.session_state.logged = False

# =========================
# GEMINI
# =========================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# QDRANT
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

def init_collection():
    try:
        info = qdrant.get_collection("docs")
        size = info.config.params.vectors.size

        if size != VECTOR_SIZE:
            qdrant.delete_collection("docs")
            raise Exception("recreate")

    except:
        qdrant.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

init_collection()

# =========================
# MODEL (CACHE FIX)
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

model = load_model()
reranker = load_reranker()

# =========================
# HELPERS
# =========================

def embed(texts):
    return model.encode(texts).tolist()

def smart_chunk(text, size=800, overlap=150):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def process_text(file, text, page):
    chunks = []
    for c in smart_chunk(text):
        if len(c) < 50:
            continue
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": c,
            "page": page,
            "source": file
        })
    return chunks

# =========================
# 🔥 EXACT SENTENCE (NEW)
# =========================

def extract_exact_sentence(paragraph, question):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)

    best = ""
    best_score = 0
    q_words = set(question.lower().split())

    for s in sentences:
        s_words = set(s.lower().split())
        score = len(q_words & s_words)

        if score > best_score:
            best_score = score
            best = s

    return best if best else paragraph[:300]

# =========================
# INGEST
# =========================

def ingest_pdf(files):
    all_chunks = []

    progress = st.progress(0)
    status = st.empty()

    for idx, file in enumerate(files):
        reader = PdfReader(file)

        for i, page in enumerate(reader.pages):
            status.text(f"{file.name} | {i+1}/{len(reader.pages)}")

            try:
                text = page.extract_text()
            except:
                text = None

            if not text:
                try:
                    with pdfplumber.open(file) as pdf:
                        text = pdf.pages[i].extract_text()
                except:
                    continue

            all_chunks.extend(process_text(file.name, text, i+1))

        progress.progress((idx+1)/len(files))

    if not all_chunks:
        st.error("❌ Žádná data")
        return

    vectors = embed([c["text"] for c in all_chunks])

    points = []
    for c, v in zip(all_chunks, vectors):
        points.append({"id": c["id"], "vector": v, "payload": c})

    qdrant.upsert("docs", points)
    st.success("✅ Hotovo")

# =========================
# SEARCH (UPDATED)
# =========================

def search(q):
    vec = embed([q])[0]

    results = qdrant.query_points("docs", query=vec, limit=10).points

    if not results:
        return []

    pairs = [(q, r.payload["text"]) for r in results]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    contexts = []

    for r, s in ranked[:5]:
        p = r.payload

        exact = extract_exact_sentence(p["text"], q)

        contexts.append({
            "text": p["text"],
            "exact": exact,
            "page": p["page"],
            "score": float(s)
        })

    return contexts

# =========================
# AI (RESTORED CORRECT VERSION)
# =========================

def ask(q, ctx):
    if not ctx:
        return "❌ Odpověď není v dokumentech dostupná."

    combined = "\n\n---\n\n".join([
        f"[Strana {c['page']}]\n{c['exact']}"
        for c in ctx
    ])

    prompt = f"""
Jsi expert na pojistné podmínky.

PRAVIDLA:
- odpovídej POUZE z textu
- nic si nevymýšlej
- pokud odpověď není v textu → napiš "není k dispozici"
- odpověď musí dávat smysl jako vysvětlení pro člověka
- nepřepisuj jen text, vysvětluj

FORMÁT:
1. Krátké shrnutí (1–2 věty)
2. Vysvětlení (odrážky)
3. Citace (přesná věta z textu)

TEXT:
{combined}

DOTAZ:
{q}
"""

    try:
        r = model_gemini.generate_content(prompt)
        return r.text.strip()
    except:
        return ctx[0]["exact"]

# =========================
# UI
# =========================

st.title("🛡️ VPP Checker")

if not st.session_state.logged:
    st.sidebar.info("🔒 Přihlas se pro upload PDF")

    pwd = st.sidebar.text_input("Heslo", type="password")

    if st.sidebar.button("Přihlásit"):
        if pwd == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.logged = True
            st.rerun()
        else:
            st.sidebar.error("Špatné heslo")

else:
    st.sidebar.success("✅ Admin")

    files = st.sidebar.file_uploader("PDF", accept_multiple_files=True)

    if st.sidebar.button("Nahrát"):
        if files:
            with st.spinner("📄 Zpracovávám PDF..."):
                ingest_pdf(files)
                st.session_state.files = {f.name: True for f in files}

# =========================
# CHAT
# =========================

q = st.chat_input("Zeptej se...")

if q:
    with st.spinner("🔍 Hledám odpověď..."):
        ctx = search(q)
        ans = ask(q, ctx)

    st.session_state.history.insert(0, {"q": q, "a": ans})

# =========================
# RENDER
# =========================

for h in st.session_state.history:
    st.markdown(f"**Ty:** {h['q']}")
    st.markdown(f"**AI:** {h['a']}")
