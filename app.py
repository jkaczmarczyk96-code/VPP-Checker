import streamlit as st
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import pdfplumber
import re
import uuid
import base64
from datetime import datetime
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai
import time

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="VPP Checker",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.set_option('client.showErrorDetails', False)

PRIMARY = "#314397"
ACCENT = "#E43238"

# =========================
# GEMINI
# =========================

@st.cache_resource
def load_gemini():
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    return genai.GenerativeModel("gemini-1.5-flash")

model_gemini = load_gemini()

# =========================
# STYLE
# =========================

st.markdown(f"""
<style>
body {{ background-color: #f5f7fb; }}
.header {{ background:{PRIMARY}; padding:20px; border-radius:12px; color:white; }}
.chat-user {{ background:#e3edff; padding:15px; border-radius:12px; margin-top:10px; }}
.chat-ai {{ background:white; padding:15px; border-radius:12px; border-left:5px solid {ACCENT}; margin-top:10px; }}
</style>
""", unsafe_allow_html=True)

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
# QDRANT
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

def init_collection():
    try:
        qdrant.get_collection("docs")
    except:
        qdrant.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

init_collection()

# =========================
# MODEL
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

model = load_model()

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

# =========================
# EMBEDDING
# =========================

@st.cache_data(show_spinner=False)
def embed_batch(texts_tuple):
    return model.encode(list(texts_tuple)).tolist()

# =========================
# PDF EXTRACT
# =========================

def safe_extract_text_pypdf(page, page_num, file_name):
    try:
        return page.extract_text()
    except Exception as e:
        print(f"[pypdf FAIL] {file_name} | strana {page_num}: {e}")
        return None

def extract_with_pdfplumber(file):
    texts = []
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    texts.append((i, text))
                except:
                    texts.append((i, None))
    except:
        pass
    return texts

# =========================
# HEADINGS
# =========================

def detect_headings(text):
    lines = text.split("\n")
    main, sub = "", ""

    for line in lines:
        line = line.strip()
        if line.isupper() and len(line) > 5:
            main = line
        elif re.match(r'^\d+(\.\d+)*\s', line):
            sub = line

    return main, sub

# =========================
# CHUNKING
# =========================

def smart_chunk(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def process_text_to_chunks(file_name, text, page_num):
    if not text:
        return []

    main, sub = detect_headings(text)
    paragraphs = smart_chunk(text)

    chunks = []

    for para in paragraphs:
        if len(para) < 50:
            continue

        chunks.append({
            "id": str(uuid.uuid4()),
            "text": para[:1500],
            "page": page_num,
            "source": file_name,
            "heading": main,
            "subheading": sub
        })

    return chunks

# =========================
# INGEST (WITH PROGRESS)
# =========================

def ingest_pdf(files):
    all_chunks = []

    progress = st.progress(0)
    status = st.empty()

    total_files = len(files)
    done_files = 0

    for file in files:

        if file.name in st.session_state.files:
            continue

        status.text(f"📄 Zpracovávám: {file.name}")

        reader = PdfReader(file)
        plumber_data = None

        total_pages = len(reader.pages)

        for i, page in enumerate(reader.pages):
            status.text(f"📄 {file.name} | strana {i+1}/{total_pages}")

            text = safe_extract_text_pypdf(page, i+1, file.name)

            if not text:
                if plumber_data is None:
                    status.text(f"⚠️ Fallback na pdfplumber: {file.name}")
                    plumber_data = extract_with_pdfplumber(file)

                _, text = plumber_data[i]

            chunks = process_text_to_chunks(file.name, text, i+1)
            all_chunks.extend(chunks)

        done_files += 1
        progress.progress(done_files / total_files)

    if not all_chunks:
        status.text("❌ Žádná data")
        return

    status.text("🔄 Embedding...")
    vectors = embed_batch(tuple([c["text"] for c in all_chunks]))

    status.text("📦 Ukládám do databáze...")

    points = []
    for c, v in zip(all_chunks, vectors):
        points.append({
            "id": c["id"],
            "vector": v,
            "payload": c
        })

    qdrant.upsert(collection_name="docs", points=points)

    progress.progress(1.0)
    status.text("✅ Hotovo")

# =========================
# QUERY REWRITE
# =========================

def rewrite_query(user_q, history):
    try:
        r = model_gemini.generate_content(f"Přeformuluj dotaz: {user_q}")
        return r.text.strip()
    except:
        return user_q

# =========================
# SEARCH
# =========================

def search_cached(q):
    vector = embed_batch((q,))[0]

    detected_source = None
    for f in st.session_state.files:
        if f.lower() in q.lower():
            detected_source = f

    query_filter = None
    if detected_source:
        query_filter = Filter(
            must=[FieldCondition(
                key="source",
                match=MatchValue(value=detected_source)
            )]
        )

    results = qdrant.query_points(
        collection_name="docs",
        query=vector,
        limit=10,
        query_filter=query_filter
    ).points

    if not results:
        return None

    pairs = [(q, r.payload["text"]) for r in results]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    contexts = []

    for r, score in ranked[:5]:
        p = r.payload

        contexts.append({
            "text": p["text"],
            "page": p["page"],
            "score": float(score)
        })

    return contexts

# =========================
# CONFIDENCE
# =========================

def compute_confidence(contexts, query):
    if not contexts:
        return 0

    return min(max(sum([c["score"] for c in contexts]) / len(contexts), 0), 1)

# =========================
# REASONING
# =========================

def build_reasoning_plan(q):
    try:
        r = model_gemini.generate_content(f"Rozděl otázku na kroky: {q}")
        return r.text.strip()
    except:
        return ""

# =========================
# AI
# =========================

def ask_ai_stream(q, contexts):
    if not contexts:
        return "❌ Nic nenalezeno"

    plan = build_reasoning_plan(q)

    combined = "\n\n".join([c["text"] for c in contexts])

    try:
        response = model_gemini.generate_content(f"{plan}\n\n{combined}\n\n{q}")
        answer = response.text.strip()
    except:
        answer = contexts[0]["text"][:200]

    confidence = int(compute_confidence(contexts, q) * 100)

    st.markdown(f"🧠 {answer}\n\n📊 Jistota: {confidence}%")

    return answer

# =========================
# UI
# =========================

st.markdown("<div class='header'><h2>🛡️ VPP Checker</h2></div>", unsafe_allow_html=True)

st.sidebar.markdown("## 🔐 Admin panel")

if not st.session_state.logged:
    pwd = st.sidebar.text_input("Heslo", type="password")

    if st.sidebar.button("Přihlásit"):
        if pwd == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.logged = True
            st.rerun()
        else:
            st.sidebar.error("Špatné heslo")

else:
    st.sidebar.success("✅ Přihlášeno jako admin")

    files = st.sidebar.file_uploader("📄 Vyber PDF", accept_multiple_files=True)

    if st.sidebar.button("🚀 Nahrát"):
        if files:
            ingest_pdf(files)
            st.session_state.files = {f.name: True for f in files}
            st.sidebar.success("Hotovo")

q = st.chat_input("Zeptej se...")

if q:
    better_q = rewrite_query(q, st.session_state.history)
    contexts = search_cached(better_q)
    answer = ask_ai_stream(q, contexts)

    st.session_state.history.insert(0, {"q": q, "a": answer})

for item in st.session_state.history:
    st.markdown(f"<div class='chat-user'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'>{item['a']}</div>", unsafe_allow_html=True)
