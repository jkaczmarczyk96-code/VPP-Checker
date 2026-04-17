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
# SAFE PDF EXTRACT
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
# INGEST
# =========================

def ingest_pdf(files):
    all_chunks = []

    for file in files:
        if file.name in st.session_state.files:
            continue

        reader = PdfReader(file)
        plumber_data = None

        for i, page in enumerate(reader.pages):
            text = safe_extract_text_pypdf(page, i+1, file.name)

            if not text:
                if plumber_data is None:
                    plumber_data = extract_with_pdfplumber(file)
                _, text = plumber_data[i]

            chunks = process_text_to_chunks(file.name, text, i+1)
            all_chunks.extend(chunks)

    vectors = embed_batch(tuple([c["text"] for c in all_chunks]))

    points = []
    for c, v in zip(all_chunks, vectors):
        points.append({
            "id": c["id"],
            "vector": v,
            "payload": c
        })

    qdrant.upsert(collection_name="docs", points=points)

# =========================
# QUERY REWRITE
# =========================

def rewrite_query(user_q, history):
    history_text = "\n".join([
        f"Q: {h['q']}\nA: {h['a']}"
        for h in history[:3]
    ])

    prompt = f"""
Přeformuluj dotaz pro vyhledávání.

{history_text}

Dotaz:
{user_q}
"""

    try:
        r = model_gemini.generate_content(prompt)
        return r.text.strip()
    except:
        return user_q

# =========================
# SEARCH + FILTER + RERANK
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
            "exact": p["text"][:300],
            "page": p["page"],
            "heading": p.get("heading",""),
            "subheading": p.get("subheading",""),
            "score": float(score)
        })

    return contexts

# =========================
# CONFIDENCE
# =========================

def compute_confidence(contexts, query):
    if not contexts:
        return 0

    scores = []
    q_words = set(query.lower().split())

    for c in contexts:
        text_words = set(c["text"].lower().split())
        overlap = len(q_words & text_words) / (len(q_words) + 1)
        scores.append(0.7 * c["score"] + 0.3 * overlap)

    return min(max(sum(scores)/len(scores), 0), 1)

# =========================
# REASONING PLAN
# =========================

def build_reasoning_plan(q, contexts):
    preview = "\n\n".join([c["text"][:300] for c in contexts[:2]])

    prompt = f"""
Rozděl otázku na kroky.

Otázka: {q}
"""

    try:
        r = model_gemini.generate_content(prompt)
        return r.text.strip()
    except:
        return ""

# =========================
# AI
# =========================

def ask_ai_stream(q, contexts):
    if not contexts:
        return "❌ Nic nenalezeno"

    plan = build_reasoning_plan(q, contexts)

    history_text = "\n".join([
        f"{h['q']} -> {h['a']}"
        for h in st.session_state.history[1:4]
    ])

    combined = "\n\n---\n\n".join([
        f"[Strana {c['page']}]\n{c['text']}"
        for c in contexts
    ])

    prompt = f"""
Jsi expert na pojistné podmínky.

Použij plán:
{plan}

Konverzace:
{history_text}

Text:
{combined}

Otázka:
{q}
"""

    try:
        response = model_gemini.generate_content(prompt)
        full_answer = response.text.strip()
    except:
        full_answer = contexts[0]["text"][:300]

    confidence = int(compute_confidence(contexts, q) * 100)

    citations = "\n\n".join([
        f"\"{c['text'][:200]}\" (str. {c['page']})"
        for c in contexts
    ])

    st.markdown(f"""
🧠 {full_answer}

📊 Jistota: {confidence} %

📄 Citace:
{citations}
""")

    return full_answer

# =========================
# UI
# =========================

st.markdown("<div class='header'><h2>🛡️ VPP Checker</h2></div>", unsafe_allow_html=True)

st.sidebar.markdown("## 🔐 Admin panel")

pwd = st.sidebar.text_input("Heslo", type="password")

if st.sidebar.button("Přihlásit"):
    if pwd == st.secrets["ADMIN_PASSWORD"]:
        st.session_state.logged = True
        st.sidebar.success("Přihlášeno")
    else:
        st.sidebar.error("Špatné heslo")

if st.session_state.logged:
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
