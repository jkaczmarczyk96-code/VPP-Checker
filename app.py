import streamlit as st
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
import uuid
import base64
from datetime import datetime
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai
import concurrent.futures
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
# GEMINI (CACHE)
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
# MODEL (CACHE)
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =========================
# EMBEDDING (CACHE)
# =========================

@st.cache_data(show_spinner=False)
def embed_batch(texts_tuple):
    return model.encode(list(texts_tuple)).tolist()

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
# EXACT SENTENCE
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
# PARALELNÍ INGEST
# =========================

def process_page(file_name, page, i):
    text = page.extract_text()
    if not text:
        return []

    main, sub = detect_headings(text)
    paragraphs = text.split("\n\n")

    chunks = []

    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:
            continue

        chunks.append({
            "id": str(uuid.uuid4()),
            "text": para[:1500],
            "page": i+1,
            "source": file_name,
            "heading": main,
            "subheading": sub
        })

    return chunks


def ingest_pdf(files):
    all_chunks = []

    progress = st.progress(0)
    status = st.empty()

    total_files = len(files)
    done_files = 0

    for file in files:

        if file.name in st.session_state.files:
            continue

        status.text(f"Zpracovávám: {file.name}")

        reader = PdfReader(file)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_page, file.name, page, i)
                for i, page in enumerate(reader.pages)
            ]

            for f in concurrent.futures.as_completed(futures):
                all_chunks.extend(f.result())

        done_files += 1
        progress.progress(done_files / total_files)

    if not all_chunks:
        return

    status.text("🔄 Embedding...")

    vectors = embed_batch(tuple([c["text"] for c in all_chunks]))

    status.text("📦 Ukládám...")

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
# SEARCH (CACHE)
# =========================

@st.cache_data(show_spinner=False)
def search_cached(q):
    vector = embed_batch((q,))[0]

    results = qdrant.query_points(
        collection_name="docs",
        query=vector,
        limit=3
    ).points

    if not results:
        return None

    contexts = []

    for r in results:
        p = r.payload
        exact = extract_exact_sentence(p["text"], q)

        contexts.append({
            "text": p["text"],
            "exact": exact,
            "page": p["page"],
            "heading": p.get("heading",""),
            "subheading": p.get("subheading","")
        })

    return contexts

# =========================
# AI (STREAMING + PRAVIDLA)
# =========================

def ask_ai_stream(q, contexts):
    combined = "\n\n".join([c["text"] for c in contexts])

    prompt = f"""
Jsi odborník na pojistné podmínky.

DŮLEŽITÉ:
- odpovídej pouze z poskytnutého textu
- nic si nevymýšlej
- pokud odpověď v textu není, napiš že není k dispozici

Odpověz strukturovaně:
- krátké shrnutí
- hlavní body (odrážky)

TEXT:
{combined}

OTÁZKA:
{q}
"""

    try:
        response = model_gemini.generate_content(prompt)
        full_answer = response.text.strip()
    except:
        full_answer = contexts[0]["exact"]

    placeholder = st.empty()
    streamed = ""

    for word in full_answer.split():
        streamed += word + " "
        placeholder.markdown(f"🧠 Odpověď:\n\n{streamed}")
        time.sleep(0.01)

    citations = "\n\n".join([
        f"\"{c['exact']}\"\n(str. {c['page']}, {c['heading']}, {c['subheading']})"
        for c in contexts[:3]
    ])

    placeholder.markdown(f"""
🧠 Odpověď:
{full_answer}

📄 Citace:
{citations}
""")

    return full_answer

# =========================
# UI + SIDEBAR (ZACHOVÁNO)
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

if not st.session_state.logged:
    st.sidebar.info("🔒 Přihlas se pro upload PDF")

if st.session_state.logged:
    files = st.sidebar.file_uploader("📄 Vyber PDF", accept_multiple_files=True)

    if st.sidebar.button("🚀 Nahrát"):
        if files:
            ingest_pdf(files)
            st.sidebar.success("Hotovo")

# =========================
# CHAT FLOW
# =========================

q = st.chat_input("Zeptej se...")

if q:
    st.session_state.history.insert(0, {"q": q, "a": "⏳ Checker hledá..."})
    st.rerun()

if st.session_state.history and st.session_state.history[0]["a"] == "⏳ Checker hledá...":
    last = st.session_state.history[0]

    with st.spinner("🔍 Prohledávám dokumenty..."):
        contexts = search_cached(last["q"])

    if contexts:
        with st.spinner("🤖 Checker přemýšlí..."):
            answer = ask_ai_stream(last["q"], contexts)
    else:
        answer = "❌ Nic nenalezeno"

    st.session_state.history[0]["a"] = answer

# =========================
# RENDER CHAT
# =========================

for item in st.session_state.history:
    st.markdown(f"<div class='chat-user'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'>{item['a']}</div>", unsafe_allow_html=True)
