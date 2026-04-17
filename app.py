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

# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="VPP Checker",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PRIMARY = "#314397"
ACCENT = "#E43238"

# =========================
# GEMINI
# =========================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# STYLE (UX FIX)
# =========================

st.markdown(f"""
<style>
body {{ background-color: #f5f7fb; }}

.header {{
    background:{PRIMARY};
    padding:20px;
    border-radius:12px;
    color:white;
}}

.chat-user {{
    background:#e3edff;
    padding:15px;
    border-radius:12px;
    margin-top:10px;
}}

.chat-ai {{
    background:white;
    padding:15px;
    border-radius:12px;
    border-left:5px solid {ACCENT};
    margin-top:10px;
}}

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

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_batch(texts):
    return model.encode(texts).tolist()

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
# INGEST (BEZE ZMĚNY)
# =========================

def ingest_pdf(files):
    all_chunks = []

    progress = st.progress(0)
    status = st.empty()

    total = len(files)
    done = 0

    for file in files:
        status.text(f"Zpracovávám: {file.name}")

        reader = PdfReader(file)

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            main, sub = detect_headings(text)
            paragraphs = text.split("\n\n")

            for para in paragraphs:
                para = para.strip()
                if len(para) < 50:
                    continue

                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": para,
                    "page": i+1,
                    "source": file.name,
                    "heading": main,
                    "subheading": sub
                })

        done += 1
        progress.progress(done / total)

    vectors = embed_batch([c["text"] for c in all_chunks])

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
# SEARCH (TOP 3)
# =========================

def search(q):
    vector = embed_batch([q])[0]

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
# AI (STRUKTUROVANÁ ODPOVĚĎ)
# =========================

def ask_ai(q, contexts):
    combined = "\n\n".join([c["text"] for c in contexts])

    prompt = f"""
Jsi odborník na pojistné podmínky.

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
        answer = response.text.strip()
    except:
        answer = contexts[0]["exact"]

    # 🔥 více citací
    citations = "\n\n".join([
        f"\"{c['exact']}\"\n(str. {c['page']}, {c['heading']}, {c['subheading']})"
        for c in contexts[:3]
    ])

    return f"""
🧠 Odpověď:
{answer}

📄 Citace:
{citations}
"""

# =========================
# UI
# =========================

st.markdown("<div class='header'><h2>🛡️ VPP Checker</h2></div>", unsafe_allow_html=True)

# =========================
# ADMIN (SKRYTÝ)
# =========================

if not st.session_state.logged:
    pwd = st.text_input("Admin heslo", type="password")

    if st.button("Přihlásit"):
        if pwd == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.logged = True
            st.success("Přihlášeno")

# ADMIN PANEL až po loginu
if st.session_state.logged:
    st.sidebar.markdown("## 📄 Upload PDF")

    files = st.sidebar.file_uploader(
        "Vyber PDF",
        accept_multiple_files=True
    )

    if st.sidebar.button("🚀 Nahrát"):
        if files:
            ingest_pdf(files)
            st.sidebar.success("Hotovo")

# =========================
# CHAT FLOW (FIX)
# =========================

q = st.chat_input("Zeptej se...")

if q:
    # 👉 ihned zobraz dotaz
    st.session_state.history.append({"q": q, "a": "⏳ Checker hledá..."})

    # 👉 rerender okamžitě
    st.rerun()

# odpověď AI
if st.session_state.history and st.session_state.history[0]["a"] == "⏳ Checker hledá...":
    last = st.session_state.history[0]

    with st.spinner("🔍 Hledám v dokumentech..."):
        contexts = search(last["q"])

    if contexts:
        with st.spinner("🤖 Generuji odpověď..."):
            answer = ask_ai(last["q"], contexts)
    else:
        answer = "❌ Nic nenalezeno"

    st.session_state.history[0]["a"] = answer
    st.rerun()

# =========================
# RENDER CHAT
# =========================

for item in st.session_state.history:
    st.markdown(f"<div class='chat-user'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'>{item['a']}</div>", unsafe_allow_html=True)
