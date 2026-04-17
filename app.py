import streamlit as st
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
import uuid
import base64
from datetime import datetime
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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
# STYLE
# =========================

st.markdown(f"""
<style>
.header {{background:{PRIMARY};padding:20px;border-radius:12px;color:white;}}
.chat-user {{background:#eef2ff;padding:15px;border-radius:12px;margin-top:10px;}}
.chat-ai {{background:white;padding:15px;border-radius:12px;border-left:5px solid {ACCENT};margin-top:10px;}}
.source {{background:#f9f9f9;padding:10px;border-radius:8px;margin-top:5px;border-left:3px solid {PRIMARY};}}
.highlight {{background:#fff3cd;padding:5px;border-radius:5px;}}
</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN
# =========================

if "logged" not in st.session_state:
    st.session_state.logged = False

# =========================
# HEADINGS
# =========================

def detect_headings(text):
    lines = text.split("\n")
    main = ""
    sub = ""

    for line in lines:
        line = line.strip()

        if line.isupper() and len(line) > 5:
            main = line

        elif re.match(r'^\d+(\.\d+)*\s', line):
            sub = line

    return main, sub

# =========================
# GOOGLE SHEETS FEEDBACK
# =========================

def save_feedback(question, answer, feedback, note=""):
    scope = ["https://spreadsheets.google.com/feeds"]

    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])

    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, scope
    )

    client = gspread.authorize(creds)

    sheet = client.open("feedback").sheet1

    sheet.append_row([
        str(datetime.now()),
        question,
        answer,
        feedback,
        note
    ])

# =========================
# DB
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text).tolist()

# =========================
# PDF
# =========================

def show_pdf(file, page):
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    st.markdown(f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}#page={page}"
    width="100%" height="500"></iframe>
    """, unsafe_allow_html=True)

def ingest_pdf(file):
    reader = PdfReader(file)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        main, sub = detect_headings(text)

        chunks = re.split(r'(?<=[.!?]) +', text)

        for chunk in chunks:
            qdrant.upsert(
                collection_name="docs",
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": embed(chunk),
                    "payload": {
                        "text": chunk,
                        "page": i+1,
                        "source": file.name,
                        "heading": main,
                        "subheading": sub
                    }
                }]
            )

# =========================
# AI
# =========================

def ask_ai(question, contexts):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_KEY']}"
    }

    prompt = f"""
Jsi odborník na pojistné podmínky.

Odpovídej pouze z textu.
Nevymýšlej si nic.

Na konci uveď citaci:
- přesný text
- stránka
- nadpis
- podnadpis

TEXT:
{chr(10).join(contexts)}

OTÁZKA:
{question}
"""

    res = requests.post(url, headers=headers, json={"inputs": prompt})

    return res.json()[0]["generated_text"]

# =========================
# SEARCH
# =========================

def search(q):
    results = qdrant.query_points(
        collection_name="docs",
        query=embed(q),
        limit=5
    ).points

    contexts = []
    sources = []

    for r in results:
        contexts.append(
            f"[str. {r.payload['page']}] {r.payload['text']} "
            f"(Nadpis: {r.payload.get('heading','')}, "
            f"Podnadpis: {r.payload.get('subheading','')})"
        )
        sources.append(r.payload)

    answer = ask_ai(q, contexts)

    return answer, sources

# =========================
# SESSION
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "files" not in st.session_state:
    st.session_state.files = {}

# =========================
# HEADER
# =========================

st.markdown("<div class='header'><h2>🛡️ VPP Checker</h2></div>", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("### 🔐 Admin")

pwd = st.sidebar.text_input("Heslo", type="password")

if st.sidebar.button("Přihlásit"):
    if pwd == st.secrets["ADMIN_PASSWORD"]:
        st.session_state.logged = True
        st.sidebar.success("OK")

if st.session_state.logged:
    file = st.sidebar.file_uploader("Nahraj PDF")

    if file:
        st.session_state.files[file.name] = file
        ingest_pdf(file)
        st.sidebar.success("Nahráno")

# =========================
# INPUT
# =========================

q = st.chat_input("Zeptej se...")

if q:
    answer, sources = search(q)

    st.session_state.history.append({
        "question": q,
        "answer": answer,
        "sources": sources
    })

# =========================
# CHAT
# =========================

for i, item in enumerate(reversed(st.session_state.history)):
    st.markdown(f"<div class='chat-user'>{item['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'>{item['answer']}</div>", unsafe_allow_html=True)

    for j, s in enumerate(item["sources"]):
        st.markdown(f"""
        <div class="source">
        {s['source']} – str. {s['page']}<br>
        <b>{s.get('heading','')}</b><br>
        {s.get('subheading','')}<br>
        <span class="highlight">{s['text'][:200]}</span>
        </div>
        """, unsafe_allow_html=True)

        if s["source"] in st.session_state.files:
            if st.button("📄 Otevřít", key=f"{i}-{j}"):
                show_pdf(st.session_state.files[s["source"]], s["page"])

    col1, col2 = st.columns(2)

    if col1.button("👍", key=f"like{i}"):
        save_feedback(item["question"], item["answer"], "like")

    if col2.button("👎", key=f"dislike{i}"):
        note = st.text_input("Co bylo špatně?", key=f"note{i}")
        if st.button("Odeslat", key=f"send{i}"):
            save_feedback(item["question"], item["answer"], "dislike", note)
