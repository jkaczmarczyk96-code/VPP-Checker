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
# SESSION
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "files" not in st.session_state:
    st.session_state.files = {}

if "logged" not in st.session_state:
    st.session_state.logged = False

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
# GOOGLE FEEDBACK
# =========================

def save_feedback(q, a, f, note=""):
    try:
        scope = ["https://spreadsheets.google.com/feeds"]

        creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])

        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict, scope
        )

        client = gspread.authorize(creds)
        sheet = client.open("feedback").sheet1

        sheet.append_row([
            str(datetime.now()), q, a, f, note
        ])
    except:
        st.warning("Feedback se nepodařilo uložit")

# =========================
# DB
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_batch(texts):
    return model.encode(texts).tolist()

# =========================
# PDF INGEST (RYCHLÝ + MULTI)
# =========================

def ingest_pdf(files):
    all_points = []

    for file in files:
        reader = PdfReader(file)

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            main, sub = detect_headings(text)
            chunks = re.split(r'(?<=[.!?]) +', text)

            for chunk in chunks:
                all_points.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk[:1000],
                    "page": i+1,
                    "source": file.name,
                    "heading": main,
                    "subheading": sub
                })

    # batch embed
    vectors = embed_batch([p["text"] for p in all_points])

    points = []
    for p, v in zip(all_points, vectors):
        points.append({
            "id": p["id"],
            "vector": v,
            "payload": p
        })

    qdrant.upsert(collection_name="docs", points=points)

# =========================
# SEARCH
# =========================

def search(q):
    vector = embed_batch([q])[0]

    results = qdrant.query_points(
        collection_name="docs",
        query=vector,
        limit=5
    ).points

    contexts, sources = [], []

    for r in results:
        payload = r.payload

        contexts.append(
            f"[str. {payload['page']}] {payload['text']} "
            f"(Nadpis: {payload.get('heading','')}, "
            f"Podnadpis: {payload.get('subheading','')})"
        )

        sources.append(payload)

    return contexts, sources

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
# PDF VIEW
# =========================

def show_pdf(file, page):
    file.seek(0)
    b64 = base64.b64encode(file.read()).decode()
    st.markdown(
        f'<iframe src="data:application/pdf;base64,{b64}#page={page}" width="100%" height="500"></iframe>',
        unsafe_allow_html=True
    )

# =========================
# HEADER
# =========================

st.markdown("<div class='header'><h2>🛡️ VPP Checker</h2></div>", unsafe_allow_html=True)

# =========================
# SIDEBAR (ADMIN)
# =========================

st.sidebar.markdown("### 🔐 Admin")

pwd = st.sidebar.text_input("Heslo", type="password")

if st.sidebar.button("Přihlásit"):
    if pwd == st.secrets["ADMIN_PASSWORD"]:
        st.session_state.logged = True
        st.sidebar.success("OK")

if st.session_state.logged:
    files = st.sidebar.file_uploader(
        "Nahraj PDF",
        accept_multiple_files=True
    )

    if files:
        for f in files:
            st.session_state.files[f.name] = f

        ingest_pdf(files)
        st.sidebar.success("Nahráno")

# =========================
# CHAT INPUT
# =========================

q = st.chat_input("Zeptej se...")

if q:
    contexts, sources = search(q)
    answer = ask_ai(q, contexts)

    st.session_state.history.append({
        "q": q,
        "a": answer,
        "sources": sources
    })

# =========================
# CHAT UI
# =========================

for i, item in enumerate(reversed(st.session_state.history)):
    st.markdown(f"<div class='chat-user'>{item['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'>{item['a']}</div>", unsafe_allow_html=True)

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
        save_feedback(item["q"], item["a"], "like")

    if col2.button("👎", key=f"dislike{i}"):
        note = st.text_input("Co bylo špatně?", key=f"note{i}")
        if st.button("Odeslat", key=f"send{i}"):
            save_feedback(item["q"], item["a"], "dislike", note)
