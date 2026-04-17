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

VECTOR_SIZE = 768  # 🔥 FIX

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
        info = qdrant.get_collection("docs")

        current_size = info.config.params.vectors.size

        if current_size != VECTOR_SIZE:
            print("⚠️ Dimenze nesedí → mažu kolekci")
            qdrant.delete_collection("docs")

            raise Exception("recreate")

    except:
        qdrant.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
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

    paragraphs = smart_chunk(text)

    chunks = []

    for para in paragraphs:
        if len(para) < 50:
            continue

        chunks.append({
            "id": str(uuid.uuid4()),
            "text": para[:1500],
            "page": page_num,
            "source": file_name
        })

    return chunks

# =========================
# INGEST
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

        for i, page in enumerate(reader.pages):
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
