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
# STYLE (NEZMENĚNO)
# =========================

st.markdown(f"""
<style>
body {{ background-color: #f5f7fb; }}
.header {{ background:{PRIMARY}; padding:20px; border-radius:12px; color:white; margin-bottom:15px; }}
.chat-user {{ background:#e3edff; color:#1a1a1a; padding:15px; border-radius:12px; margin-top:10px; }}
.chat-ai {{ background:#ffffff; color:#1a1a1a; padding:15px; border-radius:12px; border-left:5px solid {ACCENT}; margin-top:10px; }}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION (NEZMENĚNO)
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "files" not in st.session_state:
    st.session_state.files = {}

if "logged" not in st.session_state:
    st.session_state.logged = False

# =========================
# QDRANT (NEZMENĚNO)
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
# MODEL (NEZMENĚNO)
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_batch(texts):
    return model.encode(texts).tolist()

# =========================
# HEADINGS (NEZMENĚNO)
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
# 🆕 EXACT SENTENCE
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
# FEEDBACK (NEZMENĚNO)
# =========================

def save_feedback(q, a, f, note=""):
    try:
        creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(
            creds_dict,
            ["https://spreadsheets.google.com/feeds"]
        )
        client = gspread.authorize(creds)
        sheet = client.open("feedback").sheet1
        sheet.append_row([str(datetime.now()), q, a, f, note])
    except:
        st.warning("Feedback se nepodařilo uložit")

# =========================
# 🔥 INGEST (UPRAVENO NA ODSTAVCE)
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

    if not all_chunks:
        status.text("❌ Žádný text")
        return

    status.text("🔄 Vytvářím embeddingy...")

    vectors = embed_batch([c["text"] for c in all_chunks])

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
# 🔥 SEARCH (UPRAVENO)
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

    combined_text = []
    best = results[0].payload

    for r in results:
        combined_text.append(r.payload["text"])

    exact = extract_exact_sentence(best["text"], q)

    return {
        "text": "\n".join(combined_text[:3]),
        "page": best["page"],
        "source": best["source"],
        "heading": best.get("heading",""),
        "subheading": best.get("subheading",""),
        "exact": exact
    }

# =========================
# 🔥 AI (UPRAVENO)
# =========================

def ask_ai(q, context):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-base"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_KEY']}"
    }

    prompt = f"""
Jsi odborník na pojistné podmínky.

TEXT:
{context['text']}

OTÁZKA:
{q}
"""

    try:
        res = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=20)

        if res.status_code != 200:
            raise Exception()

        data = res.json()

        if isinstance(data, list):
            answer = data[0].get("generated_text", "")
        else:
            answer = ""

    except:
        answer = context["exact"]

    if not answer.strip():
        answer = "Na základě pojistných podmínek platí následující:"

    return f"""
🧠 Odpověď:
{answer.strip()}

📄 Citace:
"{context['exact']}"
(str. {context['page']}, {context['heading']}, {context['subheading']})
"""

# =========================
# ZBYTEK KÓDU = BEZE ZMĚN
# =========================
