import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
import base64
import uuid

# ⚡ CACHE MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# 🔐 DB
qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

collection = "docs_v5"

try:
    qdrant.get_collection(collection)
except:
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# 💬 SESSION
if "history" not in st.session_state:
    st.session_state.history = []

if "files" not in st.session_state:
    st.session_state.files = {}

# ⚡ EMBEDDING (rychlý batch pro search)
@st.cache_data(show_spinner=False)
def embed_cached(text):
    vec = model.encode(text)
    return [float(x) for x in vec]

def embed(text):
    return embed_cached(text)

# ✂️ CHUNKING
def split_text(text, chunk_size=400):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += " " + s
        else:
            chunks.append(current.strip())
            current = s

    if current:
        chunks.append(current.strip())

    return chunks

# 🚀 FAST INGEST (batch)
def ingest_pdf(file):
    reader = PdfReader(file)
    st.session_state.files[file.name] = file

    all_chunks = []
    metadata = []

    # načtení textu
    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if not text or len(text.strip()) < 30:
            continue

        chunks = split_text(text)

        for chunk in chunks:
            if not chunk or len(chunk.strip()) < 20:
                continue

            all_chunks.append(chunk)
            metadata.append({
                "source": file.name,
                "page": i + 1
            })

    if not all_chunks:
        return 0

    # ⚡ BATCH EMBEDDING (hlavní zrychlení)
    embeddings = model.encode(all_chunks)

    # ⚡ připrav body
    points = []
    for i, emb in enumerate(embeddings):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": [float(x) for x in emb],
            "payload": {
                "text": all_chunks[i],
                "source": metadata[i]["source"],
                "page": metadata[i]["page"]
            }
        })

    # ⚡ BATCH UPSERT (méně requestů)
    batch_size = 50
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=collection,
            points=points[i:i+batch_size]
        )

    return len(points)

# 🧠 BEST SENTENCE
def best_sentence(text, question, q_emb):
    sentences = re.split(r'(?<=[.!?]) +', text)

    best = ""
    best_score = -1

    for s in sentences:
        if len(s) < 20:
            continue

        s_emb = embed(s)
        score = sum(a*b for a, b in zip(s_emb, q_emb))

        if score > best_score:
            best_score = score
            best = s

    return best.strip()

# 📂 CACHE dokumentů
@st.cache_data
def get_documents():
    res = qdrant.scroll(collection_name=collection, limit=1000)
    return list({p.payload["source"] for p in res[0]})

# 🔍 SEARCH
def search(question, doc_filter=None):
    q_emb = embed(question)

    query_filter = None
    if doc_filter and doc_filter != "Vše":
        query_filter = {
            "must": [{"key": "source", "match": {"value": doc_filter}}]
        }

    results = qdrant.query_points(
        collection_name=collection,
        query=q_emb,
        query_filter=query_filter,
        limit=5
    ).points

    answer_parts = []
    sources = []

    for r in results:
        sentence = best_sentence(r.payload["text"], question, q_emb)

        if sentence:
            answer_parts.append(
                f"{sentence} <span style='color:#E43238'>[str. {r.payload['page']}]</span>"
            )
            sources.append((r.payload["source"], r.payload["page"], sentence))

    return " ".join(answer_parts[:3]), sources

# 📄 PDF VIEW
def show_pdf(file, page):
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    st.markdown(f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}#page={page}"
    width="100%" height="600"></iframe>
    """, unsafe_allow_html=True)

# 🎨 DESIGN
PRIMARY = "#314397"
ACCENT = "#E43238"

st.set_page_config(page_title="VPP Checker", layout="wide")

st.markdown(f"""
<style>
.block-container {{padding-top:2rem;}}
.header {{background:{PRIMARY};padding:20px;border-radius:12px;color:white;}}
.card {{background:white;padding:20px;border-radius:12px;border-left:6px solid {ACCENT};margin-top:20px;}}
.source {{background:white;padding:10px;border-radius:8px;margin-top:5px;border-left:3px solid {PRIMARY};}}
.highlight {{background:#fff3cd;}}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="header">
<h2>🛡️ VPP Checker</h2>
<p>AI kontrola pojistných podmínek</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.markdown("### 📊 Ovládání")

if st.sidebar.button("🗑️ Vymazat historii"):
    st.session_state.history = []

pwd = st.sidebar.text_input("Admin heslo", type="password")

# 🔥 MULTI UPLOAD + FAST INGEST
if pwd == st.secrets["ADMIN_PASSWORD"]:
    files = st.sidebar.file_uploader(
        "Nahraj PDF",
        type="pdf",
        accept_multiple_files=True
    )

    if files and st.sidebar.button("Nahrát všechny"):
        progress = st.progress(0)

        total_files = len(files)
        total_chunks = 0

        for idx, file in enumerate(files):
            st.sidebar.write(f"📄 {file.name}")
            chunks = ingest_pdf(file)
            total_chunks += chunks

            progress.progress((idx + 1) / total_files)

        st.sidebar.success(
            f"✅ Nahráno {total_files} souborů ({total_chunks} částí)"
        )

# 📂 FILTER
docs = get_documents()
selected_doc = st.sidebar.selectbox("Filtr dokumentu", ["Vše"] + docs)

# DOTAZ
st.markdown("### 🔍 Zadej dotaz")
q = st.text_input("")

if st.button("Odeslat") and q:
    answer, sources = search(q, selected_doc)

    st.session_state.history.append({
        "question": q,
        "answer": answer,
        "sources": sources
    })

# HISTORIE
for item in reversed(st.session_state.history):
    st.markdown(f"""
    <div class="card">
    <b>❓ {item['question']}</b><br><br>
    {item['answer']}
    </div>
    """, unsafe_allow_html=True)

    for src, page, sentence in item["sources"]:
        st.markdown(f"""
        <div class="source">
        <b>{src}</b> – str. {page}<br>
        <span class="highlight">{sentence}</span>
        </div>
        """, unsafe_allow_html=True)

        if src in st.session_state.files:
            if st.button(
                f"Otevřít {src} str. {page}",
                key=f"{src}-{page}-{sentence[:10]}"
            ):
                show_pdf(st.session_state.files[src], page)
