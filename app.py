import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pypdf import PdfReader
import re
import base64
import uuid

# =========================
# 🧠 MODELY
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )

embed_model = load_embedding_model()
llm = load_llm()

# =========================
# 🔐 QDRANT
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

collection = "docs_v7"

try:
    qdrant.get_collection(collection)
except:
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# =========================
# 💬 SESSION
# =========================

if "history" not in st.session_state:
    st.session_state.history = []

if "files" not in st.session_state:
    st.session_state.files = {}

# =========================
# 🧠 EMBEDDING
# =========================

def embed_passages(texts):
    texts = [f"passage: {t}" for t in texts]
    vectors = embed_model.encode(texts)
    return [[float(x) for x in v] for v in vectors]

def embed_query(text):
    vec = embed_model.encode([f"query: {text}"])[0]
    return [float(x) for x in vec]

# =========================
# ✂️ CHUNKING
# =========================

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

# =========================
# 🚀 INGEST (FAST)
# =========================

def ingest_pdf(file):
    reader = PdfReader(file)
    st.session_state.files[file.name] = file

    all_chunks = []
    metadata = []

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

    embeddings = embed_passages(all_chunks)

    points = []
    for i, emb in enumerate(embeddings):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": emb,
            "payload": {
                "text": all_chunks[i],
                "source": metadata[i]["source"],
                "page": metadata[i]["page"]
            }
        })

    batch_size = 50
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=collection,
            points=points[i:i+batch_size]
        )

    return len(points)

# =========================
# 🧠 LLM ODPOVĚĎ
# =========================

def generate_llm_answer(question, contexts):
    context_text = "\n\n".join(contexts[:3])

    prompt = f"""
Odpověz na otázku pouze na základě textu.

Používej citace [str. X].
Nevymýšlej si informace.

Text:
{context_text}

Otázka:
{question}

Odpověď:
"""

    result = llm(prompt)[0]["generated_text"]
    return result

# =========================
# 🔍 SEARCH
# =========================

def search(question, doc_filter=None):
    q_emb = embed_query(question)

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

    contexts = []
    sources = []

    for r in results:
        text = r.payload["text"]
        page = r.payload["page"]
        source = r.payload["source"]

        contexts.append(f"[str. {page}] {text}")
        sources.append((source, page))

    answer = generate_llm_answer(question, contexts)

    return answer, sources

# =========================
# 📄 PDF VIEW
# =========================

def show_pdf(file, page):
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    st.markdown(f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}#page={page}"
    width="100%" height="600"></iframe>
    """, unsafe_allow_html=True)

# =========================
# 🎨 DESIGN
# =========================

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

st.markdown("""
<div class="header">
<h2>🛡️ VPP Checker</h2>
<p>AI kontrola pojistných podmínek</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("### 📊 Ovládání")

if st.sidebar.button("🗑️ Vymazat historii"):
    st.session_state.history = []

pwd = st.sidebar.text_input("Admin heslo", type="password")

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

# =========================
# FILTER
# =========================

@st.cache_data
def get_documents():
    res = qdrant.scroll(collection_name=collection, limit=1000)
    return list({p.payload["source"] for p in res[0]})

docs = get_documents()
selected_doc = st.sidebar.selectbox("Filtr dokumentu", ["Vše"] + docs)

# =========================
# DOTAZ
# =========================

st.markdown("### 🔍 Zadej dotaz")
q = st.text_input("")

if st.button("Odeslat") and q:
    answer, sources = search(q, selected_doc)

    st.session_state.history.append({
        "question": q,
        "answer": answer,
        "sources": sources
    })

# =========================
# HISTORIE
# =========================

for item in reversed(st.session_state.history):
    st.markdown(f"""
    <div class="card">
    <b>❓ {item['question']}</b><br><br>
    {item['answer']}
    </div>
    """, unsafe_allow_html=True)

    for src, page in item["sources"]:
        st.markdown(f"""
        <div class="source">
        <b>{src}</b> – str. {page}
        </div>
        """, unsafe_allow_html=True)

        if src in st.session_state.files:
            if st.button(
                f"Otevřít {src} str. {page}",
                key=f"{src}-{page}"
            ):
                show_pdf(st.session_state.files[src], page)
