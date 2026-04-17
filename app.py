import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
import base64

# 🧠 MODEL
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔐 DB
qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

# 🔥 NOVÁ KOLEKCE (fix chyby)
collection = "docs_v2"

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

# 🧠 EMBEDDING
def embed(text):
    return model.encode(text).tolist()

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

# 📥 INGEST
def ingest_pdf(file):
    reader = PdfReader(file)
    st.session_state.files[file.name] = file

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if not text or len(text.strip()) < 30:
            continue

        chunks = split_text(text)

        for j, chunk in enumerate(chunks):
            qdrant.upsert(
                collection_name=collection,
                points=[{
                    "id": f"{file.name}_{i}_{j}",
                    "vector": embed(chunk),
                    "payload": {
                        "text": chunk,
                        "source": file.name,
                        "page": i + 1
                    }
                }]
            )

    st.success("PDF nahráno")

# 🧠 BEST SENTENCE
def best_sentence(text, question):
    sentences = re.split(r'(?<=[.!?]) +', text)
    q_emb = embed(question)

    best = ""
    best_score = -1

    for s in sentences:
        if len(s) < 20:
            continue

        score = sum([a*b for a, b in zip(embed(s), q_emb)])

        if score > best_score:
            best_score = score
            best = s

    return best.strip()

# 📂 DOKUMENTY
def get_documents():
    res = qdrant.scroll(collection_name=collection, limit=1000)
    docs = set()

    for p in res[0]:
        docs.add(p.payload["source"])

    return list(docs)

# 🔍 SEARCH
def search(question, doc_filter=None):
    query_filter = None

    if doc_filter and doc_filter != "Vše":
        query_filter = {
            "must": [
                {"key": "source", "match": {"value": doc_filter}}
            ]
        }

    results = qdrant.search(
        collection_name=collection,
        query_vector=embed(question),
        query_filter=query_filter,
        limit=5
    )

    answer_parts = []
    sources = []

    for r in results:
        text = r.payload["text"]
        page = r.payload["page"]
        source = r.payload["source"]

        sentence = best_sentence(text, question)

        if sentence:
            answer_parts.append(
                f"{sentence} <span style='color:#E43238'>[str. {page}]</span>"
            )
            sources.append((source, page, sentence))

    answer = " ".join(answer_parts[:3])

    return answer, sources

# 📄 PDF VIEW
def show_pdf(file, page):
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}#page={page}"
    width="100%" height="600" type="application/pdf"></iframe>
    """

    st.markdown(pdf_display, unsafe_allow_html=True)

# 🎨 DESIGN
PRIMARY = "#314397"
ACCENT = "#E43238"

st.set_page_config(page_title="VPP Checker", layout="wide")

st.markdown(f"""
<style>
body {{
    background-color: #f5f7fb;
}}

.header {{
    background: {PRIMARY};
    padding: 20px;
    border-radius: 12px;
    color: white;
}}

.card {{
    background: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid {ACCENT};
    margin-top: 20px;
}}

.source {{
    background: white;
    padding: 10px;
    border-radius: 8px;
    margin-top: 5px;
    border-left: 3px solid {PRIMARY};
}}

.highlight {{
    background-color: #fff3cd;
}}
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

if pwd == st.secrets["ADMIN_PASSWORD"]:
    file = st.sidebar.file_uploader("Nahraj PDF", type="pdf")

    if file:
        if st.sidebar.button("Nahrát"):
            ingest_pdf(file)

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

# 💬 HISTORIE
st.markdown("### 💬 Historie")

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
            if st.button(f"Otevřít {src} str. {page}"):
                show_pdf(st.session_state.files[src], page)
