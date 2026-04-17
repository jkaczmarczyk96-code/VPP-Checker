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
        max_length=512,
        do_sample=False
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

collection = "docs_final"

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
    return embed_model.encode(texts)

def embed_query(text):
    return embed_model.encode([f"query: {text}"])[0]

# =========================
# 🧠 HEADINGS
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
# 🚀 INGEST
# =========================

def ingest_pdf(file):
    reader = PdfReader(file)
    st.session_state.files[file.name] = file

    all_chunks = []
    metadata = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        main, sub = detect_headings(text)
        chunks = split_text(text)

        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({
                "page": i + 1,
                "source": file.name,
                "heading": main,
                "subheading": sub
            })

    embeddings = embed_passages(all_chunks)

    points = []
    for i, emb in enumerate(embeddings):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": [float(x) for x in emb],
            "payload": {
                "text": all_chunks[i],
                **metadata[i]
            }
        })

    for i in range(0, len(points), 50):
        qdrant.upsert(collection_name=collection, points=points[i:i+50])

    return len(points)

# =========================
# 🧠 VÝBĚR VĚT
# =========================

def extract_relevant_sentences(text, question):
    sentences = re.split(r'(?<=[.!?]) +', text)
    q_emb = embed_query(question)

    scored = []
    for s in sentences:
        if len(s) < 30:
            continue
        emb = embed_passages([s])[0]
        score = sum(a*b for a, b in zip(emb, q_emb))
        scored.append((score, s))

    scored.sort(reverse=True)
    return [s for _, s in scored[:2]]

# =========================
# 🧠 AI
# =========================

def extract_facts(question, contexts):
    prompt = f"""
Vyber přesné citace z textu.

TEXT:
{chr(10).join(contexts)}

OTÁZKA:
{question}

FAKTA:
"""
    return llm(prompt)[0]["generated_text"]

def generate_answer(question, facts):
    prompt = f"""
Jsi odborník na pojištění.

Odpověz profesionálně a srozumitelně.

FAKTA:
{facts}

OTÁZKA:
{question}

ODPOVĚĎ:
"""
    return llm(prompt)[0]["generated_text"]

# =========================
# 🔍 SEARCH
# =========================

def search(question):
    results = qdrant.query_points(
        collection_name=collection,
        query=embed_query(question),
        limit=5
    ).points

    contexts = []
    sources = []

    for r in results:
        sentences = extract_relevant_sentences(r.payload["text"], question)

        for s in sentences:
            contexts.append(f"[str. {r.payload['page']}] {s}")

        sources.append(r.payload)

    facts = extract_facts(question, contexts)
    answer = generate_answer(question, facts)

    return answer, sources

# =========================
# 📄 PDF VIEW
# =========================

def show_pdf(file, page, text):
    file.seek(0)
    base64_pdf = base64.b64encode(file.read()).decode()

    st.markdown(f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}#page={page}"
    width="100%" height="600"></iframe>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#fff3cd;padding:15px;border-left:5px solid red;">
    {text}
    </div>
    """, unsafe_allow_html=True)

# =========================
# UI
# =========================

pwd = st.sidebar.text_input("Heslo", type="password")

if pwd == st.secrets["ADMIN_PASSWORD"]:
    files = st.sidebar.file_uploader("PDF", accept_multiple_files=True)

    if files and st.sidebar.button("Nahrát"):
        for f in files:
            ingest_pdf(f)

q = st.text_input("Dotaz")

if st.button("Odeslat") and q:
    answer, sources = search(q)

    st.write(answer)

    for i, s in enumerate(sources):
        st.markdown(f"""
        **{s['source']}**  
        str. {s['page']}  
        {s['heading']}  
        {s['subheading']}
        """)

        if s["source"] in st.session_state.files:
            if st.button(f"Otevřít", key=i):
                show_pdf(
                    st.session_state.files[s["source"]],
                    s["page"],
                    s["text"][:300]
                )
