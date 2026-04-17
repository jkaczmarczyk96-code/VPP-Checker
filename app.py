import streamlit as st
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
import uuid

# 🔐 QDRANT
qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

collection = "docs"

# 🧠 EMBEDDING (lehčí model)
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text).tolist()

# ✂️ CHUNKING
def split_text(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

# 📥 NAHRÁNÍ PDF
def ingest_pdf(file):
    reader = PdfReader(file)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        chunks = split_text(text)

        for chunk in chunks:
            qdrant.upsert(
                collection_name=collection,
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": embed(chunk),
                    "payload": {
                        "text": chunk,
                        "page": i + 1,
                        "source": file.name
                    }
                }]
            )

# 🔍 DOTAZ NA BACKEND
def ask_backend(question, contexts):
    url = "https://TVŮJ-RENDER-URL.onrender.com/ask"

    res = requests.post(url, json={
        "question": question,
        "contexts": contexts
    })

    return res.json()["answer"]

# 🔍 SEARCH
def search(question):
    results = qdrant.query_points(
        collection_name=collection,
        query=embed(question),
        limit=5
    ).points

    contexts = []
    sources = []

    for r in results:
        text = r.payload["text"]
        page = r.payload["page"]

        contexts.append(f"[str. {page}] {text}")
        sources.append(r.payload)

    answer = ask_backend(question, contexts)

    return answer, sources

# 🎨 UI
st.title("🛡️ VPP Checker")

pwd = st.sidebar.text_input("Heslo", type="password")

if pwd == st.secrets["ADMIN_PASSWORD"]:
    file = st.sidebar.file_uploader("Nahraj PDF")

    if file:
        ingest_pdf(file)
        st.sidebar.success("Nahráno")

q = st.text_input("Zeptej se")

if st.button("Odeslat") and q:
    answer, sources = search(q)

    st.write(answer)

    for s in sources:
        st.write(f"{s['source']} – str. {s['page']}")