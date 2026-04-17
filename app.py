(import streamlit as st
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
import uuid

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(text):
    return model.encode(text).tolist()

def split_text(text):
    return re.split(r'(?<=[.!?]) +', text)

def ingest_pdf(file):
    reader = PdfReader(file)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue

        for chunk in split_text(text):
            qdrant.upsert(
                collection_name="docs",
                points=[{
                    "id": str(uuid.uuid4()),
                    "vector": embed(chunk),
                    "payload": {
                        "text": chunk,
                        "page": i+1,
                        "source": file.name
                    }
                }]
            )

def ask_ai(question, contexts):
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"

    headers = {
        "Authorization": f"Bearer {st.secrets['HF_API_KEY']}"
    }

    prompt = f"""
Jsi odborník na pojistné podmínky.

Odpovídej pouze z textu.

TEXT:
{chr(10).join(contexts)}

OTÁZKA:
{question}
"""

    res = requests.post(url, headers=headers, json={"inputs": prompt})

    return res.json()[0]["generated_text"]

def search(q):
    results = qdrant.query_points(
        collection_name="docs",
        query=embed(q),
        limit=5
    ).points

    contexts = []
    sources = []

    for r in results:
        contexts.append(f"[str. {r.payload['page']}] {r.payload['text']}")
        sources.append(r.payload)

    answer = ask_ai(q, contexts)

    return answer, sources

st.title("🛡️ VPP Checker")

pwd = st.sidebar.text_input("Heslo", type="password")

if pwd == st.secrets["ADMIN_PASSWORD"]:
    file = st.sidebar.file_uploader("Nahraj PDF")
    if file:
        ingest_pdf(file)
        st.sidebar.success("Hotovo")

q = st.text_input("Dotaz")

if st.button("Odeslat") and q:
    answer, sources = search(q)

    st.write(answer)

    for s in sources:
        st.write(f"{s['source']} – str. {s['page']}")
)
