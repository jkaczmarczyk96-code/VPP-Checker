import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# 🧠 AI model (zdarma)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔐 secrets
qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

collection = "docs"

# vytvoření kolekce
try:
    qdrant.get_collection(collection)
except:
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# 🧠 embedding zdarma
def embed(text):
    return model.encode(text).tolist()

# 📥 ingest PDF
def ingest_pdf(file):
    reader = PdfReader(file)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if not text or len(text.strip()) < 20:
            continue

        qdrant.upsert(
            collection_name=collection,
            points=[{
                "id": f"{file.name}_{i}",
                "vector": embed(text[:1000]),
                "payload": {
                    "text": text,
                    "source": file.name,
                    "page": i+1
                }
            }]
        )

    st.success("PDF nahráno")

# 🔍 vyhledávání (bez OpenAI)
def search(question):
    results = qdrant.search(
        collection_name=collection,
        query_vector=embed(question),
        limit=3
    )

    answer = ""
    sources = []

    for r in results:
        answer += r.payload["text"][:300] + "\n\n"
        sources.append(f"{r.payload['source']} (str. {r.payload['page']})")

    return answer + "\n\n📄 Zdroje:\n" + "\n".join(sources)

# 🎨 UI
st.title("🛡️ VPP Checker")

pwd = st.sidebar.text_input("Heslo", type="password")

if pwd == st.secrets["ADMIN_PASSWORD"]:
    file = st.sidebar.file_uploader("Nahraj PDF")

    if file:
        ingest_pdf(file)

q = st.text_input("Zeptej se:")

if st.button("Odeslat"):
    st.write(search(q))
