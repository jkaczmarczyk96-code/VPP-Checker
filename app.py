import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pypdf import PdfReader
import time

# 🔐 secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# 🧠 BATCH EMBEDDING (hlavní fix)
def embed_batch(texts):
    for attempt in range(5):
        try:
            return client.embeddings.create(
                model="text-embedding-3-small",
                input=[t[:1000] for t in texts]
            ).data

        except Exception as e:
            if "RateLimit" in str(e):
                wait = 3 + attempt * 3
                st.warning(f"⏳ Čekám {wait}s kvůli limitu...")
                time.sleep(wait)
            else:
                raise e

    raise Exception("Embedding batch selhal")

# 📥 INGEST PDF (batch verze)
def ingest_pdf(file):
    reader = PdfReader(file)

    texts = []
    metas = []

    # 📄 načtení textů
    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if not text or len(text.strip()) < 20:
            continue

        text = text[:1000]

        texts.append(text)
        metas.append({
            "id": f"{file.name}_{i}",
            "source": file.name,
            "page": i + 1
        })

    if not texts:
        st.warning("PDF neobsahuje čitelný text")
        return

    progress = st.progress(0)

    batch_size = 5

    # ⚡ batch zpracování
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_meta = metas[i:i+batch_size]

        embeddings = embed_batch(batch_texts)

        points = []
        for emb, meta, text in zip(embeddings, batch_meta, batch_texts):
            points.append({
                "id": meta["id"],
                "vector": emb.embedding,
                "payload": {
                    "text": text,
                    "source": meta["source"],
                    "page": meta["page"]
                }
            })

        qdrant.upsert(
            collection_name=collection,
            points=points
        )

        progress.progress(min((i + batch_size) / len(texts), 1.0))

        time.sleep(1)  # malé zpomalení

    st.success("PDF nahráno ✅")

# 🔍 DOTAZ (single embed OK)
def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:1000]
    ).data[0].embedding

def search(question):
    results = qdrant.search(
        collection_name=collection,
        query_vector=embed(question),
        limit=3
    )

    context = ""
    sources = []

    for r in results:
        context += r.payload["text"] + "\n\n"
        sources.append(f"{r.payload['source']} (str. {r.payload['page']})")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "Odpovídej jen z textu a ve stejném jazyce jako otázka."},
            {"role": "user", "content": context + "\n\nOtázka: " + question}
        ]
    )

    return response.choices[0].message.content + "\n\n📄 Zdroje:\n" + "\n".join(set(sources))

# 🎨 UI
PRIMARY = "#314397"
ACCENT = "#E43238"

st.set_page_config(page_title="VPP Checker")

st.markdown(f"""
<style>
body {{
    background-color: #f5f7fb;
}}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ VPP Checker")

# 🔐 admin
pwd = st.sidebar.text_input("Heslo", type="password")

if pwd == st.secrets["ADMIN_PASSWORD"]:
    file = st.sidebar.file_uploader("Nahraj PDF", type="pdf")

    if file:
        if st.sidebar.button("Nahrát do databáze"):
            ingest_pdf(file)

# 💬 chat
q = st.text_input("Zeptej se:")

if st.button("Odeslat") and q:
    answer = search(q)

    st.markdown(f"""
    <div style="background:white;padding:15px;border-left:5px solid {ACCENT};border-radius:10px;">
    {answer}
    </div>
    """, unsafe_allow_html=True)
