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

# vytvoření kolekce (jen jednou)
try:
    qdrant.get_collection(collection)
except:
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

# embedding
def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# 📥 upload PDF (opravený)
def ingest_pdf(file):
    reader = PdfReader(file)
    total_pages = len(reader.pages)

    progress = st.progress(0)
    status = st.empty()

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        # ⛔ prázdná stránka
        if not text or len(text.strip()) < 20:
            continue

        # ✂️ zkrácení textu
        text = text[:2000]

        # 🔁 kontrola duplicity
        point_id = f"{file.name}_{i}"

        try:
            qdrant.retrieve(collection_name=collection, ids=[point_id])
            continue  # už existuje
        except:
            pass

        # 📦 uložení
        qdrant.upsert(
            collection_name=collection,
            points=[{
                "id": point_id,
                "vector": embed(text),
                "payload": {
                    "text": text,
                    "source": file.name,
                    "page": i + 1
                }
            }]
        )

        # ⏳ zpomalení (PROTI RATE LIMIT)
        time.sleep(1)

        # 📊 progress bar
        progress.progress((i + 1) / total_pages)
        status.text(f"Zpracovávám stránku {i+1}/{total_pages}")

    status.text("Hotovo ✅")

# 🔍 dotaz
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
            st.sidebar.success("PDF nahráno")

# 💬 chat
q = st.text_input("Zeptej se:")

if st.button("Odeslat") and q:
    answer = search(q)

    st.markdown(f"""
    <div style="background:white;padding:15px;border-left:5px solid {ACCENT};border-radius:10px;">
    {answer}
    </div>
    """, unsafe_allow_html=True)
