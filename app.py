import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from pypdf import PdfReader

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

def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# 📥 upload PDF
def ingest_pdf(file):
    reader = PdfReader(file)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        qdrant.upsert(
            collection_name=collection,
            points=[{
                "id": f"{file.name}_{i}",
                "vector": embed(text),
                "payload": {
                    "text": text,
                    "source": file.name,
                    "page": i+1
                }
            }]
        )

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
            {"role": "system", "content": "Odpovídej jen z textu."},
            {"role": "user", "content": context + "\n\nOtázka: " + question}
        ]
    )

    return response.choices[0].message.content + "\n\n📄 Zdroje:\n" + "\n".join(set(sources))

# 🎨 UI
PRIMARY = "#314397"
ACCENT = "#E43238"

st.set_page_config(page_title="VPP Checker")

st.title("🛡️ VPP Checker")

# 🔐 admin upload
pwd = st.sidebar.text_input("Heslo", type="password")

if pwd == st.secrets["ADMIN_PASSWORD"]:
    file = st.sidebar.file_uploader("Nahraj PDF", type="pdf")

    if file:
        ingest_pdf(file)
        st.sidebar.success("PDF nahráno")

# 💬 chat
q = st.text_input("Zeptej se:")

if st.button("Odeslat"):
    answer = search(q)

    st.markdown(f"""
    <div style="background:white;padding:15px;border-left:5px solid {ACCENT};border-radius:10px;">
    {answer}
    </div>
    """, unsafe_allow_html=True)