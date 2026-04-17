import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import pdfplumber
import uuid
import re
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# =========================
# CONFIG
# =========================

VECTOR_SIZE = 768
st.set_page_config(page_title="VPP Checker", layout="wide")

# =========================
# STYLE
# =========================

st.markdown("""
<style>
.chat-user {background:#eef3ff;padding:14px;border-radius:12px;margin-top:10px;}
.chat-ai {background:white;padding:14px;border-radius:12px;border-left:4px solid #314397;margin-top:10px;}
.conf-high {color:#0f7b0f;font-weight:bold;}
.conf-mid {color:#b8860b;font-weight:bold;}
.conf-low {color:#b22222;font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION
# =========================

for key, default in {
    "history": [],
    "feedback": [],
    "feedback_done": {},
    "feedback_open": {},
    "logged": False,
    "last_logged": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# =========================
# GEMINI
# =========================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# =========================
# GOOGLE SHEETS
# =========================

@st.cache_resource
def get_client():
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    return gspread.authorize(creds)

@st.cache_resource
def get_feedback_sheet():
    return get_client().open("VPP_Feedback").worksheet("feedback")

@st.cache_resource
def get_logs_sheet():
    return get_client().open("VPP_Feedback").worksheet("logs")

@st.cache_resource
def get_analytics_sheet():
    return get_client().open("VPP_Feedback").worksheet("analytics")

def classify_error(note):
    if not note:
        return "unknown"
    n = note.lower()
    if "citace" in n:
        return "citation_error"
    if "mimo" in n:
        return "wrong_context"
    if "nepřes" in n or "špatně" in n:
        return "inaccuracy"
    return "other"

def save_feedback(q, a, rating, note=""):
    key = f"{q}_{a}_{rating}_{note}"
    if st.session_state.get("last_feedback") == key:
        return
    st.session_state["last_feedback"] = key

    error_type = classify_error(note)

    st.session_state.feedback.append({
        "q": q,
        "a": a,
        "rating": rating,
        "note": note,
        "error_type": error_type
    })

    try:
        get_feedback_sheet().append_row([
            q, a, rating, note, error_type,
            selected_insurer, selected_vpp
        ])
    except:
        pass

# =========================
# LOGGING
# =========================

def log_query(q, insurer, vpp, confidence):
    timestamp = str(datetime.now())

    try:
        get_logs_sheet().append_row([timestamp, q, insurer, vpp, confidence])
    except:
        pass

    try:
        get_analytics_sheet().append_row([timestamp, q, insurer, vpp, confidence])
    except:
        pass

# =========================
# ALERTS
# =========================

def check_alerts():
    try:
        rows = get_feedback_sheet().get_all_records()

        query_counts = {}
        vpp_counts = {}

        for r in rows:
            if r.get("rating") == "dislike":
                q = r.get("q")
                vpp = r.get("vpp")

                if q:
                    query_counts[q] = query_counts.get(q, 0) + 1
                if vpp:
                    vpp_counts[vpp] = vpp_counts.get(vpp, 0) + 1

        bad_queries = [q for q, c in query_counts.items() if c >= 3]
        bad_vpps = [v for v, c in vpp_counts.items() if c >= 5]

        return bad_queries, bad_vpps
    except:
        return [], []

# =========================
# QDRANT
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

def init_collection():
    try:
        qdrant.get_collection("docs")
    except:
        qdrant.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

init_collection()

# =========================
# MODELS
# =========================

@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-base")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

model = load_model()
reranker = load_reranker()

# =========================
# HELPERS
# =========================

def embed(texts):
    return model.encode(texts).tolist()

def smart_chunk(text, size=800, overlap=150):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def extract_exact_sentence(paragraph, question):
    sentences = re.split(r'(?<=[.!?]) +', paragraph)
    q_words = set(question.lower().split())
    return max(sentences, key=lambda s: len(q_words & set(s.lower().split())), default=paragraph)

# =========================
# INGEST
# =========================

def ingest_pdf(files, vpp_name, insurer):
    chunks = []
    for file in files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
            except:
                text = None

            if not text:
                try:
                    with pdfplumber.open(file) as pdf:
                        text = pdf.pages[i].extract_text()
                except:
                    continue

            for c in smart_chunk(text):
                if len(c) < 50:
                    continue
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": c,
                    "page": i+1,
                    "source": file.name,
                    "vpp_name": vpp_name,
                    "insurer": insurer
                })

    vectors = embed([c["text"] for c in chunks])
    points = [{"id": c["id"], "vector": v, "payload": c} for c, v in zip(chunks, vectors)]
    qdrant.upsert("docs", points)

# =========================
# FILTR
# =========================

st.sidebar.markdown("## 📂 Filtr dokumentů")

def get_insurers():
    try:
        res = qdrant.scroll("docs", limit=1000, with_payload=True)
        return sorted(set(r.payload.get("insurer") for r in res[0] if r.payload.get("insurer")))
    except:
        return []

def get_vpps(insurer):
    try:
        res = qdrant.scroll("docs", limit=1000, with_payload=True)
        return sorted(set(r.payload.get("vpp_name") for r in res[0] if r.payload.get("insurer") == insurer))
    except:
        return []

insurers = get_insurers()
selected_insurer = st.sidebar.selectbox("Pojišťovna", ["— vyber —"] + insurers)

if selected_insurer != "— vyber —":
    vpps = get_vpps(selected_insurer)
    selected_vpp = st.sidebar.selectbox("VPP", ["— vyber —"] + vpps)
else:
    selected_vpp = None

# =========================
# ADMIN + ALERTY
# =========================

st.sidebar.markdown("## 🔐 Administrace")

if not st.session_state.logged:
    pwd = st.sidebar.text_input("Heslo", type="password")
    if st.sidebar.button("Přihlásit"):
        if pwd == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.logged = True
            st.rerun()
else:
    files = st.sidebar.file_uploader("PDF", accept_multiple_files=True)
    vpp_name = st.sidebar.text_input("Název VPP")
    insurer = st.sidebar.text_input("Pojišťovna")

    if st.sidebar.button("Nahrát"):
        if files and vpp_name and insurer:
            ingest_pdf(files, vpp_name, insurer)

    bad_q, bad_v = check_alerts()

    if bad_q or bad_v:
        st.sidebar.markdown("## 🚨 Alerty")

        if bad_q:
            st.sidebar.error("Problémové dotazy:")
            for q in bad_q[:3]:
                st.sidebar.write(f"- {q}")

        if bad_v:
            st.sidebar.error("Problémové VPP:")
            for v in bad_v[:3]:
                st.sidebar.write(f"- {v}")

# =========================
# SEARCH
# =========================

def search(q):
    vec = embed([q])[0]

    conditions = []
    if selected_insurer and selected_insurer != "— vyber —":
        conditions.append(FieldCondition(key="insurer", match=MatchValue(value=selected_insurer)))
    if selected_vpp and selected_vpp != "— vyber —":
        conditions.append(FieldCondition(key="vpp_name", match=MatchValue(value=selected_vpp)))

    query_filter = Filter(must=conditions) if conditions else None

    results = qdrant.query_points("docs", query=vec, query_filter=query_filter, limit=10).points

    if not results:
        return [], 0

    pairs = [(q, r.payload["text"]) for r in results]
    scores = reranker.predict(pairs)

    ranked = list(zip(results, scores))

    for i, (r, s) in enumerate(ranked):
        for fb in st.session_state.feedback:
            snippet = fb["a"][:200] if fb.get("a") else ""
            if snippet and snippet in r.payload["text"]:
                if fb["rating"] == "like":
                    s += 0.4
                elif fb["rating"] == "dislike":
                    s -= 0.7
        ranked[i] = (r, s)

    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)

    confidence = max(0, min(100, int((float(ranked[0][1]) + 5) * 10)))

    contexts = [{
        "text": r.payload["text"],
        "exact": extract_exact_sentence(r.payload["text"], q),
        "page": r.payload["page"]
    } for r, _ in ranked[:5]]

    return contexts, confidence

# =========================
# AI
# =========================

def is_dental_query(q):
    return any(k in q.lower() for k in ["zuby","zub","dentální"])

def ask(q, ctx, confidence):
    if is_dental_query(q):
        return "👉 Použij Zuby AI v Copilotu."

    if not ctx:
        return "❌ Nenalezeno. 👉 Konzultuj s vedením směny."

    combined = "\n\n".join([c["text"] for c in ctx])
    citations = "\n".join([f"- \"{c['exact']}\" (str. {c['page']})" for c in ctx])

    conf_class = "conf-high" if confidence > 70 else "conf-mid" if confidence > 40 else "conf-low"

    if confidence < 40:
        return f"<span class='{conf_class}'>Spolehlivost: {confidence}%</span>\n\n❗ Konzultuj vedení směny.\n\n{ctx[0]['exact']}\n\n{citations}"

    try:
        r = model_gemini.generate_content(f"Odpověz pouze z textu:\n{combined}")
        answer = r.text.strip()
    except:
        return f"⚠️ Chyba generování.\n\n{ctx[0]['exact']}\n\n{citations}"

    return f"<span class='{conf_class}'>Spolehlivost: {confidence}%</span>\n\n{answer}\n\n{citations}"

# =========================
# CHAT
# =========================

st.title("🛡️ VPP Checker")

q = st.chat_input("Zeptej se...")

if q:
    ctx, conf = search(q)
    ans = ask(q, ctx, conf)
    log_query(q, selected_insurer, selected_vpp, conf)
    st.session_state.history.insert(0, {"q": q, "a": ans})

# =========================
# RENDER + FEEDBACK
# =========================

for i, h in enumerate(st.session_state.history):
    st.markdown(f"<div class='chat-user'>{h['q']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-ai'>{h['a']}</div>", unsafe_allow_html=True)

    if st.session_state.feedback_done.get(i):
        st.success("Díky!")
        continue

    col1, col2 = st.columns(2)

    if col1.button("👍", key=f"like_{i}"):
        save_feedback(h["q"], h["a"], "like")
        st.session_state.feedback_done[i] = True
        st.rerun()

    if col2.button("👎", key=f"dislike_{i}"):
        st.session_state.feedback_open[i] = True

    if st.session_state.feedback_open.get(i):
        note = st.text_input("Co bylo špatně?", key=f"note_{i}")
        if st.button("Odeslat", key=f"send_{i}"):
            save_feedback(h["q"], h["a"], "dislike", note)
            st.session_state.feedback_done[i] = True
            st.session_state.feedback_open[i] = False
            st.rerun()
