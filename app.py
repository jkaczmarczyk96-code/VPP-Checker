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

st.set_page_config(page_title="VPP Checker", layout="wide")

VECTOR_SIZE = 768

INSURERS = [
    "CZ - GČPOJ","CZ - Direct","CZ - TravelCare","CZ - Uniqa",
    "SK - TravelCare","SK - Generali","SK - ECP","SK - Wüstenrot","SK - Uniqa"
]

# =========================
# SESSION
# =========================

for k, v in {
    "messages": [],
    "feedback": [],
    "feedback_done": {},
    "feedback_open": {},
    "logged": False,
    "upload_key": str(uuid.uuid4())
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

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

def save_feedback(q,a,r,n):
    try:
        get_client().open("VPP_Feedback").worksheet("feedback").append_row(
            [q,a,r,n,"",selected_insurer,selected_vpp]
        )
    except:
        pass

def log_query(q,ins,vpp,conf):
    try:
        sheet = get_client().open("VPP_Feedback")
        row = [str(datetime.now()),q,ins,vpp,conf]
        sheet.worksheet("logs").append_row(row)
        sheet.worksheet("analytics").append_row(row)
    except:
        pass

# =========================
# ALERTY
# =========================

st.sidebar.markdown("## 🚨 Alerty")
try:
    data = get_client().open("VPP_Feedback").worksheet("feedback").get_all_records()
    bad = [r for r in data[-20:] if r.get("rating")=="dislike"]
    st.sidebar.error(f"⚠️ {len(bad)} negativních feedbacků") if bad else st.sidebar.success("✅ OK")
except:
    st.sidebar.info("Alerty nedostupné")

# =========================
# QDRANT
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

def init_collection():
    try:
        col = qdrant.get_collection("docs")
        if col.config.params.vectors.size != VECTOR_SIZE:
            qdrant.delete_collection("docs")
            qdrant.create_collection("docs",
                vectors_config=VectorParams(size=VECTOR_SIZE,distance=Distance.COSINE))
    except:
        qdrant.create_collection("docs",
            vectors_config=VectorParams(size=VECTOR_SIZE,distance=Distance.COSINE))

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

def embed_query(x): return model.encode([f"query: {x}"])[0].tolist()
def embed_doc(x): return model.encode([f"passage: {x}"])[0].tolist()

# =========================
# HELPERS
# =========================

def normalize(x): return x.strip().upper() if x else ""

def smart_chunk(t, s=800, o=150):
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i+s])
        i += s-o
    return out

def extract_sentence(p,q):
    s=re.split(r'(?<=[.!?]) +',p)
    qw=set(q.lower().split())
    return max(s,key=lambda x:len(qw & set(x.lower().split())),default=p)

def safe_scroll():
    try:
        return qdrant.scroll("docs",limit=1000,with_payload=True)[0]
    except:
        return []

def get_vpps(ins):
    return sorted(set(r.payload.get("vpp_name") for r in safe_scroll() if r.payload.get("insurer")==ins))

# =========================
# MEMORY
# =========================

def get_memory(q):
    if not st.session_state.messages:
        return ""

    current = embed_query(q)
    scored = []

    for m in st.session_state.messages:
        if m["role"] == "user":
            vec = embed_query(m["content"])
            sim = sum(a*b for a,b in zip(current, vec))
            scored.append((sim, m["content"]))

    scored = sorted(scored, reverse=True)[:2]
    return "\n".join([f"U:{x[1]}" for x in scored])

# =========================
# INGEST
# =========================

def ingest_pdf(files,vpp,ins):
    vpp=normalize(vpp)
    if not vpp:
        st.sidebar.error("Zadej VPP")
        return

    chunks=[]
    for f in files:
        reader=PdfReader(f)
        for i,p in enumerate(reader.pages):
            try: t=p.extract_text()
            except: t=None
            if not t:
                try:
                    with pdfplumber.open(f) as pdf:
                        t=pdf.pages[i].extract_text()
                except:
                    continue
            for c in smart_chunk(t):
                if len(c)<50: continue
                chunks.append({
                    "id":str(uuid.uuid4()),
                    "text":c,
                    "page":i+1,
                    "vpp_name":vpp,
                    "insurer":ins
                })

    vec=[embed_doc(c["text"]) for c in chunks]
    pts=[{"id":c["id"],"vector":v,"payload":c} for c,v in zip(chunks,vec)]
    qdrant.upsert("docs",pts)

    st.session_state["upload_key"]=str(uuid.uuid4())

# =========================
# SEARCH (HYBRID)
# =========================

def keyword_score(q, text):
    return len(set(q.lower().split()) & set(text.lower().split()))

def search(q):
    try:
        vec = embed_query(q + " pojištění výluky podmínky")

        ins = normalize(selected_insurer)
        vpp = normalize(selected_vpp)

        filt=None
        if ins!="— VYBER —" and vpp!="— VYBER —":
            filt=Filter(must=[
                FieldCondition(key="insurer",match=MatchValue(value=ins)),
                FieldCondition(key="vpp_name",match=MatchValue(value=vpp))
            ])

        res=qdrant.query_points("docs",query=vec,query_filter=filt,limit=25).points

        if not res:
            res=qdrant.query_points("docs",query=vec,limit=25).points

        if not res:
            return [],0

        enriched=[(r,keyword_score(q,r.payload["text"])) for r in res]
        pairs=[(q,r.payload["text"]) for r,_ in enriched]
        scores=reranker.predict(pairs)

        final=[(r,s+(kw*0.3)) for (r,kw),s in zip(enriched,scores)]
        ranked=sorted(final,key=lambda x:x[1],reverse=True)

        ctx=[{
            "text":r.payload["text"],
            "exact":extract_sentence(r.payload["text"],q),
            "page":r.payload["page"]
        } for r,_ in ranked[:5]]

        conf=int(min(max(ranked[0][1]*20,0),100))
        return ctx,conf

    except:
        return [],0

# =========================
# UI
# =========================

st.title("🛡️ VPP Checker")

selected_insurer=st.sidebar.selectbox("Pojišťovna",["— vyber —"]+INSURERS)

selected_vpp="— vyber —"
if selected_insurer!="— vyber —":
    selected_vpp=st.sidebar.selectbox("VPP",["— vyber —"]+get_vpps(selected_insurer))

# ADMIN
if not st.session_state.logged:
    if st.sidebar.text_input("Heslo",type="password")==st.secrets["ADMIN_PASSWORD"]:
        st.session_state.logged=True
        st.rerun()
else:
    f=st.sidebar.file_uploader("PDF",accept_multiple_files=True,key=st.session_state["upload_key"])
    v=st.sidebar.text_input("Název VPP")
    i=st.sidebar.selectbox("Pojišťovna",INSURERS)

    if st.sidebar.button("Nahrát"):
        ingest_pdf(f,v,i)

# =========================
# CHAT
# =========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Zeptej se..."):

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full = ""

        ctx, conf = search(prompt)

        if not ctx:
            placeholder.markdown("❌ Nenalezeno → kontaktuj vedení směny")
            full = "❌ Nenalezeno → kontaktuj vedení směny"

        else:
            memory = get_memory(prompt)
            combined = "\n\n".join([c["text"] for c in ctx])

            prompt_ai = f"""
Shrň odpověď a uveď body.

ODPOVÍDEJ POUZE Z TEXTU.

KONTEXT:
{memory}

TEXT:
{combined}

OTÁZKA:
{prompt}
"""

            stream = model_gemini.generate_content(prompt_ai, stream=True)

            for chunk in stream:
                if chunk.text:
                    full += chunk.text
                    placeholder.markdown(full + "▌")

            cites = "\n".join([
                f"📄 \"{c['exact']}\" (str. {c['page']})"
                for c in ctx
            ])

            full = f"{full}\n\n{cites}"
            placeholder.markdown(full)

        st.session_state.messages.append({
            "role":"assistant",
            "content":full
        })

        log_query(prompt, selected_insurer, selected_vpp, conf)
