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
    "history": [],
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
    st.session_state.feedback.append({"question":q,"answer":a,"rating":r,"note":n})
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
# SMART MEMORY
# =========================

def get_memory(q):
    if not st.session_state.history: return ""
    current = embed_query(q)
    scored=[]
    for h in st.session_state.history[:10]:
        vec = embed_query(h["q"])
        sim=sum(a*b for a,b in zip(current,vec))
        scored.append((sim,h))
    scored=sorted(scored,key=lambda x:x[0],reverse=True)[:2]
    return "\n".join([f"U:{h['q']} A:{h['a']}" for _,h in scored])

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
# SEARCH
# =========================

def search(q):
    try:
        vec=embed_query(q)

        filt=Filter(must=[
            FieldCondition(key="insurer",match=MatchValue(value=selected_insurer)),
            FieldCondition(key="vpp_name",match=MatchValue(value=selected_vpp))
        ])

        res=qdrant.query_points("docs",query=vec,query_filter=filt,limit=10).points

        if not res:
            return [],0

        pairs=[(q,r.payload["text"]) for r in res]
        scores=reranker.predict(pairs)

        ranked=sorted(zip(res,scores),key=lambda x:x[1],reverse=True)

        ctx=[{
            "text":r.payload["text"],
            "exact":extract_sentence(r.payload["text"],q),
            "page":r.payload["page"]
        } for r,_ in ranked[:5]]

        conf=int((float(ranked[0][1])+5)*10)
        return ctx,conf

    except:
        return [],0

# =========================
# AI STREAMING
# =========================

def ask_stream(q, ctx, conf, placeholder):
    if not ctx:
        placeholder.markdown("❌ Nenalezeno → kontaktuj vedení směny")
        return "❌ Nenalezeno → kontaktuj vedení směny"

    memory = get_memory(q)
    combined = "\n\n".join([c["text"] for c in ctx])

    prompt=f"""
Shrň odpověď a uveď body.

ODPOVÍDEJ POUZE Z TEXTU.

KONTEXT:
{memory}

TEXT:
{combined}

OTÁZKA:
{q}
"""

    full=""

    try:
        response = model_gemini.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                full += chunk.text
                placeholder.markdown(full)
    except:
        full = ctx[0]["exact"]
        placeholder.markdown(full)

    cites="\n".join([f"📄 \"{c['exact']}\" (str. {c['page']})" for c in ctx])
    final=f"{full}\n\n{cites}"
    placeholder.markdown(final)
    return final

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

# CHAT INPUT
q=st.chat_input("Zeptej se...")

if q and selected_vpp!="— vyber —":
    st.session_state.history.insert(0,{"q":q,"a":"..."})
    st.rerun()

# STREAMING EXECUTION
if st.session_state.history and st.session_state.history[0]["a"]=="...":
    last=st.session_state.history[0]
    placeholder=st.empty()

    ctx,conf=search(last["q"])
    ans=ask_stream(last["q"],ctx,conf,placeholder)

    log_query(last["q"],selected_insurer,selected_vpp,conf)

    st.session_state.history[0]["a"]=ans
    st.rerun()

# CHAT BUBBLES
for h in st.session_state.history:
    st.markdown(f"<div style='text-align:right'><div style='background:#2b6cb0;color:white;padding:12px;border-radius:16px;max-width:70%'>{h['q']}</div></div>",unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:left'><div style='background:#f1f5f9;color:#111;padding:12px;border-radius:16px;max-width:70%'>{h['a']}</div></div>",unsafe_allow_html=True)
