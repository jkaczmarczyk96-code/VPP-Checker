import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import pdfplumber
import uuid
import re
import time
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
    "logged": False,
    "upload_key": str(uuid.uuid4())
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# GEMINI SAFE LAYER
# =========================

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest"
]

def generate_safe(prompt, stream=True, retries=2):
    last_error = None

    for model_name in MODELS:
        model = genai.GenerativeModel(model_name)

        for _ in range(retries):
            try:
                if stream:
                    return model.generate_content(prompt, stream=True)
                else:
                    return model.generate_content(prompt)
            except Exception as e:
                last_error = e
                time.sleep(0.5)

    st.error(f"AI ERROR: {last_error}")
    return None

# =========================
# GOOGLE SHEETS
# =========================

@st.cache_resource
def get_client():
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    return gspread.authorize(creds)

def log_query(q,ins,vpp,conf):
    try:
        sheet = get_client().open("VPP_Feedback")
        row = [str(datetime.now()),q,ins,vpp,conf]
        sheet.worksheet("logs").append_row(row)
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
# QDRANT + INDEX
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
            "docs",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

    # 🔥 payload indexy (kritické)
    try:
        qdrant.create_payload_index("docs", "insurer", PayloadSchemaType.KEYWORD)
    except:
        pass

    try:
        qdrant.create_payload_index("docs", "vpp_name", PayloadSchemaType.KEYWORD)
    except:
        pass

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

def get_memory(q):
    return ""

# =========================
# INGEST
# =========================

def ingest_pdf(files,vpp,ins):
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
                    "vpp_name":normalize(vpp),
                    "insurer":normalize(ins)
                })

    vec=[embed_doc(c["text"]) for c in chunks]
    pts=[{"id":c["id"],"vector":v,"payload":c} for c,v in zip(chunks,vec)]
    qdrant.upsert("docs",pts)

    st.session_state["upload_key"]=str(uuid.uuid4())

# =========================
# SEARCH
# =========================

def keyword_score(q, text):
    return len(set(q.lower().split()) & set(text.lower().split()))

def search(q):
    try:
        vec = embed_query(q + " pojištění výluky podmínky")

        filt=None
        if selected_insurer!="— vyber —" and selected_vpp!="":
            filt=Filter(must=[
                FieldCondition(key="insurer",match=MatchValue(value=normalize(selected_insurer))),
                FieldCondition(key="vpp_name",match=MatchValue(value=normalize(selected_vpp)))
            ])

        res=qdrant.query_points("docs",query=vec,query_filter=filt,limit=25).points

        if not res:
            res=qdrant.query_points("docs",query=vec,limit=25).points

        if not res:
            return [],0

        enriched=[(r,keyword_score(q,r.payload["text"])) for r in res]
        pairs=[(q,r.payload["text"]) for r,_ in enriched]
        scores=reranker.predict(pairs)

        ranked=sorted([(r,s+(kw*0.3)) for (r,kw),s in zip(enriched,scores)],
                      key=lambda x:x[1],reverse=True)

        ctx=[{
            "text":r.payload["text"],
            "exact":extract_sentence(r.payload["text"],q),
            "page":r.payload["page"]
        } for r,_ in ranked[:5]]

        conf=int(min(max(ranked[0][1]*20,0),100))
        return ctx,conf

    except Exception as e:
        st.error(f"SEARCH ERROR: {e}")
        return [],0

# =========================
# UI
# =========================

st.title("🛡️ VPP Checker")

selected_insurer=st.sidebar.selectbox("Pojišťovna",["— vyber —"]+INSURERS)
selected_vpp=st.sidebar.text_input("VPP")

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
            combined = "\n\n".join([c["text"] for c in ctx])
            stream = generate_safe(combined, stream=True)

            if stream:
                for chunk in stream:
                    if hasattr(chunk,"text") and chunk.text:
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
