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
    "logged": False
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
    st.session_state.feedback.append({
        "question": q,
        "answer": a,
        "rating": r,
        "note": n
    })
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
    sheet = get_client().open("VPP_Feedback").worksheet("feedback")
    data = sheet.get_all_records()
    bad = [r for r in data[-20:] if r.get("rating") == "dislike"]

    if bad:
        st.sidebar.error(f"⚠️ {len(bad)} negativních feedbacků")
    else:
        st.sidebar.success("✅ Bez problémů")
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

def embed(x): return model.encode(x).tolist()

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
    return sorted(set(
        r.payload.get("vpp_name")
        for r in safe_scroll()
        if r.payload.get("insurer")==ins
    ))

# =========================
# SMART MEMORY
# =========================

def get_relevant_memory(q, top_k=2):
    if not st.session_state.history:
        return ""

    current_vec = embed([q])[0]
    scored = []

    for h in st.session_state.history[:10]:
        vec = embed([h["q"]])[0]
        score = sum(a*b for a,b in zip(current_vec, vec))
        scored.append((score, h))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]

    context = ""
    for _, h in scored:
        context += f"Uživatel: {h['q']}\nOdpověď: {h['a']}\n\n"

    return context.strip()

# =========================
# INGEST
# =========================

def ingest_pdf(files,vpp,ins):
    vpp=normalize(vpp)

    if not vpp:
        st.sidebar.error("Zadej VPP"); return

    if vpp in get_vpps(ins):
        st.sidebar.warning("VPP už existuje"); return

    progress=st.sidebar.progress(0)
    status=st.sidebar.empty()

    total=sum(len(PdfReader(f).pages) for f in files)
    done=0
    chunks=[]

    for f in files:
        reader=PdfReader(f)
        status.text(f"Zpracovávám {f.name}")

        for i,p in enumerate(reader.pages):
            try: t=p.extract_text()
            except: t=None

            if not t:
                try:
                    with pdfplumber.open(f) as pdf:
                        t=pdf.pages[i].extract_text()
                except: continue

            for c in smart_chunk(t):
                if len(c)<50: continue
                chunks.append({
                    "id":str(uuid.uuid4()),
                    "text":c,
                    "page":i+1,
                    "vpp_name":vpp,
                    "insurer":ins
                })

            done+=1
            progress.progress(min(done/total,1))

    if not chunks:
        status.text("❌ PDF chyba"); return

    status.text("🔄 Embedding...")
    vec=embed([c["text"] for c in chunks])

    pts=[{"id":c["id"],"vector":v,"payload":c} for c,v in zip(chunks,vec)]

    status.text("📦 Ukládám...")
    qdrant.upsert("docs",pts)
    status.text("✅ Hotovo")

# =========================
# SEARCH
# =========================

def search(q):
    try:
        vec=embed([q])[0]

        filt=Filter(must=[
            FieldCondition(key="insurer",match=MatchValue(value=selected_insurer)),
            FieldCondition(key="vpp_name",match=MatchValue(value=selected_vpp))
        ])

        res=qdrant.query_points("docs",query=vec,query_filter=filt,limit=10).points

        if not res: return [],0

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
# AI
# =========================

def ask(q,ctx,conf):
    if "zub" in q.lower():
        return "👉 Použij Zuby AI v Copilotu"

    if not ctx:
        return "❌ Nenalezeno → kontaktuj vedení směny"

    combined="\n\n".join([c["text"] for c in ctx])
    memory=get_relevant_memory(q)

    prompt=f"""
Shrň odpověď a uveď body.

ODPOVÍDEJ POUZE Z TEXTU.
Nevymýšlej.

KONTEXT:
{memory}

TEXT:
{combined}

OTÁZKA:
{q}
"""

    try:
        r=model_gemini.generate_content(prompt)
        answer=r.text.strip()
    except:
        answer=ctx[0]["exact"]

    cites="\n".join([f"📄 \"{c['exact']}\" (str. {c['page']})" for c in ctx])

    return f"{answer}\n\n{cites}\n\n📊 Confidence: {conf}%"

# =========================
# UI
# =========================

st.title("🛡️ VPP Checker")

st.sidebar.markdown("## 📂 Filtr dokumentů")

selected_insurer=st.sidebar.selectbox("Pojišťovna",["— vyber —"]+INSURERS)

selected_vpp="— vyber —"
if selected_insurer!="— vyber —":
    selected_vpp=st.sidebar.selectbox("VPP",["— vyber —"]+get_vpps(selected_insurer))

# ADMIN
if not st.session_state.logged:
    p=st.sidebar.text_input("Heslo",type="password")
    if st.sidebar.button("Přihlásit"):
        if p==st.secrets["ADMIN_PASSWORD"]:
            st.session_state.logged=True
            st.rerun()
else:
    f=st.sidebar.file_uploader("PDF",accept_multiple_files=True)
    v=st.sidebar.text_input("Název VPP")
    i=st.sidebar.selectbox("Pojišťovna",INSURERS)

    if st.sidebar.button("Nahrát"):
        if f and v:
            ingest_pdf(f,v,i)

# CHAT
q=st.chat_input("Zeptej se...")

if q:
    if selected_vpp=="— vyber —":
        st.warning("Vyber VPP")
    else:
        ctx,conf=search(q)
        ans=ask(q,ctx,conf)
        log_query(q,selected_insurer,selected_vpp,conf)
        st.session_state.history.insert(0,{"q":q,"a":ans})

# BUBBLES
for i,h in enumerate(st.session_state.history):
    st.markdown(f"<div style='text-align:right'><div style='background:#e3edff;padding:10px;border-radius:10px'>{h['q']}</div></div>",unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:left'><div style='background:white;padding:10px;border-radius:10px;border-left:4px solid #314397'>{h['a']}</div></div>",unsafe_allow_html=True)
