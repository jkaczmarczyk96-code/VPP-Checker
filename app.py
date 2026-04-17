# FINAL VERIFIED VERSION (INTEGRITY + OPTIMIZATION CHECKED)

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import pdfplumber
import uuid
import re
import hashlib
import google.generativeai as genai
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from functools import lru_cache
import numpy as np

# =========================
# CONFIG
# =========================

st.set_page_config(page_title="VPP Checker", layout="wide")

VECTOR_SIZE = 768
TOP_K = 20
FINAL_K = 5

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
# ALERTS
# =========================

st.sidebar.markdown("## 🚨 Alerty")
try:
    sheet = get_client().open("VPP_Feedback").worksheet("feedback")
    data = sheet.get_all_records()
    recent_bad = [r for r in data[-20:] if r.get("rating") == "dislike"]

    if recent_bad:
        st.sidebar.error(f"⚠️ {len(recent_bad)} negativních feedbacků")
    else:
        st.sidebar.success("✅ Bez problémů")
except:
    st.sidebar.info("Alerty nedostupné")

# =========================
# QDRANT
# =========================

qdrant = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"],
    timeout=3.0
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

@lru_cache(maxsize=10000)
def embed_cached(text):
    return model.encode(text).tolist()

# =========================
# HELPERS
# =========================

def normalize(x): return x.strip().upper() if x else ""


def smart_chunk(t, s=700, o=120):
    out, i = [], 0
    while i < len(t):
        out.append(t[i:i+s])
        i += s-o
    return out


def extract_sentence(p,q):
    s = re.split(r'(?<=[.!?]) +', p)
    qw = set(q.lower().split())
    return max(s, key=lambda x: len(qw & set(x.lower().split())), default=p)


def cosine_sim(a,b):
    a,b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9))


def safe_scroll():
    try:
        return qdrant.scroll("docs",limit=1000,with_payload=True)[0]
    except:
        return []


def get_vpps(ins):
    return sorted(set(r.payload.get("vpp_name") for r in safe_scroll() if r.payload.get("insurer")==ins))


def chunk_hash(text, vpp):
    return hashlib.md5((text+vpp).encode()).hexdigest()

# =========================
# INGEST
# =========================

def ingest_pdf(files,vpp,ins):
    vpp=normalize(vpp)

    if not vpp:
        st.sidebar.error("Zadej VPP")
        return

    if vpp in get_vpps(ins):
        st.sidebar.warning("VPP už existuje")
        return

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
                    "id":chunk_hash(c,vpp),
                    "text":c,
                    "page":i+1,
                    "vpp_name":vpp,
                    "insurer":ins
                })

            done+=1
            progress.progress(min(done/total,1))

    if not chunks:
        status.text("❌ PDF chyba")
        return

    status.text("🔄 Embedding...")
    vecs=model.encode([c["text"] for c in chunks]).tolist()

    pts=[{"id":c["id"],"vector":v,"payload":c} for c,v in zip(chunks,vecs)]

    status.text("📦 Ukládám...")
    qdrant.upsert("docs",pts)
    status.text("✅ Hotovo")

# =========================
# QUERY REWRITE
# =========================

def rewrite_query(q):
    try:
        r=model_gemini.generate_content(f"Upřesni dotaz stručně: {q}")
        rq=r.text.strip()
        if len(rq.split())<2:
            return q
        return rq
    except:
        return q

# =========================
# SEARCH
# =========================

def search(q):
    if selected_vpp=="— vyber —": return [],0

    try:
        q2=rewrite_query(q)

        vec1=embed_cached(q)
        vec2=embed_cached(q2)

        filt=Filter(must=[
            FieldCondition(key="insurer",match=MatchValue(value=selected_insurer)),
            FieldCondition(key="vpp_name",match=MatchValue(value=selected_vpp))
        ])

        res1=qdrant.query_points("docs",query=vec1,query_filter=filt,limit=TOP_K).points
        res2=qdrant.query_points("docs",query=vec2,query_filter=filt,limit=TOP_K).points

        res=res1+res2
        if not res: return [],0

        uniq={}
        for r in res:
            uniq[r.payload["text"][:100]]=r
        res=list(uniq.values())

        pairs=[(q,r.payload["text"]) for r in res]
        scores=reranker.predict(pairs)

        ranked=list(zip(res,scores))

        for i,(r,s) in enumerate(ranked):
            for fb in st.session_state.feedback:
                sim=cosine_sim(embed_cached(q),embed_cached(fb["question"]))
                if sim>0.7:
                    if fb["rating"]=="like": s+=0.3*sim
                    elif fb["rating"]=="dislike": s-=0.5*sim
            ranked[i]=(r,s)

        ranked=sorted(ranked,key=lambda x:x[1],reverse=True)

        ctx=[{
            "text":r.payload["text"],
            "exact":extract_sentence(r.payload["text"],q2),
            "page":r.payload["page"]
        } for r,_ in ranked[:FINAL_K]]

        conf=int(min(max(ranked[0][1]*20,0),100))

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

    prompt=f"""
Jsi asistent pro práci s pojistnými podmínkami (VPP).

PRAVIDLA:
- Odpovídej POUZE z textu
- NIC si nevymýšlej
- Pokud odpověď není → napiš: "Nelze najít v dokumentu"

STRUKTURA:
1. Krátké shrnutí
2. Body

TEXT:
{combined}

OTÁZKA:
{q}
"""

    try:
        r=model_gemini.generate_content(prompt)
        answer=r.text.strip()
    except:
        answer=ctx[0]["exact"] if ctx else ""

    cites="\n".join([f"📄 \"{c['exact']}\" (strana {c['page']})" for c in ctx]) if ctx else ""

    return f"{answer}\n\n---\n{cites}\n\n📊 Confidence: {conf}%"

# =========================
# UI
# =========================

st.title("🛡️ VPP Checker")

st.sidebar.markdown("## 📂 Filtr dokumentů")

selected_insurer=st.sidebar.selectbox("Pojišťovna",["— vyber —"]+INSURERS)

selected_vpp="— vyber —"
if selected_insurer!="— vyber —":
    vpps=get_vpps(selected_insurer)
    selected_vpp=st.sidebar.selectbox("VPP",["— vyber —"]+vpps)

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

for i,h in enumerate(st.session_state.history):
    st.markdown(f"**Ty:** {h['q']}")
    st.markdown(f"**AI:** {h['a']}")

    if st.session_state.feedback_done.get(i):
        st.success("Díky!")
        continue

    c1,c2=st.columns(2)

    if c1.button("👍",key=f"l{i}"):
        save_feedback(h["q"],h["a"],"like","")
        st.session_state.feedback_done[i]=True
        st.rerun()

    if c2.button("👎",key=f"d{i}"):
        st.session_state.feedback_open[i]=True

    if st.session_state.feedback_open.get(i):
        note=st.text_input("Co bylo špatně?",key=f"n{i}")
        if st.button("Odeslat",key=f"s{i}"):
            save_feedback(h["q"],h["a"],"dislike",note)
            st.session_state.feedback_done[i]=True
            st.session_state.feedback_open[i]=False
            st.rerun()
