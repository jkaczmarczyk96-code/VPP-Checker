# =========================
# IMPORTS
# =========================
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import *
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
import pdfplumber
import uuid, re, time
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
defaults = {
    "messages": [],
    "feedback": [],
    "feedback_done": {},
    "feedback_open": {},
    "logged": False,
    "upload_key": str(uuid.uuid4())
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================
# GEMINI
# =========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
MODELS = ["gemini-1.5-flash-latest","gemini-1.5-pro-latest"]

def generate_safe(prompt, stream=True):
    for m in MODELS:
        try:
            model = genai.GenerativeModel(m)
            res = model.generate_content(prompt, stream=stream)
            if not stream:
                try:
                    return res.text
                except:
                    try:
                        return res.candidates[0].content.parts[0].text
                    except:
                        return None
            return res
        except:
            continue
    return None

# =========================
# GOOGLE SHEETS
# =========================
@st.cache_resource
def get_client():
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    return gspread.authorize(creds)

def save_feedback(q,a,r,n,ins,vpp):
    st.session_state.feedback.append({"q":q,"a":a,"r":r})
    try:
        get_client().open("VPP_Feedback").worksheet("feedback").append_row(
            [q,a,r,n,ins,vpp]
        )
    except:
        pass

def log_query(q,ins,vpp,conf):
    try:
        row = [str(datetime.now()),q,ins,vpp,conf]
        sheet = get_client().open("VPP_Feedback")
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
    bad = [r for r in data[-20:] if "dislike" in str(r.values()).lower()]
    st.sidebar.error(f"⚠️ {len(bad)} negativních") if bad else st.sidebar.success("OK")
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
        qdrant.get_collection("docs")
    except:
        qdrant.create_collection("docs",
            vectors_config=VectorParams(size=VECTOR_SIZE,distance=Distance.COSINE))
    for f in ["insurer","vpp_name"]:
        try:
            qdrant.create_payload_index("docs",f,PayloadSchemaType.KEYWORD)
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

@st.cache_resource
def embed_text(x):
    return model.encode([x])[0]

# 🔥 cache pro feedback embeddingy
@st.cache_data
def embed_feedback(text):
    return embed_text(text)

# =========================
# HELPERS
# =========================
def normalize(x): return x.strip().upper() if x else ""

def smart_chunk(t,s=800,o=150):
    out,i=[],0
    while i<len(t):
        out.append(t[i:i+s])
        i+=s-o
    return out

def extract_sentence(p,q):
    s=re.split(r'(?<=[.!?]) +',p)
    qw=set(q.lower().split())
    ranked=sorted(s,key=lambda x:len(qw & set(x.lower().split())),reverse=True)
    return ranked[0] if ranked else p

def keyword_score(q,t):
    return len(set(q.lower().split()) & set(t.lower().split()))

def safe_scroll():
    try:
        return qdrant.scroll("docs",limit=1000,with_payload=True)[0]
    except:
        return []

def get_vpps(ins):
    ins=normalize(ins)
    try:
        res=safe_scroll()
        return sorted(set(r.payload.get("vpp_name") for r in res if r.payload.get("insurer")==ins))
    except:
        return []

def get_memory(q):
    msgs=[m["content"] for m in st.session_state.messages if m["role"]=="user"]
    scored=sorted(msgs,key=lambda x:len(set(q.lower().split()) & set(x.lower().split())),reverse=True)
    return "\n".join(scored[:2])

def rewrite_query(q):
    res = generate_safe(f"Rozšiř dotaz pro vyhledávání:\n{q}", False)
    return res if isinstance(res,str) and len(res)>5 else q

# =========================
# INGEST
# =========================
def ingest_pdf(files,vpp,ins):
    vpp=normalize(vpp)

    progress=st.sidebar.progress(0)
    status=st.sidebar.empty()
    stats=st.sidebar.empty()

    total_pages=sum(len(PdfReader(f).pages) for f in files)
    done,total_chunks=0,0
    start=time.time()

    chunks=[]

    for f in files:
        reader=PdfReader(f)
        status.text(f"📄 {f.name}")

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
                    "insurer":normalize(ins)
                })
                total_chunks+=1

            done+=1
            progress.progress(done/total_pages)

            elapsed=time.time()-start
            eta=(elapsed/done)*(total_pages-done) if done else 0

            stats.markdown(f"""
**📊 Stats**
- Pages: {done}/{total_pages}
- Chunks: {total_chunks}
- ETA: {round(eta,1)}s
""")

    vec=[embed_doc(c["text"]) for c in chunks]
    pts=[{"id":c["id"],"vector":v,"payload":c} for c,v in zip(chunks,vec)]
    qdrant.upsert("docs",pts)

    status.text("✅ Hotovo")
    st.session_state.upload_key=str(uuid.uuid4())

# =========================
# AI HELPERS
# =========================
def feedback_boost(text):
    try:
        text_vec = embed_text(text)
        boost = 0
        for fb in st.session_state.feedback:
            if not fb.get("a"): continue
            fb_vec = embed_feedback(fb["a"][:300])
            sim = sum(a*b for a,b in zip(text_vec, fb_vec))
            if sim > 0.6:
                boost += 2 if fb["r"]=="like" else -2
        return boost
    except:
        return 0

def compute_confidence(ranked):
    try:
        top = ranked[0][1]
        second = ranked[1][1] if len(ranked)>1 else 0
        return int(max(min((top*10)+(top-second)*20,100),0))
    except:
        return 0

# =========================
# SEARCH
# =========================
def search(q,insurer,vpp):
    debug=[]
    try:
        q=rewrite_query(q)
        vec=embed_query(q+" výluky pojištění")

        filt=None
        if insurer!="— vyber —" and vpp!="— vyber —":
            filt=Filter(must=[
                FieldCondition(key="insurer",match=MatchValue(value=normalize(insurer))),
                FieldCondition(key="vpp_name",match=MatchValue(value=normalize(vpp)))
            ])

        res=qdrant.query_points("docs",query=vec,query_filter=filt,limit=25).points
        if not res:
            res=qdrant.query_points("docs",query=vec,limit=25).points
        if not res:
            return [],0,[]

        scored=[]
        for r in res:
            text=r.payload["text"]
            score=keyword_score(q,text)+feedback_boost(text)
            scored.append((r,score))

        pairs=[(q,r.payload["text"]) for r,_ in scored]
        rs=reranker.predict(pairs)

        ranked=sorted([(r,s+rscore) for (r,s),rscore in zip(scored,rs)],key=lambda x:x[1],reverse=True)

        for r,s in ranked[:5]:
            debug.append({
                "score":round(s,2),
                "page":r.payload["page"],
                "text":r.payload["text"][:120]
            })

        ctx=[{
            "text":r.payload["text"],
            "exact":extract_sentence(r.payload["text"],q),
            "page":r.payload["page"]
        } for r,_ in ranked[:5]]

        return ctx,compute_confidence(ranked),debug
    except:
        return [],0,[]

# =========================
# UI
# =========================
st.title("🛡️ VPP Checker")

selected_insurer=st.sidebar.selectbox("Pojišťovna",["— vyber —"]+INSURERS)

selected_vpp="— vyber —"
if selected_insurer!="— vyber —":
    vpps=get_vpps(selected_insurer)
    if vpps:
        selected_vpp=st.sidebar.selectbox("VPP",["— vyber —"]+vpps)
    else:
        st.sidebar.warning("Žádná VPP")

# ADMIN
if not st.session_state.logged:
    if st.sidebar.text_input("Heslo",type="password")==st.secrets["ADMIN_PASSWORD"]:
        st.session_state.logged=True
        st.rerun()
else:
    f=st.sidebar.file_uploader("PDF",accept_multiple_files=True,key=st.session_state.upload_key)
    v=st.sidebar.text_input("Název VPP")
    i=st.sidebar.selectbox("Pojišťovna",INSURERS)

    if st.sidebar.button("Nahrát"):
        ingest_pdf(f,v,i)

# CHAT
ctx,debug=[],[]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt:=st.chat_input("Zeptej se..."):
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("assistant"):
        ph=st.empty()
        full=""

        ctx,conf,debug=search(prompt,selected_insurer,selected_vpp)

        if not ctx:
            full="❌ Nenalezeno → kontaktuj vedení směny"
            ph.markdown(full)
        else:
            memory=get_memory(prompt)
            combined="\n\n".join([c["text"] for c in ctx])

            prompt_ai=f"""
Jsi specialista na pojistné podmínky.

PRAVIDLA:
- Pouze TEXT
- Bez domýšlení
- Pokud nevíš: "Nelze dohledat v dokumentu"
- Stručně

KONTEXT:
{memory}

TEXT:
{combined}

OTÁZKA:
{prompt}
"""

            stream=generate_safe(prompt_ai,True)
            if stream:
                for chunk in stream:
                    if hasattr(chunk,"text") and chunk.text:
                        full+=chunk.text
                        ph.markdown(full+"▌")

            cites="\n".join([f"📄 \"{c['exact']}\" (str. {c['page']})" for c in ctx])
            full=f"{full}\n\n{cites}"
            ph.markdown(full)

        st.caption(f"Confidence: {conf}%")
        st.session_state.messages.append({"role":"assistant","content":full})

        log_query(prompt,selected_insurer,selected_vpp,conf)

        # FEEDBACK
        msg_id=len(st.session_state.messages)

        if msg_id not in st.session_state.feedback_done:
            col1,col2=st.columns(2)

            if col1.button("👍",key=f"l{msg_id}"):
                save_feedback(prompt,full,"like","",selected_insurer,selected_vpp)
                st.session_state.feedback_done[msg_id]=True

            if col2.button("👎",key=f"d{msg_id}"):
                st.session_state.feedback_open[msg_id]=True

            if st.session_state.feedback_open.get(msg_id,False):
                note=st.text_area("Co bylo špatně?",key=f"note_{msg_id}")
                if st.button("Odeslat",key=f"s{msg_id}"):
                    save_feedback(prompt,full,"dislike",note,selected_insurer,selected_vpp)
                    st.session_state.feedback_done[msg_id]=True
                    st.session_state.feedback_open[msg_id]=False

# DEBUG
with st.expander("🔍 Debug"):
    st.write(debug)

