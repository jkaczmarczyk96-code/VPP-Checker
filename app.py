# =========================
# NORMALIZACE
# =========================

def normalize_text(text):
    if not text:
        return ""
    return text.strip().upper()


# =========================
# SAFE SCROLL
# =========================

def safe_scroll():
    try:
        res = qdrant.scroll("docs", limit=1000, with_payload=True)
        return res[0] if res else []
    except:
        return []


# =========================
# GET INSURERS / VPPS
# =========================

def get_insurers():
    try:
        data = safe_scroll()
        return sorted(set(
            r.payload.get("insurer")
            for r in data
            if r.payload.get("insurer")
        ))
    except:
        return []


def get_vpps(insurer):
    try:
        data = safe_scroll()
        return sorted(set(
            r.payload.get("vpp_name")
            for r in data
            if r.payload.get("insurer") == insurer
            and r.payload.get("vpp_name")
        ))
    except:
        return []


# =========================
# INGEST (FIX DUPLICIT)
# =========================

def ingest_pdf(files, vpp_name, insurer):
    vpp_name = normalize_text(vpp_name)

    if not vpp_name:
        st.sidebar.error("Zadej název VPP")
        return

    # 🔥 kontrola duplicity
    existing_vpps = get_vpps(insurer)
    if vpp_name in existing_vpps:
        st.sidebar.warning("⚠️ Toto VPP už existuje")
        return

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
                    "page": i + 1,
                    "source": file.name,
                    "vpp_name": vpp_name,
                    "insurer": insurer
                })

    if not chunks:
        st.sidebar.error("❌ Nepodařilo se načíst text z PDF")
        return

    vectors = embed([c["text"] for c in chunks])

    points = [
        {"id": c["id"], "vector": v, "payload": c}
        for c, v in zip(chunks, vectors)
    ]

    try:
        qdrant.upsert("docs", points)
        st.sidebar.success("✅ VPP nahráno")
    except:
        st.sidebar.error("❌ Chyba při ukládání")


# =========================
# FILTR
# =========================

st.sidebar.markdown("## 📂 Filtr dokumentů")

selected_insurer = st.sidebar.selectbox(
    "Pojišťovna",
    ["— vyber —"] + INSURERS,
    key="filter_insurer"
)

selected_vpp = None

if selected_insurer != "— vyber —":
    vpps = get_vpps(selected_insurer)

    if not vpps:
        st.sidebar.info("Žádné VPP pro tuto pojišťovnu")

    selected_vpp = st.sidebar.selectbox(
        "VPP",
        ["— vyber —"] + vpps,
        key="filter_vpp"
    )


# =========================
# ADMIN
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

    insurer = st.sidebar.selectbox(
        "Pojišťovna",
        ["— vyber —"] + INSURERS,
        key="admin_insurer"
    )

    if st.sidebar.button("Nahrát"):
        if not files:
            st.sidebar.error("Nahraj PDF")
        elif not vpp_name:
            st.sidebar.error("Zadej název VPP")
        elif insurer == "— vyber —":
            st.sidebar.error("Vyber pojišťovnu")
        else:
            ingest_pdf(files, vpp_name, insurer)

    # ALERTY
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
# CHAT GUARD (VPP POVINNÉ)
# =========================

q = st.chat_input("Zeptej se...")

if q:
    if selected_insurer == "— vyber —" or selected_vpp == "— vyber —":
        st.warning("Vyber pojišťovnu a VPP")
    else:
        ctx, conf = search(q)
        ans = ask(q, ctx, conf)

        log_query(q, selected_insurer, selected_vpp, conf)

        st.session_state.history.insert(0, {"q": q, "a": ans})
