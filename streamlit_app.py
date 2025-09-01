from pathlib import Path
import traceback
import streamlit as st
import verbatim

DOCS_DIR = verbatim.DEFAULT_DOCS_DIR
DB_PATH  = verbatim.DEFAULT_DB_PATH

st.set_page_config(page_title="Verbatim RAG Q&A", page_icon="🔎", layout="centered")
st.title("🔎 Verbatim RAG")

DOCS_DIR.mkdir(parents=True, exist_ok=True)
st.caption(f"Index DB:  {'exists' if Path(DB_PATH).exists() else 'new will be created on build'}")

st.divider()

# ---------------- Upload & (re)build ----------------
st.subheader("Add PDFs and build the index")
uploaded = st.file_uploader("Drop PDF(s) here", type=["pdf"], accept_multiple_files=True)

col_a, col_b = st.columns(2)
with col_a:
    add_clicked = st.button("Save PDFs to folder", use_container_width=True)
with col_b:
    build_clicked = st.button("Build / Update index", type="primary", use_container_width=True)

if add_clicked:
    if not uploaded:
        st.warning("Please select at least one PDF.")
    else:
        saved = 0
        for uf in uploaded:
            out = DOCS_DIR / uf.name
            out.write_bytes(uf.read())
            saved += 1
        st.success(f"Saved {saved} PDF(s) to {DOCS_DIR}.")

if build_clicked:
    try:
        out = verbatim.build_index(DOCS_DIR, DB_PATH)
        st.success(f"Index built  →  {out}")
    except Exception as e:
        st.error("Index build failed.")
        st.code(traceback.format_exc(), language="bash")

st.divider()

# ---------------- Q&A ----------------
st.subheader("Ask a question")

if "history" not in st.session_state:
    st.session_state.history = [] 

q = st.text_input("Your question", placeholder="What is RBI?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and q.strip():
    st.session_state.history.append({"role": "user", "content": q})
    try:
        resp = verbatim.ask(q, DB_PATH, k=16)
        answer = verbatim.format_answer(resp)
        st.session_state.history.append({"role": "assistant", "content": answer})
    except Exception:
        st.session_state.history.append({
            "role": "assistant",
            "content": "Backend error:\n```\n" + traceback.format_exc() + "\n```",
        })

if st.session_state.history:
    st.divider()
    st.subheader("History")
    for msg in reversed(st.session_state.history):
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(msg["content"])
