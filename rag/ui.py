"""
The browser app, launched by `rag ui`.

Participants don't run this file directly — the CLI starts it for them. It
shows the answer AND the evidence chunks so you can *see* the grounding.
"""

import streamlit as st

from rag import config
from rag import pipeline

st.set_page_config(page_title="College Document AI Assistant", page_icon="🎓")


@st.cache_resource
def _warm():
    """Make sure the index exists and the embedder is loaded, once."""
    pipeline.ensure_index()
    pipeline.get_embedder()
    return True


st.title("🎓 College Document AI Assistant")
st.caption(
    "RAG = find the right paragraph, then ask the LLM to answer using only "
    "that paragraph."
)

# ── Sidebar: bring in your OWN document (the "wow" finale) ───────────────
with st.sidebar:
    st.header("📄 Use your own document")
    st.caption(
        "Upload a PDF or text file and watch the assistant's knowledge change "
        "— with no retraining."
    )

    current = sorted(
        p.name
        for p in config.DOCUMENTS_DIR.glob("*")
        if p.suffix.lower() in {".pdf", ".txt"}
    ) if config.DOCUMENTS_DIR.exists() else []
    st.write("**Currently loaded:**")
    st.write("\n".join(f"- {n}" for n in current) or "_(none)_")

    uploads = st.file_uploader(
        "Add a file", type=["pdf", "txt"], accept_multiple_files=True
    )
    replace = st.checkbox(
        "Replace the current documents", value=True,
        help="On: full knowledge swap. Off: add alongside the existing docs.",
    )

    if st.button("Add & rebuild", type="primary", disabled=not uploads):
        with st.spinner("Embedding your document and rebuilding the index..."):
            if replace:
                pipeline.clear_documents()
            for up in uploads:
                pipeline.save_uploaded_file(up.name, up.getvalue())
            pipeline.build_index()
        st.success("Done! The assistant now answers from your document.")
        st.rerun()

try:
    _warm()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

count = pipeline._get_collection().count()
st.info(f"Knowledge base: **{count} chunks** loaded from {config.DOCUMENTS_DIR}/")

question = st.text_input(
    "Ask a question about the college:",
    placeholder="How much attendance is required?",
)

if question:
    with st.spinner("Searching documents and asking the LLM..."):
        result = pipeline.ask(question)

    st.subheader("Answer")
    if result["llm_ok"]:
        st.write(result["answer"])
    else:
        st.warning(result["note"])

    st.subheader("Retrieved chunks (the evidence)")
    for i, c in enumerate(result["chunks"], 1):
        with st.expander(
            f"Chunk {i} · source: {c['source']} · distance: {c['distance']:.3f}"
        ):
            st.write(c["text"])

# ── Browse the whole vector database ─────────────────────────────────────
st.divider()
with st.expander("🗂️  Browse the vector database (every stored chunk)"):
    st.caption(
        "This is the full ChromaDB collection — every chunk the assistant can "
        "retrieve from. Each row is one chunk: its text and (optionally) its "
        "embedding vector."
    )
    show_vec = st.checkbox("Show embedding vectors", value=False)
    rows = pipeline.list_all_chunks(include_vectors=show_vec)

    search = st.text_input("Filter rows (type a word):", placeholder="e.g. library")
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r["text"].lower() or s in r["source"].lower()]

    st.write(f"**{len(rows)} chunk(s)**")
    st.dataframe(rows, use_container_width=True, hide_index=True)
