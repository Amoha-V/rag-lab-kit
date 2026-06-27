"""
The RAG pipeline — fully abstracted.


Pipeline:  documents -> chunks -> embeddings -> vector DB
           question  -> embedding -> retrieve chunks -> LLM -> answer
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from . import config

# ChromaDB 0.5.x prints a harmless but confusing telemetry error
# ("capture() takes 1 positional argument..."). Silence that logger so the
# workshop output stays clean. (We also disable telemetry in _client().)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)



# ──────────────────────────────────────────────────────────────────────────
# Reading & chunking documents
# ──────────────────────────────────────────────────────────────────────────
def read_documents(documents_dir: Path) -> list[dict]:
    """Read every .pdf and .txt file in a folder -> [{text, source}]."""
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        return []

    docs: list[dict] = []
    for path in sorted(documents_dir.iterdir()):
        if path.suffix.lower() == ".pdf":
            from pypdf import PdfReader  # imported only when a PDF is present

            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8")
        else:
            continue
        text = text.strip()
        if text:
            docs.append({"text": text, "source": path.name})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into overlapping chunks -> [{text, source, chunk_index}]."""
    size, overlap = config.CHUNK_SIZE, config.CHUNK_OVERLAP
    chunks: list[dict] = []
    for doc in docs:
        words = doc["text"].split()
        current: list[str] = []
        current_len = 0
        index = 0
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= size:
                text = " ".join(current)
                chunks.append({"text": text, "source": doc["source"], "chunk_index": index})
                index += 1
                tail = text[-overlap:]
                current = tail.split()
                current_len = len(tail)
        if current:
            chunks.append({"text": " ".join(current), "source": doc["source"], "chunk_index": index})
    return chunks


# ──────────────────────────────────────────────────────────────────────────
# Embeddings & vector database
# ──────────────────────────────────────────────────────────────────────────
_embedder = None


def get_embedder():
    """Load the embedding model once, then reuse it.

    Uses ChromaDB's built-in ONNX MiniLM (same all-MiniLM-L6-v2, 384 dims) which
    runs on onnxruntime — so the kit needs NO PyTorch / sentence-transformers.
    """
    global _embedder
    if _embedder is None:
        from chromadb.utils import embedding_functions

        _embedder = embedding_functions.DefaultEmbeddingFunction()
    return _embedder


def _embed(texts) -> list[list[float]]:
    """Embed a list of strings → list of 384-float vectors (plain Python)."""
    vectors = get_embedder()(list(texts))
    return [[float(x) for x in v] for v in vectors]


def _client():
    import chromadb
    from chromadb.config import Settings

    # Turn off ChromaDB's anonymous telemetry — it's noisy and prints harmless
    # but confusing errors in a workshop.
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def build_index(documents_dir: Path | None = None) -> dict:
    """Read -> chunk -> embed -> store. Returns a small summary dict."""
    documents_dir = Path(documents_dir) if documents_dir else config.DOCUMENTS_DIR

    docs = read_documents(documents_dir)
    if not docs:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in '{documents_dir}'. "
            "Add a document and try again."
        )

    chunks = chunk_documents(docs)
    texts = [c["text"] for c in chunks]
    embeddings = _embed(texts)

    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION_NAME)
    collection.add(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": c["source"], "chunk": c["chunk_index"]} for c in chunks],
    )

    return {
        "documents": [d["source"] for d in docs],
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "db_path": str(config.CHROMA_DIR),
    }


def clear_documents() -> list[str]:
    """Remove every .pdf/.txt from the documents folder. Returns names removed."""
    removed = []
    if config.DOCUMENTS_DIR.exists():
        for path in config.DOCUMENTS_DIR.iterdir():
            if path.suffix.lower() in {".pdf", ".txt"}:
                removed.append(path.name)
                path.unlink()
    return removed


def add_documents(paths, replace: bool = False) -> dict:
    """Copy one or more .pdf/.txt files into documents/ and rebuild the index.

    This powers the "wow" finale: swap in your own document and watch the
    assistant's knowledge change — with no retraining.

    paths   : list of file paths (or a single path)
    replace : if True, delete the existing documents first (full swap)
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p).expanduser() for p in paths]

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        if p.suffix.lower() not in {".pdf", ".txt"}:
            raise ValueError(f"Only .pdf or .txt files are supported: {p.name}")

    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    removed = clear_documents() if replace else []

    added = []
    for p in paths:
        dest = config.DOCUMENTS_DIR / p.name
        shutil.copyfile(p, dest)
        added.append(p.name)

    summary = build_index()
    summary["added"] = added
    summary["removed"] = removed
    return summary


def save_uploaded_file(name: str, data: bytes) -> Path:
    """Save raw uploaded bytes into documents/ (used by the browser app)."""
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.DOCUMENTS_DIR / Path(name).name
    dest.write_bytes(data)
    return dest


def _get_collection():
    return _client().get_collection(config.COLLECTION_NAME)


def index_exists() -> bool:
    try:
        return _get_collection().count() > 0
    except Exception:
        return False


def ensure_index() -> None:
    """Build the index automatically if it isn't there yet.

    This is the key to abstraction: a participant can run `rag ask`
    first and the whole ingest pipeline just happens behind the scenes.
    """
    if not index_exists():
        build_index()


def project_2d() -> list[dict]:
    """Project every stored 384-dim vector down to 2D so it can be plotted.

    PCA via NumPy SVD (no scikit-learn). Returns one point per chunk with x, y,
    its source, chunk index, and a short text label — the data the `rag map`
    command turns into a scatter plot of the vector space.
    """
    import numpy as np

    collection = _get_collection()
    data = collection.get(include=["embeddings", "metadatas", "documents"])
    vectors = np.array(data["embeddings"], dtype=float)
    if len(vectors) < 2:
        raise ValueError("Need at least 2 chunks to draw a map. Run `rag ingest`.")

    # PCA = center the data, then project onto the top-2 right singular vectors.
    centered = vectors - vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T

    points = []
    for i in range(len(vectors)):
        words = data["documents"][i].split()[:3]
        points.append(
            {
                "x": float(coords[i][0]),
                "y": float(coords[i][1]),
                "source": data["metadatas"][i]["source"],
                "chunk": data["metadatas"][i]["chunk"],
                "label": " ".join(words),
            }
        )
    return points


def list_all_chunks(include_vectors: bool = False) -> list[dict]:
    """Return EVERY stored chunk as a row — powers the browse table in the app."""
    collection = _get_collection()
    include = ["documents", "metadatas"]
    if include_vectors:
        include.append("embeddings")
    data = collection.get(include=include)

    rows = []
    for i in range(len(data["ids"])):
        row = {
            "id": data["ids"][i],
            "source": data["metadatas"][i]["source"],
            "chunk": data["metadatas"][i]["chunk"],
            "text": data["documents"][i],
        }
        if include_vectors:
            vec = data["embeddings"][i]
            row["vector (first 8 of 384)"] = ", ".join(
                f"{float(x):.3f}" for x in vec[:8]
            ) + " …"
        rows.append(row)

    rows.sort(key=lambda r: (r["source"], r["chunk"]))
    return rows


def inspect_db(limit: int = 5, show_vectors: bool = False) -> dict:
    """Peek inside the vector database so participants can SEE what's stored.

    Returns counts, the per-document chunk breakdown, and a few sample chunks
    (optionally with a preview of the actual embedding vector).
    """
    collection = _get_collection()
    total = collection.count()

    # Per-source breakdown (the DB is tiny, so fetching all metadata is fine).
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    per_source: dict[str, int] = {}
    for m in all_meta:
        per_source[m["source"]] = per_source.get(m["source"], 0) + 1

    include = ["documents", "metadatas"]
    if show_vectors:
        include.append("embeddings")
    sample = collection.get(limit=limit, include=include)

    rows = []
    for i in range(len(sample["ids"])):
        row = {
            "id": sample["ids"][i],
            "source": sample["metadatas"][i]["source"],
            "chunk": sample["metadatas"][i]["chunk"],
            "text": sample["documents"][i],
        }
        if show_vectors:
            vec = sample["embeddings"][i]
            row["vector_length"] = len(vec)
            row["vector_preview"] = [round(float(x), 4) for x in vec[:8]]
        rows.append(row)

    return {
        "collection": config.COLLECTION_NAME,
        "db_path": str(config.CHROMA_DIR),
        "total_chunks": total,
        "per_source": per_source,
        "samples": rows,
    }


# ──────────────────────────────────────────────────────────────────────────
# Retrieval & generation
# ──────────────────────────────────────────────────────────────────────────
def retrieve(
    question: str,
    k: int | None = None,
    source: str | None = None,
    contains: str | None = None,
) -> list[dict]:
    """Find the chunks most similar to the question.

    Demonstrates ChromaDB's three query types together:
      • similarity  — always (query_embeddings + n_results)
      • metadata    — source=... → where={"source": ...}
      • keyword     — contains=... → where_document={"$contains": ...}
    """
    k = k or config.TOP_K
    q_emb = _embed([question])[0]

    query_kwargs = {"query_embeddings": [q_emb], "n_results": k}
    if source:
        query_kwargs["where"] = {"source": source}
    if contains:
        query_kwargs["where_document"] = {"$contains": contains}

    results = _get_collection().query(**query_kwargs)
    out = []
    for text, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        out.append({"text": text, "source": meta["source"], "distance": dist})
    return out


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(c["text"] for c in chunks)
    return (
        "You are a helpful college assistant. Answer the question using ONLY "
        "the context below. If the answer is not in the context, say you don't "
        "know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def _chat(messages) -> str:
    """Send chat messages to the active LLM backend (Groq or Hugging Face)."""
    if not config.LLM_KEY:
        raise RuntimeError("no-api-key")

    if config.LLM_BACKEND == "hf":
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=config.LLM_KEY)
        resp = client.chat_completion(
            messages=messages, model=config.LLM_MODEL,
            max_tokens=512, temperature=0.2,
        )
        return resp.choices[0].message.content.strip()

    # default: Groq
    from groq import Groq

    client = Groq(api_key=config.LLM_KEY)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL, messages=messages, temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def generate(question: str, chunks: list[dict]) -> str:
    """Ask the cloud LLM to write an answer grounded in the chunks."""
    return _chat([{"role": "user", "content": _build_prompt(question, chunks)}])


def ask(question: str, k: int | None = None) -> dict:
    """The full pipeline for one question.

    Auto-builds the index if needed, retrieves, then generates. If the LLM
    (Groq) is unavailable, still returns the retrieved chunks with a note —
    because retrieval already works without an LLM.
    """
    ensure_index()
    chunks = retrieve(question, k=k)
    try:
        answer = generate(question, chunks)
        llm_ok = True
        note = None
    except Exception as exc:
        answer = None
        llm_ok = False
        if str(exc) == "no-api-key":
            note = (
                f"No {config.KEY_ENV_NAME} set, so the LLM step is skipped — but "
                "notice retrieval already works without it. Get a free key at "
                f"{config.KEY_URL} and add it to the .env file."
            )
        else:
            note = (
                f"Could not reach the {config.LLM_BACKEND.upper()} API. Showing "
                "retrieved chunks instead (retrieval works without the LLM). "
                f"Check your key / internet. ({exc})"
            )
    return {"answer": answer, "chunks": chunks, "llm_ok": llm_ok, "note": note}


def ask_llm_only(question: str) -> dict:
    """Ask the LLM DIRECTLY — no retrieval, no documents, no RAG.

    This is the 'before' picture: the model answers from its training alone, so
    for questions about your specific institution it makes things up or says it
    doesn't know. Contrast with ask() to see what retrieval adds.
    """
    if not config.LLM_KEY:
        return {
            "answer": None,
            "note": (
                f"No {config.KEY_ENV_NAME} set. Add a free key from {config.KEY_URL} "
                "to the .env file to run this."
            ),
        }
    try:
        # the raw question only — no retrieval, no context
        answer = _chat([{"role": "user", "content": question}])
        return {"answer": answer, "note": None}
    except Exception as exc:
        return {"answer": None, "note": f"Could not reach the {config.LLM_BACKEND.upper()} API. ({exc})"}


# ──────────────────────────────────────────────────────────────────────────
# Teaching demos (replace the old notebooks, but as commands)
# ──────────────────────────────────────────────────────────────────────────
def demo_embeddings(base: str, candidates: list[str]) -> dict:
    """Show that similar sentences have similar vectors."""
    import numpy as np

    vecs = _embed([base] + list(candidates))
    base_vec = np.array(vecs[0])

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored = [
        {"text": t, "similarity": cosine(base_vec, vecs[i + 1])}
        for i, t in enumerate(candidates)
    ]
    return {
        "base": base,
        "vector_length": len(base_vec),
        "vector_preview": [round(float(x), 4) for x in base_vec[:8]],
        "comparisons": sorted(scored, key=lambda s: -s["similarity"]),
    }
