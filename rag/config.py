"""

Paths are resolved relative to the folder you run `rag` from, so you can
keep your documents and database wherever you launch the workshop.
"""

import os
from pathlib import Path

# Silence a harmless HuggingFace tokenizers warning about forking.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load_dotenv() -> None:
    """Tiny .env loader (no extra dependency).

    Reads KEY=VALUE lines from a `.env` file next to where you run `rag`, so an
    instructor can ship one file with the API key and participants need zero
    setup. Real environment variables always win over the file.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()

# ── Where your files live (relative to your current folder) ─────────────
DOCUMENTS_DIR = Path(os.environ.get("RAG_DOCS", "documents"))
CHROMA_DIR = Path(os.environ.get("RAG_DB", "chroma_db"))

# ── Embedding model: text -> 384 numbers. ──────────────────────────────
# Provided by ChromaDB's built-in ONNX runtime (no PyTorch). ~80 MB, once.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2 (ONNX)"

# ── Chunking: split documents into ~CHUNK_SIZE-char pieces for precise
#    retrieval; keep CHUNK_OVERLAP chars across boundaries for context. ──
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# ── Retrieval: how many chunks to pull back per question. ───────────────
TOP_K = 3

# ── Vector DB collection name. ──────────────────────────────────────────
COLLECTION_NAME = "college_docs"

# ── LLM backend (cloud — nothing to download, no GPU) ───────────────────
#    Choose the provider with RAG_LLM:
#       RAG_LLM=groq   (default)  → needs GROQ_API_KEY  (https://console.groq.com)
#       RAG_LLM=hf                → needs HF_TOKEN       (https://huggingface.co/settings/tokens)
#    Use this if one provider is blocked/unreachable on your network.
LLM_BACKEND = os.environ.get("RAG_LLM", "groq").strip().lower()


def _clean(value, placeholders):
    value = value.strip()
    return "" if value in placeholders else value


# Groq settings
GROQ_API_KEY = _clean(
    os.environ.get("GROQ_API_KEY", ""),
    {"gsk_your_key_here", "gsk_paste_your_key_here", "gsk_..."},
)
GROQ_MODEL = "llama-3.1-8b-instant"

# Hugging Face settings
HF_TOKEN = _clean(
    os.environ.get("HF_TOKEN", ""),
    {"hf_your_token_here", "hf_..."},
)
HF_MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Resolve the active backend into one set of values the rest of the code uses.
if LLM_BACKEND == "hf":
    LLM_MODEL = HF_MODEL
    LLM_KEY = HF_TOKEN
    KEY_ENV_NAME = "HF_TOKEN"
    KEY_URL = "https://huggingface.co/settings/tokens"
else:
    LLM_BACKEND = "groq"
    LLM_MODEL = GROQ_MODEL
    LLM_KEY = GROQ_API_KEY
    KEY_ENV_NAME = "GROQ_API_KEY"
    KEY_URL = "https://console.groq.com"

# RAG_MODEL overrides the model for whichever backend is active.
_override = os.environ.get("RAG_MODEL")
if _override:
    LLM_MODEL = _override
