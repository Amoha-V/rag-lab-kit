"""
rag — a teaching kit for Retrieval-Augmented Generation (RAG).

Participants never edit code. They install the package once and drive the
whole pipeline with a handful of high-level commands:

    rag setup     # one-time: prepare models
    rag ask "..." # ask a question (auto-builds the index if needed)
    rag chat      # interactive chat
    rag ui        # browser app

Everything below the surface — chunking, embeddings, the vector database,
retrieval, the LLM call — is abstracted away in rag.pipeline.
"""

__version__ = "1.0.0"
