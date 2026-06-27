"""
The `rag` command-line interface.

This is the ONLY thing participants interact with. Each command runs a whole
stage of the pipeline for them — nothing to open, nothing to edit.

    rag setup            one-time: download the models
    rag info             show settings + whether the DB is built
    rag ingest           (re)build the vector DB from documents/
    rag ask "question"   ask one question (auto-builds DB if needed)
    rag chat             interactive chat
    rag ui               launch the browser app
    rag demo embeddings  show that similar text -> similar vectors
    rag demo search      retrieval-only demo (no LLM)
"""

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__, config


# ── pretty printing helpers ──────────────────────────────────────────────
def _hr():
    print("─" * 64)


def _print_chunks(chunks):
    print("\nRetrieved chunks (the evidence):")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] source={c['source']}  distance={c['distance']:.3f}")
        snippet = c["text"].strip().replace("\n", " ")
        print(f"      {snippet[:160]}{'…' if len(snippet) > 160 else ''}")


# ── commands ─────────────────────────────────────────────────────────────
_DEFAULT_ENV = """\
# ────────────────────────────────────────────────────────────────────
#  RAG Lab — your settings.  (rag setup also fills the key in for you.)
# ────────────────────────────────────────────────────────────────────
#  Default LLM backend is Groq. Get a FREE key: https://console.groq.com
GROQ_API_KEY=gsk_your_key_here

#  --- To use Hugging Face instead of Groq (e.g. if Groq is blocked) ---
#  Uncomment the next line and add a free token from
#  https://huggingface.co/settings/tokens
#  RAG_LLM=hf
#  HF_TOKEN=hf_your_token_here
#  HF_MODEL=meta-llama/Llama-3.1-8B-Instruct

#  Optional: RAG_MODEL=<a model name for the active backend>
"""


def _ensure_env_file():
    """Create an editable .env if one doesn't exist yet. Returns (path, created)."""
    env = Path(".env")
    if env.exists():
        return env, False
    example = Path(".env.example")
    env.write_text(example.read_text() if example.exists() else _DEFAULT_ENV)
    return env, True


def _set_env_key(env_path, key, value):
    """Write key=value into the .env file, replacing any existing line."""
    lines = env_path.read_text().splitlines() if env_path.exists() else []
    out, found = [], False
    for line in lines:
        if line.strip().startswith(f"{key}="):
            out.append(f"{key}={value}")
            found = True
        else:
            out.append(line)
    if not found:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n")


def _prompt_for_key(env_path):
    """Ask the user to paste their key and save it to .env. Returns the key or ''."""
    if not sys.stdin.isatty():
        return ""  # non-interactive (e.g. CI) — skip the prompt
    label = "Groq API key" if config.LLM_BACKEND == "groq" else "Hugging Face token"
    print(f"     Get a free {label} at {config.KEY_URL} (takes ~1 minute).")
    try:
        entered = input(f"     Paste your {label} here (or press Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""
    if not entered:
        return ""
    _set_env_key(env_path, config.KEY_ENV_NAME, entered)
    return entered


def cmd_setup(args):
    print("Preparing the RAG Lab. This is a one-time step.\n")

    print("1/4  Checking Python packages...")
    missing = []
    for mod in ("chromadb", "streamlit", "groq", "huggingface_hub", "pypdf", "matplotlib"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"     ✗ missing: {', '.join(missing)}")
        print("       Run:  pip install -e .   (or: pip install -r requirements.txt)")
        return 1
    print("     ✓ all packages present")

    print("2/4  Caching the ONNX embedding model (downloads once, ~80 MB)...")
    from .pipeline import get_embedder

    get_embedder()
    print("     ✓ embedding model ready (no PyTorch needed)")

    print("3/4  Making sure a .env file exists to hold your API key...")
    env_path, created = _ensure_env_file()
    if created:
        print(f"     ✓ created {env_path.resolve()}")
    else:
        print(f"     ✓ found {env_path.resolve()}")

    print(f"4/4  Setting up the {config.KEY_ENV_NAME} (LLM backend: {config.LLM_BACKEND})...")
    if config.LLM_KEY:
        print(f"     ✓ {config.KEY_ENV_NAME} already set")
    else:
        entered = _prompt_for_key(env_path)
        if entered:
            config.LLM_KEY = entered
            tail = entered[-4:] if len(entered) > 4 else entered
            print(f"     ✓ saved to {env_path.name} (…{tail})")
        else:
            print("     • skipped — to add it later, open this file and paste your key:")
            print(f"         {env_path.resolve()}")
            print("       The lab still demonstrates retrieval without the LLM.")

    _hr()
    print("Setup done. Try:")
    print('    rag llm ask "What is the attendance policy of ABC Polytechnic?"')
    return 0


def cmd_info(args):
    from .pipeline import index_exists

    _hr()
    print(f"rag v{__version__}")
    print(f"  documents folder : {config.DOCUMENTS_DIR.resolve()}")
    print(f"  database folder  : {config.CHROMA_DIR.resolve()}")
    print(f"  embedding model  : {config.EMBED_MODEL_NAME}")
    print(f"  LLM backend      : {config.LLM_BACKEND}  ({config.LLM_MODEL})")
    print(f"  {config.KEY_ENV_NAME:<16} : {'set ✓' if config.LLM_KEY else 'NOT set — see `rag setup`'}")
    print(f"  chunk size       : {config.CHUNK_SIZE} (overlap {config.CHUNK_OVERLAP})")
    print(f"  top-k retrieved  : {config.TOP_K}")
    print(f"  index built?     : {'yes' if index_exists() else 'no — will build on first question'}")
    _hr()
    return 0


def cmd_ingest(args):
    from .pipeline import build_index

    docs_dir = Path(args.docs) if args.docs else None
    print("Building the knowledge base (read → chunk → embed → store)...")
    try:
        summary = build_index(docs_dir)
    except FileNotFoundError as exc:
        print(f"  ✗ {exc}")
        return 1
    print(f"  ✓ documents : {', '.join(summary['documents'])}")
    print(f"  ✓ chunks    : {summary['num_chunks']}")
    print(f"  ✓ stored in : {summary['db_path']}")
    print("\nReady. Try:  rag ask \"Where is the principal's office?\"")
    return 0


def cmd_add(args):
    from .pipeline import add_documents

    mode = "replacing the current documents" if args.replace else "adding to the documents"
    print(f"Bringing in your own file(s), {mode}...")
    try:
        summary = add_documents(args.files, replace=args.replace)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  ✗ {exc}")
        return 1

    if summary["removed"]:
        print(f"  ✓ removed   : {', '.join(summary['removed'])}")
    print(f"  ✓ added     : {', '.join(summary['added'])}")
    print(f"  ✓ now knows : {', '.join(summary['documents'])}")
    print(f"  ✓ chunks    : {summary['num_chunks']}")
    print("\nThe assistant's knowledge just changed — no retraining. Try:")
    print('    rag ask "..."   (ask something from your new document)')
    return 0


def cmd_db(args):
    from .pipeline import inspect_db, index_exists

    if not index_exists():
        print("No database yet. Build it with:  rag ingest")
        return 1

    info = inspect_db(limit=args.limit, show_vectors=args.vectors)
    _hr()
    print(f"Vector database: '{info['collection']}'")
    print(f"  stored at    : {info['db_path']}  (a ChromaDB / SQLite folder)")
    print(f"  total chunks : {info['total_chunks']}")
    print("  by document  :")
    for src, n in sorted(info["per_source"].items()):
        print(f"      {n:>4}  {src}")

    print(f"\nShowing {len(info['samples'])} sample chunk(s):")
    for row in info["samples"]:
        print(f"\n  • {row['id']}  (source: {row['source']}, chunk {row['chunk']})")
        if args.vectors:
            print(
                f"    vector: {row['vector_length']} numbers, "
                f"first 8 = {row['vector_preview']}"
            )
        text = row["text"].strip().replace("\n", " ")
        if not args.full and len(text) > 200:
            text = text[:200] + "…"
        print(f"    text: {text}")
    _hr()
    print("Tip: each chunk = some text + its 384-number embedding. A question is")
    print("     embedded the same way, then matched to the closest chunks here.")
    return 0


def cmd_llm(args):
    from .pipeline import ask_llm_only

    question = " ".join(args.question).strip()
    if not question:
        print("Please provide a question, e.g.  rag llm ask \"...\"")
        return 1

    result = ask_llm_only(question)
    _hr()
    print(f"Q: {question}")
    print("(LLM ALONE — no documents, no retrieval, no RAG)\n")
    if result["answer"]:
        print(f"A: {result['answer']}")
    else:
        print(result["note"])
    _hr()
    print("Notice: the model answered from its training only. For questions about")
    print("YOUR documents it guesses or refuses. Now compare:  rag ask \"...\"")
    return 0


def cmd_search(args):
    from .pipeline import retrieve, ensure_index

    query = " ".join(args.query).strip()
    if not query:
        print("Please provide a query, e.g.  rag search \"library timings\"")
        return 1

    ensure_index()
    matches = retrieve(query, k=args.top_k, source=args.source, contains=args.contains)

    filters = []
    if args.source:
        filters.append(f"source = {args.source}")
    if args.contains:
        filters.append(f'text contains "{args.contains}"')
    filter_note = f"  [filters: {', '.join(filters)}]" if filters else ""

    _hr()
    print(f"Query: {query}{filter_note}")
    print("(ChromaDB query — retrieval only, no LLM)\n")
    if not matches:
        print("  (no chunks matched the filters)")
    for i, m in enumerate(matches, 1):
        snippet = m["text"].strip().replace("\n", " ")
        print(f"  [{i}] distance={m['distance']:.3f}  source={m['source']}")
        print(f"      {snippet[:180]}{'…' if len(snippet) > 180 else ''}")
    _hr()
    print("Three ChromaDB query types in one call:")
    print("  • similarity  (always)        → query_embeddings + n_results")
    print("  • metadata    (--source)      → where={'source': ...}")
    print("  • keyword     (--contains)    → where_document={'$contains': ...}")
    return 0


def cmd_map(args):
    from .pipeline import project_2d, index_exists

    if not index_exists():
        print("No database yet. Build it with:  rag ingest")
        return 1

    try:
        points = project_2d()
    except ValueError as exc:
        print(f"  ✗ {exc}")
        return 1

    import matplotlib

    matplotlib.use("Agg")  # save to file, no GUI needed
    import matplotlib.pyplot as plt

    # Colour each document differently.
    sources = sorted({p["source"] for p in points})
    cmap = plt.get_cmap("tab10")
    colour = {s: cmap(i % 10) for i, s in enumerate(sources)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for s in sources:
        pts = [p for p in points if p["source"] == s]
        ax.scatter([p["x"] for p in pts], [p["y"] for p in pts],
                   color=colour[s], label=s, s=90, alpha=0.8)

    # Label points when there aren't too many to stay readable.
    if len(points) <= 40:
        for p in points:
            ax.annotate(f"{p['chunk']}: {p['label']}", (p["x"], p["y"]),
                        fontsize=8, alpha=0.75,
                        xytext=(5, 4), textcoords="offset points")

    ax.set_title("Vector database map — each dot is one chunk\n"
                 "(384-dim embeddings projected to 2D with PCA; "
                 "closer dots = more similar meaning)")
    ax.set_xlabel("PCA dimension 1")
    ax.set_ylabel("PCA dimension 2")
    ax.legend(title="document", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)

    print(f"  ✓ saved {len(points)} chunks to {args.output}")
    if not args.no_open:
        _open_file(args.output)
    print("\nClose dots share meaning. After `rag add ...`, run this again to")
    print("watch your new document appear as a fresh cluster.")
    return 0


def _open_file(path):
    """Open a file in the OS default app — works on Windows, macOS, and Linux."""
    try:
        if sys.platform == "win32":
            import os

            os.startfile(path)  # Windows-only; the correct way (not "start")
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception:
        pass  # opening is a convenience; never fail the command over it


def cmd_ask(args):
    from .pipeline import ask

    question = " ".join(args.question).strip()
    if not question:
        print("Please provide a question, e.g.  rag ask \"...\"")
        return 1

    result = ask(question, k=args.top_k)
    _hr()
    print(f"Q: {question}\n")
    if result["llm_ok"]:
        print(f"A: {result['answer']}")
    else:
        print(f"(LLM unavailable) {result['note']}")
    _print_chunks(result["chunks"])
    _hr()
    return 0


def cmd_chat(args):
    from .pipeline import ask, ensure_index

    print("Ensuring the knowledge base is ready...")
    ensure_index()
    print("College RAG chatbot. Ask a question, or type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q", ""}:
            break
        result = ask(question, k=args.top_k)
        if result["llm_ok"]:
            print(f"\nBot: {result['answer']}")
        else:
            print(f"\nBot: (LLM unavailable) {result['note']}")
        sources = sorted({c["source"] for c in result["chunks"]})
        print(f"     (sources: {', '.join(sources)})\n")
    return 0


def cmd_ui(args):
    ui_path = Path(__file__).parent / "ui.py"
    print("Launching the browser app (Streamlit)...")
    print("Press Ctrl+C in this terminal to stop it.\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(ui_path)], check=True
        )
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as exc:
        print(f"Could not launch Streamlit: {exc}")
        return 1
    return 0


# Sample data for the embeddings demo.
_DEMO_BASE = "The library opens at 9 AM."
_DEMO_CANDIDATES = [
    "What time does the library open?",
    "Library timings",
    "The hostel gate closes at 10 PM.",
]


def cmd_demo(args):
    from .pipeline import demo_embeddings

    base = args.text or _DEMO_BASE
    out = demo_embeddings(base, _DEMO_CANDIDATES)
    _hr()
    print(f'Base sentence: "{out["base"]}"')
    print(f"Embedding length: {out['vector_length']} numbers")
    print(f"First 8 numbers : {out['vector_preview']}")
    print("\nSimilarity to base (1.0 = same meaning, 0.0 = unrelated):")
    for c in out["comparisons"]:
        print(f"  {c['similarity']:.3f}   {c['text']}")
    print("\nTakeaway: similar meaning → similar vectors.")
    print("\nNext: see retrieval on the REAL handbook with  rag search \"...\"")
    _hr()
    return 0


# ── argument parser ──────────────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        prog="rag",
        description="Build your own College Document AI Assistant (RAG).",
    )
    p.add_argument("--version", action="version", version=f"rag {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="one-time: download the models").set_defaults(
        func=cmd_setup
    )
    sub.add_parser("info", help="show settings and index status").set_defaults(
        func=cmd_info
    )

    pi = sub.add_parser("ingest", help="(re)build the vector DB from documents/")
    pi.add_argument("--docs", help="folder of .pdf/.txt files (default: documents/)")
    pi.set_defaults(func=cmd_ingest)

    pdb = sub.add_parser("db", help="peek inside the vector database")
    pdb.add_argument("--limit", type=int, default=5, help="how many chunks to show")
    pdb.add_argument("--vectors", action="store_true", help="also show a vector preview")
    pdb.add_argument("--full", action="store_true", help="show full chunk text")
    pdb.set_defaults(func=cmd_db)

    pm = sub.add_parser("map", help="visualise the vectors as a 2D map (PNG)")
    pm.add_argument("-o", "--output", default="vectors_map.png", help="image file to write")
    pm.add_argument("--no-open", action="store_true", help="don't auto-open the image")
    pm.set_defaults(func=cmd_map)

    pn = sub.add_parser(
        "add", help="add your own .pdf/.txt and rebuild (the 'wow' finale)"
    )
    pn.add_argument("files", nargs="+", help="path(s) to your .pdf or .txt file(s)")
    pn.add_argument(
        "--replace",
        action="store_true",
        help="remove the current documents first (full knowledge swap)",
    )
    pn.set_defaults(func=cmd_add)

    pl = sub.add_parser(
        "llm", help="talk to the LLM directly — NO retrieval (the 'before RAG' demo)"
    )
    llm_sub = pl.add_subparsers(dest="llm_command", required=True)
    pla = llm_sub.add_parser("ask", help="ask the raw LLM (no RAG)")
    pla.add_argument("question", nargs="+", help="your question, in quotes")
    pla.set_defaults(func=cmd_llm)

    pa = sub.add_parser("ask", help="ask one question (full RAG)")
    pa.add_argument("question", nargs="+", help="your question, in quotes")
    pa.add_argument("--top-k", type=int, default=None, help="chunks to retrieve")
    pa.set_defaults(func=cmd_ask)

    psr = sub.add_parser(
        "search", help="query ChromaDB directly (retrieval only, no LLM)"
    )
    psr.add_argument("query", nargs="+", help="your search query, in quotes")
    psr.add_argument("--top-k", type=int, default=None, help="chunks to retrieve")
    psr.add_argument("--source", default=None, help="metadata filter: only this document")
    psr.add_argument("--contains", default=None, help="keyword filter: text must contain this")
    psr.set_defaults(func=cmd_search)

    pc = sub.add_parser("chat", help="interactive chat")
    pc.add_argument("--top-k", type=int, default=None, help="chunks to retrieve")
    pc.set_defaults(func=cmd_chat)

    sub.add_parser("ui", help="launch the browser app").set_defaults(func=cmd_ui)

    pd = sub.add_parser("demo", help="teaching demo: embeddings")
    pd.add_argument("kind", choices=["embeddings"])
    pd.add_argument("text", nargs="?", help="optional custom sentence")
    pd.set_defaults(func=cmd_demo)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
