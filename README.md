# RAG Lab Kit

A free, hands-on session for learning Retrieval-Augmented Generation (RAG) —
**no-code**. You go from *"the LLM hallucinates about my documents"* to *"the
LLM answers correctly, grounded in my own documents"* — entirely through one
`rag` command.

> **RAG in one line:** find the right paragraph, then ask the LLM to answer
> using only that paragraph.

```
PDF/Text → Chunks → Embeddings → Vector DB → (question) → Retrieve → LLM → Answer
```

## What's inside

- A real, working RAG pipeline: **ChromaDB** (built-in ONNX embeddings — no
  PyTorch download) for the vector store, **Groq** (or Hugging Face) as the LLM.
- A single CLI (`rag ...`) and an optional **Streamlit** browser UI (`rag ui`).
- Two ready-made sample documents (a college handbook + department manual) so
  it works the moment you install it — no setup of your own data required.
- **Bring-your-own-document** support: drop in any PDF/text file and watch the
  assistant's answers change instantly, with no retraining.

## Quickstart

**Option A — Download (no git needed)**

Grab the zip directly: **https://tinyurl.com/5dhazb8m**
Unzip it, then `cd` into the `rag-lab-kit` folder.

**Option B — Clone**

```bash
git clone <this-repo-url>
cd rag-lab-kit
```

**1. Install (one-time)**

| Platform | Command |
|---|---|
| macOS / Linux | `bash setup.sh` |
| Windows — Command Prompt | `setup.bat` |
| Windows — PowerShell | `.\setup.bat` |

This creates a virtual environment, installs dependencies, and prompts for a
**free** Groq API key ([console.groq.com](https://console.groq.com)).

**2. Activate (every new terminal)**

| Platform | Command |
|---|---|
| macOS / Linux | `source .venv/bin/activate` |
| Windows — cmd | `.venv\Scripts\activate.bat` |
| Windows — PowerShell | `.\.venv\Scripts\Activate.ps1` |

**3. Check it's ready**

```bash
rag info
```

**4. Run the lab**

Open **`LAB_MANUAL.md`** and follow it step by step — it's a full guided
walkthrough (~30–45 min) that builds intuition for every part of RAG. Keep
**`CHEATSHEET.md`** open alongside it for a one-page command reference.

## Core commands

```bash
rag llm ask "..."          # LLM alone -> guesses/hallucinates (the problem)
rag demo embeddings        # see text turn into vectors
rag search "..."           # retrieval only, no LLM
rag ask "..."              # full RAG -> grounded, correct answer (the fix)
rag chat                   # interactive chat in the terminal
rag ui                     # browser app (Streamlit)
rag add <file> --replace   # swap in your own PDF/text and rebuild
rag map                    # 2D plot of the document vectors
```

## Requirements

- Python **3.10 – 3.12** (3.13+ is not yet supported by the pinned ML stack)
- A free **Groq** API key ([console.groq.com](https://console.groq.com)), or a
  free **Hugging Face** token as an alternative backend — see `.env.example`

## Project structure

```
rag-lab-kit/
├── setup.sh / setup.bat       # one-time installer (macOS/Linux / Windows)
├── requirements.txt           # pip install -e . also works
├── pyproject.toml             # defines the `rag` CLI entry point
├── .env.example                # copy to .env and add your API key
├── LAB_MANUAL.md               # the guided, step-by-step lab
├── CHEATSHEET.md                # one-page command reference
├── documents/                   # default sample document(s)
├── extra_documents/             # a second sample document to swap in
└── rag/                          # the package: cli.py, pipeline.py, config.py, ui.py
```

## Troubleshooting

See the **Quick fixes** table at the bottom of `CHEATSHEET.md` for the most
common issues (`rag: command not found`, missing API key, no compatible
Python, etc.).

## License

Released under the [MIT License](LICENSE) — free to use, copy, and adapt for
teaching or learning.
