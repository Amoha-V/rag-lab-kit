# 🎓 RAG Lab — One-Page Cheat-Sheet

**RAG = find the right paragraph, then ask the LLM to answer using only that paragraph.**

---

### Setup (once)
```bash
cd rag-lab
```
**macOS / Linux**
```bash
bash setup.sh                    # asks for your Groq key (console.groq.com)
source .venv/bin/activate        # every new terminal
```
**Windows — Command Prompt (cmd)**
```bat
setup.bat
.venv\Scripts\activate.bat
```
**Windows — PowerShell** (note the `.\` prefix)
```powershell
.\setup.bat
.\.venv\Scripts\Activate.ps1
```
Then on any OS:
```bash
rag info                         # expect: GROQ_API_KEY set ✓
```
> Skipped the key prompt? Run `rag setup` again, or paste it into the `.env` file.
> PowerShell blocks `Activate.ps1`? Run once: `Set-ExecutionPolicy -Scope Process Bypass`

### Switch LLM provider (if Groq is blocked on your network)
In the `.env` file, use **Hugging Face** instead of Groq:
```
RAG_LLM=hf
HF_TOKEN=hf_your_token_here     # free: huggingface.co/settings/tokens
```
Then `rag info` shows `LLM backend: hf`. Default is `groq` (console.groq.com).

### The pipeline
```
PDF → Chunks → Embeddings → Vector DB → (question) → Retrieve → LLM → Answer
```

### Core commands (the story)
```bash
rag llm ask "attendance policy of ABC Polytechnic?"   # LLM ALONE → guesses (the problem)
rag demo embeddings                          # text → 384 numbers; similar = close
rag search "library timings"        # retrieval on real DB, NO LLM
rag ask "attendance policy of ABC Polytechnic?"   # full RAG → correct! (the fix)
rag chat                                     # interactive (quit to exit)
rag ui                                       # browser app
```

### See & query the database
```bash
rag search "library timings"                 # query ChromaDB (no LLM)
rag search "timings" --contains library      # keyword filter
rag search "timings" --source college_handbook.txt   # metadata filter
rag db --vectors                             # list chunks + their vectors
rag map                                       # 2D picture of the vectors
```

### The finale — your own document
```bash
rag add ~/Downloads/your_file.pdf --replace  # swap in your file + rebuild
rag ask "..."                                # ask something only it knows
rag map                                       # see it as a new cluster
```
*Or:* `rag ui` → **"📄 Use your own document"** → drag a PDF → **Add & rebuild**.

> **Why it works:** we didn't retrain the model — we changed its **library**, not its **brain**.

---

### Quick fixes
| Problem | macOS / Linux | Windows |
|---|---|---|
| `rag: command not found` | `source .venv/bin/activate` | `.venv\Scripts\activate.bat` |
| activation won't work / PATH issues | `.venv/bin/rag info` (no activation) | `.venv\Scripts\rag info` (no activation) |
| `No GROQ_API_KEY` | add key to `.env`; check `rag info` | same |
| `Invalid API Key` | make a new key at console.groq.com | same |
| no Python 3.10–3.12 | `brew install python@3.12`, re-run | install Python 3.12 from python.org (tick "Add to PATH"), re-run `setup.bat` |

> **No-activation escape hatch:** you can always run the tool by full path without
> activating — `.venv\Scripts\rag <cmd>` (Windows) or `.venv/bin/rag <cmd>` (macOS/Linux).

*You only ever provide two things: the **documents** (once) and a **question** (each time).*
*Stack: Python · ChromaDB (ONNX embeddings, no PyTorch) · Groq · Streamlit*
