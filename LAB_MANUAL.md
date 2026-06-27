# Lab Manual — Build Your Own College Document AI Assistant

You will build a working AI assistant **without writing or editing any code.**
Everything is one `rag` command. Work at your own pace; each part says
what to type and what you should see.

**The big idea, in one line:**
> RAG = find the right paragraph, then ask the LLM to answer using only that paragraph.

---

## Before you start

**First time?** Run the installer from inside the `rag-lab` folder:

- **macOS / Linux:** `bash setup.sh`
- **Windows — Command Prompt (cmd):** `setup.bat`
- **Windows — PowerShell:** `.\setup.bat`  *(PowerShell needs the `.\` prefix)*

In every **new** terminal, activate the environment first:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows — cmd:** `.venv\Scripts\activate.bat`
- **Windows — PowerShell:** `.\.venv\Scripts\Activate.ps1`

Then check it's ready:
```bash
rag info
```

If `rag` is "command not found", you haven't activated the environment — run the
activate command above. See the **Quick fixes** table in `CHEATSHEET.md`.

---

## Part 1 — First, see the PROBLEM (the LLM alone)

Ask the LLM **directly** — no documents, no retrieval, just the raw model:

```bash
rag llm ask "What is the attendance policy of ABC Polytechnic?"
```

❌ **It guesses or makes something up.** The model was never trained on your
college's handbook, so it can't know the real 75% rule. *Remember this answer.*

```
LLM alone  →  doesn't know YOUR documents  →  guesses / hallucinates
```

We'll fix this step by step — and at the end, ask the **exact same question**
the right way and compare.

---

## Part 2 — The plan to fix it (the 5 boxes)

To make the model answer from *our* documents, we first turn the documents into
something searchable, then feed the right piece to the LLM:

```
PDF  →  Chunks  →  Embeddings  →  Vector DB  →  (retrieve)  →  LLM
```

- **Embedding** = GPS coordinates for meaning
- **Vector DB** = Google Maps for documents
- **Retrieval** = finding nearby documents
- **LLM** = writes the final answer in plain English

---

## Part 3 — See embeddings

```bash
rag demo embeddings
```

✅ **You should see:** a sentence turned into **384 numbers**, and two
library-related sentences scoring **high** similarity while a hostel sentence
scores **low** — even though the words differ.

Try your own sentence:

```bash
rag demo embeddings "The exam starts at 10 AM"
```

> **Takeaway:** similar meaning → similar vectors.

---

## Part 4 — See retrieval on the real handbook (no LLM yet)

Query the actual document database directly — retrieval only, no LLM:

```bash
rag search "library timings"
```

✅ **You should see:** the matching chunk from the handbook (the **library**
section) come back first — found by **meaning**, with no LLM, even though
"timings" isn't in the text. *(This is the first "wow".)*

Try more — and the filters too:

```bash
rag search "when does the hostel close?"
rag search "fees" --contains payment       # keyword filter
rag search "rules" --top-k 5               # more results
```

> The vector DB **always** returns the closest sentence, even if nothing truly
> matches — which is why the next stage adds an LLM that can say *"I don't know."*

---

## Part 5 — Full RAG: now ask the SAME question the right way

Remember Part 1, where the LLM alone guessed? Ask the **exact same question**,
but this time with retrieval (full RAG):

```bash
rag llm ask "What is the attendance policy of ABC Polytechnic?"   # before — guesses
rag ask "What is the attendance policy of ABC Polytechnic?"   # after  — correct!
```

✅ **Same model, same question — but `rag ask` is right** because it first
retrieved the real handbook text and the LLM answered using only that. That gap
between the two answers *is* what RAG buys you.

The first `rag ask` automatically builds the knowledge base, then answers. Try more:

```bash
rag ask "Where is the principal's office?"
rag ask "What are the examination rules?"
rag ask "When does the library close?"
```

✅ **Notice** the **Retrieved chunks** printed under each answer — that's the
exact evidence the LLM was given. The assistant is *grounded*, not guessing.

Prefer a conversation or a browser app?

```bash
rag chat      # type questions; 'quit' to exit
rag ui        # opens a web page; Ctrl+C to stop
```

---

## Part 6 — Swap in YOUR OWN document (the payoff)

Bring your own PDF or text file — your timetable, a rulebook, your notes,
anything — and watch the assistant's knowledge change instantly.

**One command** (this copies your file in, replaces the old documents, and
rebuilds — all in one step):

```bash
# macOS / Linux:
rag add ~/Downloads/my_file.pdf --replace
# Windows:
rag add C:\Users\you\Downloads\my_file.pdf --replace

rag ask "..."          # ask something only YOUR document can answer
```

Leave off `--replace` to **add** your file alongside the existing ones instead
of replacing them.

**Or do it in the browser** — run `rag ui`, open the **"📄 Use your own
document"** panel on the left, drag in a PDF, click **Add & rebuild**, and ask.

Prefer the bundled example? Use the second sample document:

```bash
rag add extra_documents/department_manual.txt --replace
rag ask "When does the project lab close?"
rag ask "What is the internship requirement?"
```

✅ **The assistant's knowledge changed completely** — and you never retrained
the LLM. You only changed the documents it retrieves from.

> **This is the whole point of RAG.**

---

## Check yourself

You've succeeded if you can answer these:

1. Why can't a plain LLM answer questions about *your* college's documents?
2. What is an embedding, in one sentence?
3. Why do we need a vector database?
4. What are the steps between a user's question and the final answer?
5. How would you make this assistant answer about a different college?

---

## Bonus (if you finish early)

- `rag ask "..." --top-k 1` vs `--top-k 5` — when is one chunk too little context?
- `rag search "fees" --contains payment` — combine meaning search with a keyword filter.
- Add your own `.txt` to `documents/`, run `rag ingest`, and ask about it.
- `RAG_MODEL=llama-3.3-70b-versatile rag ask "..."` — same retrieval, a smarter writer.
