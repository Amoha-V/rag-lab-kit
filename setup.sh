#!/usr/bin/env bash
# ============================================================
#  RAG Lab setup for macOS / Linux  (robust: tries fallbacks)
#  Run once:   bash setup.sh
# ============================================================
set -u

echo "============================================================"
echo " RAG Lab setup (macOS / Linux)"
echo "============================================================"

# ── 1. Find a compatible Python (3.12 / 3.11 / 3.10) ─────────
# Native deps (tokenizers, hnswlib) only ship wheels for 3.10–3.12.
PYBIN=""
for cand in python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
done
# fallback: a bare python3 that happens to be 3.10–3.12
if [ -z "$PYBIN" ] && command -v python3 >/dev/null 2>&1; then
    if python3 -c 'import sys;exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)'; then
        PYBIN="python3"
    fi
fi
# fallback: install 3.12 via Homebrew on macOS
if [ -z "$PYBIN" ] && command -v brew >/dev/null 2>&1; then
    echo "==> No Python 3.10–3.12 found. Installing python@3.12 via Homebrew..."
    brew install python@3.12 && PYBIN="$(brew --prefix)/bin/python3.12"
fi
if [ -z "$PYBIN" ]; then
    echo "ERROR: Need Python 3.10, 3.11, or 3.12 (not 3.13+). Install it, then re-run:"
    echo "  macOS:                 brew install python@3.12"
    echo "  Linux (Debian/Ubuntu): sudo apt install python3.12 python3.12-venv"
    echo "  Linux (Fedora):        sudo dnf install python3.12"
    exit 1
fi
echo "Using Python: $("$PYBIN" --version 2>&1)"

# ── 2. Create the virtual environment (try several ways) ─────
if [ -x ".venv/bin/python" ]; then
    echo ".venv already exists - reusing it."
else
    echo "Creating virtual environment..."
    "$PYBIN" -m venv .venv || true
    if [ ! -x ".venv/bin/python" ]; then
        echo "  ... standard venv failed, retrying with --copies"
        rm -rf .venv
        "$PYBIN" -m venv --copies .venv || true
    fi
    if [ ! -x ".venv/bin/python" ]; then
        echo "  ... retrying via virtualenv"
        "$PYBIN" -m pip install --user virtualenv >/dev/null 2>&1 || true
        "$PYBIN" -m virtualenv .venv >/dev/null 2>&1 || true
    fi
fi
if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: Could not create a virtual environment with $PYBIN."
    exit 1
fi

# ── 3. Install using the venv's python DIRECTLY (no activation) ──
VPY=".venv/bin/python"
echo "Installing the rag package (a few minutes the first time)..."
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -e . || "$VPY" -m pip install -e . --no-cache-dir

if ! "$VPY" -m rag.cli --version >/dev/null 2>&1; then
    echo "ERROR: the rag package did not install correctly. Scroll up for the error."
    exit 1
fi

# ── 4. .env + embedding model + Groq key prompt ──────────────
[ -f .env ] || { [ -f .env.example ] && cp .env.example .env; }
echo "Running built-in setup (caches the model, asks for your Groq key)..."
"$VPY" -m rag.cli setup || true

echo ""
echo "============================================================"
echo " Setup complete!"
echo ""
echo " Activate the environment in each NEW terminal:"
echo "     source .venv/bin/activate"
echo ""
echo " ...or skip activation and run it directly any time:"
echo "     .venv/bin/rag info"
echo "     .venv/bin/rag ask \"How much attendance is required?\""
echo "============================================================"
