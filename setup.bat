@echo off
REM ============================================================
REM  RAG Lab setup for Windows  (robust: tries many fallbacks)
REM  Run once:   setup.bat   (cmd)   or   .\setup.bat   (PowerShell)
REM ============================================================
setlocal enabledelayedexpansion

echo ============================================================
echo  RAG Lab setup (Windows)
echo ============================================================

REM ── 1. Find a compatible Python (3.12 / 3.11 / 3.10) ─────────
REM    The native deps (tokenizers, hnswlib) only ship wheels for
REM    3.10-3.12; 3.13+ would try to build from source and fail.
set "PYBIN="
for %%V in (3.12 3.11 3.10) do (
    if not defined PYBIN (
        py -%%V -c "import sys" >nul 2>&1 && set "PYBIN=py -%%V"
    )
)
REM fallback: a bare python/python3 that happens to be 3.10-3.12
if not defined PYBIN (
    for %%P in (python python3) do (
        if not defined PYBIN (
            %%P -c "import sys;exit(0 if (3,10)<=sys.version_info[:2]<=(3,12) else 1)" >nul 2>&1 && set "PYBIN=%%P"
        )
    )
)
if not defined PYBIN goto :no_python
echo Using Python: %PYBIN%
%PYBIN% --version

REM ── 2. Create the virtual environment (try several ways) ─────
if exist ".venv\Scripts\python.exe" (
    echo .venv already exists - reusing it.
) else (
    echo Creating virtual environment...
    %PYBIN% -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo   ... standard venv failed, retrying with --copies
        rmdir /s /q .venv >nul 2>&1
        %PYBIN% -m venv --copies .venv
    )
    if not exist ".venv\Scripts\python.exe" (
        echo   ... retrying via virtualenv
        %PYBIN% -m pip install --user virtualenv >nul 2>&1
        %PYBIN% -m virtualenv .venv >nul 2>&1
    )
)
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Could not create a virtual environment with %PYBIN%.
    echo   Try repairing your Python install, or see CHEATSHEET.md for a
    echo   no-venv option ^(py -3.12 -m pip install -e .^).
    pause
    exit /b 1
)

REM ── 3. Install using the venv's python DIRECTLY (no activation) ──
set "VPY=.venv\Scripts\python.exe"
echo Installing the rag package (a few minutes the first time)...
"%VPY%" -m pip install --upgrade pip
"%VPY%" -m pip install -e .
if errorlevel 1 (
    echo   ... first install attempt failed, retrying without cache
    "%VPY%" -m pip install -e . --no-cache-dir
)

REM verify the package actually imports / the command exists
"%VPY%" -m rag.cli --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: the rag package did not install correctly. Scroll up for the error.
    pause
    exit /b 1
)

REM ── 4. .env + embedding model + Groq key prompt ──────────────
if not exist .env if exist .env.example copy .env.example .env >nul
echo Running built-in setup (caches the model, asks for your Groq key)...
"%VPY%" -m rag.cli setup

echo.
echo ============================================================
echo  Setup complete!
echo.
echo  Activate the environment in each NEW terminal:
echo     Command Prompt : .venv\Scripts\activate.bat
echo     PowerShell     : .\.venv\Scripts\Activate.ps1
echo.
echo  ...or skip activation and run it directly any time:
echo     .venv\Scripts\rag info
echo     .venv\Scripts\rag ask "How much attendance is required?"
echo ============================================================
pause
exit /b 0

:no_python
echo.
echo No compatible Python (3.10, 3.11, or 3.12) was found.
where winget >nul 2>&1
if errorlevel 1 goto :manual_python
echo Installing Python 3.12 automatically via winget (a few minutes)...
winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
echo.
echo Python 3.12 installed. IMPORTANT: close this window, open a NEW terminal,
echo cd back into this folder, and run setup.bat again.
pause
exit /b 1

:manual_python
echo winget is not available, so install Python 3.12 manually:
echo   https://www.python.org/downloads/release/python-3120/
echo   ^(tick "Add Python to PATH" during install^), then re-run setup.bat
pause
exit /b 1
