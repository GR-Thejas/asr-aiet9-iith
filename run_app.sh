#!/usr/bin/env bash
# Speak & Transcribe launcher for macOS / Linux.  Usage: ./run_app.sh [streamlit options]
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "[setup] First run: creating a Python environment and installing packages (5-15 minutes)."
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    if [ "$(uname)" = "Darwin" ]; then
        .venv/bin/python -m pip install torch
    else
        .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    .venv/bin/python -m pip install -r requirements.txt
fi

echo
echo "Starting the app. Open http://localhost:8501 in your browser. Press Ctrl+C to stop."
exec .venv/bin/python -m streamlit run app.py "$@"
