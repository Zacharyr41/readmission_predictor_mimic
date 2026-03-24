#!/usr/bin/env bash
# Launch the NeuroGraph conversational chat UI.
# Usage: bash scripts/run_chat.sh [streamlit args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
exec .venv/bin/streamlit run src/conversational/app.py "$@"
