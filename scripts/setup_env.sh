#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Run: source .venv/bin/activate'
