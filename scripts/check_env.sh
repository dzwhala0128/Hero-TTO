#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1 TEXTGRAD_DISABLE_LOGGING=1
python -B scripts/check_data.py
python -B - <<'PY'
import sys,importlib.metadata
from pipeline.common import ROOT
import textgrad
from pipeline.runner import parser
from pipeline.validator import validate
print('Python:',sys.version.split()[0],'; native TextGrad:',textgrad.__version__)
for name in ['httpx','requests','platformdirs','diskcache']:
 print(name,importlib.metadata.version(name))
print('Offline core imports passed. GPU/retrieval/model services: not verified.')
PY
if [[ ${1:-} == --online ]]; then
  if [[ -f .env ]]; then set -a; source .env; set +a; fi
  python -B - <<'PY'
from pipeline.runner import settings
from pipeline.engine import TraceEngine
from pipeline.common import ROOT
for role in ['hero','textgrad']:
 e=TraceEngine(ROOT/'outputs/service_check',role,settings(role));e.verify_model();print(role,e.model_string,'/v1/models alias verified')
PY
fi
