#!/usr/bin/env bash
# 로컬 Streamlit 데모 실행
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if [[ -x "../.venv/bin/python" ]]; then
  PYTHON="../.venv/bin/python"
fi

echo "[1/2] 시나리오 CLI 검증..."
"$PYTHON" run_scenarios.py

echo ""
echo "[2/2] Streamlit 웹 데모 시작..."
exec "$PYTHON" -m streamlit run app.py --server.headless true
