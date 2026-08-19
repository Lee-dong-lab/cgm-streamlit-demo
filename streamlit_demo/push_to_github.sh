#!/usr/bin/env bash
# GitHub 저장소 생성 + push (최초 1회)
# 사전: ~/.local/bin/gh auth login  (또는 시스템 gh)
set -euo pipefail

REPO_NAME="${1:-cgm-streamlit-demo}"
VISIBILITY="${2:-public}"   # public | private
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GH="${GH:-$(command -v gh || echo "$HOME/.local/bin/gh")}"

cd "$ROOT"

if ! "$GH" auth status >/dev/null 2>&1; then
  echo "GitHub 로그인이 필요합니다. 아래 명령을 먼저 실행하세요:"
  echo "  $GH auth login"
  exit 1
fi

ACCOUNT=$("$GH" api user -q .login)
REMOTE_URL="https://github.com/${ACCOUNT}/${REPO_NAME}.git"

if git remote get-url origin >/dev/null 2>&1; then
  echo "origin 이미 존재 → push만 수행"
  git push -u origin main
elif "$GH" repo view "${ACCOUNT}/${REPO_NAME}" >/dev/null 2>&1; then
  echo "저장소가 이미 존재함 → remote 연결 후 push: ${ACCOUNT}/${REPO_NAME}"
  git remote add origin "$REMOTE_URL"
  git push -u origin main
else
  echo "저장소 생성: $REPO_NAME ($VISIBILITY)"
  "$GH" repo create "$REPO_NAME" \
    --"$VISIBILITY" \
    --source=. \
    --remote=origin \
    --push \
    --description "CGM AI Streamlit simulation demo (Model3 LightGBM)"
fi

echo ""
echo "완료. 저장소 URL:"
"$GH" repo view "${ACCOUNT}/${REPO_NAME}" --json url -q .url 2>/dev/null || git remote get-url origin
