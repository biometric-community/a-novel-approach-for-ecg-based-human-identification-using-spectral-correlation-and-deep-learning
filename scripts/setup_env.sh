#!/usr/bin/env bash
# Prefer the monorepo shared venv (.venv). No per-project .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REPO_ROOT="$ROOT"
while [[ "$REPO_ROOT" != "/" ]]; do
  if [[ -f "$REPO_ROOT/docs/PAPERS.md" ]] || [[ -d "$REPO_ROOT/.cursor/skills" && -d "$REPO_ROOT/.venv" ]]; then
    break
  fi
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
SHARED_VENV="${TBIOT_SHARED_VENV:-$REPO_ROOT/.venv}"

if [[ ! -x "$SHARED_VENV/bin/python" ]]; then
  echo "==> shared venv missing; bootstrapping via research-analyze-paper setup"
  bash "$REPO_ROOT/.cursor/skills/research-analyze-paper/scripts/setup.sh"
fi

# Remove leftover project-local .venv (symlink or stale dir) — shared env only.
if [[ -L .venv ]]; then
  rm -f .venv
elif [[ -d .venv ]]; then
  echo "==> removing non-shared project .venv (was not a symlink); backing up"
  mv .venv ".venv.bak.$(date +%Y%m%d%H%M%S)"
fi

# shellcheck disable=SC1091
source "$SHARED_VENV/bin/activate"
python -m pip install --upgrade pip
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
fi

python - <<'PY'
try:
    import torch
    print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
except ImportError:
    print("torch not installed (optional until ML deps are pip-installed)")
PY

echo "Environment ready (shared only): $SHARED_VENV"
echo "Activate with: source $SHARED_VENV/bin/activate"
