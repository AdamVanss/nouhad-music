#!/usr/bin/env bash
# One command for a friend who already has Node.
# Installs the stem/face app and the Cinematch movie recommender. No Docker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

say() { printf '\n== %s\n' "$1"; }

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node and npm are required. Install Node, then run this script again." >&2
  exit 1
fi

PY=()
if command -v py >/dev/null 2>&1 && py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
  PY=(py -3)
else
  for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1 && "$cmd" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
      PY=("$cmd")
      break
    fi
  done
fi
if [[ ${#PY[@]} -eq 0 ]]; then
  echo "Python 3.10 or newer is required. Install it, then run this script again." >&2
  exit 1
fi

say "Python: ${PY[*]}"
if [[ ! -d .venv ]]; then
  "${PY[@]}" -m venv .venv
fi
if [[ -f .venv/Scripts/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/Scripts/activate
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

say "Python packages"
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install demucs fastapi "uvicorn[standard]" python-multipart pillow "opencv-python-headless<5"
python -m pip install -r movie-recommendation-system/requirements.txt nltk

say "Movie data and similarity matrix"
python movie-recommendation-system/build_artifacts.py

say "Web app"
cd "$ROOT/web"
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi
cd "$ROOT"

say "Ready"
cat <<EOF
Both apps are installed.

Stem separator and face classifier, two terminals, from this folder:

  source .venv/Scripts/activate    # Windows Git Bash
  source .venv/bin/activate        # macOS or Linux
  python -m uvicorn src.serve:app --port 8000

  cd web
  npm run dev

Movie recommender, a third terminal, from this folder (venv still active):

  cd movie-recommendation-system
  streamlit run app.py

The movie app opens at http://localhost:8501
The stem and face app opens at http://localhost:3000
EOF
