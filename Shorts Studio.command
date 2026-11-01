#!/bin/bash
# Doppio clic per aprire Shorts Studio. Al primo avvio crea un ambiente Python
# dedicato (.venv-app) e installa le dipendenze: serve internet una volta.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python3}
if [ ! -x .venv-app/bin/python ]; then
  echo "Primo avvio: installo le dipendenze (qualche minuto)..."
  "$PY" -m venv .venv-app || { echo "Serve Python 3 (python.org)"; read -r; exit 1; }
  .venv-app/bin/pip install --upgrade pip >/dev/null
  .venv-app/bin/pip install -e ".[app]" || { echo "Installazione fallita"; read -r; exit 1; }
fi
exec .venv-app/bin/python -m face_reframe.app "$@"
