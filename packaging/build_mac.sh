#!/bin/bash
# Builds dist/Shorts Studio.app (ffmpeg and the models inside).
# Run on the Mac where you'll use it: the app matches that CPU (Apple Silicon / Intel).
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if [ ! -x .venv-build/bin/python ]; then
  "$PY" -m venv .venv-build
fi
.venv-build/bin/pip install --upgrade pip
.venv-build/bin/pip install -e ".[app]" pyinstaller
# app icon from a generated 1024px PNG (skipped silently if iconutil is missing)
if command -v iconutil >/dev/null 2>&1 && [ ! -f packaging/ShortsStudio.icns ]; then
  .venv-build/bin/python packaging/make_icon.py packaging/ShortsStudio.iconset && \
    iconutil -c icns packaging/ShortsStudio.iconset -o packaging/ShortsStudio.icns || true
fi
.venv-build/bin/pyinstaller --noconfirm --clean packaging/shorts_studio.spec
echo
echo "Fatto: dist/Shorts Studio.app"
echo "Primo avvio: clic destro sull'app > Apri (non è firmata)."
