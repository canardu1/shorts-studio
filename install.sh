#!/bin/bash
# Shorts Studio — macOS installer.
#
#   curl -fsSL https://raw.githubusercontent.com/canardu1/shorts-studio/main/install.sh | bash
#
# Installs everything in ~/.shorts-studio (its own Python, no admin rights, nothing
# else on the system is touched) and creates "Shorts Studio" in ~/Applications.
# Run it again to update. To uninstall:
#   rm -rf ~/.shorts-studio ~/Applications/"Shorts Studio.app"
#
# Options (environment variables):
#   SHORTS_STUDIO_REF   branch or tag to install (default: main)
#   SHORTS_STUDIO_HOME  install folder (default: ~/.shorts-studio)
#   SHORTS_STUDIO_APPS  where the app goes (default: ~/Applications)
#   SHORTS_STUDIO_SRC   install from a local source folder instead of GitHub (for testing)

set -euo pipefail

REPO="canardu1/shorts-studio"
REF="${SHORTS_STUDIO_REF:-main}"
HOME_DIR="${SHORTS_STUDIO_HOME:-$HOME/.shorts-studio}"
APPS_DIR="${SHORTS_STUDIO_APPS:-$HOME/Applications}"
APP="$APPS_DIR/Shorts Studio.app"
PYVER="3.12"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
step() { printf '\n\033[1;31m▸\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31mError:\033[0m %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "this installer is for macOS. On Linux/Windows use: pip install -e \".[app]\" from the source code."
MACOS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
[ "$MACOS_MAJOR" -ge 11 ] || fail "macOS 11 (Big Sur) or newer is required."

bold "Shorts Studio — installation"
echo "Folder:   $HOME_DIR"
echo "App:      $APP"

mkdir -p "$HOME_DIR" "$APPS_DIR"
cd "$HOME_DIR"

# 1. uv: a small tool that downloads a private Python and installs the packages fast
step "Setting up Python (once)…"
UV="$HOME_DIR/bin/uv"
if [ ! -x "$UV" ]; then
  curl -fsSL https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME_DIR/bin" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null \
    || fail "could not download uv. Check your connection and try again."
fi
export UV_PYTHON_INSTALL_DIR="$HOME_DIR/python"
export UV_CACHE_DIR="$HOME_DIR/cache"
export UV_PYTHON_BIN_DIR="$HOME_DIR/bin"   # keep python3.x out of ~/.local/bin
"$UV" python install "$PYVER" >/dev/null || fail "Python installation failed."

# 2. the app's source code
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/src"
if [ -n "${SHORTS_STUDIO_SRC:-}" ]; then
  step "Copying the code from ${SHORTS_STUDIO_SRC}…"
  (cd "$SHORTS_STUDIO_SRC" && tar --exclude .git --exclude '.venv*' --exclude build --exclude dist -cf - .) | tar -xf - -C "$TMP/src"
else
  step "Downloading Shorts Studio (${REF})…"
  URL="https://github.com/$REPO/archive/refs/heads/$REF.tar.gz"
  curl -fsSL "$URL" -o "$TMP/src.tar.gz" 2>/dev/null \
    || curl -fsSL "https://github.com/$REPO/archive/refs/tags/$REF.tar.gz" -o "$TMP/src.tar.gz" \
    || fail "source download failed ($URL)."
  tar -xzf "$TMP/src.tar.gz" -C "$TMP/src" --strip-components 1
fi
rm -rf "$HOME_DIR/src"
mv "$TMP/src" "$HOME_DIR/src"

# 3. a private environment with the app and its libraries
step "Installing the libraries (a few minutes the first time, ~1 GB)…"
if [ ! -x "$HOME_DIR/venv/bin/python" ]; then
  "$UV" venv --python "$PYVER" "$HOME_DIR/venv" >/dev/null
fi
VIRTUAL_ENV="$HOME_DIR/venv" "$UV" pip install --upgrade --reinstall-package face-reframe "$HOME_DIR/src[app]" \
  || fail "library installation failed."
"$UV" cache prune --ci >/dev/null 2>&1 || true   # downloaded wheels are no longer needed
"$HOME_DIR/venv/bin/python" -c "import face_reframe.app.window" \
  || fail "the app won't start: try again, or open an issue at https://github.com/$REPO/issues"

# 4. the app bundle (made on this Mac, so macOS doesn't block it)
step "Creating the app in ${APPS_DIR}…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cat > "$APP/Contents/MacOS/Shorts Studio" <<EOF
#!/bin/bash
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:\$PATH"
exec "$HOME_DIR/venv/bin/python" -m face_reframe.app "\$@" >>"$HOME_DIR/app.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/Shorts Studio"
VERSION=$("$HOME_DIR/venv/bin/python" -c "import importlib.metadata as m; print(m.version('face-reframe'))" 2>/dev/null || echo "0")
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Shorts Studio</string>
  <key>CFBundleDisplayName</key><string>Shorts Studio</string>
  <key>CFBundleIdentifier</key><string>com.canardu1.shortsstudio</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>Shorts Studio</string>
  <key>CFBundleIconFile</key><string>ShortsStudio</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
EOF
if command -v iconutil >/dev/null 2>&1; then
  ICONSET="$TMP/ShortsStudio.iconset"
  if QT_QPA_PLATFORM=offscreen "$HOME_DIR/venv/bin/python" "$HOME_DIR/src/packaging/make_icon.py" "$ICONSET" >/dev/null 2>&1; then
    iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/ShortsStudio.icns" 2>/dev/null || true
  fi
fi
touch "$APP"

bold ""
bold "Done! Shorts Studio is in $APPS_DIR."
echo "Open it from Finder (your Applications folder) or Spotlight: search \"Shorts Studio\"."
echo "To update later, run the same command again."
[ "${SHORTS_STUDIO_NO_OPEN:-}" = "1" ] || open "$APP" 2>/dev/null || true
