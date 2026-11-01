#!/usr/bin/env bash
# SpectraFit installer for macOS and Linux.
#
#   Open a Terminal in this folder and run:   bash install.sh
#
# What it does (nothing outside your home folder is touched):
#   1. installs "uv", a small tool that downloads the right Python, if you don't have it
#   2. creates a private Python environment in ~/.spectrafit/env
#   3. installs SpectraFit and napari into it
#   4. adds a "SpectraFit" launcher (Desktop on macOS, applications menu on Linux)
#      and the "spectrafit" command for the Terminal
# To remove everything again:  bash uninstall.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${SPECTRAFIT_HOME:-$HOME/.spectrafit}"
ENV="$PREFIX/env"
PYTHON_VERSION="${SPECTRAFIT_PYTHON:-3.12}"
BIN="$HOME/.local/bin"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mInstallation failed: %s\033[0m\n' "$*" >&2; exit 1; }

say "SpectraFit installer — this takes a few minutes and needs an internet connection."
mkdir -p "$PREFIX" "$BIN"

# ---------------------------------------------------------------- 1. uv
if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
elif [ -x "$HOME/.cargo/bin/uv" ]; then
  UV="$HOME/.cargo/bin/uv"
else
  say "1/4  Installing uv (Python manager)…"
  if command -v curl >/dev/null 2>&1 && curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh; then
    UV="$HOME/.local/bin/uv"
  elif command -v python3 >/dev/null 2>&1 && { python3 -m pip install --user --quiet uv 2>/dev/null || python3 -m pip install --user --quiet --break-system-packages uv; }; then
    UV="$(python3 -c 'import site,os;print(os.path.join(site.USER_BASE,"bin","uv"))')"
  else
    fail "could not install uv. Install it from https://docs.astral.sh/uv/ and run this script again."
  fi
fi
[ -x "$UV" ] || fail "uv not found after installation ($UV)."
echo "Using uv: $UV"

# ---------------------------------------------------------------- 2. environment
say "2/4  Creating the Python $PYTHON_VERSION environment in $ENV …"
"$UV" venv --quiet --allow-existing --python "$PYTHON_VERSION" "$ENV" || fail "could not create the environment."

# ---------------------------------------------------------------- 3. packages
say "3/4  Installing SpectraFit and napari (the image viewer)…"
"$UV" pip install --quiet --python "$ENV/bin/python" "$HERE[gui]" || fail "package installation failed (see the messages above)."
"$ENV/bin/spectrafit" chromophores >/dev/null || fail "SpectraFit does not start."

# ---------------------------------------------------------------- 4. launchers
say "4/4  Adding launchers…"
ln -sf "$ENV/bin/spectrafit" "$BIN/spectrafit"
cp "$HERE/README.html" "$PREFIX/README.html" 2>/dev/null || true  # opened by the 'User guide' button

if [ "$(uname)" = "Darwin" ]; then
  LAUNCHER="$HOME/Desktop/SpectraFit.command"
  cat > "$LAUNCHER" <<EOF
#!/bin/bash
# Double-click to start SpectraFit.
exec "$ENV/bin/spectrafit" gui
EOF
  chmod +x "$LAUNCHER"
  echo "Desktop launcher: $LAUNCHER  (double-click it; a Terminal window opens alongside the program)"
else
  APPS="$HOME/.local/share/applications"
  mkdir -p "$APPS"
  cat > "$APPS/spectrafit.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=SpectraFit
Comment=Spectral unmixing of hyperspectral images
Exec=$ENV/bin/spectrafit gui
Terminal=false
Categories=Science;Education;
EOF
  if [ -d "$HOME/Desktop" ]; then
    cp "$APPS/spectrafit.desktop" "$HOME/Desktop/SpectraFit.desktop"
    chmod +x "$HOME/Desktop/SpectraFit.desktop"
    command -v gio >/dev/null 2>&1 && gio set "$HOME/Desktop/SpectraFit.desktop" metadata::trusted true 2>/dev/null || true
  fi
  echo "Launcher added to your applications menu (and Desktop)."
fi

case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo "Note: to use the 'spectrafit' command in a Terminal, add $BIN to your PATH (or use $ENV/bin/spectrafit)." ;;
esac

say "Done. Start SpectraFit with the launcher, or type:  spectrafit"
echo "Try the example first:  spectrafit example ~/spectrafit_example   then load ~/spectrafit_example/example_settings.yaml in the program."
