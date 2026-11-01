#!/usr/bin/env bash
# Removes SpectraFit's environment and launchers. Your data, results and the chromophores you
# added (~/.spectrafit/chromophores) are kept unless you pass --all.
set -euo pipefail
PREFIX="${SPECTRAFIT_HOME:-$HOME/.spectrafit}"
rm -rf "$PREFIX/env"
rm -f "$PREFIX/README.html"
rm -f "$HOME/.local/bin/spectrafit" "$HOME/Desktop/SpectraFit.command" "$HOME/Desktop/SpectraFit.desktop" \
      "$HOME/.local/share/applications/spectrafit.desktop"
if [ "${1:-}" = "--all" ]; then
  rm -rf "$PREFIX"
  echo "SpectraFit removed, including your chromophore library."
else
  echo "SpectraFit removed. Your added chromophores are still in $PREFIX/chromophores (use --all to delete them too)."
fi
