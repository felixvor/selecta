#!/usr/bin/env bash
# Einmalig (und bei jedem Start ueber Selecta.bat) pro Laptop: legt eine venv
# in der WSL-eigenen Filesystem an und installiert Selecta editable aus
# diesem Ordner. Braucht Internet nur, wenn Pakete noch nicht installiert
# sind (pip installiert bereits erfuellte Requirements ohne Netzzugriff).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/.local/share/selecta/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Lege venv an: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -e "$SCRIPT_DIR"

if [ "$#" -gt 0 ]; then
    MUSIC_DIR="$(wslpath -a "$1")"
    exec "$VENV_DIR/bin/selecta" "$MUSIC_DIR"
else
    exec "$VENV_DIR/bin/selecta"
fi
