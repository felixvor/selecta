#!/usr/bin/env bash
# Einmalig (und bei jedem Start ueber Selecta.bat) pro Laptop: legt eine venv
# in der WSL-eigenen Filesystem an und installiert Selecta editable aus
# diesem Ordner. Braucht Internet nur, wenn Pakete noch nicht installiert
# sind (pip installiert bereits erfuellte Requirements ohne Netzzugriff).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$HOME/.local/share/selecta/venv"

if ! python3 --version 2>&1 | grep -q "Python 3\.14\."; then
    echo "Fehler: Python 3.14 wird benoetigt (essentia-tensorflow gibt es nur dafuer)." >&2
    echo "Gefunden: $(python3 --version 2>&1)" >&2
    echo "Fix: WSL-Distro auf aktuelles Ubuntu-LTS aktualisieren (wsl --install -d Ubuntu)." >&2
    exit 1
fi

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
