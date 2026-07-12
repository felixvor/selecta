# VHS-Drehbuch fuer das README-Demo-GIF -- reproduzierbar statt Screen-Recording.
#
# Braucht vhs (+ ttyd, ffmpeg): statische Binaries in ~/.local/bin reichen;
# vhs' Chromium braucht LD_LIBRARY_PATH auf ~/.local/chromium-libs (siehe
# CLAUDE.md). Ausfuehren aus dem Repo-Root in WSL (selecta-venv aktiv):
#
#   python scripts/demo_library.py ~/Music \
#       --seed-audio /mnt/g/Media/Musik/House/HumanMusic 4 \
#       --state /tmp/selecta-home
#   vhs scripts/demo.tape
#
# ~/Music (WSL-Home) haelt 7 Demo-Libraries in Echteinsatz-Groesse
# (~1800 Dummy-Dateien) -- authentische Pfade im Launcher-Bild.
# --seed-audio mischt 4 echte (umbenannte, tag-gestrippte) Dateien in die
# House-Crate, damit der Analyse-Teil echte ✓-Chip-Zeilen zeigt. Vor jedem
# Re-Render demo_library.py erneut ausfuehren -- der Lauf im GIF schreibt
# die 4 Zeilen in die Crate-CSV, sonst ist beim naechsten Mal nichts offen.
#
# Gezeigter Flow: Launcher (7 Libraries, Space-Toggle) -> Analyse (4 echte
# Tracks, Cut waehrend der Modell-Ladezeit) -> Enter startet die Session ->
# Suche -> Ranking -> Energie hoch/runter -> Kern-Loop (naechster Track
# wird neue Query) -> Ctrl+T -> Transition-Ziel pinnen -> Bruecke
# uebernehmen.

Output docs/demo.gif

Set FontSize 15
Set Width 1200
Set Height 720
Set Padding 12
Set TypingSpeed 80ms

# --- Setup (unsichtbar): isoliertes HOME, damit der Launcher nicht die
# echte libraries.json des Nutzers liest ---
Hide
Type "export HOME=/tmp/selecta-home && clear && selecta"
Enter
Sleep 4s
Show

# --- Launcher: Warmup-Library per Space dazuschalten ---
Sleep 2.5s
Down
Sleep 0.8s
Down
Sleep 0.8s
Space
Sleep 1.5s
Up
Sleep 0.6s
Up
Sleep 1s

# --- Analyse der House-Crate: 4 neue Tracks, Rest komplett ---
Ctrl+A
Sleep 2s
Enter
# Modell-Laden + erste Etappen kurz zeigen, dann Cut bis zum Ende
Sleep 12s
Hide
Sleep 240s
Show
Sleep 4s
Escape
Sleep 1.5s

# --- Session starten (beide aktiven Libraries) ---
Enter
Sleep 3s

# --- Suche + Ranking ---
Type@110ms "velvet"
Sleep 1.2s
Enter
Sleep 3s

# --- Energie-Achse: zwei Stufen hoch, wieder runter ---
Right
Sleep 1.2s
Right
Sleep 2.5s
Left
Sleep 0.8s
Left
Sleep 1.5s

# --- Kern-Loop: durch die Top-10 navigieren, passenden Track als neue
# Query uebernehmen (zweimal -- so entsteht ein Set) ---
Down
Sleep 0.9s
Down
Sleep 0.9s
Down
Sleep 1.4s
Enter
Sleep 3s
Down
Sleep 1s
Down
Sleep 1.4s
Enter
Sleep 3s

# --- Transition: Ziel pinnen, Bruecke uebernehmen ---
Ctrl+T
Sleep 1.5s
Type@110ms "pressure"
Sleep 1.2s
Enter
Sleep 4s
Enter
Sleep 3.5s

Escape
Sleep 2s
