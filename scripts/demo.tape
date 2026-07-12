# VHS-Drehbuch fuer das README-Demo-GIF -- reproduzierbar statt Screen-Recording.
#
# Braucht vhs (+ ttyd, ffmpeg): https://github.com/charmbracelet/vhs
#   statische Binaries nach ~/.local/bin reichen (kein sudo noetig).
#
# Ausfuehren aus dem Repo-Root in WSL (selecta-venv aktiv):
#   python scripts/demo_library.py /tmp/selecta-demo
#   vhs scripts/demo.tape
#
# Gezeigter Flow: Track suchen -> Query waehlen (Ranking) -> Energie hoch ->
# Ctrl+T -> Transition-Ziel pinnen (Bar + Brueckenkandidaten) -> Bruecke
# uebernehmen (wird neues A, Ziel bleibt).

Output docs/demo.gif

Set FontSize 15
Set Width 1200
Set Height 720
Set Padding 12
Set TypingSpeed 80ms

Hide
Type "selecta /tmp/selecta-demo/House"
Enter
Sleep 3s
Show

Sleep 1.5s
Type@110ms "velvet"
Sleep 1.2s
Enter
Sleep 3s

# Energie-Achse: zwei Stufen nach oben (haertere/schnellere Kandidaten)
Right
Sleep 1.2s
Right
Sleep 2.5s
Left
Sleep 0.8s
Left
Sleep 1.5s

Ctrl+T
Sleep 1.5s
Type@110ms "pressure"
Sleep 1.2s
Enter
Sleep 4s

# Top-Brueckenkandidat uebernehmen: wird neues A, Ziel bleibt gepinnt
Enter
Sleep 3.5s

Escape
Sleep 2s
