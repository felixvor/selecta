# VHS-Drehbuch fuer das README-Demo-GIF -- reproduzierbar statt Screen-Recording.
#
# Braucht vhs (+ ttyd, ffmpeg): https://github.com/charmbracelet/vhs
#   z.B.  sudo apt install ffmpeg  &&  go install github.com/charmbracelet/vhs@latest
#         (oder das vhs-Release-Binary nach ~/.local/bin legen)
#
# Ausfuehren aus dem Repo-Root in WSL (selecta-venv aktiv):
#   python scripts/demo_library.py /tmp/selecta-demo/demo-crate
#   vhs scripts/demo.tape
#
# Gezeigter Flow: Track suchen -> Query waehlen (Ranking) -> Ctrl+T ->
# Transition-Ziel pinnen (Bar + Brueckenkandidaten) -> Bruecke uebernehmen.

Output docs/demo.gif

Set FontSize 15
Set Width 1200
Set Height 720
Set Padding 12
Set TypingSpeed 80ms

Hide
Type "selecta /tmp/selecta-demo/demo-crate"
Enter
Sleep 3s
Show

Sleep 1.5s
Type@110ms "velvet"
Sleep 1.2s
Enter
Sleep 3s

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
