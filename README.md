# ◤ SELECTA ◢

Terminal-Helper zum Auflegen: findet zu einem Track die passendsten nächsten — über
Audio-Embeddings (Essentia `discogs_track_embeddings`, kontrastiv auf Track-Ähnlichkeit
trainiert), re-ranked nach BPM, Key und Mood.

## Voraussetzungen

Python 3.11+. `essentia-tensorflow` gibt es nur als Linux-Wheel — auf macOS
läuft SELECTA daher nicht, unter Windows über WSL2 (siehe unten).

## Setup (Linux / WSL)

```bash
cd ~/selecta
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Setup (Windows über WSL2)

Einmalig WSL2 mit einer Ubuntu-Distro installieren (`wsl --install -d Ubuntu-22.04`).
Danach diesen Ordner (z.B. per USB-Stick) irgendwohin kopieren und `Selecta.bat`
doppelklicken — richtet beim ersten Start automatisch venv + Abhängigkeiten in
WSL ein (`setup.sh`) und startet danach die TUI. Ein Ordner lässt sich auch
direkt auf `Selecta.bat` ziehen, um ihn als Musik-Ordner zu übergeben.

## Starten

```bash
selecta /pfad/zu/deiner/Musik       # TUI, Ordner als Argument
selecta                             # TUI, fragt nach dem Ordner
selecta analyze --music-dir DIR     # headless analysieren (z.B. übernacht)
```

Die Analyse-Ergebnisse liegen als `library_analysis.csv` im Musik-Ordner
(Resume-fähig, bestehende CSVs werden automatisch um Embeddings ergänzt).
Modelle landen einmalig in `./models` (~43 MB).

## Bedienung

Tippen filtert die Library, **Enter** übernimmt den markierten Track als Query —
die Liste zeigt dann die ähnlichsten Tracks. **Enter/Klick auf ein Ergebnis**
macht es zur neuen Query (so hangelt man sich durchs Set).

| Taste | Funktion |
|---|---|
| `↑` `↓` | Liste navigieren (Detail-Zeile unten zeigt den markierten Track) |
| `←` `→` | Energie-Achse −3…+3: verschiebt das Suchziel (BPM/Arousal/Härte) |
| `,` `.` | BPM-Feintuning ±4 aufs Suchziel |
| `Ctrl+A` | Ordner analysieren (Bestätigung → Progress + Log; neue Tracks, Backfill alter CSVs) |
| `Ctrl+T` | Transition-Planer: Kette von Song A nach Song B |
| `Esc` | Suche leeren / Screen zurück |
| `Ctrl+C` / `Ctrl+Q` | Beenden |

Buchstaben tippen immer in die Suche. `←→,.` wirken nur bei **leerem Suchfeld** —
mit Text im Feld bewegen sie den Cursor bzw. tippen normal.

Ergebnis-Spalten: BPM mit Δ (grün = direkt mixbar), KEY (grün = harmonisch),
SCORE (Embedding-Cosine minus Penalties), ΔENERG/ΔHÄRTE/ΔMOOD als Δ×10-Ganzzahl
(+4 = +0.4) relativ zur Query.

## Entwicklung

```bash
pip install -e ".[dev]"
pytest
```

Tuning-Konstanten (Energie-Schrittweiten, Gewichte, Top-N): `selecta/config.py`.

## Abhängigkeiten & Lizenz

Nutzt [Essentia](https://essentia.upf.edu) (`essentia-tensorflow`), das von der
Music Technology Group (UPF) unter AGPLv3 (nichtkommerziell) bzw. einer
kommerziellen Lizenz vertrieben wird — siehe
[Lizenzinfo](https://essentia.upf.edu/licensing_information.html), falls du
dieses Projekt kommerziell einsetzen willst.
