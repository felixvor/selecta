# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Was SELECTA ist

Terminal-Tool (Textual-TUI) für DJs: Ein Track dient als Query, die eigene
Library wird per Audio-Embedding-Cosine ähnlichkeitsgerankt, re-ranked über
BPM-, Key- und Mood-Penalties. Bewusst **stateless** — keine Playlists, keine
Sets, kein gespeicherter Pfad; das Set lebt in Rekordbox/Traktor.
Sprachkonvention: **UI-Strings, CLI-Ausgaben und README auf Englisch**
(OSS-Sichtbarkeit); Kommentare, Docstrings und Tests auf Deutsch.

## Umgebung & Kommandos

**Die App läuft nur unter Linux/WSL** — `essentia-tensorflow` gibt es nur als
Linux-Wheel, gepinnt auf Python 3.14. Auf diesem Windows-Rechner liegt die
venv in WSL unter `~/.local/share/selecta/venv` (angelegt von `setup.sh`,
das `Selecta.bat` bei jedem Start ausführt).

```bash
# Tests (unter Windows via WSL ausführen; Repo liegt auf /mnt/g):
wsl.exe -e bash -c 'source ~/.local/share/selecta/venv/bin/activate && cd /mnt/g/Media/Musik/selecta && python -m pytest -q'

# Einzelner Test:
... python -m pytest tests/test_similarity.py -q -k "bridge"

# App starten (in WSL):
selecta                                 # TUI, startet im LibraryScreen (Launcher)
selecta /mnt/g/Media/Musik/House        # TUI, ein Ordner ad-hoc (Library-Liste unangetastet)
selecta analyze --music-dir DIR         # headless Analyse (auch von der TUI als Subprozess genutzt)

# Dev-Install (in WSL):
pip install -e ".[dev]"
```

Die Tests brauchen **kein** Essentia/TensorFlow — alle Essentia-Imports sind
lazy (in Funktionen/`__init__`), getestet wird gegen synthetische CSVs
(`tests/conftest.py`: 3-dim-Embedding-Cluster house/techno/ambient).
TUI-Tests laufen headless über Textuals `run_test()`/Pilot
(`asyncio_mode = "auto"`, kein `@pytest.mark.asyncio` nötig).

Kein Linter/Formatter konfiguriert. Git: `eol=lf` erzwungen (`.gitattributes`),
weil `setup.sh` in WSL läuft — CRLF in Shell-Skripten hat schon einmal das
Setup gebrochen.

## Architektur

Datenfluss: `analysis.py` schreibt pro Musikordner eine
`library_analysis.csv` (Schema: `config.CSV_FIELDNAMES`; Embedding als
base64-float32) → `library.py` lädt einen **oder mehrere** Ordner als
`Library` (Tracks + zeilenweise L2-normalisierte Matrix; Dedupe über den
absoluten Pfad, wegen verschachtelter Libraries) → `similarity.py` rankt →
`app.py` rendert. Die Liste der gemerkten Libraries (Pfad + aktiv-Flag)
liegt in `~/.local/share/selecta/libraries.json`.

- **`config.py`** — alle Tuning-Konstanten: Modell-URLs, CSV-Schema,
  Ranking-Gewichte (`W_BPM/W_KEY/W_MOOD`), Energie-Schrittweiten,
  Score-Farbschwellen. Neue Stellschrauben gehören hierher.
- **`analysis.py`** — Modell-Download (essentia.upf.edu, mit Retry; auch
  die `.json`-Label-Dateien der Genre/Vibe-Heads), `EssentiaAnalyzer`
  (Discogs-EffNet-Track-Embedding 512-dim + Mood-/AV-Heads +
  `genre_discogs400` (400 Discogs-Styles) + `mtg_jamendo_moodtheme`
  (56 Tags, gefiltert über `VIBE_WHITELIST`)), Ableitung
  `pick_genres`/`pick_vibes`, Analyse-Loop `run_analysis()` mit
  log/progress/cancelled-**Callbacks**, damit derselbe Loop headless (CLI)
  und aus der TUI läuft. Resume über die CSV; `missing_parts()` (lebt in
  `library.py`, damit die Statusanzeigen dasselbe Kriterium nutzen)
  entscheidet pro Zeile: kein Embedding **oder kein effnet_embedding** →
  volle (teure) Analyse; kein BPM → billiger Tag-Re-Check +
  `RhythmExtractor2013`-Fallback.
- **`library.py`** — CSV-Persistenz (`load_csv_data`/`compact_csv` =
  dedupliziert + prune), ID3-Tag-Lesen (nur mp3), `prefix_aware_score` für
  die Tipp-Suche (letztes Wort zählt als Präfix, Rest fuzzy via rapidfuzz).
  `Library` nimmt einen Pfad oder eine Liste; `missing_parts()` ist DAS
  Vollständigkeits-Kriterium (auch für den Analyse-Lauf); `dir_status()`
  und `Library.status()` zählen vollständig/gesamt pro Ordner ohne
  Embeddings zu decodieren und müssen mit `missing_parts()` übereinstimmen
  — sonst zeigt das Analyse-Modal "0 open", obwohl der Lauf noch Zeilen
  anfasst (war ein echter Bug).
- **`similarity.py`** — `rank_similar` (Score = Embedding-Cosine −
  gewichtete Penalties), `rank_bridge` (Transition A→B), Key-Parsing
  (Open Key `7m`/`12d` **und** Camelot `8A`/`8B`), BPM-Distanz mit
  Halb-/Doppeltempo. Notations-Utilities `note_to_camelot`/`note_to_openkey`/
  `key_to_pitch_class` (kanonischer Vergleich beider Räder — Achtung:
  Open Key `8m` = Bb-Moll, Camelot `8A` = A-Moll, gleiche Nummer ≠ gleicher
  Key; die Räder sind um 7 Positionen = 1 Halbton gegeneinander verschoben).
- **`app.py`** — Textual-App: Chip-Zeile unter dem Track-Label
  (`chip_line`/`fmt_track_cell`: Genres als farbige Pills, Vibes/Jahr
  gedimmt; Zeilenhöhe 2 nur wenn Chips vorhanden), `LibraryScreen`
  (Launcher mit ASCII-Logo: Libraries an/abwählen [Space/Klick], anlegen
  [A, `AddLibraryModal`], entfernen [D], analysieren [Ctrl+A], Enter
  startet alle aktiven; Persistenz `load_libraries`/`save_libraries`),
  `MainScreen` (Suche/Ranking/Transition; Ctrl+L pusht den LibraryScreen
  erneut mit `return_paths=True` und `_after_library_change()` hängt
  Query/Ziel auf die neuen Track-Dicts um), `AnalyzeModal` (bekommt eine
  **Liste** von Ordnern und analysiert sie sequenziell als je einen
  Subprozess). Transition-Modus: `#transition-bar` über der Tabelle
  (`fmt_transition_bar`: A links, B rechts, Direkt-Score dazwischen;
  sichtbar ab der Ziel-Auswahl, damit A nie aus dem Blick gerät) und
  `◆B`-Marker statt Ranknummer auf der Zeile des Ziels; die Statuszeile
  zeigt nur noch knapp "Transition" (keine Doppelung zur Bar).
  CSS inline in `SelectaApp.CSS`.
- **`scripts/`** — nicht Teil des Pakets: `energy_eval.py` (Diagnose der
  Energie-Achse auf echten Library-CSVs: churn/discovery/direction-Metriken
  über Ranking-Varianten — Grundlage der z-Subspace-Designentscheidung),
  `demo_library.py` (fiktive Demo-Tracks + handgebaute Embedding-Cluster
  für README-Assets; erzeugt zwei Crates: demo-crate + warmup-crate, damit
  Launcher-Screenshots nicht nach Ein-Ordner-Tool aussehen),
  `make_screens.py` (SVG-Screenshots via Pilot → `docs/`; nach
  UI-Änderungen neu ausführen), `demo.tape` (VHS-Drehbuch fürs README-GIF;
  braucht installiertes `vhs`).
- **`cli.py`** — Entry Point `selecta` (TUI) und Subcommand `selecta analyze`
  mit verstecktem `--porcelain`-Flag. Porcelain-Protokoll: `::progress i n`
  (Fortschrittsbalken) und `::status {json}` (Events `track`/`stage`/`done`
  aus `run_analysis`; `done` trägt Genres/Vibes/BPM/Scores und speist die
  Chip-Ergebniszeile im Analyse-Log, `stage` die Live-Statuszeile mit den
  `EssentiaAnalyzer.STAGES`-Etappen). Ohne `status`-Callback (headless,
  nicht-porcelain) druckt `run_analysis` stattdessen lesbare `✓/~/≡/✗`-
  Zeilen über `log()` (`_human_line`). Ein echter Sub-Fortschritt pro Track
  ist nicht möglich — TensorFlow meldet innerhalb eines Forward-Pass
  nichts; die Etappengrenzen sind die ehrliche Auflösung.

## Bewusste Designentscheidungen (nicht "verbessern" ohne Grund)

- **Analyse läuft als Subprozess, nicht als Worker-Thread**: Essentia/TF
  halten den GIL sekundenlang, ein Thread würde die TUI einfrieren. Das
  `AnalyzeModal` startet `python -m selecta analyze --porcelain` und streamt
  stdout.
- **Key wird geschätzt, aber sichtbar als Schätzung** (`compute_key`,
  Profil `config.KEY_PROFILE=edmm`, fest 440 Hz, Notation
  `config.KEY_NOTATION`): Der frühere Befund "KeyExtractor liegt zu 85 %
  einen Halbton daneben" war ein Vergleichs-Artefakt (Open-Key- vs.
  Camelot-Nummern, um 7 Positionen = 1 Halbton versetzt; `scripts/
  key_eval.py` misst 0 % Halbton-Fehler, ~75 % exakt). Geschätzte Keys
  tragen das `key_estimated`-Flag, werden gedimmt mit `~`-Präfix
  angezeigt, fließen voll ins Ranking und werden von jedem später
  auftauchenden DJ-Tag überschrieben (Flag fällt). Tuning-Korrektur
  bewusst NICHT (verschlechtert die Quote). Fehlender Key triggert wie
  BPM den billigen `tags`-Pfad. Das Jahr bleibt Tag-only (nie berechnet),
  `vibes` darf leer sein — Vollständigkeits-Marker des Genre/Vibe-Schemas
  ist allein `effnet_embedding`, und `pick_genres` übernimmt Top-1 immer,
  damit `genres` nach einer Analyse nie leer ist.
- **`effnet_embedding` (1280-dim, gemittelt) wird mit persistiert**, damit
  künftige neue Classification-Heads als reiner CSV-Backfill laufen können
  (Head auf dem Mittel ≈ Mittel der Head-Outputs; für Tags ok) — ohne
  erneuten Audio-Decode. Genre/Vibe-Tags sind **display-only**, sie fließen
  nicht ins Ranking (Genre-Nähe steckt schon im Embedding-Cosine).
- **DJ-Software-Tags gewinnen immer**: Bei jedem Analyse-Lauf werden
  BPM/Key-Tags aus der Datei re-gelesen und überschreiben CSV-Werte —
  Rekordbox-Analyse gilt als hochwertiger als die eigene Schätzung.
- **Energie-Achse = Target-Shifting**, kein Score-Bonus für Extreme (sonst
  gewinnt immer der härteste Track der Library). Bereich ±6, weil dort die
  Zielverschiebung physisch sättigt. Bei `energy != 0` wird die Mood-Distanz
  z-normiert (Library-Std pro Dimension, `mood_scales`) und auf die
  tatsächlich verschobenen Dimensionen beschränkt (`ENERGY_DIM_MASK`:
  aggressive/relaxed/arousal — danceable ist in DJ-Libraries praktisch
  konstant, valence hat mit Energie nichts zu tun). Datengetrieben gewählt
  (`scripts/energy_eval.py`, 1426 echte Tracks): Monotonie der
  Energie-Antwort 0.79→0.96, doppelt so viele neu aufgedeckte Tracks,
  Embedding-Ähnlichkeit der Top-10 fällt nur 0.944→0.918. Bei `energy == 0`
  bleibt bewusst die rohe 5-dim-Distanz — das Default-Ranking ist
  unangetastet.
- **Transition-Sortierung nach `min(score_a, score_b)`** — der Engpass
  entscheidet, ob eine Brücke funktioniert. B läuft selbst als Kandidat mit
  (`score_b == 1.0`) und steht oben, sobald der Direktsprung am besten ist.
- **BPM-Filter (`,`/`.`) springt auf real vorhandene BPM-Werte** der
  Library, nicht in festen Schritten (`_next_bpm_offset`).
- **Tastatur-Konvention (fzf-Stil)**: Druckbare Tasten tippen IMMER in die
  Suche (nie Aktionen — sonst würde "**A**mbush" tippen die Analyse
  starten); Aktionen liegen auf Ctrl-Chords (`Ctrl+A`, `Ctrl+T`);
  `←→,.` sind nur bei leerem Suchfeld Aktionstasten. Umgesetzt über
  `SearchInput._on_key`/`ResultsTable._on_key`, die `ActionKey`-Messages an
  den Screen posten.
- **`Ctrl+C` beendet die App** (in Raw-Mode nur eine Taste, kein SIGINT) —
  verlustfrei, die CSV ist immer persistiert.
- **Keine eigene Pfad-Autovervollständigung** (wieder entfernt — sie war
  fragil und nie so gut wie die Shell): Der `AddLibraryModal` ist ein
  bewusst dummes Input-Feld; Drag & Drop eines Ordners ins Terminal pastet
  den Pfad (`resolve_music_dir` entfernt die dabei mitkommenden
  Anführungszeichen und übersetzt `C:\…` via wslpath). Nicht wieder
  einbauen — Pfadeingabe ist ein Einmal-Ereignis pro Library.
- **LibraryScreen ist EIN Screen, kein Menübaum** — Launcher-Charakter:
  Enter-Binding mit `priority=True` (sonst schluckt die DataTable das
  Enter als RowSelected); Klick toggelt, Enter startet. "Stateless"
  bezieht sich auf Sets/Playlists — die Library-Liste (libraries.json)
  ist Konfiguration und ist die einzige persistierte App-Einstellung.
- **Embedding-Modell**: `discogs_track_embeddings` mit Output
  `PartitionedCall:0` — der kontrastiv trainierte 512-dim Projektionsraum,
  nicht das Genre-Klassifikations-Embedding (`discogs-effnet`,
  `PartitionedCall:1`; letzteres füttert die Mood- und Genre/Vibe-Heads).
  Achtung: `genre_discogs400` ist anders exportiert als die übrigen Heads
  und braucht explizit `input="serving_default_model_Placeholder"`,
  `output="PartitionedCall:0"`; `mtg_jamendo_moodtheme` ist multi-label
  (`output="model/Sigmoid"`, kleine Aktivierungen, Schwelle in `config.py`).
- **TF-Logging**: `TF_CPP_MIN_LOG_LEVEL` muss VOR jedem Essentia-Import
  gesetzt sein (steht am Modulanfang von `analysis.py`); C++-seitiger
  Lärm wird zusätzlich über `filtered_stderr` weggefiltert.

## Stolperfallen

- Pfade: Die App läuft in WSL — Windows-Pfade (`C:\…`) werden in
  `resolve_music_dir()` via `wslpath` nach `/mnt/…` übersetzt. CSV-Zeilen
  speichern absolute Pfade im WSL-Format.
- `pyproject.toml` pinnt exakte Versionen (auch Textual `8.2.8`) — Textual-
  API-Verhalten (private `_on_key`-Overrides, `_suggestion`) hängt daran.
- Die Modelle (`models/`, ~48 MB inkl. Label-JSONs) sind nicht im Git
  (`.gitignore`); sie werden beim ersten Analyse-Lauf heruntergeladen.
  essentia.upf.edu hat gelegentlich SSL-Handshake-Timeouts — der Download
  hat Retries, ggf. einfach erneut starten.
- CSVs aus der Zeit vor dem Genre/Vibe-Schema (ohne `effnet_embedding`)
  werden beim nächsten Analyse-Lauf automatisch einmal voll re-analysiert —
  das ist gewollt (Migration), kein Bug.
- `~/.local/share/selecta/last_dir` (Vorgänger-Format, ein einziger
  gemerkter Pfad) wird von `load_libraries()` nur noch als einmalige
  Migration gelesen, wenn `libraries.json` fehlt.



## Offene TODOs (Nutzer):

- GIF-Erweiterung (optional): Launcher/Analyse-Lauf mit ins Drehbuch
  nehmen (Library anlegen, analysieren mit Skip zu 100%). Heikel: der
  Launcher liest die ECHTE libraries.json des Nutzers — dafür müsste
  demo.tape mit isoliertem HOME laufen.

Erledigt (2026-07-12): Live-Zähler im Analyselog statt statischem
"X of Y"-Satz; '? BPM ?'-Bug (Voll-Analyse rechnet BPM/Key jetzt selbst,
`_fill_missing_bpm_key`); Enter kopiert das Track-Label per OSC-52 in die
Zwischenablage; Energie-Achse auf z-Subspace-Distanz umgestellt (siehe
Designentscheidungen); Score-Zerlegung in der Detail-Zeile
(`fmt_why_line`); Demo-Assets: 2 prozedural gefüllte Crates (~235
Tracks), Launcher-Shot mit 7 fiktiven DJ-Libraries (Status-Cache wird in
make_screens direkt gesetzt, die Pfade existieren nicht);
`docs/demo.gif` erzeugt und in der README verlinkt. Verworfen: portable
Build ohne WSL (Essentia-Ersatz nötig, Embeddings würden inkompatibel).

GIF-Regenerierung: vhs/ttyd/ffmpeg liegen als statische Binaries in
`~/.local/bin` (WSL, ohne sudo installiert); vhs' Chromium braucht
`LD_LIBRARY_PATH=$HOME/.local/chromium-libs/usr/lib/x86_64-linux-gnu`
(per apt-get download entpackte Libs, kein Systemeingriff). Ablauf:
`python scripts/demo_library.py /tmp/selecta-demo && vhs scripts/demo.tape`.
