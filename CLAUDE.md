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
  gewichtete Penalties), `rank_bridge` (Transition A→B; jedes Ergebnis
  trägt `parts_a`/`parts_b` = `pair_score_parts()`-Dicts mit den rohen
  Einzeltermen, Grundlage der Bridge-Warum-Zeile), Key-Parsing
  (Open Key `7m`/`12d` **und** Camelot `8A`/`8B`), BPM-Distanz mit
  Halb-/Doppeltempo. Notations-Utilities `note_to_camelot`/`note_to_openkey`/
  `key_to_pitch_class` (kanonischer Vergleich beider Räder — Achtung:
  Open Key `8m` = Bb-Moll, Camelot `8A` = A-Moll, gleiche Nummer ≠ gleicher
  Key; die Räder sind um 7 Positionen = 1 Halbton gegeneinander verschoben).
  `pair_score()` ist ein dünner Wrapper um `pair_score_parts()["score"]`.
- **`app.py`** — Textual-App: Chip-Zeile unter dem Track-Label
  (`chip_line`/`fmt_track_cell`: Genres als farbige Pills, Vibes/Jahr
  gedimmt; Zeilenhöhe 2 nur wenn Chips vorhanden; `played`-Flag dimmt das
  Label zusätzlich), `LibraryScreen`
  (Launcher mit ASCII-Logo: Libraries an/abwählen [Space/Klick], anlegen
  [A, `AddLibraryModal`], entfernen [D], analysieren [Ctrl+A], Enter
  startet alle aktiven; Persistenz `load_libraries`/`save_libraries`),
  `MainScreen` (Suche/Ranking/Transition; `self.played` = Session-
  Gedächtnis gespielter Filepaths, siehe Designentscheidungen; Ctrl+L
  pusht den LibraryScreen erneut mit `return_paths=True` und
  `_after_library_change()` hängt Query/Ziel auf die neuen Track-Dicts
  um), `AnalyzeModal` (bekommt eine **Liste** von Ordnern und analysiert
  sie sequenziell als je einen Subprozess). Transition-Modus:
  `#transition-bar` über der Tabelle (`fmt_transition_bar`: A links,
  B rechts, Direkt-Score dazwischen; sichtbar ab der Ziel-Auswahl, damit
  A nie aus dem Blick gerät) und `◆B`-Marker statt Ranknummer auf der
  Zeile des Ziels; die Statuszeile zeigt nur noch knapp "Transition"
  (keine Doppelung zur Bar). Bridge-Zellen `fmt_bpm_cell_ab`/
  `fmt_key_cell_ab` (Slash-Delta bzw. zwei Farbpunkte zu A/B, siehe
  Designentscheidungen) und `fmt_bridge_why_line` (Detail-Zeile mit
  Score-Zerlegung zu beiden Seiten). CSS inline in `SelectaApp.CSS`.
- **`map.py`** — Library-Map als selbstständige HTML-Datei (kein CDN,
  Canvas-JS inline, dunkler Hackerlook), bewusst **nur als CLI-Gimmick**
  (`selecta map`), NICHT in die TUI verdrahtet — siehe Designentscheidungen
  und TODOs. `project_2d()`: pacmap > umap > reine-numpy-PCA-Fallback
  (lazy imports, `pip install selecta[map]` für pacmap — optional,
  Kern-Installation/Tests laufen ohne). Projiziert wird NUR das
  L2-normierte Track-Embedding (`Library.matrix`), niemals Metadaten —
  die sind Anzeige-Kanäle (Farbe = Top-1-Genre über `_genre_color_hex`,
  eine map.py-eigene Kopie von `app.genre_chip_color`s Hash-Logik, um
  einen Kreisimport mit app.py zu vermeiden; Punktgröße = BPM; Tooltip =
  Chip-Infos).
- **`scripts/`** — nicht Teil des Pakets: `energy_eval.py` (Diagnose der
  Energie-Achse auf echten Library-CSVs: churn/discovery/direction-Metriken
  über Ranking-Varianten — Grundlage der z-Subspace-Designentscheidung),
  `demo_library.py` (fiktive Demo-Tracks + handgebaute Embedding-Cluster
  für README-Assets; erzeugt zwei Crates: demo-crate + warmup-crate, damit
  Launcher-Screenshots nicht nach Ein-Ordner-Tool aussehen),
  `make_screens.py` (SVG-Screenshots via Pilot → `docs/`; nach
  UI-Änderungen neu ausführen), `demo.tape` (VHS-Drehbuch fürs README-GIF;
  braucht installiertes `vhs`).
- **`cli.py`** — Entry Point `selecta` (TUI), Subcommand `selecta analyze`
  mit verstecktem `--porcelain`-Flag, Subcommand `selecta map --music-dir
  DIR [--music-dir DIR2 ...] [--out FILE] [--no-open]` (siehe map.py).
  Porcelain-Protokoll: `::progress i n`
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
- **Nur der Tag-Pfad ist parallelisiert** (`config.ANALYSIS_WORKERS` =
  Kerne/2, ProcessPoolExecutor in `run_analysis`): RhythmExtractor/
  KeyExtractor sind single-threaded und I/O-lastig → skaliert fast linear.
  Die Voll-Analyse bleibt bewusst sequenziell (TF parallelisiert einen
  Forward-Pass intern schon; mehr Prozesse = nur mehr RAM/Modell-Ladezeit
  und die Stage-Anzeige bräuchte ein Multi-Track-Konzept). CSV schreibt
  ausschließlich der Hauptprozess; Ergebnisse werden in Submit-Reihenfolge
  eingesammelt, Log/CSV bleiben deterministisch.
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
- **Suche schaltet die Ähnlichkeit nicht ab**: Tippen mit gesetzter Query
  zeigt Suchtreffer MIT Score-Spalten (`_show_scored_filter`; TOP_N ist
  reine Anzeige-Trunkierung, die Scores existieren für die ganze Library),
  sortiert nach Suchrelevanz statt Score — wer tippt, sucht einen Namen.
  Bewusst ohne harten BPM-Filter. In der Transition-Ziel-Auswahl zeigt
  `SCORE→A` den direkten Sprung von der aktuellen Query zum Kandidaten.
- **Transition-Sortierung nach `min(score_a, score_b)`** — der Engpass
  entscheidet, ob eine Brücke funktioniert. B läuft selbst als Kandidat mit
  (`score_b == 1.0`) und steht oben, sobald der Direktsprung am besten ist.
- **Bridge-Zellen kompakt statt zwei Spaltensätze**: BPM zu A UND B wird
  als eine Slash-Zelle gezeigt (`129 +2/−4`, `fmt_bpm_cell_ab`), Key als
  zwei Farbpunkte (`7m ●●`, `fmt_key_cell_ab`) — dieselben Farbschwellen
  wie die normalen Zellen, nur zweimal in einer Zelle. Mood-Distanz zu
  beiden Seiten steht NICHT in Spalten, sondern ausschließlich in der
  Detail-Zeile (`fmt_bridge_why_line`) — Breite bleibt fast wie im
  einfachen Ranking. Auf der `◆B`-Zeile (Kandidat ist das Ziel selbst)
  zeigt die B-Seite bewusst `—`/`·` statt `+0/−0`-Rauschen.
- **Suche bleibt im Transition-Modus die gleiche Suche**: Tippen mit
  gepinntem Ziel B bricht den Modus NICHT ab, sondern zeigt Fuzzy-Treffer
  mit den Bridge-Spalten (`_show_scored_filter`, Zweig `transition_target
  is not None`) — man kann jederzeit einen zu A passenden Kandidaten
  suchen, ohne die Transition zu verlassen.
- **Session-Gedächtnis „gespielt"** (`MainScreen.played`, nur RAM): jeder
  Track, der einmal Query (A) wurde, bekommt in allen Listen einen
  gedimmten `✓`-Präfix vor der Ranknummer (`fmt_rank_cell`) und ein
  gedimmtes Label — reine Anzeige, KEIN Ranking-Malus. Nur Anvisieren als
  Transition-Ziel zählt noch nicht als gespielt, erst wenn der Track
  selbst zu A wird. Passt zur „stateless"-Philosophie: Session-Kontext
  wie die Energie-Stufe, kein persistiertes Set.
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
- **Library-Map bewusst (noch) nicht in die TUI verdrahtet**: Ctrl+G +
  `action_map`/`_run_map` gab es kurz, wurde aber wieder ausgebaut — laut
  Nutzer braucht die Karte erst Suche/Filter nach Artist und eine
  vernünftige Legende, um wirklich nützlich zu sein, statt nur eine nette
  Spielerei zu sein. `selecta map` bleibt als CLI-Kommando bestehen
  (Subprozess-Architektur unverändert richtig: PaCMAP/UMAP/PCA können
  sekundenlang laufen). Falls die Karte doch in die TUI zurückkommt:
  NICHT Ctrl+M (im Terminal byte-identisch mit Enter/CR, nicht
  unterscheidbar) — siehe TODOs für den Rückbau-Stand.
  Projektion läuft NUR über das rohe Track-Embedding, nie über Metadaten
  gemischt — sonst weiß man bei zwei benachbarten Punkten nicht mehr, ob
  sie klanglich nah sind oder nur zufällig dieselbe BPM/Genre haben.
  Kein HDBSCAN/eigene Clusterung: Farbe kommt aus dem ohnehin vorhandenen
  Top-1-Genre. Kein Plotly/D3 — ~200 Zeilen eigenes Canvas-JS, damit die
  Datei wirklich ohne jede externe Ressource funktioniert (Test prüft
  `http(s)://` kommt nirgends vor). `pacmap` ist ein optionales Extra
  (`pip install selecta[map]`) und zieht `numpy<2.5`, was pip als
  Konflikt mit der exakten `numpy==2.5.1`-Pinnung meldet, aber trotzdem
  installiert (geprüft, funktioniert) — ohne das Extra läuft `project_2d`
  automatisch über den numpy-only-PCA-Fallback.

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
- Demo-GIF nach der Bridge-Zellen-Erweiterung noch nicht neu gerendert
  (auf Nutzerwunsch übersprungen) — vor dem nächsten Release-Push
  nachholen, sonst zeigt es den alten Transition-Screen ohne
  Slash-BPM/Key-Punkte.
- **Library-Map zurück in die TUI (Backlog, bewusst zurückgestellt):**
  `selecta map` ist ein CLI-Gimmick, der Nutzer fand die Karte "as-is"
  noch nicht nützlich genug für eine eigene Taste. Fehlt laut ihm für
  einen echten Mehrwert: Suche/Filter nach Artist direkt in der
  HTML-Seite (JS-seitig, kein Server — z.B. ein Textfeld, das Punkte
  dimmt/hervorhebt), eine bessere farbliche Hervorhebung (aktuell nur
  Top-1-Genre-Hash, evtl. mehrdeutig bei vielen Genres/Farbkollisionen).
  Erst wenn das steht, wieder einen Ctrl-Chord in app.py verdrahten
  (Code-Grundgerüst dafür existierte schon einmal, siehe Git-Historie
  Commit 89009cc — `action_map`/`_run_map` als Vorlage, falls
  Wiederverwendung sinnvoll ist).
- **Order-Set-Idee (Konzept, nicht umgesetzt):** Ctrl+O startet einen
  "Ordering"-Modus — mehrere Tracks (bis N) werden nacheinander
  ausgewählt/abgewählt (Suche/Ranking bleiben nutzbar, orientieren sich
  an "zuletzt hinzugefügt"), sichtbar in einer Leiste; "Fertig" berechnet
  die Reihenfolge mit minimalen Sprüngen zwischen den gewählten Tracks
  (im Kern ein kleines TSP über `pair_score`/`pair_score_parts`) und
  zeigt die Distanz zwischen jedem Schritt, optional mit
  Brücken-Vorschlägen für die größten verbleibenden Sprünge. Nutzer:
  "ich weiß, welche Songs ich spielen will, aber nicht in welcher
  Reihenfolge." Ausdrücklich als Diskussionsvorschlag markiert (nicht
  1:1 umzusetzen) — offene Fragen vor dem Bauen: eigener Modus oder
  Erweiterung des Transition-Modus (der ja schon A→B-Brücken kann)?
  Eigene UI-Leiste für die Auswahl oder reicht die Statuszeile? Lohnt
  sich exaktes TSP (bei kleinem N problemlos, z.B. Held-Karp) oder reicht
  ein gieriger Nearest-Neighbor-Ansatz, der besser zur "schlank, sofort
  nachvollziehbar"-Linie des Tools passt? Muss noch mit dem Nutzer
  abgestimmt werden, bevor hier Code entsteht.

Erledigt (2026-07-13): `IMPLEMENTATION_PLAN.md` (Gespielt-Markierung,
Library-Map, Transition-Deltas) umgesetzt — Session-Haken für gespielte
Tracks (`MainScreen.played`, reine Anzeige); Bridge-Zellen kompakt
(Slash-BPM, Key-Punkte, `fmt_bridge_why_line`), Suche bleibt im
Transition-Modus nutzbar; `selecta/map.py` + `selecta map`-Subcommand
(2D-Landkarte der Track-Embeddings, PaCMAP/UMAP/PCA-Fallback, kein CDN).
Ctrl+G-TUI-Anbindung noch am selben Tag wieder ausgebaut (Nutzer: Karte
braucht erst Suche/Filter/bessere Farben, siehe Backlog-TODO oben) —
`selecta map` bleibt als reines CLI-Kommando. 115 Tests grün. Plan-Datei
nach Umsetzung entfernt (Zweck erfüllt, Details stehen jetzt hier bzw.
im Code).

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
(per apt-get download entpackte Libs, kein Systemeingriff). Ablauf steht
im Kopf von `scripts/demo.tape`: erst `demo_library.py ~/Music
--seed-audio <echter Ordner> 4 --state /tmp/selecta-home` (7 Libraries
in Echteinsatz-Groesse; die 4 Seeds sind echte, umbenannte und
tag-gestrippte Dateien fuer den Analyse-Teil), dann `vhs
scripts/demo.tape`. Vor JEDEM Re-Render demo_library.py neu ausfuehren
(der Lauf im GIF schreibt die Seeds in die CSV). `~/Music` und
`/tmp/selecta-home` sind reine Demo-Artefakte und koennen jederzeit weg.

order set funktion: strg o startet "ordering" modus, man kann nun lieder selecten oder deselecten. selected songs werden übersichtlich oben oder an der seite rechts angezeigt. suche und ähnlichkeitsmaß funktionieren weiter und orientieren sich an "zuletzt hinzugefügt". bis zu N songs können hinzugefügt werden. dann "fertig" -> es rechnet / ladebalken (wenn nätig), ergebnis: wie die beste reihenfolge der songs aussehen würde um die sprünge zwischen den songs minimal zu halten. konzeptidee. ich weiß welche songs ich spielen will aber nicht in welcher reihenfolge. distanz zwischen jedem song wird anzeigt im ergebnis, (insert funktion um große sprünge die bleiben zu brücken?) ziemlich umfangreiche idee, große baustelle, muss nicht 1zu1 so umgesetzt werden, mit dem nutzer diskutieren und absprechen wie hier die eleganteste und charmantese lösung aussieht. muss zur vision und feeling des aktuellen programms passen, schlank easy nachvollziehbar. geht das überhaupt oder lieber scratchen?