# Implementierungsplan: Gespielt-Markierung + Library-Map + Transition-Deltas

Drei Features, unabhängig voneinander umsetzbar — **in dieser Reihenfolge**.
Teil A ist klein und risikofrei, Teil B ist das größere Stück, Teil C ist
UI-Feinarbeit im bestehenden Transition-Modus (KEIN Refactor).

Lies zuerst die CLAUDE.md (Konventionen!). Die wichtigsten Regeln, die hier
mehrfach greifen:

- UI-Strings/CLI-Ausgaben **Englisch**, Kommentare/Docstrings/Testnamen **Deutsch**.
- Druckbare Tasten tippen IMMER in die Suche; Aktionen liegen auf Ctrl-Chords.
- Tests laufen ohne Essentia/TensorFlow (synthetische CSVs, `tests/conftest.py`);
  neue Abhängigkeiten dürfen NIE beim Import von `selecta.*` gezogen werden
  (lazy imports in Funktionen).
- `pyproject.toml` pinnt exakte Versionen.
- Nach UI-Änderungen: `python scripts/make_screens.py` NICHT vergessen
  (Screenshots), aber erst am Ende, wenn alles grün ist.
- Testlauf: `wsl.exe -e bash -c 'source ~/.local/share/selecta/venv/bin/activate
  && cd /mnt/g/Media/Musik/selecta && python -m pytest -q'`

---

## Teil A — Gespielte Tracks markieren (Session-Gedächtnis, nur RAM)

**Ziel:** Jeder Track, der in dieser Session einmal Query war, ist ab dann in
allen Listen als „gespielt" erkennbar. Kein Persistieren, kein Ranking-Malus —
reine Anzeige. App-Neustart = leeres Gedächtnis (bewusst, stateless).

### Schritte

1. **`MainScreen.__init__`** (selecta/app.py): neues Feld
   `self.played: set[str] = set()`  — Filepaths aller bisherigen Queries.
   Deutscher Kommentar: Session-Gedächtnis, bewusst nur RAM.

2. **`MainScreen.select_track()`**: direkt am Anfang
   `self.played.add(track["filepath"])` — VOR den Transition-Fallunterscheidungen,
   damit auch Transition-Ziele/Brücken als gespielt zählen, sobald sie Query
   werden. Das Transition-Ziel beim Pinnen zählt NICHT als gespielt (es wurde
   noch nicht gespielt, nur anvisiert).

3. **Anzeige — zwei Stellen, ein Prinzip** („Rank-Spalte trägt den Haken,
   Label wird gedimmt, Rest der Zeile bleibt farbig"):
   - `fmt_track_cell(track)` bekommt einen zweiten Parameter
     `played: bool = False`. Wenn True: Label-Style `"bold dim"` statt
     `"bold"`. Chips unverändert lassen.
   - In den Zeilen-Bauschleifen (`show_filter`, `_show_scored_filter`,
     `show_results`, `_show_bridge_results`): Rank-Zelle wird
     `Text(f"✓{i+1}", style="dim")` wenn `t["filepath"] in self.played`,
     sonst wie bisher. Achtung in `_show_bridge_results`: die `◆B`-Zelle des
     Ziels hat Vorrang vor dem Haken.
   - Alle Aufrufer von `fmt_track_cell` anpassen (auch
     `fmt_analysis_log_line` ruft es NICHT — nicht anfassen).

4. **Kein** Eintrag in der Keybar (es gibt nichts zu drücken), aber die
   Spaltenbreite der `#`-Spalte verkraftet `✓100` — mit 100 Zeilen (TOP_N)
   testen.

### Tests (tests/test_app.py, Namen auf Deutsch)

- `test_gespielte_tracks_bekommen_haken`: Query wählen („groove a" + Enter),
  dann zweite Query aus der Ergebnisliste (Enter). Danach Suche leeren →
  Filter-Modus: die Rank-Zelle des ersten Tracks beginnt mit `✓`.
  (An die Zellen kommt man über `table.get_row_at(row)`.)
- `test_gespielt_ist_kein_ranking_malus`: Ergebnisliste vor/nach dem Spielen
  eines Kandidaten vergleichen — Reihenfolge identisch.

---

## Teil B — Library-Map (`selecta map`, Ctrl+G in der TUI)

**Ziel:** 2D-Karte aller Tracks der aktiven Libraries als **eine
selbständige HTML-Datei** (kein CDN, kein Framework), dunkler „Hackerlook",
Hover-Tooltip mit denselben Infos wie die TUI-Chipzeile. Öffnet automatisch
im Standardbrowser.

### Architektur-Entscheidungen (nicht neu diskutieren, sind gefallen)

- **Projektion: nur das 512-dim Track-Embedding.** Keine Metadaten in die
  Distanz mischen. BPM/Genre/Key sind visuelle Kanäle (Farbe/Größe/Tooltip).
- **Kein HDBSCAN/keine Clusterung.** Farbe = Top-1-Genre über die bestehende
  `genre_chip_color()`-Zuordnung (Konsistenz mit der TUI!).
- **Kein Plotly/D3.** Handgeschriebenes Canvas-JS inline im HTML-Template
  (~200 Zeilen). Schwarzer Grund, Neon-Punkte, Monospace-Tooltip, Zoom
  (Mausrad) + Pan (Drag).
- **Als Subprozess/CLI-Subcommand**, nicht im TUI-Prozess rechnen — gleiche
  Begründung wie bei der Analyse (schwere Numerik würde die TUI einfrieren;
  außerdem gibt es den Headless-Weg gratis dazu).
- **Wichtig, Tastenwahl:** `Ctrl+M` geht NICHT — ^M ist im Terminal identisch
  mit Enter (Carriage Return), Textual kann das nicht unterscheiden.
  Stattdessen **`Ctrl+G`** („geo/graph"). In Keybar und README so
  dokumentieren.

### Abhängigkeiten (pyproject.toml, optionales Extra)

Neues Extra `map` — NICHT in die Kern-Dependencies:

```toml
[project.optional-dependencies]
map = ["pacmap==<aktuellste, die unter Python 3.14/Linux installierbar ist>"]
```

Entscheidungsbaum beim Pinnen (ausprobieren, in dieser Reihenfolge):
1. `pacmap` (beste globale Struktur — „Inseln" bleiben Inseln).
2. Falls pacmap unter Python 3.14 nicht installierbar/kaputt: `umap-learn`.
3. **Immer** als Code-Fallback: PCA auf 2D über `numpy.linalg.svd`
   (keine Zusatz-Dependency). Wenn weder pacmap noch umap importierbar sind,
   läuft die Map mit PCA und druckt einen Hinweis:
   `"pacmap not installed — falling back to PCA (pip install selecta[map])"`.

Der Import von pacmap/umap passiert **lazy in der Funktion**, nie am
Modulkopf (Tests + Kern-Installation ohne Extra müssen sauber bleiben).

### Neue Datei `selecta/map.py`

```python
def project_2d(matrix: np.ndarray) -> np.ndarray:
    """(N x dim) -> (N x 2). pacmap > umap > PCA-Fallback (lazy imports).
    Rueckgabe auf [0,1]^2 min-max-normalisiert."""

def build_map_html(tracks: list[dict], coords: np.ndarray) -> str:
    """Fuellt das HTML-Template. Pro Track ein JSON-Objekt:
    x, y, label (track_label), bpm, key (display_key-Logik: ~ bei
    key_estimated), genres, vibes, year, arousal, aggressive, danceable,
    color (genre_chip_color des Top-1-Genres, auf Hex gemappt)."""

def write_map(music_dirs, out_path=None) -> Path:
    """Library laden (bestehende Library-Klasse!), projizieren, HTML
    schreiben. Default-Zielpfad: erster music_dir / 'selecta_map.html'
    -- liegt damit in WSL-Faellen auf /mnt/..., sonst kommt der
    Windows-Browser nicht an die Datei."""

def open_in_browser(path: Path) -> None:
    """Oeffnen, robust fuer WSL: erst wslview, dann explorer.exe mit
    wslpath -w (nur fuer /mnt-Pfade moeglich), dann xdg-open, zuletzt
    webbrowser.open. Fehler nur loggen, nie crashen."""
```

Hinweise:
- Rich-Farbnamen aus `GENRE_CHIP_COLORS` (config.py) müssen für die Map als
  Hex vorliegen. Kleines festes Mapping-Dict `RICH_TO_HEX` in map.py anlegen
  (10 Einträge, Werte aus der Rich-Farbtabelle abschreiben). NICHT versuchen,
  Rich zur Laufzeit zu befragen — unnötige Kopplung.
- Punktgröße im JS aus BPM ableiten (min→3px, max→7px, fehlend→3px), nicht
  in Python vorrechnen — die Daten liegen ja im JSON.
- JSON in ein `<script type="application/json">`-Tag einbetten,
  `json.dumps(..., ensure_ascii=False)`; die Titel enthalten Umlaute/„&".
  `</script>`-Sequenz in Strings escapen (`<\/`).

### HTML-Template (inline als Python-String in map.py, kein separates File)

Muss-Kriterien:
- Ein einziges `<canvas>`, volle Fenstergröße, `background: #0a0a0a`.
- Kopfzeile (position: fixed): `SELECTA MAP — <N> tracks — <projection>`
  in Monospace, magenta/grün-Akzente wie das TUI-Logo.
- Punkte mit leichtem Glow (`shadowBlur`), Farbe aus dem Track-JSON.
- Hover (Distanz < 8px im Screen-Space): Tooltip-Box neben dem Cursor,
  Monospace, Reihenfolge: `Artist - Title`, dann `128.0 BPM · 7m · 2021`,
  dann Genres, dann Vibes, dann `arous/aggr/dance`-Zeile. Bei
  `key_estimated`: `~7m` und gedimmt (opacity).
- Zoom auf Mausposition (Mausrad), Pan per Drag. Kein Reset-Button nötig,
  Doppelklick = View zurücksetzen.
- Legende unten links: die vorkommenden Top-Genres mit ihren Farben
  (max. ~12, Rest unter „other" in grau).
- Alles inline, keine externen Requests (muss offline im file://-Kontext
  laufen).

### CLI (selecta/cli.py)

Neuer Subcommand:

```
selecta map --music-dir DIR [--music-dir DIR2 ...] [--out FILE] [--no-open]
```

- Mehrere `--music-dir` erlaubt (wie die Library mehrere Ordner nimmt).
- Druckt den geschriebenen Pfad; öffnet den Browser außer bei `--no-open`.
- `--porcelain` braucht es NICHT (kein Fortschritts-Streaming; die
  Projektion hat keinen sinnvollen Zwischenfortschritt).

### TUI-Anbindung (selecta/app.py)

- `MainScreen`: Binding/ActionKey `ctrl+g` → `action_map()`.
  In `SearchInput._on_key`/`ResultsTable._on_key` muss `ctrl+g` wie die
  anderen Ctrl-Chords als `ActionKey` durchgereicht werden (Konvention
  beachten, siehe `_on_action_key`).
- `action_map()`: Statuszeile auf `"creating map …"`, dann Subprozess
  `python -m selecta map --music-dir ... (alle aktiven) --no-open`,
  bei Exit-Code 0 `open_in_browser()` vom TUI-Prozess aus aufrufen und
  `self.app.notify(f"Map written to {path}")`. Kein Modal, kein Menü —
  eine Taste, ein Ergebnis. Als `@work`-Worker wie `_stream_analysis`
  (async subprocess), damit die TUI bedienbar bleibt.
- Keybar-Eintrag: `^g map` (kurz halten).

### Tests (ohne pacmap/umap — CI/Testumgebung hat nur das Kern-Setup!)

tests/test_map.py, Namen deutsch:
- `test_pca_fallback_projiziert_auf_2d`: 6 synthetische Tracks
  (conftest-Cluster), `project_2d` per erzwungenem Fallback (monkeypatch:
  Import von pacmap/umap schlägt fehl) → Shape (6,2), Werte in [0,1].
- `test_html_enthaelt_alle_tracks_und_kein_cdn`: `build_map_html` →
  jeder `track_label` kommt im HTML vor; `http://` und `https://` kommen
  NICHT vor (Offline-Garantie); `</script>`-Escaping mit einem Titel wie
  `Foo</script>Bar` prüfen.
- `test_default_zielpfad_liegt_im_music_dir`: `write_map(tmp_path)` →
  Datei existiert unter `tmp_path/selecta_map.html`.
- App-Test: `test_ctrl_g_startet_map_worker` — monkeypatch auf den
  Subprozess-Aufruf (nicht wirklich rechnen), prüfen dass ActionKey
  ankommt und die Statuszeile sich ändert.

### Stolperfallen (gesammelt, bitte ernst nehmen)

1. `Ctrl+M` = Enter im Terminal. Deshalb Ctrl+G. Nicht „nur mal testen".
2. HTML-Zielpfad muss für den Windows-Browser erreichbar sein → Default in
   den Musikordner (liegt praktisch immer auf /mnt/...). NIE nach /tmp
   schreiben und dann explorer.exe aufrufen.
3. pacmap/numba: Erst-Import kompiliert JIT-Code und kann 10-30 s dauern —
   passiert im Subprozess, also ok, aber nicht wundern.
4. `Library` lädt nur Zeilen MIT Embedding — Tracks ohne Analyse fehlen auf
   der Karte. Das ist korrekt so; im HTML-Header die Zahl ehrlich anzeigen
   (`823 of 1306 analyzed tracks mapped` wenn Library.status() weniger
   Vollständige meldet als Dateien existieren).
5. 1000+ Punkte mit Glow: Canvas-Redraw pro Mousemove ist zu teuer, wenn
   naiv. Punkte einmal auf ein Offscreen-Canvas zeichnen, beim
   Pan/Zoom/Hover nur blitten + Highlight drüber.
6. Umlaute/Sonderzeichen in Titeln (echte Libraries: „Sørensen", „&") —
   `ensure_ascii=False` + korrektes `<meta charset="utf-8">`.

---

## Teil C — Transition-Modus: BPM/Key/Mood-Deltas zu A UND B, kompakt

**Ziel:** Brückenkandidaten zeigen nicht nur `SCORE→A`/`SCORE→B`, sondern
auch, WORAN ein Kandidat zu welcher Seite hängt (BPM-, Key-, Mood-Abstand
je zu A und zu B) — ohne die Tabellenbreite zu verdoppeln. Außerdem soll
sich der Transition-Modus wie die Hauptsuche anfühlen: Suche/Enter-Loop
funktionieren identisch, es gibt eben nur zusätzlich ein Ziel B.

**Ausdrücklich KEIN Refactor.** Der Modus bleibt wie er ist (gleicher
Screen, `transition_target` gesetzt). Die bestehende Semantik ist schon
richtig: Enter re-anchort A und behält B; Enter auf B beendet die
Transition. Es geht nur um Anzeige.

### Kompakt-Prinzip: Slash-Zellen + Detail-Zeile, keine neuen Spalten

1. **BPM-Zelle im Bridge-Modus**: statt `129 (+2)` (nur vs. A) jetzt
   `129 +2/−4` — erster Wert Delta zu A, zweiter zu B. Färbung pro Delta
   einzeln (bestehende Schwellen aus `fmt_bpm_cell` wiederverwenden:
   grün ≤2 %, gelb ≤6 %, sonst default). Neue Funktion
   `fmt_bpm_cell_ab(track, a, b)` neben der bestehenden — die bestehende
   NICHT umbauen (sie wird von drei anderen Modi genutzt).
   Halb-/Doppeltempo wie überall über `relative_bpm_distance` bewerten,
   angezeigt wird aber das rohe Delta.

2. **Key-Zelle im Bridge-Modus**: `7m ●●` — der Key plus zwei Punkte:
   erster Punkt = harmonische Kompatibilität zu A, zweiter zu B. Farben
   aus derselben Logik wie `fmt_key_cell` (grün = harmonic_distance ≤1,
   gelb = 2, rot/dim sonst; `None` → grauer Punkt `·`). Neue Funktion
   `fmt_key_cell_ab(track, a, b)`. Das ist die kompakteste ehrliche
   Darstellung — zwei Farbpunkte statt zwei Spalten.

3. **Mood: NICHT als Spalte.** Mood-Distanz zu A und B kommt in die
   **Detail-Zeile** (unten, cursorabhängig): Bridge-Modus bekommt eine
   eigene Warum-Zeile `fmt_bridge_why_line(result, a, b)`:
   `→A 0.912 = cos .95 − bpm .01 − key .00 − mood .02   →B 0.874 = cos .93 − …`
   Dafür in `rank_bridge` (similarity.py) die Einzelterme mitgeben:
   pro Kandidat `bpm_pen_a/b`, `key_pen_a/b`, `mood_dist_a/b`, `cos_a/b`
   (analog zu dem, was `rank_similar` seit der Warum-Zeile liefert —
   `pair_score` dazu in eine Variante zerlegen, die die Terme zurückgibt,
   z.B. `pair_score_parts(a, b) -> dict`; `pair_score` selbst bleibt als
   dünner Wrapper erhalten, damit bestehende Aufrufer/Tests unberührt
   bleiben). `_row_results` im Bridge-Modus befüllen (bisher None).

4. **Suche im Transition-Modus = gleiche Suche, nur mit B-Spalten:**
   `_show_scored_filter` prüft `self.transition_target`: wenn gesetzt
   (und nicht `_selecting_target`), zeigen Suchtreffer die
   Bridge-Spalten (`SCORE→A`/`SCORE→B` + Slash-BPM + Punkt-Key) statt der
   normalen Score-Spalten — über `rank_bridge`-Ergebnisse für genau die
   Fuzzy-Treffer (per Filepath-Lookup wie beim scored filter, `top` auf
   Library-Größe). Sortierung bleibt Suchrelevanz. Damit gilt: „Suche
   bleibt gleich, man hat halt ein Target" — abbrechen und neu suchen ist
   nie nötig.

### Spaltenbreite (das war die Sorge)

Vorher: `#  TRACK  BPM  KEY  SCORE→A  SCORE→B`
Nachher: identische Spaltenzahl — BPM-Zelle wird ~4 Zeichen breiter
(`+2/−4`), KEY-Zelle 3 Zeichen (` ●●`). Mood kostet 0 Breite (Detail-
Zeile). Kein zweiter Spaltensatz.

### Tests (tests/test_app.py bzw. test_similarity.py)

- `test_pair_score_parts_summiert_zum_pair_score`: für zwei conftest-Tracks
  `pair_score_parts()['score'] == pair_score()` (Float-Toleranz).
- `test_bridge_zeigt_slash_deltas`: Transition aufbauen (wie
  `test_..._transition`-Bestand), BPM-Zelle des ersten Kandidaten enthält
  `/`; KEY-Zelle enthält `●`.
- `test_bridge_detailzeile_zeigt_beide_seiten`: `#detail`-Static enthält
  `→A` und `→B` nach Cursor-Bewegung im Bridge-Modus.
- `test_suche_im_transition_modus_zeigt_score_b`: Query + Ziel pinnen,
  dann tippen → Spalten-Labels enthalten `SCORE→B` (Suche bricht den
  Modus nicht ab).

### Stolperfallen

1. `fmt_bpm_cell`/`fmt_key_cell` werden von Filter-/Result-/Target-Modus
   geteilt — NICHT anfassen, neue `*_ab`-Varianten daneben stellen.
2. `◆B`-Zeile (das Ziel als Kandidat): Slash-Deltas zu B wären dort
   `+0/−0`-Rauschen — für die Zeile des Ziels die B-Seite als `—` rendern.
3. Die Transition-Bar oben zeigt weiterhin nur den Direkt-Score A→B —
   unverändert lassen, sie ist bewusst knapp.
4. `rank_bridge` wird auch von der Suche (Punkt 4) mit `top=len(tracks)`
   gerufen — der Python-Loop läuft dann über die ganze Library pro
   Tastendruck. Bei ~2000 Tracks ok (wie scored filter); nichts
   vorzeitig optimieren.

---

### Abnahme-Checkliste

- [ ] `python -m pytest -q` grün, ohne pacmap installiert.
- [ ] `selecta map --music-dir /mnt/g/Media/Musik/House/HumanMusic` schreibt
      HTML und öffnet den Windows-Browser (auf diesem Rechner in WSL).
- [ ] Karte zeigt sichtbare Genre-Inseln (House vs. PsyTek-Ordner zusammen
      laden — zwei klar getrennte Kontinente = Erfolg).
- [ ] Hover zeigt Chip-Infos inkl. `~`-Key bei geschätzten Keys.
- [ ] TUI: Ctrl+G → „creating map …" → Browser öffnet, TUI bleibt bedienbar.
- [ ] Tippen von „g" ohne Ctrl landet weiterhin in der Suche.
- [ ] README: kurzer Abschnitt „Library map" mit Screenshot (PNG des
      Browser-Fensters, docs/map.png) + Ctrl+G in der Key-Tabelle.
- [ ] Teil C: Bridge-Tabelle zeigt `+2/−4`-BPM und `●●`-Key; Detail-Zeile
      `→A …  →B …`; Tippen im Transition-Modus behält B und zeigt
      `SCORE→B`; Tabellenbreite bleibt einspaltig pro Wert.
- [ ] CLAUDE.md: Map-Abschnitt in Architektur (selecta/map.py) ergänzen,
      Teil-C-Verhalten unter Designentscheidungen dokumentieren,
      TODO-Einträge entfernen.
- [ ] `python scripts/make_screens.py` + GIF-Regenerierung laut CLAUDE.md,
      da sich der Transition-Screenshot ändert.
