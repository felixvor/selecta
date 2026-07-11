"""SELECTA Textual-App: Such-Screen, Analyse-Modal, Transition-Modus.

Bedienlogik MainScreen (fzf-Stil: Suche haelt den Fokus, Aktionen auf Chords):
- Tippen filtert die Library; druckbare Tasten tippen IMMER (nie Aktionen,
  sonst wuerde "Ambush" die Analyse starten).
- Ctrl+A = Analyse -- jederzeit, auch beim Tippen.
- Ctrl+T pinnt ein Transition-Ziel B (Fuzzy-Suche + Enter): Die Liste zeigt
  dann Brueckenkandidaten mit Score zu beiden Seiten, sortiert nach der
  schwaecheren. Enter re-anchort A und behaelt B; Enter auf B selbst,
  Esc oder erneutes Ctrl+T beenden den Modus. Bewusst stateless -- kein
  gespeicherter Pfad, das Set lebt in der DJ-Software.
- Bei LEEREM Suchfeld sind ←/→ (Energie) und ,/. (BPM) Aktions-Tasten;
  mit Text im Feld bewegen sie den Cursor bzw. tippen.
- ↑/↓/Enter laufen immer auf der Ergebnisliste (Cursor + Auswahl).
- Enter/Klick auf eine Zeile macht den Track zur neuen Query.
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.suggester import Suggester
from textual.widgets import DataTable, Input, ProgressBar, RichLog, Static

from .config import (
    ENERGY_MAX,
    ENERGY_MIN,
    SCORE_COLOR_STEPS,
    TOP_N,
)
from .library import Library, fuzzy_search, track_label
from .similarity import (
    harmonic_distance,
    pair_score,
    rank_bridge,
    rank_similar,
    relative_bpm_distance,
    _to_float,
)

LOGO = "◤ SELECTA ◢"

# Tasten, die nur bei leerem Suchfeld (bzw. auf der Tabelle) Aktionen ausloesen
# -- keine Buchstaben, mit denen ein Titel beginnen koennte.
ACTION_KEYS = {"left", "right", "comma", "full_stop"}
# Chords, die IMMER Aktionen ausloesen (fzf-Konvention: Aktionen auf Ctrl).
CTRL_ACTION_KEYS = {"ctrl+a", "ctrl+t"}

FILTER_COLUMNS = ("#", "TRACK", "BPM", "KEY")
RESULT_COLUMNS = ("#", "TRACK", "BPM", "KEY", "SCORE", "ΔENERG", "ΔHÄRTE", "ΔMOOD")
BRIDGE_COLUMNS = ("#", "TRACK", "BPM", "KEY", "SCORE→A", "SCORE→B")

SEARCH_PLACEHOLDER = "Track suchen … (tippen zum Filtern)"
TARGET_PLACEHOLDER = "Transition-Ziel suchen … (tippen zum Filtern)"


# ---------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------

def fmt_delta10(value) -> Text:
    """Mood-Delta als Δ×10-Ganzzahl: +0.4 -> '+4' (spart das redundante '0.')."""
    if value is None:
        return Text("·", style="dim")
    scaled = round(value * 10)
    style = "orange1" if scaled > 0 else ("cyan" if scaled < 0 else "dim")
    return Text(f"{scaled:+d}" if scaled else "0", style=style)


def score_style(score: float) -> str:
    for threshold, style in SCORE_COLOR_STEPS:
        if score >= threshold:
            return style
    return "red"


def fmt_score_cell(score: float) -> Text:
    return Text(f"{score:.3f}", style=score_style(score))


def fmt_bpm_cell(track: dict, query: dict | None) -> Text:
    bpm = _to_float(track.get("bpm"))
    if bpm is None:
        return Text("?", style="dim")
    if query is None:
        return Text(f"{bpm:.0f}")
    q_bpm = _to_float(query.get("bpm"))
    if q_bpm is None:
        return Text(f"{bpm:.0f}")
    rel = relative_bpm_distance(q_bpm, bpm)
    style = "green" if rel <= 0.02 else ("yellow" if rel <= 0.06 else "dim")
    return Text(f"{bpm:.0f} ({bpm - q_bpm:+.0f})", style=style)


def fmt_key_cell(track: dict, query: dict | None) -> Text:
    key = (track.get("key") or "").strip()
    if not key:
        return Text("?", style="dim")
    if query is None:
        return Text(key)
    rel = harmonic_distance(query.get("key"), key)
    if rel is None:
        return Text(key, style="dim")
    style = "green" if rel <= 1 else ("yellow" if rel == 2 else "dim")
    return Text(key, style=style)


def fmt_detail_line(track: dict) -> str:
    def v(name, digits=2):
        value = _to_float(track.get(name))
        return "?" if value is None else f"{value:.{digits}f}"

    return (
        f"aggr {v('aggressive')}  happy {v('happy')}  sad {v('sad')}  relax {v('relaxed')}  "
        f"party {v('party')}  dance {v('danceable')}  aprch {v('approachability')}  "
        f"engag {v('engagement')}  arous {v('arousal', 1)}  valen {v('valence', 1)}"
        f"   [dim]{track['filepath']}[/]"
    )


def query_title(track: dict) -> str:
    bpm = track.get("bpm") or "?"
    key = track.get("key") or "?"
    return f"Ähnlich zu: {track_label(track)}  [{bpm} BPM | {key}]"


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

class ActionKey(Message):
    """Aktions-Taste (Energie/BPM/Analyse/Transition) aus Input oder Tabelle."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__()


class SearchInput(Input):
    """Input, das Aktions-Chords immer und die Pfeil-/BPM-Tasten bei leerem
    Feld an den Screen abgibt (Ctrl+A wuerde sonst von Inputs eigenem
    Cursor-Handling geschluckt)."""

    async def _on_key(self, event: events.Key) -> None:
        if event.key in CTRL_ACTION_KEYS or (not self.value and event.key in ACTION_KEYS):
            event.stop()
            event.prevent_default()
            self.post_message(ActionKey(event.key))
            return
        await super()._on_key(event)


class ResultsTable(DataTable):
    """Tabelle, die Aktions-Tasten meldet und Tipp-Eingaben zurueck ins
    Suchfeld leitet (falls sie per Klick den Fokus bekommen hat)."""

    async def _on_key(self, event: events.Key) -> None:
        key = event.key
        if key in ACTION_KEYS or key in CTRL_ACTION_KEYS:
            event.stop()
            event.prevent_default()
            self.post_message(ActionKey(key))
            return
        if (event.is_printable and event.character) or key == "backspace":
            inputs = self.screen.query(SearchInput)
            if inputs:
                search = inputs.first()
                search.focus()
                if key == "backspace":
                    search.action_delete_left()
                else:
                    search.insert_text_at_cursor(event.character)
                event.stop()
                event.prevent_default()
                return
        await super()._on_key(event)


# ---------------------------------------------------------------------------
# MainScreen: die Suche
# ---------------------------------------------------------------------------

class MainScreen(Screen):
    BINDINGS = [
        Binding("up", "cursor(-1)", show=False),
        Binding("down", "cursor(1)", show=False),
        Binding("pageup", "cursor(-10)", show=False),
        Binding("pagedown", "cursor(10)", show=False),
        Binding("escape", "clear_search", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.query_track: dict | None = None
        self.transition_target: dict | None = None  # gepinntes Ziel B (Ctrl+T)
        self._selecting_target = False  # Ctrl+T gedrueckt, B noch nicht gewaehlt
        self.energy = 0
        self.bpm_offset = 0
        self._row_tracks: list[dict] = []
        self._results_shown = False
        self._status_cache: tuple[int, int] | None = None

    # --- Aufbau ---

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="topbar"):
                yield Static(id="logo")
                yield Static(id="status", markup=True)
            yield SearchInput(placeholder=SEARCH_PLACEHOLDER, id="search")
            yield ResultsTable(id="results")
            yield Static(id="detail", markup=True)
            yield Static(id="keybar", markup=True)

    def on_mount(self) -> None:
        table = self.query_one(ResultsTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self.query_one("#logo", Static).update(Text(LOGO, style="bold magenta"))
        self.query_one("#keybar", Static).update(
            "[b]Enter[/b] wählen  [b]↑↓[/b] navigieren  [b]←→[/b] Energie  [b],[/b][b].[/b] BPM-Filter  "
            "[b]^a[/b] Analyse  [b]^t[/b] Transition  [b]Esc[/b] leeren  [b]^c[/b] Ende"
        )
        self._update_header()
        self.refresh_status()
        self.show_filter("")
        self.query_one(SearchInput).focus()

    @property
    def library(self) -> Library:
        return self.app.library

    # --- Header / Statuszeile ---

    def refresh_status(self) -> None:
        """Ordner-Scan (os.walk, auf /mnt/g potenziell langsam) im Thread --
        nur beim Start und nach einer Analyse noetig."""
        self.run_worker(self._scan_status, thread=True, exclusive=True, group="status")

    def _scan_status(self) -> None:
        status = self.library.status()
        self.app.call_from_thread(self._apply_status, status)

    def _apply_status(self, status: tuple[int, int]) -> None:
        self._status_cache = status
        self._update_header()

    def _update_header(self) -> None:
        """Reines Rendern aus dem Cache -- laeuft bei jedem Energie/BPM-Tastendruck."""
        if self._status_cache is None:
            badge = "[dim]● prüfe Ordner …[/]"
        else:
            analyzed, total = self._status_cache
            if total == 0:
                badge = "[red]● keine Audiodateien[/]"
            elif analyzed == total:
                badge = f"[green]● {analyzed}/{total} analysiert[/]"
            elif analyzed == 0:
                badge = f"[red]● 0/{total} analysiert — \\[a] drücken[/]"
            else:
                badge = f"[yellow]● {analyzed}/{total} analysiert ({total - analyzed} fehlen)[/]"

        if self.energy > 0:
            energy = f"[orange1]{'▶' * self.energy} ({self.energy:+d})[/]"
        elif self.energy < 0:
            energy = f"[cyan]{'◀' * -self.energy} ({self.energy:+d})[/]"
        else:
            energy = "[dim]─ (0)[/]"
        q_bpm = _to_float(self.query_track.get("bpm")) if self.query_track else None
        anchor = round(q_bpm) if q_bpm else None
        if self.bpm_offset > 0 and anchor:
            bpm = f"[orange1]≥{anchor + self.bpm_offset:.0f}[/]"
        elif self.bpm_offset < 0 and anchor:
            bpm = f"[orange1]≤{anchor + self.bpm_offset:.0f}[/]"
        elif self.bpm_offset:
            bpm = f"[orange1]{self.bpm_offset:+.0f}[/]"
        else:
            bpm = "[dim]±0[/]"

        # Im Transition-Modus ersetzt das gepinnte Ziel die Energie-Anzeige
        # (Energie ist dort deaktiviert); der Direkt-Score A->B zeigt, wie
        # gross die Luecke noch ist, die ueberbrueckt wird.
        if self.transition_target is not None and self.query_track is not None:
            direct = pair_score(self.query_track, self.transition_target)
            mode = (f"Transition → {escape(track_label(self.transition_target))} "
                    f"[{score_style(direct)}]{direct:.3f}[/]")
        elif self._selecting_target:
            mode = "[orange1]Transition-Ziel wählen …[/]"
        else:
            mode = f"Energie {energy}"

        self.query_one("#status", Static).update(
            f"[dim]{self.library.music_dir}[/]   {badge}   {mode}   BPM {bpm}"
        )

    # --- Listen-Befuellung ---

    def _fill_table(self, columns, rows, tracks, border_title):
        table = self.query_one(ResultsTable)
        table.clear(columns=True)
        table.add_columns(*columns)
        for row in rows:
            table.add_row(*row)
        self._row_tracks = tracks
        if tracks:
            table.move_cursor(row=0)
        table.border_title = border_title
        self._update_detail()

    def show_filter(self, needle: str) -> None:
        """Filter-Modus: Library nach Tippstring durchsuchen (leer = alle)."""
        self._results_shown = False
        lib = self.library
        if needle.strip():
            tracks = [lib.tracks[i] for i in fuzzy_search(needle, lib.labels)]
        else:
            order = sorted(range(len(lib.tracks)), key=lambda i: lib.labels[i].lower())
            tracks = [lib.tracks[i] for i in order]

        rows = [
            (str(i + 1), track_label(t), fmt_bpm_cell(t, None), fmt_key_cell(t, None))
            for i, t in enumerate(tracks)
        ]
        title = f"{len(tracks)} Treffer" if needle.strip() else f"Library ({len(tracks)} Tracks)"
        if self._selecting_target:
            title = f"Transition-Ziel wählen — {title}"
        self._fill_table(FILTER_COLUMNS, rows, tracks, title)

    def show_results(self) -> None:
        """Ergebnis-Modus: Ranking zur aktuellen Query (inkl. Energie/BPM-Shift);
        mit gepinntem Transition-Ziel stattdessen die Brueckenkandidaten."""
        if self.query_track is None:
            self.show_filter("")
            return
        self._results_shown = True
        q = self.query_track
        if self.transition_target is not None:
            self._show_bridge_results()
            return
        results = rank_similar(q, self.library, energy=self.energy, bpm_offset=self.bpm_offset, top=TOP_N)
        rows = []
        tracks = []
        for i, r in enumerate(results):
            t = r["track"]
            tracks.append(t)
            rows.append((
                str(i + 1),
                track_label(t),
                fmt_bpm_cell(t, q),
                fmt_key_cell(t, q),
                Text(f"{r['score']:.3f}", style="bold"),
                fmt_delta10(r["d_arousal"]),
                fmt_delta10(r["d_aggressive"]),
                fmt_delta10(r["d_valence"]),
            ))
        title = query_title(q)
        if self.bpm_offset and not results:
            title += "  — 0 Treffer im BPM-Filter"
        self._fill_table(RESULT_COLUMNS, rows, tracks, title)

    def _show_bridge_results(self) -> None:
        """Transition-Modus: Kandidaten zwischen Query (A) und Ziel (B),
        sortiert nach der schwaecheren der beiden Uebergangs-Seiten."""
        q, target = self.query_track, self.transition_target
        results = rank_bridge(q, target, self.library, bpm_offset=self.bpm_offset, top=TOP_N)
        rows = []
        tracks = []
        for i, r in enumerate(results):
            t = r["track"]
            tracks.append(t)
            rows.append((
                str(i + 1),
                track_label(t),
                fmt_bpm_cell(t, q),
                fmt_key_cell(t, q),
                fmt_score_cell(r["score_a"]),
                fmt_score_cell(r["score_b"]),
            ))
        title = f"Transition: {track_label(q)} ⇄ {track_label(target)}"
        if self.bpm_offset and not results:
            title += "  — 0 Treffer im BPM-Filter"
        self._fill_table(BRIDGE_COLUMNS, rows, tracks, title)

    def _update_detail(self) -> None:
        detail = self.query_one("#detail", Static)
        table = self.query_one(ResultsTable)
        if self._row_tracks and 0 <= table.cursor_row < len(self._row_tracks):
            detail.update(fmt_detail_line(self._row_tracks[table.cursor_row]))
        else:
            detail.update("[dim]keine Auswahl[/]")

    # --- Auswahl ---

    def select_track(self, track: dict) -> None:
        """Track wird neue Query -- Kern-Loop der Graph-Navigation.

        Im Transition-Modus: waehrend der Ziel-Auswahl pinnt Enter das Ziel
        (Query bleibt); mit gepinntem Ziel re-anchort Enter die Query und
        behaelt das Ziel -- ausser der Track IST das Ziel, dann ist die
        Transition fertig und der Modus endet."""
        search = self.query_one(SearchInput)
        if self._selecting_target:
            self._selecting_target = False
            self.transition_target = track
            search.placeholder = SEARCH_PLACEHOLDER
        elif (self.transition_target is not None
              and track["filepath"] == self.transition_target["filepath"]):
            self.transition_target = None
            self.query_track = track
        else:
            self.query_track = track
        with search.prevent(Input.Changed):
            search.value = ""
        search.focus()
        self.show_results()
        self._update_header()

    def _select_cursor_row(self) -> None:
        table = self.query_one(ResultsTable)
        if self._row_tracks and 0 <= table.cursor_row < len(self._row_tracks):
            self.select_track(self._row_tracks[table.cursor_row])

    # --- Events ---

    @on(Input.Changed)
    def _on_search_changed(self, event: Input.Changed) -> None:
        if event.value.strip():
            self.show_filter(event.value)
        elif self._selecting_target:
            self.show_filter("")  # Ziel-Auswahl bleibt in der Library-Liste
        elif self.query_track is not None:
            self.show_results()
        else:
            self.show_filter("")

    @on(Input.Submitted)
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        self._select_cursor_row()

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self._row_tracks):
            self.select_track(self._row_tracks[event.cursor_row])

    @on(DataTable.RowHighlighted)
    def _on_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._update_detail()

    @on(ActionKey)
    def _on_action_key(self, message: ActionKey) -> None:
        key = message.key
        if key == "ctrl+a":
            self.action_analyze()
        elif key == "ctrl+t":
            self.action_transition()
        elif key in ("left", "right") and self._results_shown and self.transition_target is None:
            # Energie-Achse im Transition-Modus aus: ihr Ziel-Shifting
            # kollidiert mit dem Brueckenziel zwischen A und B.
            step = 1 if key == "right" else -1
            self.energy = max(ENERGY_MIN, min(ENERGY_MAX, self.energy + step))
            self.show_results()
            self._update_header()
        elif key in ("comma", "full_stop") and self._results_shown:
            direction = 1 if key == "full_stop" else -1
            self.bpm_offset = self._next_bpm_offset(direction)
            self.show_results()
            self._update_header()

    def _next_bpm_offset(self, direction: int) -> float:
        """Naechster BPM-Filter-Schritt: springt exakt auf den naechsten in
        der Library tatsaechlich vorhandenen BPM-Wert, statt eine fixe
        Schrittweite zu addieren. Dadurch wird nie ein Track uebersprungen
        (bei zu grobem Schritt) und nie ein Tastendruck verschwendet (bei
        zu feinem Schritt, wenn im Fenster gerade kein Track liegt). Ist
        bereits der schnellste/langsamste Track erreicht, bleibt der Wert
        stehen -- das begrenzt den Filter automatisch auf den Datenbereich."""
        q_bpm = _to_float(self.query_track.get("bpm")) if self.query_track else None
        if not q_bpm:
            return self.bpm_offset
        anchor = round(q_bpm)
        cutoff = anchor + self.bpm_offset
        candidates = sorted({
            _to_float(t.get("bpm"))
            for t in self.library.tracks
            if t["filepath"] != self.query_track["filepath"] and _to_float(t.get("bpm"))
        })
        if direction > 0:
            nxt = next((b for b in candidates if b > cutoff), None)
        else:
            nxt = next((b for b in reversed(candidates) if b < cutoff), None)
        return self.bpm_offset if nxt is None else nxt - anchor

    # --- Actions (Screen-Bindings) ---

    def action_cursor(self, delta: int) -> None:
        table = self.query_one(ResultsTable)
        if self._row_tracks:
            new_row = max(0, min(len(self._row_tracks) - 1, table.cursor_row + delta))
            table.move_cursor(row=new_row)

    def action_clear_search(self) -> None:
        """Esc-Schichten: erst Suchtext leeren, dann Ziel-Auswahl abbrechen,
        dann gepinntes Transition-Ziel loeschen."""
        search = self.query_one(SearchInput)
        if search.value:
            search.value = ""  # loest Changed aus -> zurueck zu Ergebnissen/Library
        elif self._selecting_target:
            self._exit_target_selection()
        elif self.transition_target is not None:
            self.transition_target = None
            self.show_results()
            self._update_header()
        search.focus()

    def action_analyze(self) -> None:
        def done(_result) -> None:
            self.library.reload()
            self.refresh_status()
            if self._results_shown:
                self.show_results()
            else:
                self.show_filter(self.query_one(SearchInput).value)

        self.app.push_screen(AnalyzeModal(status=self._status_cache), done)

    def action_transition(self) -> None:
        """Ctrl+T: Ziel-Auswahl starten bzw. den Transition-Modus verlassen."""
        if self._selecting_target:
            self._exit_target_selection()
        elif self.transition_target is not None:
            self.transition_target = None
            self.show_results()
            self._update_header()
        elif self.query_track is None:
            self.app.notify("Transition braucht eine Query — erst Track mit Enter wählen.",
                            severity="warning")
        else:
            self._selecting_target = True
            search = self.query_one(SearchInput)
            search.placeholder = TARGET_PLACEHOLDER
            with search.prevent(Input.Changed):
                search.value = ""
            search.focus()
            self.show_filter("")
            self._update_header()

    def _exit_target_selection(self) -> None:
        self._selecting_target = False
        self.query_one(SearchInput).placeholder = SEARCH_PLACEHOLDER
        if self.query_track is not None:
            self.show_results()
        else:
            self.show_filter("")
        self._update_header()


# ---------------------------------------------------------------------------
# AnalyzeModal: Bestaetigung, dann Essentia-Lauf als Subprozess
#
# Subprozess statt Worker-Thread: Essentia/TensorFlow halten in ihren
# C-Extensions den GIL ueber Sekunden -- ein Thread wuerde die UI einfrieren.
# Der Subprozess ist das headless `selecta analyze --porcelain`, dessen
# stdout hier gestreamt wird.
# ---------------------------------------------------------------------------

class AnalyzeModal(ModalScreen):
    BINDINGS = [
        Binding("escape", "close_or_cancel", "Abbrechen/Schließen"),
        Binding("enter", "start", "Start"),
    ]

    def __init__(self, status: tuple[int, int] | None = None):
        super().__init__()
        self._status = status  # Cache vom MainScreen -- kein erneuter Ordner-Scan
        self._state = "confirm"  # confirm -> running -> finished
        self._proc: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="analyze-box"):
            yield Static("[b]Analyse[/b]", id="analyze-title")
            yield Static(id="analyze-info", markup=True)
            yield ProgressBar(id="analyze-progress", show_eta=False)
            yield RichLog(id="analyze-log", max_lines=500, wrap=True)
            yield Static(id="analyze-hint", markup=True)

    def on_mount(self) -> None:
        if self._status is not None:
            analyzed, total = self._status
            counts = f"{analyzed} von {total} Tracks analysiert, [b]{total - analyzed} offen[/b]."
        else:
            counts = "Analysiert neue Tracks und ergänzt fehlende Embeddings."
        self.query_one("#analyze-info", Static).update(
            f"[dim]{self.app.library.music_dir}[/]\n{counts}"
        )
        self.query_one(ProgressBar).display = False
        self.query_one(RichLog).display = False
        self.query_one("#analyze-hint", Static).update(
            "[b]Enter[/b] startet die Analyse   [b]Esc[/b] zurück"
        )

    def action_start(self) -> None:
        if self._state != "confirm":
            return
        self._state = "running"
        self.query_one(ProgressBar).display = True
        self.query_one(RichLog).display = True
        self.query_one("#analyze-title", Static).update("[b]Analyse läuft[/b]")
        self.query_one("#analyze-hint", Static).update(
            "[dim]Esc bricht ab — fertige Tracks bleiben gespeichert (Resume)[/dim]"
        )
        self._stream_analysis()

    @work(exclusive=True)
    async def _stream_analysis(self) -> None:
        log = self.query_one(RichLog)
        bar = self.query_one(ProgressBar)
        cmd = [
            sys.executable, "-u", "-m", "selecta", "analyze",
            "--music-dir", str(self.app.library.music_dir),
            "--models-dir", str(self.app.models_dir),
            "--porcelain",
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        assert self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            if text.startswith("::progress "):
                _, done, total = text.split()
                bar.update(total=max(int(total), 1), progress=int(done))
            elif text:
                log.write(text)

        code = await self._proc.wait()
        self._state = "finished"
        if code == 0:
            self.query_one("#analyze-title", Static).update("[b green]Analyse beendet[/b green]")
        else:
            self.query_one("#analyze-title", Static).update(
                f"[b yellow]Analyse abgebrochen/fehlgeschlagen (Code {code})[/b yellow]"
            )
        self.query_one("#analyze-hint", Static).update("[dim]Esc schließt[/dim]")

    def action_close_or_cancel(self) -> None:
        if self._state == "running":
            if self._proc is not None and self._proc.returncode is None:
                self._proc.terminate()
            self.query_one("#analyze-hint", Static).update("[yellow]Breche ab …[/yellow]")
        elif self._state == "finished":
            self.dismiss(True)
        else:
            self.dismiss(False)


def resolve_music_dir(raw: str) -> Path:
    """Windows-Pfade (C:\\... oder C:/...) via wslpath nach /mnt/... uebersetzen,
    z.B. beim Eintippen im DirScreen-Prompt. Alles andere (/mnt/..., ~, relative
    Pfade) bleibt unveraendert -- WSL kennt kein C:\\, die App laeuft dort drin."""
    raw = raw.strip()
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        try:
            converted = subprocess.run(
                ["wslpath", "-a", raw], capture_output=True, text=True, check=True
            ).stdout.strip()
            if converted:
                return Path(converted)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
    return Path(raw).expanduser()


class PathSuggester(Suggester):
    """Ghost-Text-Vervollstaendigung fuer Ordnerpfade (naechster passender
    Unterordner, alphabetisch erster Treffer -- Rechts/End uebernimmt ihn)."""

    def __init__(self) -> None:
        super().__init__(use_cache=False, case_sensitive=True)

    async def get_suggestion(self, value: str) -> str | None:
        value = value.strip()
        if not value:
            return None
        path = resolve_music_dir(value)
        parent, partial = (path, "") if value.endswith(("/", "\\")) else (path.parent, path.name)
        try:
            if not parent.is_dir():
                return None
            matches = sorted(
                p.name
                for p in parent.iterdir()
                if p.is_dir() and not p.name.startswith(".") and p.name.startswith(partial)
            )
        except OSError:
            return None
        if not matches:
            return None
        return str(parent / matches[0])


class PathInput(Input):
    """Input mit Tab-Vervollstaendigung wie in einer normalen Konsole --
    Textual bindet die Ghost-Text-Uebernahme sonst nur an Rechts-Pfeil/Ende,
    nicht an Tab (das navigiert normalerweise den Fokus weiter)."""

    BINDINGS = [Binding("tab", "accept_suggestion", "Vervollstaendigen", show=False)]

    def action_accept_suggestion(self) -> None:
        if self.cursor_at_end and self._suggestion:
            self.value = self._suggestion
            self.cursor_position = len(self.value)
        else:
            self.screen.focus_next()


LAST_DIR_FILE = Path.home() / ".local" / "share" / "selecta" / "last_dir"


def load_last_dir() -> str:
    try:
        return LAST_DIR_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_last_dir(music_dir: Path) -> None:
    try:
        LAST_DIR_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_DIR_FILE.write_text(str(music_dir), encoding="utf-8")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Ordner-Abfrage beim Start ohne Argument
# ---------------------------------------------------------------------------

class DirScreen(Screen):
    def compose(self) -> ComposeResult:
        with Vertical(id="dir-box"):
            yield Static(Text(LOGO, style="bold magenta"))
            yield Static("Musik-Ordner eingeben (Windows- oder /mnt-Pfad, Tab/→ vervollstaendigt):")
            yield PathInput(
                value=load_last_dir(),
                placeholder="/mnt/g/Media/Musik/…",
                id="dir-input",
                suggester=PathSuggester(),
            )
            yield Static(id="dir-error", markup=True)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def _on_submitted(self, event: Input.Submitted) -> None:
        path = resolve_music_dir(event.value)
        if path.is_dir():
            self.app.open_library(path)
        else:
            self.query_one("#dir-error", Static).update(f"[red]Ordner nicht gefunden: {path}[/]")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class SelectaApp(App):
    TITLE = "SELECTA"
    # Ctrl+C zusaetzlich zu Ctrl+Q: im Raw-Mode der TUI ist Ctrl+C nur eine
    # Taste (kein SIGINT), und Ctrl+Q wird z.B. vom VSCode-Terminal geschluckt.
    # Beenden verliert nichts -- die CSV ist immer persistiert.
    BINDINGS = [
        Binding("ctrl+q", "quit", "Ende", priority=True),
        Binding("ctrl+c", "quit", "Ende", priority=True),
    ]

    CSS = """
    Screen {
        background: $surface;
    }
    #topbar {
        dock: top;
        height: 1;
        background: $panel;
        padding: 0 1;
    }
    #logo {
        width: auto;
        margin-right: 2;
    }
    #status {
        width: 1fr;
        content-align: right middle;
    }
    #search, #dir-input {
        border: tall $accent;
        margin: 0 1;
    }
    ResultsTable {
        height: 1fr;
        margin: 0 1;
        border: round $accent 50%;
    }
    #detail {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
        offset-y: -1;
    }
    #keybar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: $panel-darken-1;
        color: $text-muted;
    }
    AnalyzeModal {
        align: center middle;
    }
    #analyze-box {
        width: 80%;
        height: 80%;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    #analyze-progress {
        margin: 1 0;
    }
    #analyze-log {
        height: 1fr;
        border: round $accent 30%;
    }
    DirScreen {
        align: center middle;
    }
    #dir-box {
        width: 70;
        height: auto;
        border: thick $accent;
        background: $panel;
        padding: 1 2;
    }
    """

    def __init__(self, music_dir=None, models_dir: Path = Path("models")):
        super().__init__()
        self._initial_dir = music_dir
        self.models_dir = models_dir
        self.library: Library | None = None

    def on_mount(self) -> None:
        self.theme = "textual-dark"
        if self._initial_dir:
            self.open_library(Path(self._initial_dir))
        else:
            self.push_screen(DirScreen())

    def open_library(self, music_dir: Path) -> None:
        self.library = Library(music_dir)
        save_last_dir(music_dir)
        if isinstance(self.screen, DirScreen):
            self.pop_screen()
        self.push_screen(MainScreen())
