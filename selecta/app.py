"""SELECTA Textual-App: Such-Screen, Analyse-Modal, Transition-Planer.

Bedienlogik MainScreen (fzf-Stil: Suche haelt den Fokus, Aktionen auf Chords):
- Tippen filtert die Library; druckbare Tasten tippen IMMER (nie Aktionen,
  sonst wuerde "Ambush" die Analyse starten).
- Ctrl+A = Analyse, Ctrl+T = Transition -- jederzeit, auch beim Tippen.
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

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.suggester import Suggester
from textual.widgets import DataTable, Input, ProgressBar, RichLog, Select, Static

from .config import (
    BPM_FINETUNE_STEP,
    ENERGY_MAX,
    ENERGY_MIN,
    TOP_N,
    TRANSITION_MAX_TRACKS,
)
from .library import Library, fuzzy_search, track_label
from .similarity import (
    harmonic_distance,
    plan_transition,
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
            # Nur der Such-Screen hat ein SearchInput; im Transition-Screen
            # laufen Tipp-Eingaben nicht ueber die Tabelle.
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
            yield SearchInput(placeholder="Track suchen … (tippen zum Filtern)", id="search")
            yield ResultsTable(id="results")
            yield Static(id="detail", markup=True)
            yield Static(id="keybar", markup=True)

    def on_mount(self) -> None:
        table = self.query_one(ResultsTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self.query_one("#logo", Static).update(Text(LOGO, style="bold magenta"))
        self.query_one("#keybar", Static).update(
            "[b]Enter[/b] wählen  [b]↑↓[/b] navigieren  [b]←→[/b] Energie  [b],[/b][b].[/b] BPM  "
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
        bpm = f"[orange1]{self.bpm_offset:+d}[/]" if self.bpm_offset else "[dim]±0[/]"

        self.query_one("#status", Static).update(
            f"[dim]{self.library.music_dir}[/]   {badge}   Energie {energy}   BPM {bpm}"
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
        self._fill_table(FILTER_COLUMNS, rows, tracks, title)

    def show_results(self) -> None:
        """Ergebnis-Modus: Ranking zur aktuellen Query (inkl. Energie/BPM-Shift)."""
        if self.query_track is None:
            self.show_filter("")
            return
        self._results_shown = True
        q = self.query_track
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
        self._fill_table(RESULT_COLUMNS, rows, tracks, query_title(q))

    def _update_detail(self) -> None:
        detail = self.query_one("#detail", Static)
        table = self.query_one(ResultsTable)
        if self._row_tracks and 0 <= table.cursor_row < len(self._row_tracks):
            detail.update(fmt_detail_line(self._row_tracks[table.cursor_row]))
        else:
            detail.update("[dim]keine Auswahl[/]")

    # --- Auswahl ---

    def select_track(self, track: dict) -> None:
        """Track wird neue Query -- Kern-Loop der Graph-Navigation."""
        self.query_track = track
        search = self.query_one(SearchInput)
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
        elif key in ("left", "right") and self._results_shown:
            step = 1 if key == "right" else -1
            self.energy = max(ENERGY_MIN, min(ENERGY_MAX, self.energy + step))
            self.show_results()
            self._update_header()
        elif key in ("comma", "full_stop") and self._results_shown:
            step = BPM_FINETUNE_STEP if key == "full_stop" else -BPM_FINETUNE_STEP
            self.bpm_offset += step
            self.show_results()
            self._update_header()

    # --- Actions (Screen-Bindings) ---

    def action_cursor(self, delta: int) -> None:
        table = self.query_one(ResultsTable)
        if self._row_tracks:
            new_row = max(0, min(len(self._row_tracks) - 1, table.cursor_row + delta))
            table.move_cursor(row=new_row)

    def action_clear_search(self) -> None:
        search = self.query_one(SearchInput)
        if search.value:
            search.value = ""  # loest Changed aus -> zurueck zu Ergebnissen/Library
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
        def done(track) -> None:
            if track is not None:
                self.select_track(track)

        self.app.push_screen(TransitionScreen(initial_a=self.query_track), done)


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


# ---------------------------------------------------------------------------
# TransitionScreen: A -> B ueber k Zwischentracks
# ---------------------------------------------------------------------------

class TransitionScreen(Screen):
    BINDINGS = [
        Binding("escape", "back", "Zurück"),
        Binding("up", "cursor(-1)", show=False),
        Binding("down", "cursor(1)", show=False),
    ]

    def __init__(self, initial_a: dict | None = None):
        super().__init__()
        self.track_a: dict | None = initial_a
        self.track_b: dict | None = None
        self._row_tracks: list[dict] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="topbar"):
                yield Static(Text(LOGO, style="bold magenta"), id="logo")
                yield Static("[b]Transition[/b] — von A nach B", id="status", markup=True)
            yield Input(placeholder="Song A tippen + Enter …", id="input-a")
            yield Static(id="resolved-a", markup=True)
            yield Input(placeholder="Song B tippen + Enter …", id="input-b")
            yield Static(id="resolved-b", markup=True)
            with Horizontal(id="k-row"):
                yield Static("Zwischentracks:", id="k-label")
                yield Select(
                    [("auto", 0)] + [(str(i), i) for i in range(1, TRANSITION_MAX_TRACKS + 1)],
                    value=0, allow_blank=False, id="k-select",
                )
            yield ResultsTable(id="transition-table")
            yield Static(id="detail", markup=True)
            yield Static(
                "[b]Enter[/b] Track als neue Suche übernehmen  [b]Esc[/b] zurück",
                id="keybar", markup=True,
            )

    def on_mount(self) -> None:
        table = self.query_one("#transition-table", ResultsTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        if self.track_a is not None:
            self._show_resolved("a", self.track_a)
        self.query_one("#input-a" if self.track_a is None else "#input-b", Input).focus()

    @property
    def library(self) -> Library:
        return self.app.library

    def _resolve(self, needle: str) -> dict | None:
        if not needle.strip() or not self.library.tracks:
            return None
        indices = fuzzy_search(needle, self.library.labels, limit=1)
        return self.library.tracks[indices[0]] if indices else None

    def _show_resolved(self, which: str, track: dict | None) -> None:
        target = self.query_one(f"#resolved-{which}", Static)
        if track is None:
            target.update("[dim]— nicht gefunden —[/]")
        else:
            bpm = track.get("bpm") or "?"
            key = track.get("key") or "?"
            target.update(f"  [green]→ {track_label(track)}[/]  [dim]\\[{bpm} BPM | {key}][/]")

    @on(Input.Submitted)
    def _on_input_submitted(self, event: Input.Submitted) -> None:
        track = self._resolve(event.value)
        if event.input.id == "input-a":
            self.track_a = track
            self._show_resolved("a", track)
            if track is not None:
                self.query_one("#input-b", Input).focus()
        else:
            self.track_b = track
            self._show_resolved("b", track)
        self._recompute()

    @on(Select.Changed)
    def _on_k_changed(self, event: Select.Changed) -> None:
        self._recompute()

    def _recompute(self) -> None:
        table = self.query_one("#transition-table", ResultsTable)
        table.clear(columns=True)
        self._row_tracks = []
        if self.track_a is None or self.track_b is None:
            return
        if self.track_a["filepath"] == self.track_b["filepath"]:
            self.query_one("#detail", Static).update("[yellow]A und B sind derselbe Track[/]")
            return

        k = self.query_one("#k-select", Select).value or None
        rows = plan_transition(self.track_a, self.track_b, self.library, num_tracks=k)

        table.add_columns("", "TRACK", "BPM", "KEY", "ΔEMB", "ΔBPM", "KEY-REL")
        for i, row in enumerate(rows):
            t = row["track"]
            self._row_tracks.append(t)
            deltas = row["deltas"]
            if i == 0:
                pos = Text("A", style="bold magenta")
            elif i == len(rows) - 1:
                pos = Text("B", style="bold magenta")
            else:
                pos = Text(str(i))

            if deltas is None:
                emb_cell, dbpm_cell, key_cell = Text("—", style="dim"), Text("—", style="dim"), Text("—", style="dim")
            else:
                d = deltas["emb_dist"]
                emb_style = "green" if d < 0.1 else ("yellow" if d < 0.2 else "red")
                emb_cell = Text(f"{d:.3f}", style=emb_style)
                dbpm = deltas["d_bpm"]
                dbpm_cell = Text("?" if dbpm is None else f"{dbpm:+.0f}")
                rel = deltas["key_rel"]
                if rel is None:
                    key_cell = Text("?", style="dim")
                elif rel <= 1:
                    key_cell = Text("✓ harmonisch", style="green")
                elif rel == 2:
                    key_cell = Text("~ mixbar", style="yellow")
                else:
                    key_cell = Text("✗ Bruch", style="red")

            table.add_row(
                pos, track_label(t),
                Text(str(t.get("bpm") or "?")), Text(str(t.get("key") or "?")),
                emb_cell, dbpm_cell, key_cell,
            )
        if self._row_tracks:
            table.move_cursor(row=0)
            # Fokus auf die Kette: ↑/↓ + Enter uebernehmen direkt einen Track.
            table.focus()

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self._row_tracks):
            self.dismiss(self._row_tracks[event.cursor_row])

    def action_cursor(self, delta: int) -> None:
        table = self.query_one("#transition-table", ResultsTable)
        if self._row_tracks:
            table.move_cursor(row=max(0, min(len(self._row_tracks) - 1, table.cursor_row + delta)))

    def action_back(self) -> None:
        self.dismiss(None)


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


# ---------------------------------------------------------------------------
# Ordner-Abfrage beim Start ohne Argument
# ---------------------------------------------------------------------------

class DirScreen(Screen):
    def compose(self) -> ComposeResult:
        with Vertical(id="dir-box"):
            yield Static(Text(LOGO, style="bold magenta"))
            yield Static("Musik-Ordner eingeben (Windows- oder /mnt-Pfad, → vervollstaendigt):")
            yield Input(placeholder="/mnt/g/Media/Musik/…", id="dir-input", suggester=PathSuggester())
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
    #search, #input-a, #input-b, #dir-input {
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
    #resolved-a, #resolved-b {
        height: 1;
        margin: 0 1;
    }
    #k-row {
        height: 3;
        margin: 0 1;
    }
    #k-label {
        width: auto;
        content-align: left middle;
        height: 3;
        margin-right: 1;
    }
    #k-select {
        width: 14;
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
        if isinstance(self.screen, DirScreen):
            self.pop_screen()
        self.push_screen(MainScreen())
