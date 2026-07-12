"""SELECTA Textual-App: Such-Screen, Analyse-Modal, Transition-Modus.

Bedienlogik MainScreen (fzf-Stil: Suche haelt den Fokus, Aktionen auf Chords):
- Tippen filtert die Library; druckbare Tasten tippen IMMER (nie Aktionen,
  sonst wuerde "Ambush" die Analyse starten).
- Ctrl+A = Analyse -- jederzeit, auch beim Tippen.
- Ctrl+T pinnt ein Transition-Ziel B (Fuzzy-Suche + Enter): Die Liste zeigt
  dann Brueckenkandidaten mit Score zu beiden Seiten, sortiert nach der
  schwaecheren. Enter re-anchort A und behaelt B; Enter auf B selbst,
  Esc oder erneutes Ctrl+T beenden den Modus. Bewusst stateless -- keine
  Playlists/Sets, das Set lebt in der DJ-Software; gemerkt wird nur die
  Library-Liste (libraries.json, siehe LibraryScreen).
- Ctrl+L geht zurueck zum LibraryScreen (Libraries an/abwaehlen, anlegen,
  analysieren); Enter dort uebernimmt die Auswahl in die laufende Session.
- Bei LEEREM Suchfeld sind ←/→ (Energie) und ,/. (BPM) Aktions-Tasten;
  mit Text im Feld bewegen sie den Cursor bzw. tippen.
- ↑/↓/Enter laufen immer auf der Ergebnisliste (Cursor + Auswahl).
- Enter/Klick auf eine Zeile macht den Track zur neuen Query.
"""

import asyncio
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Input, ProgressBar, RichLog, Static

from .config import (
    ENERGY_MAX,
    ENERGY_MIN,
    GENRE_CHIP_COLORS,
    SCORE_COLOR_STEPS,
    TAG_SEPARATOR,
    TOP_N,
    W_BPM,
    W_KEY,
    W_MOOD,
)
from .library import Library, dir_status, fuzzy_search, track_label
from .similarity import (
    harmonic_distance,
    pair_score,
    parse_key,
    rank_bridge,
    rank_similar,
    relative_bpm_distance,
    _to_float,
)

LOGO = "◤ SELECTA ◢"

# Grosses Logo fuer den LibraryScreen -- bewusst eine Spur zu viel Liebe,
# NFO-/Soundsystem-Style. Breite 57 Zeichen; Farbverlauf pro Zeile ueber
# LOGO_GRADIENT.
LOGO_BIG = """\
███████╗███████╗██╗     ███████╗ ██████╗████████╗ █████╗
██╔════╝██╔════╝██║     ██╔════╝██╔════╝╚══██╔══╝██╔══██╗
███████╗█████╗  ██║     █████╗  ██║        ██║   ███████║
╚════██║██╔══╝  ██║     ██╔══╝  ██║        ██║   ██╔══██║
███████║███████╗███████╗███████╗╚██████╗   ██║   ██║  ██║
╚══════╝╚══════╝╚══════╝╚══════╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝"""
LOGO_GRADIENT = [
    "bright_magenta", "magenta", "medium_orchid",
    "medium_purple", "slate_blue1", "royal_blue1",
]
LOGO_TAGLINE = "░▒▓ your local audio intelligence ▓▒░"


def render_logo() -> Text:
    logo = Text(justify="center")
    # auf einheitliche Breite padden: justify zentriert jede Zeile einzeln,
    # ungleich lange Zeilen wuerden das Logo sonst um eine Spalte versetzen
    lines = LOGO_BIG.splitlines()
    width = max(len(line) for line in lines)
    for line, style in zip(lines, LOGO_GRADIENT):
        logo.append(line.ljust(width) + "\n", style=f"bold {style}")
    logo.append(LOGO_TAGLINE, style="dim magenta")
    return logo


# Tasten, die nur bei leerem Suchfeld (bzw. auf der Tabelle) Aktionen ausloesen
# -- keine Buchstaben, mit denen ein Titel beginnen koennte.
ACTION_KEYS = {"left", "right", "comma", "full_stop"}
# Chords, die IMMER Aktionen ausloesen (fzf-Konvention: Aktionen auf Ctrl).
CTRL_ACTION_KEYS = {"ctrl+a", "ctrl+t", "ctrl+l"}

FILTER_COLUMNS = ("#", "TRACK", "BPM", "KEY")
RESULT_COLUMNS = ("#", "TRACK", "BPM", "KEY", "SCORE", "ΔENERG", "ΔHARD", "ΔMOOD")
BRIDGE_COLUMNS = ("#", "TRACK", "BPM", "KEY", "SCORE→A", "SCORE→B")

SEARCH_PLACEHOLDER = "Search tracks … (type to filter)"
TARGET_PLACEHOLDER = "Search transition target … (type to filter)"


# ---------------------------------------------------------------------------
# Formatierung
# ---------------------------------------------------------------------------

def genre_chip_color(name: str) -> str:
    """Stabiler Hash Style-Name -> Chip-Farbe (hash() ist pro Prozess
    randomisiert und wuerde die Farben bei jedem Start wuerfeln)."""
    return GENRE_CHIP_COLORS[zlib.crc32(name.encode("utf-8")) % len(GENRE_CHIP_COLORS)]


def chip_line(track: dict) -> Text | None:
    """Chip-Zeile unter dem Track-Label: Genres als farbige Pills, Vibes und
    Jahr gedimmt dahinter. None, wenn es nichts zu zeigen gibt (alte CSV ohne
    Genre-Analyse) -- die Zeile bleibt dann einzeilig."""
    genres = [g for g in (track.get("genres") or "").split(TAG_SEPARATOR) if g]
    extras = [v for v in (track.get("vibes") or "").split(TAG_SEPARATOR) if v]
    year = (track.get("year") or "").strip()
    if year:
        extras.append(year)
    if not genres and not extras:
        return None
    line = Text("  ")
    for genre in genres:
        # Farbiger Text auf dunklem Pill statt schwarz auf leuchtender
        # Flaeche -- die Chips sollen unter dem Track-Label zuruecktreten,
        # nicht heller strahlen als der Titel selbst.
        line.append(f" {genre} ", style=f"{genre_chip_color(genre)} on grey19")
        line.append(" ")
    if extras:
        line.append(" · ".join(extras), style="dim")
    return line


def fmt_track_cell(track: dict) -> tuple[Text, int]:
    """TRACK-Zelle inkl. benoetigter Zeilenhoehe: Label, darunter die
    Chip-Zeile (falls vorhanden -> Hoehe 2, sonst 1). Das Label ist fett --
    Schriftgroessen gibt es im Terminal nicht, die Hierarchie Label ueber
    Chips entsteht ueber Gewicht (bold) und Helligkeit (Chips gedimmt)."""
    label = Text(track_label(track), style="bold")
    chips = chip_line(track)
    if chips is None:
        return label, 1
    label.append("\n")
    label.append(chips)
    return label, 2


def fmt_analysis_log_line(info: dict) -> Text:
    """Ergebniszeile im Analyse-Log (aus einem ::status-done-Event des
    Subprozesses). Voll-Analyse -> Name plus dieselbe Chip-Zeile wie in der
    Trackliste plus Kern-Scores; sonst eine kompakte Einzeiler-Variante."""
    name = info.get("name", "?")
    kind = info.get("kind")
    if kind == "error":
        return Text(f"✗ {name}: {info.get('error', '')}", style="red")
    if kind == "complete":
        return Text(f"≡ {name}", style="dim")
    if kind == "tags":
        key = info.get("key") or "?"
        if info.get("key_estimated"):
            key = f"~{key}"
        return Text(
            f"~ {name}  BPM {info.get('bpm') or '?'} · key {key} backfilled  ({info.get('secs', '?')}s)",
            style="yellow",
        )
    line = Text()
    line.append("✓ ", style="bold green")
    line.append(name, style="bold")
    line.append(f"  ({info.get('secs', '?')}s)", style="dim")
    line.append("\n")
    chips = chip_line(info)  # info traegt genres/vibes/year wie ein Track-Dict
    if chips is not None:
        line.append(chips)
        line.append("   ")
    else:
        line.append("   ")
    line.append(
        f"{info.get('bpm') or '?'} BPM · {info.get('key') or '?'}"
        f" · arous {info.get('arousal') or '?'}"
        f" · aggr {info.get('aggressive') or '?'}"
        f" · dance {info.get('danceable') or '?'}",
        style="dim",
    )
    return line


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


def display_key(track: dict) -> str:
    """Key fuer die Anzeige: '?' wenn leer, ~-Praefix wenn von uns
    geschaetzt statt aus einem DJ-Software-Tag (der Wert ist Platzhalter,
    bis Rekordbox/Traktor ihn ueberschreibt)."""
    key = (track.get("key") or "").strip()
    if not key:
        return "?"
    return f"~{key}" if track.get("key_estimated") else key


def fmt_key_cell(track: dict, query: dict | None) -> Text:
    key = (track.get("key") or "").strip()
    if not key:
        return Text("?", style="dim")
    shown = display_key(track)
    # Geschaetzte Keys gedimmt: die harmonische Farbe bleibt lesbar, aber
    # der Wert drängt sich nicht als verlaesslich auf.
    estimated = bool(track.get("key_estimated"))
    if query is None:
        return Text(shown, style="dim" if estimated else "")
    rel = harmonic_distance(query.get("key"), key)
    if rel is None:
        return Text(shown, style="dim")
    style = "green" if rel <= 1 else ("yellow" if rel == 2 else "dim")
    if estimated:
        style = f"dim {style}"
    return Text(shown, style=style)


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


def fmt_why_line(result: dict) -> str:
    """Score-Zerlegung fuer die Detail-Zeile im Ergebnis-Modus: dieselben
    Terme wie _combined_score, als lesbare Rechnung. Der groesste Abzug
    traegt orange, Nullen/Neutrales ist gedimmt -- so liest man beim
    Cursor-Bewegen direkt ab, WAS einen Kandidaten nach unten drueckt
    (und beim Tuning von W_BPM/W_KEY/W_MOOD deren Wirkung)."""
    costs = {
        "bpm": None if result.get("bpm_pen") is None else W_BPM * min(result["bpm_pen"], 1.0),
        "key": None if result.get("key_pen") is None else W_KEY * (result["key_pen"] / 8.0),
        "mood": W_MOOD * ((result.get("mood_dist") or 0.0) / 2.5),
    }
    # None heisst neutral -- aber WER den Wert nicht hat, macht den
    # Unterschied: fehlt er dem Kandidaten, ist '?' eine Eigenschaft dieser
    # Zeile (gelb); hat die QUERY keinen Wert, ist der Term fuer ALLE
    # Zeilen neutral und wird nur gedimmt ausgelassen ('–') -- sonst sieht
    # eine Liste voller '?' nach Bug aus, obwohl nur der Query-Track
    # keinen Tag hat.
    track = result["track"]
    has_track_value = {
        "bpm": _to_float(track.get("bpm")) is not None,
        "key": parse_key(track.get("key") or "") is not None,
        "mood": True,
    }
    known = [v for v in costs.values() if v is not None]
    biggest = max(known) if known else 0.0
    parts = [f"score [b]{result['score']:.3f}[/]  =  cos {result['cos_sim']:.3f}"]
    for name, v in costs.items():
        if v is None:
            if has_track_value[name]:
                parts.append(f"[dim]− {name} –[/]")  # Query-seitig kein Wert
            else:
                parts.append(f"[yellow]− {name} ?[/]")  # Track hat keinen Wert
        elif v < 0.005 or v < biggest:
            style = "dim" if v < 0.005 else ""
            parts.append(f"[{style or 'white'}]− {name} {v:.2f}[/]")
        else:
            parts.append(f"[orange1]− {name} {v:.2f}[/]")
    return "  ".join(parts) + f"   [dim]{result['track']['filepath']}[/]"


def query_title(track: dict) -> str:
    bpm = track.get("bpm") or "?"
    return f"Similar to: {track_label(track)}  [{bpm} BPM | {display_key(track)}]"


def fmt_transition_bar(query: dict, target: dict | None, direct: float | None) -> Text:
    """Inhalt der Transition-Bar: die eine Zeile, die im Transition-Modus
    immer sagt, wo man steht -- Query A links, Ziel B rechts, dazwischen der
    Direkt-Score als 'verbleibende Luecke'. Waehrend der Ziel-Auswahl
    (target=None) bleibt B offen, aber A bleibt sichtbar -- vorher verschwand
    beim Ctrl+T-Druck die Information, von welchem Track aus man sucht."""
    bar = Text(no_wrap=True, overflow="ellipsis")
    bar.append(" A ", style="bold black on cyan")
    bar.append(" ")
    bar.append(track_label(query), style="bold")
    if target is None:
        bar.append("  ──▶  ", style="dim")
        bar.append(" B ", style="bold black on orange1")
        bar.append(" ")
        bar.append("select target …", style="orange1")
        return bar
    bar.append("  ──", style="dim")
    if direct is not None:
        bar.append(f" {direct:.3f} ", style=f"bold {score_style(direct)}")
    bar.append("──▶  ", style="dim")
    bar.append(" B ", style="bold black on orange1")
    bar.append(" ")
    bar.append(track_label(target), style="bold")
    return bar


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
        # Result-Dicts parallel zu _row_tracks (nur im Similarity-Modus) --
        # Quelle der Score-Zerlegung in der Detail-Zeile.
        self._row_results: list[dict] | None = None
        self._results_shown = False
        self._status_cache: tuple[int, int] | None = None

    # --- Aufbau ---

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(id="topbar"):
                yield Static(id="logo")
                yield Static(id="status", markup=True)
            yield SearchInput(placeholder=SEARCH_PLACEHOLDER, id="search")
            yield Static(id="transition-bar")
            yield ResultsTable(id="results")
            yield Static(id="detail", markup=True)
            yield Static(id="keybar", markup=True)

    def on_mount(self) -> None:
        table = self.query_one(ResultsTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        self.query_one("#logo", Static).update(Text(LOGO, style="bold magenta"))
        self.query_one("#keybar", Static).update(
            "[b]Enter[/b] select+copy  [b]↑↓[/b] navigate  [b]←→[/b] energy  [b],[/b][b].[/b] BPM filter  "
            "[b]^a[/b] analyze  [b]^t[/b] transition  [b]^l[/b] libraries  [b]Esc[/b] clear  [b]^c[/b] quit"
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
            badge = "[dim]● scanning folders …[/]"
        else:
            analyzed, total = self._status_cache
            if total == 0:
                badge = "[red]● no audio files[/]"
            elif analyzed == total:
                badge = f"[green]● {analyzed}/{total} analyzed[/]"
            elif analyzed == 0:
                badge = f"[red]● 0/{total} analyzed — press ^a[/]"
            else:
                badge = f"[yellow]● {analyzed}/{total} analyzed ({total - analyzed} open)[/]"

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

        # Im Transition-Modus ersetzt ein knapper Marker die Energie-Anzeige
        # (Energie ist dort deaktiviert); A, B und der Direkt-Score stehen in
        # der Transition-Bar ueber der Liste -- hier nicht doppelt.
        if self.transition_target is not None and self.query_track is not None:
            mode = "[orange1]Transition[/]"
        elif self._selecting_target:
            mode = "[orange1]Selecting transition target …[/]"
        else:
            mode = f"Energy {energy}"

        dirs = self.library.music_dirs
        if len(dirs) == 1:
            location = str(dirs[0])
        else:
            location = f"{len(dirs)} Libraries: " + " + ".join(d.name or str(d) for d in dirs)
        self.query_one("#status", Static).update(
            f"[dim]{escape(location)}[/]   {badge}   {mode}   BPM {bpm}"
        )
        self._update_transition_bar()

    def _update_transition_bar(self) -> None:
        """Bar nur zeigen, wenn der Transition-Modus etwas zu sagen hat --
        sonst kostet sie eine Zeile Listenplatz."""
        bar = self.query_one("#transition-bar", Static)
        if self.query_track is not None and self.transition_target is not None:
            direct = pair_score(self.query_track, self.transition_target)
            bar.update(fmt_transition_bar(self.query_track, self.transition_target, direct))
            bar.display = True
        elif self.query_track is not None and self._selecting_target:
            bar.update(fmt_transition_bar(self.query_track, None, None))
            bar.display = True
        else:
            bar.display = False

    # --- Listen-Befuellung ---

    def _fill_table(self, columns, rows, tracks, border_title, heights=None, results=None):
        table = self.query_one(ResultsTable)
        table.clear(columns=True)
        table.add_columns(*columns)
        for i, row in enumerate(rows):
            table.add_row(*row, height=heights[i] if heights else 1)
        self._row_tracks = tracks
        self._row_results = results
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

        rows = []
        heights = []
        for i, t in enumerate(tracks):
            cell, height = fmt_track_cell(t)
            heights.append(height)
            rows.append((str(i + 1), cell, fmt_bpm_cell(t, None), fmt_key_cell(t, None)))
        title = f"{len(tracks)} matches" if needle.strip() else f"Library ({len(tracks)} tracks)"
        if self._selecting_target:
            title = f"Select transition target — {title}"
        self._fill_table(FILTER_COLUMNS, rows, tracks, title, heights)

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
        heights = []
        for i, r in enumerate(results):
            t = r["track"]
            tracks.append(t)
            cell, height = fmt_track_cell(t)
            heights.append(height)
            rows.append((
                str(i + 1),
                cell,
                fmt_bpm_cell(t, q),
                fmt_key_cell(t, q),
                Text(f"{r['score']:.3f}", style="bold"),
                fmt_delta10(r["d_arousal"]),
                fmt_delta10(r["d_aggressive"]),
                fmt_delta10(r["d_valence"]),
            ))
        title = query_title(q)
        if self.bpm_offset and not results:
            title += "  — 0 matches within BPM filter"
        self._fill_table(RESULT_COLUMNS, rows, tracks, title, heights, results=results)

    def _show_bridge_results(self) -> None:
        """Transition-Modus: Kandidaten zwischen Query (A) und Ziel (B),
        sortiert nach der schwaecheren der beiden Uebergangs-Seiten."""
        q, target = self.query_track, self.transition_target
        results = rank_bridge(q, target, self.library, bpm_offset=self.bpm_offset, top=TOP_N)
        rows = []
        tracks = []
        heights = []
        for i, r in enumerate(results):
            t = r["track"]
            tracks.append(t)
            cell, height = fmt_track_cell(t)
            heights.append(height)
            # Das Ziel B laeuft selbst als Kandidat mit -- in der Liste
            # bekommt es statt der Ranknummer einen Marker, passend zum
            # B-Badge in der Transition-Bar.
            if t["filepath"] == target["filepath"]:
                rank_cell = Text("◆B", style="bold orange1")
            else:
                rank_cell = Text(str(i + 1))
            rows.append((
                rank_cell,
                cell,
                fmt_bpm_cell(t, q),
                fmt_key_cell(t, q),
                fmt_score_cell(r["score_a"]),
                fmt_score_cell(r["score_b"]),
            ))
        title = f"Bridge candidates ({len(results)})"
        if self.bpm_offset and not results:
            title = "Bridge candidates — 0 matches within BPM filter"
        self._fill_table(BRIDGE_COLUMNS, rows, tracks, title, heights)

    def _update_detail(self) -> None:
        detail = self.query_one("#detail", Static)
        table = self.query_one(ResultsTable)
        row = table.cursor_row
        if not self._row_tracks or not (0 <= row < len(self._row_tracks)):
            detail.update("[dim]no selection[/]")
        elif self._row_results is not None and row < len(self._row_results):
            # Ergebnis-Modus: Score-Zerlegung statt roher Mood-Werte --
            # die stehen als Deltas ohnehin in den Spalten.
            detail.update(fmt_why_line(self._row_results[row]))
        else:
            detail.update(fmt_detail_line(self._row_tracks[row]))

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
        # Label in die System-Zwischenablage (OSC-52, geht durch WSL/SSH) --
        # der natuerliche naechste Handgriff ist die Suche im DJ-Tool.
        self.app.copy_to_clipboard(track_label(track))
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
        elif key == "ctrl+l":
            self.action_libraries()
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

        self.app.push_screen(
            AnalyzeModal(self.library.music_dirs, status=self._status_cache), done
        )

    def action_libraries(self) -> None:
        """Ctrl+L: Library-Screen ueber die Session legen -- Libraries
        umschalten, ohne die Session zu verlieren (Query/Energie bleiben,
        sofern der Track noch in der neuen Auswahl liegt)."""
        def done(paths: list[Path] | None) -> None:
            if paths is None:
                return  # Esc -- nichts geaendert
            self.app.library = Library(paths)
            self._after_library_change()

        self.app.push_screen(LibraryScreen(return_paths=True), done)

    def _after_library_change(self) -> None:
        """Session-Zustand gegen die neue Library abgleichen: Query und
        Transition-Ziel auf die neuen Track-Dicts umhaengen; ist die Query
        nicht mehr dabei, zurueck in die Filteransicht."""
        by_path = {t["filepath"]: t for t in self.library.tracks}
        if self.query_track is not None:
            self.query_track = by_path.get(self.query_track["filepath"])
        if self.transition_target is not None:
            self.transition_target = by_path.get(self.transition_target["filepath"])
        if self.query_track is None:
            self.transition_target = None
            self._selecting_target = False
            self.query_one(SearchInput).placeholder = SEARCH_PLACEHOLDER
        self._status_cache = None
        self.refresh_status()
        if self.query_track is not None and self._results_shown:
            self.show_results()
        else:
            self._results_shown = False
            self.show_filter(self.query_one(SearchInput).value)
        self._update_header()

    def action_transition(self) -> None:
        """Ctrl+T: Ziel-Auswahl starten bzw. den Transition-Modus verlassen."""
        if self._selecting_target:
            self._exit_target_selection()
        elif self.transition_target is not None:
            self.transition_target = None
            self.show_results()
            self._update_header()
        elif self.query_track is None:
            self.app.notify("Transition needs a query — select a track with Enter first.",
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
        Binding("escape", "close_or_cancel", "Cancel/Close"),
        Binding("enter", "start", "Start"),
    ]

    def __init__(self, music_dirs, status: tuple[int, int] | None = None):
        super().__init__()
        self._music_dirs = [Path(d) for d in music_dirs]
        self._status = status  # Cache vom Aufrufer -- kein erneuter Ordner-Scan
        self._state = "confirm"  # confirm -> running -> finished
        self._cancelled = False
        self._current_name = ""  # Track der Live-Statuszeile (::status track)
        self._proc: asyncio.subprocess.Process | None = None
        # Live-Zaehler aus den done-Events -- der statische "X of Y analyzed"-
        # Satz aus dem Confirm-Zustand wuerde waehrend des Laufs luegen
        # (er aenderte sich nie); waehrend des Laufs zaehlt diese Zeile mit.
        self._counts = {"full": 0, "tags": 0, "complete": 0, "error": 0}

    def compose(self) -> ComposeResult:
        with Vertical(id="analyze-box"):
            yield Static("[b]Analysis[/b]", id="analyze-title")
            yield Static(id="analyze-info", markup=True)
            yield ProgressBar(id="analyze-progress", show_eta=False)
            yield Static(id="analyze-current")
            yield RichLog(id="analyze-log", max_lines=500, wrap=True)
            yield Static(id="analyze-hint", markup=True)

    def on_mount(self) -> None:
        if self._status is not None:
            analyzed, total = self._status
            counts = f"{analyzed} of {total} tracks analyzed, [b]{total - analyzed} open[/b]."
        else:
            counts = "Analyzes new tracks and fills in missing embeddings."
        dirs = "\n".join(f"[dim]{escape(str(d))}[/]" for d in self._music_dirs)
        self.query_one("#analyze-info", Static).update(f"{dirs}\n{counts}")
        self.query_one(ProgressBar).display = False
        self.query_one("#analyze-current", Static).display = False
        self.query_one(RichLog).display = False
        self.query_one("#analyze-hint", Static).update(
            "[b]Enter[/b] starts the analysis   [b]Esc[/b] back"
        )

    def action_start(self) -> None:
        if self._state != "confirm":
            return
        self._state = "running"
        self.query_one(ProgressBar).display = True
        self.query_one("#analyze-current", Static).display = True
        self.query_one(RichLog).display = True
        self.query_one("#analyze-title", Static).update("[b]Analysis running[/b]")
        self.query_one("#analyze-hint", Static).update(
            "[dim]Esc cancels — finished tracks stay saved (resume)[/dim]"
        )
        self._stream_analysis()

    @work(exclusive=True)
    async def _stream_analysis(self) -> None:
        """Die Libraries laufen nacheinander als je ein Subprozess durch --
        der Fortschrittsbalken gilt pro Library, das Log kuendigt jede an."""
        log = self.query_one(RichLog)
        bar = self.query_one(ProgressBar)
        last_code = 0
        for music_dir in self._music_dirs:
            if self._cancelled:
                break
            if len(self._music_dirs) > 1:
                log.write(f"── {music_dir} ──")
            cmd = [
                sys.executable, "-u", "-m", "selecta", "analyze",
                "--music-dir", str(music_dir),
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
                elif text.startswith("::status "):
                    try:
                        event = json.loads(text[len("::status "):])
                    except ValueError:
                        continue
                    self._handle_status(event, log)
                elif text:
                    log.write(text)
            last_code = await self._proc.wait()
            if last_code != 0:
                break

        self.query_one("#analyze-current", Static).update("")
        self._state = "finished"
        if last_code == 0 and not self._cancelled:
            self.query_one("#analyze-title", Static).update("[b green]Analysis finished[/b green]")
        else:
            self.query_one("#analyze-title", Static).update(
                f"[b yellow]Analysis cancelled/failed (code {last_code})[/b yellow]"
            )
        self.query_one("#analyze-hint", Static).update("[dim]Esc closes[/dim]")

    def _handle_status(self, event: dict, log: RichLog) -> None:
        """Live-Statuszeile (track/stage) und Ergebniszeilen (done) aus den
        ::status-Events des Analyse-Subprozesses."""
        current = self.query_one("#analyze-current", Static)
        kind = event.get("event")
        if kind == "track":
            self._current_name = event.get("name", "?")
            line = Text("▶ ", style="cyan")
            line.append(self._current_name, style="bold")
            line.append("  starting …", style="dim")
            current.update(line)
        elif kind == "stage":
            line = Text("▶ ", style="cyan")
            line.append(self._current_name, style="bold")
            line.append(f"  [{event.get('step', '?')}/{event.get('steps', '?')}] ", style="dim")
            line.append(event.get("label", ""), style="cyan")
            current.update(line)
        elif kind == "done":
            current.update("")
            log.write(fmt_analysis_log_line(event))
            self._counts[event.get("kind", "complete")] = (
                self._counts.get(event.get("kind", "complete"), 0) + 1
            )
            self._update_live_counts()

    def _update_live_counts(self) -> None:
        c = self._counts
        scanned = sum(c.values())
        parts = [f"[b]{scanned}[/b] scanned", f"{c['full']} analyzed"]
        if c["tags"]:
            parts.append(f"{c['tags']} backfilled")
        parts.append(f"{c['complete']} complete")
        if c["error"]:
            parts.append(f"[red]{c['error']} errors[/red]")
        dirs = "\n".join(f"[dim]{escape(str(d))}[/]" for d in self._music_dirs)
        self.query_one("#analyze-info", Static).update(f"{dirs}\n{' · '.join(parts)}")

    def action_close_or_cancel(self) -> None:
        if self._state == "running":
            self._cancelled = True
            if self._proc is not None and self._proc.returncode is None:
                self._proc.terminate()
            self.query_one("#analyze-hint", Static).update("[yellow]Cancelling …[/yellow]")
        elif self._state == "finished":
            self.dismiss(True)
        else:
            self.dismiss(False)


def resolve_music_dir(raw: str) -> Path:
    """Windows-Pfade (C:\\... oder C:/...) via wslpath nach /mnt/... uebersetzen,
    z.B. beim Einfuegen im Library-hinzufuegen-Dialog. Alles andere (/mnt/...,
    ~, relative Pfade) bleibt unveraendert -- WSL kennt kein C:\\, die App
    laeuft dort drin. Umschliessende Anfuehrungszeichen werden entfernt:
    Terminals pasten sie beim Drag & Drop eines Ordners mit."""
    raw = raw.strip().strip("'\"")
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


# ---------------------------------------------------------------------------
# Gemerkte Libraries (libraries.json)
# ---------------------------------------------------------------------------

CONFIG_DIR = Path.home() / ".local" / "share" / "selecta"
LIBRARIES_FILE = CONFIG_DIR / "libraries.json"
# Vorgaenger-Format (ein einziger gemerkter Pfad) -- wird beim ersten Start
# ohne libraries.json einmalig als erste Library uebernommen.
LAST_DIR_FILE = CONFIG_DIR / "last_dir"


def load_libraries() -> list[dict]:
    """Gemerkte Libraries: [{"path": str, "active": bool}, ...]."""
    try:
        raw = json.loads(LIBRARIES_FILE.read_text(encoding="utf-8"))
        return [
            {"path": str(entry["path"]), "active": bool(entry.get("active", True))}
            for entry in raw.get("libraries", [])
            if entry.get("path")
        ]
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    try:
        last = LAST_DIR_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        last = ""
    return [{"path": last, "active": True}] if last else []


def save_libraries(entries: list[dict]) -> None:
    try:
        LIBRARIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        LIBRARIES_FILE.write_text(
            json.dumps({"libraries": entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


# ---------------------------------------------------------------------------
# LibraryScreen: Start-Launcher -- Libraries verwalten und Auswahl starten
# ---------------------------------------------------------------------------

class AddLibraryModal(ModalScreen):
    """Neue Library anlegen: ein bewusst dummes Pfad-Eingabefeld. Statt
    eigener Tab-Vervollstaendigung (fragil, nie so gut wie die Shell)
    uebernimmt das Terminal die Arbeit: einen Ordner aus dem Dateimanager
    ins Fenster ziehen pastet den Pfad (ggf. mit Anfuehrungszeichen, die
    resolve_music_dir entfernt)."""

    BINDINGS = [Binding("escape", "cancel", show=False)]

    def compose(self) -> ComposeResult:
        with Vertical(id="add-box"):
            yield Static("[b]Add library[/b]")
            yield Static(
                "[b]Drag & drop[/b] a folder from your file manager into this "
                "window, or paste a path (Windows or /mnt path):",
                markup=True,
            )
            yield Input(placeholder="/mnt/g/Media/Musik/…", id="add-input")
            yield Static(id="add-error", markup=True)
            yield Static("[b]Enter[/b] confirm   [b]Esc[/b] cancel", markup=True)

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Input.Submitted)
    def _on_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            self.dismiss(None)
            return
        path = resolve_music_dir(event.value)
        if path.is_dir():
            self.dismiss(path)
        else:
            self.query_one("#add-error", Static).update(
                f"[red]Folder not found: {escape(str(path))}[/]"
            )

    def action_cancel(self) -> None:
        self.dismiss(None)


class LibraryScreen(Screen):
    """Launcher: gemerkte Libraries an-/abwaehlen, anlegen, entfernen,
    analysieren -- Enter startet die Suche ueber alle aktiven. Bewusst EIN
    Screen statt Menuebaum; die Pfadeingabe ist auf das Anlegen (seltener
    Vorgang) verbannt.

    Zwei Verwendungen: beim Start ohne Pfad-Argument als Basis-Screen
    (Enter oeffnet den MainScreen), und aus dem MainScreen per Ctrl+L
    gepusht (return_paths=True: dismiss() liefert die aktiven Pfade,
    None = unveraendert geschlossen)."""

    BINDINGS = [
        Binding("space", "toggle", show=False),
        Binding("a", "add", show=False),
        Binding("d", "remove", show=False),
        Binding("ctrl+a", "analyze", show=False),
        # priority: sonst schluckt die fokussierte DataTable das Enter
        # (RowSelected) -- Klick auf eine Zeile toggelt, Enter startet.
        Binding("enter", "start", show=False, priority=True),
        Binding("escape", "back", show=False),
    ]

    def __init__(self, return_paths: bool = False):
        super().__init__()
        self._return_paths = return_paths
        self.entries = load_libraries()
        self._statuses: dict[str, tuple[int, int] | str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="library-box"):
            yield Static(render_logo(), id="library-logo")
            yield DataTable(id="library-table")
            yield Static(id="library-hint", markup=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns(" ", "LIBRARY", "TRACKS", "PATH")
        self.query_one("#library-hint", Static).update(
            "[b]Space[/b]/click toggle  [b]A[/b] add  [b]D[/b] remove  "
            "[b]^a[/b] analyze  [b]Enter[/b] play  [b]^c[/b] quit"
        )
        self._render_entries()
        table.focus()
        self._scan_statuses()

    # --- Darstellung ---

    def _render_entries(self) -> None:
        table = self.query_one(DataTable)
        cursor = table.cursor_row
        table.clear()
        for entry in self.entries:
            path = entry["path"]
            active = entry["active"]
            checkbox = Text("[x]", style="bold green") if active else Text("[ ]", style="dim")
            name = Text(Path(path).name or path, style="bold" if active else "dim")
            table.add_row(checkbox, name, self._status_cell(path), Text(path, style="dim"))
        if self.entries:
            table.move_cursor(row=min(max(cursor, 0), len(self.entries) - 1))

    def _status_cell(self, path: str) -> Text:
        status = self._statuses.get(path)
        if status is None:
            return Text("…", style="dim")
        if status == "missing":
            return Text("folder missing", style="red")
        analyzed, total = status
        if total == 0:
            return Text("no audio files", style="red")
        if analyzed == total:
            return Text(f"{analyzed}/{total}", style="green")
        if analyzed == 0:
            return Text(f"0/{total}", style="red")
        return Text(f"{analyzed}/{total}", style="yellow")

    def _scan_statuses(self) -> None:
        """Ordner-Scans (os.walk, auf /mnt/g potenziell langsam) im Thread;
        die Zellen fuellen sich nach und nach."""
        self.run_worker(self._scan_worker, thread=True, exclusive=True, group="libstatus")

    def _scan_worker(self) -> None:
        for entry in list(self.entries):
            path = entry["path"]
            if path in self._statuses:
                continue
            status = dir_status(path) if Path(path).is_dir() else "missing"
            self.app.call_from_thread(self._apply_status, path, status)

    def _apply_status(self, path: str, status) -> None:
        self._statuses[path] = status
        self._render_entries()

    def _cursor_entry(self) -> dict | None:
        row = self.query_one(DataTable).cursor_row
        if 0 <= row < len(self.entries):
            return self.entries[row]
        return None

    # --- Aktionen ---

    def _toggle_entry(self, entry: dict | None) -> None:
        if entry is None:
            return
        entry["active"] = not entry["active"]
        save_libraries(self.entries)
        self._render_entries()

    def action_toggle(self) -> None:
        self._toggle_entry(self._cursor_entry())

    @on(DataTable.RowSelected)
    def _on_row_selected(self, event: DataTable.RowSelected) -> None:
        # Nur per Maus erreichbar -- Enter faengt das priority-Binding ab.
        if 0 <= event.cursor_row < len(self.entries):
            self._toggle_entry(self.entries[event.cursor_row])

    def action_add(self) -> None:
        def done(path: Path | None) -> None:
            if path is None:
                return
            if any(entry["path"] == str(path) for entry in self.entries):
                self.app.notify("Library is already in the list.", severity="warning")
                return
            for entry in self.entries:
                other = Path(entry["path"])
                if path.is_relative_to(other) or other.is_relative_to(path):
                    self.app.notify(
                        f"Note: nested with {other} — shared tracks are "
                        "deduplicated in the search.",
                        severity="warning",
                    )
            self.entries.append({"path": str(path), "active": True})
            save_libraries(self.entries)
            self._render_entries()
            self._scan_statuses()

        self.app.push_screen(AddLibraryModal(), done)

    def action_remove(self) -> None:
        entry = self._cursor_entry()
        if entry is None:
            return
        self.entries.remove(entry)
        save_libraries(self.entries)
        self._render_entries()
        self.app.notify("Entry removed — the CSV inside the folder is kept.")

    def action_analyze(self) -> None:
        entry = self._cursor_entry()
        if entry is None:
            return
        if not Path(entry["path"]).is_dir():
            self.app.notify("Folder not found — drive connected?", severity="error")
            return

        def done(_result) -> None:
            self._statuses.pop(entry["path"], None)
            self._render_entries()
            self._scan_statuses()

        status = self._statuses.get(entry["path"])
        status = status if isinstance(status, tuple) else None
        self.app.push_screen(AnalyzeModal([entry["path"]], status=status), done)

    def action_start(self) -> None:
        active = [entry["path"] for entry in self.entries if entry["active"]]
        missing = [p for p in active if not Path(p).is_dir()]
        if missing:
            self.app.notify(
                f"Folder not found: {missing[0]} — drive connected?",
                severity="error",
            )
            return
        if not active:
            hint = "press A to add one first." if not self.entries else "activate one first (Space)."
            self.app.notify(f"No active library — {hint}", severity="warning")
            return
        paths = [Path(p) for p in active]
        if self._return_paths:
            self.dismiss(paths)
        else:
            self.app.open_library(paths)

    def action_back(self) -> None:
        if self._return_paths:
            self.dismiss(None)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class SelectaApp(App):
    TITLE = "SELECTA"
    # Ctrl+C zusaetzlich zu Ctrl+Q: im Raw-Mode der TUI ist Ctrl+C nur eine
    # Taste (kein SIGINT), und Ctrl+Q wird z.B. vom VSCode-Terminal geschluckt.
    # Beenden verliert nichts -- die CSV ist immer persistiert.
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
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
    #search, #add-input {
        border: tall $accent;
        margin: 0 1;
    }
    #transition-bar {
        height: 1;
        margin: 0 1;
        padding: 0 1;
        background: $panel;
        display: none;
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
    #analyze-current {
        height: 1;
        margin-bottom: 1;
    }
    #analyze-log {
        height: 1fr;
        border: round $accent 30%;
    }
    LibraryScreen {
        align: center middle;
    }
    #library-box {
        width: 80;
        height: auto;
        max-height: 100%;
        padding: 1 2;
    }
    #library-logo {
        width: 100%;
        margin-bottom: 1;
    }
    #library-table {
        height: auto;
        max-height: 14;
        border: round $accent 50%;
    }
    #library-hint {
        margin-top: 1;
        width: 100%;
        content-align: center middle;
        color: $text-muted;
    }
    AddLibraryModal {
        align: center middle;
    }
    #add-box {
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
            # Ad-hoc-Modus (`selecta /pfad`): direkt in die Suche, ohne die
            # gemerkte Library-Liste anzufassen.
            self.open_library([Path(self._initial_dir)])
        else:
            self.push_screen(LibraryScreen())

    def open_library(self, music_dirs: list[Path]) -> None:
        self.library = Library(music_dirs)
        self.push_screen(MainScreen())
