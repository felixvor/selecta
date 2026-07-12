"""Headless-Smoke-Tests der TUI ueber Textuals Pilot (kein echtes Terminal noetig)."""

import pytest

import selecta.app as app_module
from selecta.app import (
    AddLibraryModal,
    LibraryScreen,
    MainScreen,
    ResultsTable,
    SearchInput,
    SelectaApp,
    chip_line,
    fmt_track_cell,
    genre_chip_color,
    load_libraries,
    resolve_music_dir,
    save_libraries,
)
from selecta.library import compact_csv
from tests.conftest import make_row


@pytest.fixture
def app(synthetic_library):
    a = SelectaApp(music_dir=synthetic_library.music_dirs[0])
    return a


async def test_start_zeigt_library_liste(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        assert isinstance(screen, MainScreen)
        table = screen.query_one(ResultsTable)
        assert table.row_count == 6  # ganze Library alphabetisch


async def test_tippen_filtert_und_enter_waehlt_query(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a")
        table = screen.query_one(ResultsTable)
        assert table.row_count >= 1
        await pilot.press("enter")
        assert screen.query_track is not None
        assert screen.query_track["title"] == "Groove A"
        # Ergebnisliste: alle anderen Tracks, Query nicht enthalten
        assert table.row_count == 5
        assert screen._results_shown
        # Suchfeld ist geleert und fokussiert -> direkt weitertippen moeglich
        assert screen.query_one(SearchInput).value == ""


async def test_enter_auf_ergebnis_wird_neue_query(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        first_query = screen.query_track["filepath"]
        await pilot.press("enter")  # Top-Ergebnis uebernehmen
        assert screen.query_track["filepath"] != first_query


async def test_energie_taste_rerankt(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        neutral_order = [t["filepath"] for t in screen._row_tracks]
        await pilot.press("right", "right", "right")
        assert screen.energy == 3
        pushed_order = [t["filepath"] for t in screen._row_tracks]
        assert neutral_order.index("house_fast.mp3") > pushed_order.index("house_fast.mp3")
        # Grenze haelt
        await pilot.press("right", "right", "right", "right")
        assert screen.energy == 6


async def test_bpm_filter_springt_auf_naechsten_track(app):
    """',' / '.' springen exakt auf den naechsten in der Library
    vorhandenen BPM-Wert -- kein fixer Schritt, kein Ueberspringen."""
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")  # Query = house_a, 124 BPM
        await pilot.press("full_stop")
        assert screen.bpm_offset == 2  # naechster schnellerer Track: house_b (126)
        await pilot.press("full_stop")
        assert screen.bpm_offset == 8  # naechster: house_fast (132)
        await pilot.press("comma")
        assert screen.bpm_offset == 2  # zurueck auf 126 (Filter lockern)
        await pilot.press("comma", "comma")
        assert screen.bpm_offset == -44  # via house_slow (118) weiter zu ambient (80)
        await pilot.press("comma")
        assert screen.bpm_offset == -44  # unterer Rand erreicht -- bleibt stehen


async def test_buchstaben_tippen_immer(app):
    """'a'/'t' sind Buchstaben, keine Aktionen -- sonst koennte man keinen
    Track suchen, der mit A anfaengt (z.B. 'Ambush')."""
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"at")  # leeres Feld: erste Buchstaben eines Titels
        assert screen.query_one(SearchInput).value == "at"
        assert isinstance(app.screen, MainScreen)


async def test_ctrl_a_oeffnet_analyse_auch_beim_tippen(app):
    from selecta.app import AnalyzeModal

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press(*"amb", "ctrl+a")
        assert isinstance(app.screen, AnalyzeModal)


async def test_escape_leert_suche(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press(*"xyz")
        assert screen.query_one(SearchInput).value == "xyz"
        await pilot.press("escape")
        assert screen.query_one(SearchInput).value == ""
        assert screen._results_shown  # zurueck bei den Ergebnissen


async def test_transition_ziel_pinnen_und_re_anchor(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")   # A = house_a
        await pilot.press("ctrl+t")
        assert screen._selecting_target
        await pilot.press(*"hammer", "enter")     # B = techno
        assert not screen._selecting_target
        assert screen.transition_target["title"] == "Hammer"
        table = screen.query_one(ResultsTable)
        assert table.row_count == 5               # alle ausser A, inkl. B
        # Enter auf den Top-Kandidaten: wird neues A, Ziel bleibt gepinnt
        await pilot.press("enter")
        assert screen.transition_target is not None
        assert screen.query_track["filepath"] != "house_a.mp3"


async def test_transition_ziel_waehlen_beendet_modus(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press("ctrl+t")
        await pilot.press(*"hammer", "enter")
        # B ueber die Suche direkt anspringen -> Transition fertig, B ist Query
        await pilot.press(*"hammer", "enter")
        assert screen.transition_target is None
        assert screen.query_track["title"] == "Hammer"


async def test_transition_esc_schichten(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press("ctrl+t")
        await pilot.press("escape")               # bricht die Ziel-Auswahl ab
        assert not screen._selecting_target
        assert screen.transition_target is None
        assert screen._results_shown
        await pilot.press("ctrl+t")
        await pilot.press(*"hammer", "enter")
        await pilot.press("escape")               # loescht das gepinnte Ziel
        assert screen.transition_target is None
        assert screen._results_shown


def test_fmt_transition_bar_inhalt():
    """Inhalt der Bar: A immer sichtbar; ohne Ziel der Auswahl-Hinweis,
    mit Ziel B plus Direkt-Score."""
    from selecta.app import fmt_transition_bar

    a = {"artist": "HouseArtist", "title": "Groove A", "filepath": "a.mp3"}
    b = {"artist": "TechArtist", "title": "Hammer", "filepath": "b.mp3"}
    selecting = fmt_transition_bar(a, None, None).plain
    assert "Groove A" in selecting and "select target" in selecting
    pinned = fmt_transition_bar(a, b, 0.723).plain
    assert "Groove A" in pinned and "Hammer" in pinned and "0.723" in pinned


async def test_transition_bar_sichtbarkeit_folgt_dem_modus(app):
    """Die Bar macht sichtbar, von welchem Track (A) aus gesucht wird --
    vorher verschwand diese Info, sobald Ctrl+T gedrueckt war."""
    from textual.widgets import Static

    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        bar = screen.query_one("#transition-bar", Static)
        assert not bar.display                     # ohne Transition: versteckt
        await pilot.press(*"groove a", "enter")
        assert not bar.display
        await pilot.press("ctrl+t")                # Ziel-Auswahl laeuft: A sichtbar
        assert bar.display
        await pilot.press(*"hammer", "enter")      # B = techno gepinnt
        assert bar.display
        # Re-Anchor per Enter: A wechselt, Bar bleibt
        await pilot.press("enter")
        assert screen.query_track["filepath"] != "house_a.mp3"
        assert bar.display
        # Enter auf B beendet die Transition -> Bar verschwindet
        await pilot.press(*"hammer", "enter")
        assert screen.transition_target is None
        assert not bar.display


async def test_transition_liste_markiert_ziel_b(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press("ctrl+t")
        await pilot.press(*"hammer", "enter")
        table = screen.query_one(ResultsTable)
        b_index = next(i for i, t in enumerate(screen._row_tracks)
                       if t["filepath"] == "techno.mp3")
        rank_cell = table.get_row_at(b_index)[0]
        assert rank_cell.plain == "◆B"
        # alle anderen Zeilen tragen normale Ranknummern
        other = table.get_row_at(0 if b_index != 0 else 1)[0]
        assert other.plain.isdigit()


async def test_transition_energie_inaktiv(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press("ctrl+t")
        await pilot.press(*"hammer", "enter")
        await pilot.press("right")
        assert screen.energy == 0                 # Energie im Transition-Modus aus


async def test_analyze_modal_bestaetigung_esc_startet_nichts(app):
    from selecta.app import AnalyzeModal

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+a")
        modal = app.screen
        assert isinstance(modal, AnalyzeModal)
        assert modal._state == "confirm"
        assert modal._proc is None  # kein Subprozess vor Bestaetigung
        await pilot.press("escape")
        assert isinstance(app.screen, MainScreen)


# --- Chip-Zeile (Genre/Vibe/Jahr) --------------------------------------------

def test_chip_line_mit_tags():
    track = {"genres": "Acid House|Deep House", "vibes": "dark|groovy", "year": "1994"}
    line = chip_line(track)
    text = line.plain
    assert " Acid House " in text and " Deep House " in text
    assert "dark · groovy · 1994" in text


def test_chip_line_leer_bei_alter_csv():
    # Zeile aus einer CSV vor dem Genre-Schema: keine zweite Zeile, Hoehe 1.
    track = {"filepath": "x.mp3", "genres": "", "vibes": "", "year": ""}
    assert chip_line(track) is None
    _cell, height = fmt_track_cell(track)
    assert height == 1


def test_fmt_track_cell_hoehe_2_mit_chips():
    track = {"filepath": "x.mp3", "artist": "A", "title": "T", "genres": "Techno", "vibes": "", "year": ""}
    cell, height = fmt_track_cell(track)
    assert height == 2
    assert cell.plain.startswith("A - T\n")


def test_genre_chip_color_stabil():
    assert genre_chip_color("Acid House") == genre_chip_color("Acid House")


def test_fmt_key_cell_markiert_geschaetzten_key():
    """Geschaetzte Keys (key_estimated) tragen ein ~-Praefix und sind
    gedimmt -- am Pult muss sichtbar sein, welchem Wert man trauen kann."""
    from selecta.app import display_key, fmt_key_cell

    tagged = {"key": "9m", "key_estimated": ""}
    estimated = {"key": "9m", "key_estimated": "1"}
    assert display_key(tagged) == "9m"
    assert display_key(estimated) == "~9m"
    assert display_key({"key": ""}) == "?"
    assert fmt_key_cell(tagged, None).plain == "9m"
    assert fmt_key_cell(estimated, None).plain == "~9m"
    # mit Query: harmonische Farbe bleibt, aber gedimmt
    query = {"key": "9m"}
    cell = fmt_key_cell(estimated, query)
    assert cell.plain == "~9m"
    assert "dim" in str(cell.style)


def test_fmt_analysis_log_line_volle_analyse():
    from selecta.app import fmt_analysis_log_line

    info = {"event": "done", "kind": "full", "name": "house_a.mp3",
            "genres": "Acid House|Deep House", "vibes": "dark|groovy", "year": "1994",
            "bpm": "124", "key": "7m", "arousal": 6.0, "aggressive": 0.2,
            "danceable": 0.99, "secs": 5.3, "error": ""}
    text = fmt_analysis_log_line(info).plain
    assert text.startswith("✓ house_a.mp3  (5.3s)\n")  # Zeile 1: Name + Dauer
    assert " Acid House " in text and " Deep House " in text  # Zeile 2: Chips
    assert "dark · groovy · 1994" in text
    assert "124 BPM · 7m" in text and "arous 6.0" in text


def test_fmt_analysis_log_line_kompakte_faelle():
    from selecta.app import fmt_analysis_log_line

    assert fmt_analysis_log_line(
        {"kind": "complete", "name": "x.mp3"}).plain == "≡ x.mp3"
    assert "BPM 128 · key 7m backfilled" in fmt_analysis_log_line(
        {"kind": "tags", "name": "x.mp3", "bpm": "128", "key": "7m", "secs": 2.0}).plain
    assert "key ~9m" in fmt_analysis_log_line(
        {"kind": "tags", "name": "x.mp3", "bpm": "128", "key": "9m",
         "key_estimated": "1", "secs": 2.0}).plain
    assert fmt_analysis_log_line(
        {"kind": "error", "name": "x.mp3", "error": "kaputt"}).plain == "✗ x.mp3: kaputt"


async def test_liste_zeigt_chips_und_bleibt_bedienbar(app):
    """Tracks mit Chips bekommen 2-zeilige Rows; Auswahl per Enter
    funktioniert unveraendert."""
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        table = screen.query_one(ResultsTable)
        assert table.row_count == 6
        heights = {screen._row_tracks[i]["filepath"]: table.ordered_rows[i].height
                   for i in range(table.row_count)}
        assert heights["house_a.mp3"] == 2   # hat Genres/Vibes/Jahr
        assert heights["techno.mp3"] == 1    # ohne Tags einzeilig
        await pilot.press(*"groove a", "enter")
        assert screen.query_track["title"] == "Groove A"


async def test_ctrl_c_beendet(app):
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+c")
        assert app.return_value is None and app._exit


# --- Library-Liste (libraries.json) ------------------------------------------

@pytest.fixture
def libraries_file(tmp_path, monkeypatch):
    """Isoliert die Persistenz vom echten ~/.local/share/selecta."""
    lib_file = tmp_path / "libraries.json"
    monkeypatch.setattr(app_module, "LIBRARIES_FILE", lib_file)
    monkeypatch.setattr(app_module, "LAST_DIR_FILE", tmp_path / "last_dir")
    return lib_file


@pytest.fixture
def two_music_dirs(tmp_path):
    """Zwei Ordner mit eigener CSV: house (2 Tracks) und techno (1 Track)."""
    dir_a = tmp_path / "house"
    dir_a.mkdir()
    compact_csv(dir_a / "library_analysis.csv", dict([
        make_row("house_a.mp3", "HouseArtist", "Groove A", 124, "7m", [1.0, 0.1, 0.0]),
        make_row("house_b.mp3", "HouseArtist", "Groove B", 126, "8m", [0.95, 0.15, 0.0]),
    ]))
    dir_b = tmp_path / "techno"
    dir_b.mkdir()
    compact_csv(dir_b / "library_analysis.csv", dict([
        make_row("techno.mp3", "TechArtist", "Hammer", 140, "2m", [0.1, 1.0, 0.0]),
    ]))
    return dir_a, dir_b


def test_libraries_roundtrip(libraries_file):
    entries = [{"path": "/mnt/g/house", "active": True}, {"path": "/mnt/g/techno", "active": False}]
    save_libraries(entries)
    assert load_libraries() == entries


def test_libraries_kaputtes_json_ergibt_leere_liste(libraries_file):
    libraries_file.write_text("{kaputt", encoding="utf-8")
    assert load_libraries() == []


def test_libraries_migration_von_last_dir(libraries_file, tmp_path):
    # Vorgaenger-Version hat nur einen Pfad in last_dir gemerkt -- der wird
    # beim ersten Start ohne libraries.json zur ersten Library.
    (tmp_path / "last_dir").write_text("/mnt/g/house", encoding="utf-8")
    assert load_libraries() == [{"path": "/mnt/g/house", "active": True}]


def test_resolve_music_dir_entfernt_dragdrop_anfuehrungszeichen(tmp_path):
    # Terminals pasten beim Drag & Drop den Pfad in Anfuehrungszeichen
    assert resolve_music_dir(f"'{tmp_path}'") == tmp_path
    assert resolve_music_dir(f'"{tmp_path}"') == tmp_path


# --- LibraryScreen ------------------------------------------------------------

async def test_start_ohne_pfad_zeigt_library_screen_und_enter_startet(two_music_dirs, libraries_file):
    dir_a, dir_b = two_music_dirs
    save_libraries([{"path": str(dir_a), "active": True}, {"path": str(dir_b), "active": True}])
    app = SelectaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        assert isinstance(app.screen, LibraryScreen)
        await pilot.press("enter")
        assert isinstance(app.screen, MainScreen)
        assert len(app.library.tracks) == 3  # beide Libraries zusammen


async def test_library_screen_space_toggelt_und_persistiert(two_music_dirs, libraries_file):
    dir_a, dir_b = two_music_dirs
    save_libraries([{"path": str(dir_a), "active": True}, {"path": str(dir_b), "active": True}])
    app = SelectaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("down", "space")  # techno abwaehlen
        assert load_libraries()[1]["active"] is False
        await pilot.press("enter")
        assert isinstance(app.screen, MainScreen)
        assert len(app.library.tracks) == 2  # nur house


async def test_library_screen_add_und_remove(two_music_dirs, libraries_file):
    dir_a, dir_b = two_music_dirs
    save_libraries([{"path": str(dir_a), "active": True}])
    app = SelectaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("a")
        assert isinstance(app.screen, AddLibraryModal)
        app.screen.query_one("#add-input").value = str(dir_b)
        await pilot.press("enter")
        assert isinstance(app.screen, LibraryScreen)
        assert [e["path"] for e in load_libraries()] == [str(dir_a), str(dir_b)]
        # D entfernt nur den Listeneintrag, nicht die CSV im Ordner
        await pilot.press("down", "d")
        assert [e["path"] for e in load_libraries()] == [str(dir_a)]
        assert (dir_b / "library_analysis.csv").exists()


async def test_library_screen_ohne_aktive_library_startet_nicht(two_music_dirs, libraries_file):
    dir_a, _dir_b = two_music_dirs
    save_libraries([{"path": str(dir_a), "active": False}])
    app = SelectaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("enter")
        assert isinstance(app.screen, LibraryScreen)  # bleibt stehen + Notify


async def test_ctrl_l_schaltet_libraries_in_laufender_session_um(two_music_dirs, libraries_file):
    dir_a, dir_b = two_music_dirs
    save_libraries([{"path": str(dir_a), "active": True}, {"path": str(dir_b), "active": False}])
    app = SelectaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("enter")  # Start nur mit house
        assert len(app.library.tracks) == 2
        await pilot.press("ctrl+l")
        assert isinstance(app.screen, LibraryScreen)
        await pilot.press("down", "space", "enter")  # techno dazu, uebernehmen
        screen = app.screen
        assert isinstance(screen, MainScreen)
        assert len(app.library.tracks) == 3
        assert screen.query_one(ResultsTable).row_count == 3


async def test_ctrl_l_esc_laesst_alles_unveraendert(two_music_dirs, libraries_file):
    dir_a, dir_b = two_music_dirs
    save_libraries([{"path": str(dir_a), "active": True}, {"path": str(dir_b), "active": True}])
    app = SelectaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("enter")
        library_before = app.library
        await pilot.press(*"groove a", "enter")  # Query setzen
        await pilot.press("ctrl+l", "escape")
        screen = app.screen
        assert isinstance(screen, MainScreen)
        assert app.library is library_before
        assert screen.query_track["title"] == "Groove A"


async def test_ctrl_l_query_bleibt_wenn_track_noch_dabei(two_music_dirs, libraries_file):
    dir_a, dir_b = two_music_dirs
    save_libraries([{"path": str(dir_a), "active": True}, {"path": str(dir_b), "active": False}])
    app = SelectaApp()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("enter")
        await pilot.press(*"groove a", "enter")  # Query = house_a
        await pilot.press("ctrl+l")
        await pilot.press("down", "space", "enter")  # techno dazu
        screen = app.screen
        assert screen.query_track["filepath"] == "house_a.mp3"  # Session lebt weiter
        assert screen._results_shown
        assert screen.query_one(ResultsTable).row_count == 2  # alle ausser Query


async def test_suche_mit_query_behaelt_scores(app):
    """Tippen mit gesetzter Query schaltet die Aehnlichkeit nicht mehr ab:
    Suchtreffer zeigen die Score-Spalten (RESULT_COLUMNS) und die
    Detail-Zeile die Score-Zerlegung."""
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")   # Query setzen
        await pilot.press(*"hammer")              # suchen statt Ranking
        table = screen.query_one(ResultsTable)
        assert table.row_count == 1
        labels = [str(col.label) for col in table.columns.values()]
        assert "SCORE" in labels
        # _row_results traegt das Result-Dict fuer die Warum-Zeile
        assert screen._row_results is not None
        assert screen._row_results[0]["track"]["title"] == "Hammer"


async def test_zielauswahl_zeigt_direkten_score(app):
    """In der Transition-Ziel-Auswahl zeigt die Suche SCORE→A -- den
    direkten Sprung von der aktuellen Query zum Kandidaten."""
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press("ctrl+t")
        await pilot.press(*"hammer")
        table = screen.query_one(ResultsTable)
        labels = [str(col.label) for col in table.columns.values()]
        assert "SCORE→A" in labels
        assert table.row_count == 1
