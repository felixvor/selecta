"""Headless-Smoke-Tests der TUI ueber Textuals Pilot (kein echtes Terminal noetig)."""

import pytest

from selecta.app import MainScreen, ResultsTable, SearchInput, SelectaApp, TransitionScreen


@pytest.fixture
def app(synthetic_library):
    a = SelectaApp(music_dir=synthetic_library.music_dir)
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
        await pilot.press("right")
        assert screen.energy == 3


async def test_bpm_feintuning_keys(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press("full_stop")
        assert screen.bpm_offset == 4
        await pilot.press("comma", "comma")
        assert screen.bpm_offset == -4


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


async def test_transition_screen_flow(app):
    async with app.run_test(size=(120, 40)) as pilot:
        screen = app.screen
        await pilot.press(*"groove a", "enter")
        await pilot.press("ctrl+t")
        assert isinstance(app.screen, TransitionScreen)
        # A ist mit der aktuellen Query vorbelegt, B eingeben:
        await pilot.press(*"hammer", "enter")
        tscreen = app.screen
        table = tscreen.query_one("#transition-table", ResultsTable)
        assert table.row_count >= 3  # A + mindestens 1 + B
        assert tscreen._row_tracks[0]["title"] == "Groove A"
        assert tscreen._row_tracks[-1]["title"] == "Hammer"
        # Zwischentrack uebernehmen -> zurueck in der Suche mit neuer Query
        await pilot.press("down", "enter")
        assert isinstance(app.screen, MainScreen)
        assert app.screen.query_track["filepath"] == tscreen._row_tracks[1]["filepath"]


async def test_transition_esc_zurueck(app):
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+t")
        assert isinstance(app.screen, TransitionScreen)
        await pilot.press("escape")
        assert isinstance(app.screen, MainScreen)


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


async def test_ctrl_c_beendet(app):
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+c")
        assert app.return_value is None and app._exit
