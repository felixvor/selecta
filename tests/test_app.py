"""Headless-Smoke-Tests der TUI ueber Textuals Pilot (kein echtes Terminal noetig)."""

import pytest

from selecta.app import MainScreen, ResultsTable, SearchInput, SelectaApp


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


async def test_ctrl_c_beendet(app):
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("ctrl+c")
        assert app.return_value is None and app._exit
