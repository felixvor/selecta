from selecta.analysis import pick_genres, pick_vibes
from selecta.library import missing_parts as _missing_parts


def test_missing_row_needs_embedding():
    assert _missing_parts(None) == {"embedding"}


def test_error_row_needs_embedding():
    row = {"error": "boom", "embedding": ""}
    assert _missing_parts(row) == {"embedding"}


def test_ok_row_without_embedding_needs_embedding():
    # theoretischer Alt-Schema-Fall (CSV vor Einfuehrung der Embedding-Spalte)
    row = {"error": "", "embedding": ""}
    assert _missing_parts(row) == {"embedding"}


def test_old_schema_row_without_effnet_needs_full_analysis():
    # CSV von vor dem Genre/Vibe-Schema: Track-Embedding da, aber kein
    # effnet_embedding -> einmalige volle Re-Analyse.
    row = {"error": "", "embedding": "xyz", "bpm": "128", "key": "7m"}
    assert _missing_parts(row) == {"embedding"}


def test_ok_row_missing_bpm_needs_tags():
    row = {"error": "", "embedding": "xyz", "effnet_embedding": "abc", "bpm": "", "key": "7m"}
    assert _missing_parts(row) == {"tags"}


def test_ok_row_with_bpm_but_no_key_needs_tags():
    # Seit der Key-Schaetzung (compute_key) kann eine Zeile ohne Key fertig
    # werden -- fehlender Key triggert daher wie BPM den billigen tags-Pfad.
    row = {"error": "", "embedding": "xyz", "effnet_embedding": "abc", "bpm": "128.0", "key": ""}
    assert _missing_parts(row) == {"tags"}


def test_row_with_estimated_key_is_done():
    # Ein geschaetzter Key (key_estimated=1) zaehlt als gesetzt -- die Zeile
    # bleibt fertig und wird nur noch vom Tag-Re-Check angefasst.
    row = {"error": "", "embedding": "xyz", "effnet_embedding": "abc",
           "bpm": "128.0", "key": "9m", "key_estimated": "1"}
    assert _missing_parts(row) == set()


def test_row_with_empty_vibes_is_done():
    # vibes darf legitim leer bleiben (kein klarer Vibe ueber der Schwelle)
    # und haelt die Zeile nicht offen -- Marker ist das effnet_embedding.
    row = {"error": "", "embedding": "xyz", "effnet_embedding": "abc",
           "bpm": "128", "key": "7m", "genres": "Techno", "vibes": ""}
    assert _missing_parts(row) == set()


def test_fully_tagged_row_is_done():
    row = {"error": "", "embedding": "xyz", "effnet_embedding": "abc", "bpm": "128", "key": "7m"}
    assert _missing_parts(row) == set()


# --- Genre-/Vibe-Ableitung ---------------------------------------------------

GENRE_LABELS = ["Electronic---Acid House", "Electronic---Tech House", "Rock---Indie Rock"]


def test_pick_genres_top1_und_zweiter_ueber_schwelle():
    assert pick_genres([0.6, 0.2, 0.01], GENRE_LABELS) == "Acid House|Tech House"


def test_pick_genres_zweiter_unter_schwelle_faellt_raus():
    assert pick_genres([0.6, 0.05, 0.01], GENRE_LABELS) == "Acid House"


def test_pick_genres_top1_auch_bei_niedriger_konfidenz():
    # Top-1 immer -- sonst waere 'genres' bei unsicheren Tracks leer.
    assert pick_genres([0.04, 0.03, 0.01], GENRE_LABELS) == "Acid House"


def test_pick_genres_dedupliziert_gleiche_stylenamen():
    labels = ["Electronic---House", "Rock---House"]
    assert pick_genres([0.5, 0.4], labels) == "House"


VIBE_LABELS = ["dark", "children", "deep", "fast"]


def test_pick_vibes_whitelist_und_schwelle():
    # 'children' ist nicht in der Whitelist (staerkste Aktivierung egal),
    # 'fast' liegt unter der Schwelle; Reihenfolge: staerkste zuerst.
    assert pick_vibes([0.5, 0.9, 0.62, 0.05], VIBE_LABELS) == "deep|dark"


def test_pick_vibes_darf_leer_sein():
    assert pick_vibes([0.01, 0.02, 0.03, 0.04], VIBE_LABELS) == ""


# --- Status-Events / Log-Zeilen von run_analysis ------------------------------

def test_run_analysis_emittiert_done_events(tmp_path):
    """Vollstaendige Zeile: run_analysis laeuft ohne Modelle durch und meldet
    pro Datei ein done-Event (kind=complete) statt einer Log-Zeile."""
    from selecta.analysis import run_analysis
    from selecta.library import compact_csv
    from tests.conftest import make_row

    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    track = music_dir / "done.mp3"
    track.write_bytes(b"")  # kaputtes mp3 reicht -- Tags lesen schlaegt still fehl
    compact_csv(music_dir / "library_analysis.csv",
                dict([make_row(str(track), "X", "A", 124, "7m", [1, 0, 0])]))

    events = []
    done, errors = run_analysis(music_dir, tmp_path / "models",
                                log=lambda _msg: None, status=events.append)
    assert (done, errors) == (0, 0)
    dones = [e for e in events if e["event"] == "done"]
    assert len(dones) == 1
    assert dones[0]["kind"] == "complete"
    assert dones[0]["name"] == "done.mp3"
    assert dones[0]["genres"] == ""  # make_row-Default, kommt aus der CSV-Zeile


def test_run_analysis_ohne_status_loggt_lesbare_zeile(tmp_path):
    from selecta.analysis import run_analysis
    from selecta.library import compact_csv
    from tests.conftest import make_row

    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    track = music_dir / "done.mp3"
    track.write_bytes(b"")
    compact_csv(music_dir / "library_analysis.csv",
                dict([make_row(str(track), "X", "A", 124, "7m", [1, 0, 0])]))

    lines = []
    run_analysis(music_dir, tmp_path / "models", log=lines.append)
    assert any(line.startswith("≡ done.mp3") for line in lines)


def test_human_line_volle_analyse():
    from selecta.analysis import _done_event, _human_line
    from pathlib import Path

    row = {"genres": "Acid House|Deep House", "vibes": "dark|groovy", "year": "1994",
           "bpm": "124", "key": "7m", "arousal": 6.0, "aggressive": 0.2,
           "danceable": 0.99, "error": ""}
    info = _done_event("full", Path("house_a.mp3"), row, 5.31)
    line = _human_line(info)
    assert line.startswith("✓ house_a.mp3")
    assert "Acid House | Deep House" in line
    assert "dark groovy" in line and "1994" in line
    assert "124 BPM 7m" in line and "(5.3s)" in line


def test_human_line_markiert_geschaetzten_key():
    from selecta.analysis import _done_event, _human_line
    from pathlib import Path

    row = {"bpm": "128.0", "key": "9m", "key_estimated": "1", "error": ""}
    info = _done_event("tags", Path("x.mp3"), row, 2.0)
    line = _human_line(info)
    assert "BPM 128.0 · key ~9m backfilled" in line


def test_analyzer_stages_passen_zur_stage_anzahl():
    """Die STAGES-Liste ist die Referenz fuer die [k/n]-Anzeige -- analyze()
    meldet jede Etappe genau einmal (geprueft ohne TensorFlow ueber die
    Klassenkonstante)."""
    from selecta.analysis import EssentiaAnalyzer

    assert len(EssentiaAnalyzer.STAGES) == 6
    assert len(set(EssentiaAnalyzer.STAGES)) == 6
