import numpy as np

from selecta.library import (
    Library,
    compact_csv,
    decode_embedding,
    encode_embedding,
    load_csv_data,
    track_label,
)
from tests.conftest import make_row


def test_embedding_roundtrip():
    vec = np.random.randn(512).astype(np.float32)
    assert np.allclose(vec, decode_embedding(encode_embedding(vec)))


def test_compact_csv_dedupliziert(tmp_path):
    csv_path = tmp_path / "library_analysis.csv"
    rows = dict([make_row("a.mp3", "X", "Y", 126, "7m", [1, 0, 0])])
    compact_csv(csv_path, rows)
    compact_csv(csv_path, load_csv_data(csv_path))  # zweiter Lauf: idempotent
    with open(csv_path) as f:
        assert len(f.readlines()) == 2  # Header + 1 Zeile


def test_compact_csv_prune_missing(tmp_path):
    csv_path = tmp_path / "library_analysis.csv"
    existing = tmp_path / "existiert.mp3"
    existing.write_bytes(b"")
    rows = dict([
        make_row(str(existing), "X", "A", 126, "7m", [1, 0, 0]),
        make_row(str(tmp_path / "geloescht.mp3"), "X", "B", 126, "7m", [0, 1, 0]),
    ])
    compact_csv(csv_path, rows, prune_missing=True)
    reloaded = load_csv_data(csv_path)
    assert str(existing) in reloaded
    assert len(reloaded) == 1


def test_library_laedt_nur_ok_mit_embedding(tmp_path):
    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    csv_path = music_dir / "library_analysis.csv"
    fp_ok, row_ok = make_row("ok.mp3", "X", "A", 126, "7m", [1, 0, 0])
    fp_err, row_err = make_row("err.mp3", "X", "B", 126, "7m", [0, 1, 0])
    row_err["status"] = "error: kaputt"
    fp_alt, row_alt = make_row("alt.mp3", "X", "C", 126, "7m", [0, 0, 1])
    row_alt["embedding"] = ""  # Zeile aus altem Tool ohne Embedding
    compact_csv(csv_path, {fp_ok: row_ok, fp_err: row_err, fp_alt: row_alt})

    lib = Library(music_dir)
    assert [t["filepath"] for t in lib.tracks] == ["ok.mp3"]
    assert lib.matrix.shape == (1, 3)
    assert np.allclose(np.linalg.norm(lib.matrix, axis=1), 1.0)  # zeilennormalisiert


def test_library_status_zaehlt_dateien_im_ordner(tmp_path, synthetic_library):
    # synthetic_library hat 6 CSV-Zeilen, aber keine echten Dateien im Ordner
    analyzed, total = synthetic_library.status()
    assert (analyzed, total) == (0, 0)

    # lege eine "analysierte" und eine neue Datei an
    music_dir = synthetic_library.music_dir
    (music_dir / "house_a.mp3").write_bytes(b"")
    (music_dir / "neu_dazu.mp3").write_bytes(b"")
    # CSV-Pfade sind relativ ("house_a.mp3"), Dateien absolut -- Library.status
    # matcht ueber str(pfad); baue eine Library mit absoluten Pfaden:
    rows = dict([make_row(str(music_dir / "house_a.mp3"), "X", "A", 124, "7m", [1, 0, 0])])
    compact_csv(music_dir / "library_analysis.csv", rows)
    lib = Library(music_dir)
    analyzed, total = lib.status()
    assert (analyzed, total) == (1, 2)


def test_track_label_fallback_dateiname():
    assert track_label({"artist": "A", "title": "T", "filepath": "x.mp3"}) == "A - T"
    assert track_label({"artist": "", "title": "", "filepath": "/pfad/Cooler Song.mp3"}) == "Cooler Song"


LABELS = [
    "Kolter - Step han",
    "Kolter - GTFU (Simon Says)",
    "Kolter - Trapped - Radio-Edit",
    "Sweely - 3 Dub TV",
    "Traumer - Get Out - Edit",
]


def test_fuzzy_search_volle_woerter():
    from selecta.library import fuzzy_search
    assert LABELS[fuzzy_search("kolter step", LABELS)[0]] == "Kolter - Step han"


def test_fuzzy_search_letztes_wort_als_praefix():
    """Beim Tippen ist das letzte Wort meist unvollstaendig -- 'kolter st'
    muss 'Step han' eindeutig vor den anderen Kolter-Tracks ranken."""
    from selecta.library import fuzzy_search
    assert LABELS[fuzzy_search("kolter st", LABELS)[0]] == "Kolter - Step han"
    assert LABELS[fuzzy_search("kolter t", LABELS)[0]] == "Kolter - Trapped - Radio-Edit"


def test_fuzzy_search_tippfehler():
    from selecta.library import fuzzy_search
    assert LABELS[fuzzy_search("koltr step", LABELS)[0]] == "Kolter - Step han"


def test_fuzzy_search_cutoff_und_limit():
    from selecta.library import fuzzy_search
    assert fuzzy_search("zzz nicht vorhanden qqq", LABELS) == []
    assert len(fuzzy_search("kolter", LABELS, limit=2)) == 2
