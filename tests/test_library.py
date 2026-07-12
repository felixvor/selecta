import numpy as np

from selecta.library import (
    Library,
    compact_csv,
    decode_embedding,
    dir_status,
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


def test_library_laedt_nur_zeilen_mit_embedding(tmp_path):
    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    csv_path = music_dir / "library_analysis.csv"
    fp_ok, row_ok = make_row("ok.mp3", "X", "A", 126, "7m", [1, 0, 0])
    fp_err, row_err = make_row("err.mp3", "X", "B", 126, "7m", [0, 1, 0])
    row_err["embedding"] = ""
    row_err["error"] = "kaputt"  # Debug-Info, beeinflusst das Laden nicht
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
    music_dir = synthetic_library.music_dirs[0]
    (music_dir / "house_a.mp3").write_bytes(b"")
    (music_dir / "neu_dazu.mp3").write_bytes(b"")
    # CSV-Pfade sind relativ ("house_a.mp3"), Dateien absolut -- Library.status
    # matcht ueber str(pfad); baue eine Library mit absoluten Pfaden:
    rows = dict([make_row(str(music_dir / "house_a.mp3"), "X", "A", 124, "7m", [1, 0, 0])])
    compact_csv(music_dir / "library_analysis.csv", rows)
    lib = Library(music_dir)
    analyzed, total = lib.status()
    assert (analyzed, total) == (1, 2)


def test_library_mehrere_ordner_werden_zusammengefuehrt(tmp_path):
    dir_a = tmp_path / "house"
    dir_a.mkdir()
    dir_b = tmp_path / "techno"
    dir_b.mkdir()
    # derselbe absolute Pfad in beiden CSVs (verschachtelte Libraries):
    # darf in der Suche nur einmal auftauchen
    shared = str(tmp_path / "beide.mp3")
    compact_csv(dir_a / "library_analysis.csv", dict([
        make_row("a.mp3", "X", "A", 124, "7m", [1, 0, 0]),
        make_row(shared, "X", "S", 130, "8m", [0, 1, 0]),
    ]))
    compact_csv(dir_b / "library_analysis.csv", dict([
        make_row("b.mp3", "X", "B", 140, "2m", [0, 0, 1]),
        make_row(shared, "X", "S", 130, "8m", [0, 1, 0]),
    ]))

    lib = Library([dir_a, dir_b])
    assert sorted(t["filepath"] for t in lib.tracks) == sorted(["a.mp3", shared, "b.mp3"])
    assert lib.matrix.shape == (3, 3)
    # Einzelpfad (Ad-hoc-Modus) funktioniert weiterhin ohne Liste
    assert len(Library(dir_a).tracks) == 2


def test_dir_status_zaehlt_ohne_embeddings_zu_decodieren(tmp_path):
    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    (music_dir / "fertig.mp3").write_bytes(b"")
    (music_dir / "neu.mp3").write_bytes(b"")
    rows = dict([make_row(str(music_dir / "fertig.mp3"), "X", "A", 124, "7m", [1, 0, 0])])
    compact_csv(music_dir / "library_analysis.csv", rows)
    assert dir_status(music_dir) == (1, 2)


def test_dir_status_nutzt_analyse_kriterium(tmp_path):
    """Regression '0 offen': Zeilen, die der Analyse-Lauf noch anfassen wuerde
    (fehlendes effnet_embedding bzw. fehlender BPM), duerfen nicht als fertig
    zaehlen -- sonst zeigt das Analyse-Modal '0 offen' und analysiert dann doch."""
    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    for name in ("fertig.mp3", "alt_schema.mp3", "ohne_bpm.mp3"):
        (music_dir / name).write_bytes(b"")
    fp_ok, row_ok = make_row(str(music_dir / "fertig.mp3"), "X", "A", 124, "7m", [1, 0, 0])
    fp_alt, row_alt = make_row(str(music_dir / "alt_schema.mp3"), "X", "B", 126, "8m", [0, 1, 0])
    row_alt["effnet_embedding"] = ""  # CSV von vor dem Genre/Vibe-Schema
    fp_nobpm, row_nobpm = make_row(str(music_dir / "ohne_bpm.mp3"), "X", "C", "", "7m", [0, 0, 1])
    compact_csv(music_dir / "library_analysis.csv",
                {fp_ok: row_ok, fp_alt: row_alt, fp_nobpm: row_nobpm})
    assert dir_status(music_dir) == (1, 3)

    # Library.status() nutzt dasselbe Kriterium
    assert Library(music_dir).status() == (1, 3)


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


def test_gemischte_embedding_dimensionen_crashen_nicht(tmp_path):
    """Regression: Zeilen mit abweichender Embedding-Dimension (z.B. nach
    einem Modellwechsel) haben np.stack und damit den App-Start gecrasht.
    Die Mehrheits-Dimension gewinnt, der Rest wird ignoriert."""
    from selecta.library import Library, compact_csv
    from tests.conftest import make_row

    rows = dict([
        make_row("a.mp3", "A", "One", 124, "7m", [1.0, 0.0, 0.0]),
        make_row("b.mp3", "B", "Two", 126, "8m", [0.9, 0.1, 0.0]),
        make_row("c.mp3", "C", "Odd", 128, "9m", [0.5] * 512),
    ])
    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    compact_csv(music_dir / "library_analysis.csv", rows)

    lib = Library(music_dir)
    assert len(lib.tracks) == 2
    assert lib.matrix.shape == (2, 3)
