"""Tests fuer selecta/map.py -- laufen OHNE pacmap/umap installiert (Kern-
Testumgebung hat nur die Pflicht-Dependencies), pruefen also faktisch immer
den PCA-Fallback. Essentia/TensorFlow werden nirgends beruehrt."""

import numpy as np
import pytest

from selecta.map import build_map_html, project_2d, write_map
from selecta.library import compact_csv
from tests.conftest import make_row


def test_pca_fallback_projiziert_auf_2d(synthetic_library):
    coords, projection = project_2d(synthetic_library.matrix, log=lambda _msg: None)
    assert coords.shape == (len(synthetic_library.tracks), 2)
    assert np.all(coords >= 0.0) and np.all(coords <= 1.0)
    assert projection == "PCA"  # weder pacmap noch umap in der Testumgebung installiert


def test_project_2d_leere_und_einzelne_library():
    assert project_2d(np.zeros((0, 3)))[0].shape == (0, 2)
    assert project_2d(np.array([[1.0, 0.0, 0.0]]))[0].shape == (1, 2)


def test_html_enthaelt_alle_tracks_und_kein_cdn(synthetic_library):
    coords, projection = project_2d(synthetic_library.matrix, log=lambda _msg: None)
    html = build_map_html(synthetic_library.tracks, coords, projection=projection)
    for track in synthetic_library.tracks:
        from selecta.library import track_label
        assert track_label(track) in html
    assert "http://" not in html
    assert "https://" not in html


def test_html_escaped_script_ende_im_titel(tmp_path):
    rows = dict([
        make_row("a.mp3", "Foo</script>Bar", "Title", 124, "7m", [1.0, 0.0, 0.0]),
    ])
    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    compact_csv(music_dir / "library_analysis.csv", rows)
    from selecta.library import Library
    lib = Library(music_dir)
    coords, projection = project_2d(lib.matrix, log=lambda _msg: None)
    html = build_map_html(lib.tracks, coords, projection=projection)
    assert "</script>Bar" not in html.replace('<script type="application/json"', "")
    # der problematische Teilstring darf nur escaped (mit Backslash) vorkommen
    assert "<\\/script>Bar" in html


def test_default_zielpfad_liegt_im_music_dir(synthetic_library):
    out = write_map(synthetic_library.music_dirs, log=lambda _msg: None)
    assert out == synthetic_library.music_dirs[0] / "selecta_map.html"
    assert out.exists()


def test_write_map_ohne_analysierte_tracks_wirft(tmp_path):
    empty_dir = tmp_path / "leer"
    empty_dir.mkdir()
    with pytest.raises(RuntimeError):
        write_map(empty_dir, log=lambda _msg: None)
