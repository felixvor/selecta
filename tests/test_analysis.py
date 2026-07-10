from selecta.analysis import _missing_parts


def test_missing_row_needs_embedding():
    assert _missing_parts(None) == {"embedding"}


def test_error_row_needs_embedding():
    row = {"status": "error: boom", "embedding": ""}
    assert _missing_parts(row) == {"embedding"}


def test_ok_row_without_embedding_needs_embedding():
    # theoretischer Alt-Schema-Fall (CSV vor Einfuehrung der Embedding-Spalte)
    row = {"status": "ok", "embedding": ""}
    assert _missing_parts(row) == {"embedding"}


def test_ok_row_missing_bpm_needs_tags():
    row = {"status": "ok", "embedding": "xyz", "bpm": "", "key": ""}
    assert _missing_parts(row) == {"tags"}


def test_ok_row_with_bpm_but_no_key_is_done():
    # Key wird nie selbst berechnet -- sonst waere die Zeile fuer immer
    # "offen" und wuerde bei jedem Analyse-Lauf erneut angefasst.
    row = {"status": "ok", "embedding": "xyz", "bpm": "128.0", "key": ""}
    assert _missing_parts(row) == set()


def test_fully_tagged_row_is_done():
    row = {"status": "ok", "embedding": "xyz", "bpm": "128", "key": "7m"}
    assert _missing_parts(row) == set()
