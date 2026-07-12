import numpy as np
import pytest

from selecta.similarity import (
    NOTE_PC,
    harmonic_distance,
    key_to_pitch_class,
    mood_vector,
    note_to_camelot,
    note_to_openkey,
    pair_score,
    parse_key,
    rank_bridge,
    rank_similar,
    relative_bpm_distance,
    shifted_target,
)
from tests.conftest import get_track


# --- Key-Parsing / harmonische Distanz --------------------------------------

def test_note_to_camelot_ankerpunkte():
    # Referenz: Camelot-Wheel (mixedinkey.com) -- handverifizierte Anker,
    # kein Roundtrip: der wuerde einen systematischen Off-by-one verstecken.
    assert note_to_camelot("C", "major") == "8B"
    assert note_to_camelot("A", "minor") == "8A"
    assert note_to_camelot("G", "major") == "9B"
    assert note_to_camelot("E", "minor") == "9A"
    assert note_to_camelot("F", "major") == "7B"
    assert note_to_camelot("D", "minor") == "7A"
    assert note_to_camelot("F#", "minor") == "11A"
    assert note_to_camelot("Gb", "major") == "2B"
    assert note_to_camelot("Eb", "minor") == "2A"
    assert note_to_camelot("B", "major") == "1B"
    assert note_to_camelot("G#", "minor") == "1A"


def test_note_to_openkey_ankerpunkte():
    # Referenz: Open-Key-Notation (Traktor/Beatport)
    assert note_to_openkey("C", "major") == "1d"
    assert note_to_openkey("A", "minor") == "1m"
    assert note_to_openkey("G", "major") == "2d"
    assert note_to_openkey("F#", "minor") == "4m"
    assert note_to_openkey("Eb", "major") == "10d"
    assert note_to_openkey("F", "minor") == "9m"
    assert note_to_openkey("D", "minor") == "12m"


def test_note_to_camelot_unbekannte_eingabe():
    assert note_to_camelot("H", "major") == ""  # deutsches 'H' bewusst nicht gemappt
    assert note_to_camelot("A", "dorian") == ""


def test_key_to_pitch_class_unterscheidet_notationen():
    # Dieselbe Nummer, verschiedene Raeder: Open Key '8m' = Bb-Moll,
    # Camelot '8A' = A-Moll.
    assert key_to_pitch_class("8m") == (NOTE_PC["Bb"], "minor")
    assert key_to_pitch_class("8A") == (NOTE_PC["A"], "minor")
    assert key_to_pitch_class("1d") == (NOTE_PC["C"], "major")
    assert key_to_pitch_class("8B") == (NOTE_PC["C"], "major")
    assert key_to_pitch_class("kaputt") is None


def test_key_notation_roundtrip_alle_24_tonarten():
    # Volle Drehung ueber beide Raeder: Notenname -> Notation ->
    # Pitch-Class muss die Original-Tonika ergeben.
    tonics = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    for tonic in tonics:
        for scale in ("major", "minor"):
            expected = (NOTE_PC[tonic], scale)
            assert key_to_pitch_class(note_to_camelot(tonic, scale)) == expected
            assert key_to_pitch_class(note_to_openkey(tonic, scale)) == expected

def test_parse_key_open_key_und_camelot():
    assert parse_key("7m") == (7, "minor")
    assert parse_key("12d") == (12, "major")
    assert parse_key("8A") == (8, "minor")
    assert parse_key("8B") == (8, "major")


def test_parse_key_unparsbar():
    assert parse_key("") is None
    assert parse_key(None) is None
    assert parse_key("xyz") is None
    assert parse_key("13m") is None
    assert parse_key("0d") is None


def test_harmonic_distance_stufen():
    assert harmonic_distance("7m", "7m") == 0
    assert harmonic_distance("7m", "7d") == 1     # Relativtonart (gleiche Radnummer)
    assert harmonic_distance("7m", "8m") == 1     # Nachbarquinte
    assert harmonic_distance("12m", "1m") == 1    # Wraparound Quintenzirkel
    assert harmonic_distance("7m", "8d") == 2     # Nachbar, anderer Modus
    assert harmonic_distance("7m", "1m") > 2      # deutlicher Bruch


def test_harmonic_distance_gemischte_notationen():
    # Kanonischer Vergleich ueber Pitch-Classes: Open Key 1m und Camelot 8A
    # sind beide A-Moll -- die rohen Radnummern (1 vs. 8) wuerden faelschlich
    # einen grossen Abstand ergeben (die Raeder sind um 7 Positionen versetzt).
    assert harmonic_distance("1m", "8A") == 0
    assert harmonic_distance("1d", "8B") == 0     # C-Dur in beiden Notationen
    assert harmonic_distance("2m", "9A") == 0     # E-Moll
    assert harmonic_distance("1m", "9A") == 1     # A-Moll <-> E-Moll: Nachbarquinte
    assert harmonic_distance("7m", None) is None


# --- BPM ----------------------------------------------------------------------

def test_bpm_distance_halb_und_doppeltempo():
    assert relative_bpm_distance(126, 126) == 0.0
    assert relative_bpm_distance(126, 63) == 0.0
    assert relative_bpm_distance(126, 252) == 0.0
    assert relative_bpm_distance(126, 140) > 0
    assert relative_bpm_distance(None, 140) is None
    assert relative_bpm_distance(126, None) is None


# --- Target-Shifting ------------------------------------------------------------

def test_shifted_target_neutral_entspricht_query(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    target_bpm, target_mood = shifted_target(q, energy=0)
    assert target_bpm == 124.0
    assert np.allclose(target_mood, mood_vector(q), atol=1e-6)


def test_shifted_target_verschiebt_bpm_und_mood(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    target_bpm, target_mood = shifted_target(q, energy=2)
    assert target_bpm == 124.0 + 7.0
    neutral_bpm, neutral_mood = shifted_target(q, energy=0)
    assert target_mood[0] > neutral_mood[0]  # aggressiver
    assert target_mood[1] < neutral_mood[1]  # weniger relaxed
    assert target_mood[3] > neutral_mood[3]  # mehr arousal


# --- Ranking ---------------------------------------------------------------------

def test_rank_similar_naechster_cluster_gewinnt(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    results = rank_similar(q, synthetic_library)
    assert results[0]["track"]["filepath"] == "house_b.mp3"
    filepaths = [r["track"]["filepath"] for r in results]
    assert "house_a.mp3" not in filepaths  # Query selbst nie im Ergebnis
    # ambient (orthogonales Embedding, 80 BPM) muss ganz unten liegen
    assert filepaths[-1] == "ambient.mp3"


def test_rank_similar_energie_schiebt_schnellen_track_hoch(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    neutral = [r["track"]["filepath"] for r in rank_similar(q, synthetic_library)]
    pushed = [r["track"]["filepath"] for r in rank_similar(q, synthetic_library, energy=3)]
    assert neutral.index("house_fast.mp3") > pushed.index("house_fast.mp3")


def test_rank_similar_negative_energie_schiebt_warmup_hoch(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    neutral = [r["track"]["filepath"] for r in rank_similar(q, synthetic_library)]
    chilled = [r["track"]["filepath"] for r in rank_similar(q, synthetic_library, energy=-3)]
    assert neutral.index("house_slow.mp3") > chilled.index("house_slow.mp3")


def test_rank_similar_liefert_anzeige_deltas(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    top = rank_similar(q, synthetic_library)[0]
    assert top["d_bpm"] == 2.0            # 126 - 124
    assert abs(top["d_arousal"] - 0.3) < 1e-6
    assert top["key_pen"] == 1            # 7m -> 8m
    assert 0 < top["cos_sim"] <= 1.0


def test_rank_similar_deltas_bleiben_query_relativ_bei_energie(synthetic_library):
    """Die Anzeige-Deltas beziehen sich auf die Query, nicht aufs verschobene Ziel."""
    q = get_track(synthetic_library, "house_a.mp3")
    neutral = {r["track"]["filepath"]: r for r in rank_similar(q, synthetic_library)}
    pushed = {r["track"]["filepath"]: r for r in rank_similar(q, synthetic_library, energy=3)}
    for fp in neutral:
        assert neutral[fp]["d_bpm"] == pushed[fp]["d_bpm"]
        assert neutral[fp]["d_arousal"] == pushed[fp]["d_arousal"]


def test_rank_similar_top_begrenzt(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    assert len(rank_similar(q, synthetic_library, top=2)) == 2


def test_rank_similar_bpm_offset_filtert_langsamere_tracks(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")  # 124 BPM
    results = rank_similar(q, synthetic_library, bpm_offset=7)  # nur >= 131
    filepaths = {r["track"]["filepath"] for r in results}
    assert filepaths == {"house_fast.mp3", "techno.mp3"}


def test_rank_similar_bpm_offset_filtert_schnellere_tracks(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")  # 124 BPM
    results = rank_similar(q, synthetic_library, bpm_offset=-6)  # nur <= 118
    filepaths = {r["track"]["filepath"] for r in results}
    assert filepaths == {"house_slow.mp3", "ambient.mp3"}


def test_rank_similar_bpm_offset_extrem_liefert_keine_treffer(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    assert rank_similar(q, synthetic_library, bpm_offset=1000) == []


# --- Transition (Bruecke von A nach B) ----------------------------------------------

def test_rank_bridge_sortiert_nach_engpass(synthetic_library):
    a = get_track(synthetic_library, "house_a.mp3")
    b = get_track(synthetic_library, "techno.mp3")
    results = rank_bridge(a, b, synthetic_library)
    filepaths = [r["track"]["filepath"] for r in results]
    assert "house_a.mp3" not in filepaths  # A selbst nie dabei
    assert "techno.mp3" in filepaths       # B laeuft als Kandidat mit
    mins = [min(r["score_a"], r["score_b"]) for r in results]
    assert mins == sorted(mins, reverse=True)  # Engpass-Sortierung
    # house_fast liegt in Embedding, BPM und Arousal zwischen House und
    # Techno -- beste Einzelbruecke
    assert results[0]["track"]["filepath"] == "house_fast.mp3"


def test_rank_bridge_ziel_selbst_rankt_nach_direktsprung(synthetic_library):
    a = get_track(synthetic_library, "house_a.mp3")
    b = get_track(synthetic_library, "techno.mp3")
    results = rank_bridge(a, b, synthetic_library)
    b_row = next(r for r in results if r["track"]["filepath"] == "techno.mp3")
    assert b_row["score_b"] == pytest.approx(1.0, abs=1e-5)
    assert b_row["score_a"] == pytest.approx(pair_score(a, b), abs=1e-6)


def test_rank_bridge_bpm_filter(synthetic_library):
    a = get_track(synthetic_library, "house_a.mp3")  # 124 BPM
    b = get_track(synthetic_library, "techno.mp3")
    results = rank_bridge(a, b, synthetic_library, bpm_offset=7)  # nur >= 131
    filepaths = {r["track"]["filepath"] for r in results}
    assert filepaths == {"house_fast.mp3", "techno.mp3"}
