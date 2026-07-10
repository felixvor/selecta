import numpy as np

from selecta.similarity import (
    auto_num_tracks,
    harmonic_distance,
    mood_vector,
    parse_key,
    plan_transition,
    rank_similar,
    relative_bpm_distance,
    shifted_target,
)
from tests.conftest import get_track


# --- Key-Parsing / harmonische Distanz --------------------------------------

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
    assert harmonic_distance("7m", "7d") == 1     # Paralleltonart
    assert harmonic_distance("7m", "8m") == 1     # Nachbarquinte
    assert harmonic_distance("12m", "1m") == 1    # Wraparound Quintenzirkel
    assert harmonic_distance("7m", "8d") == 2     # Nachbar, anderer Modus
    assert harmonic_distance("7m", "1m") > 2      # deutlicher Bruch
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
    target_bpm, target_mood = shifted_target(q, energy=0, bpm_offset=0)
    assert target_bpm == 124.0
    assert np.allclose(target_mood, mood_vector(q), atol=1e-6)


def test_shifted_target_verschiebt_bpm_und_mood(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    target_bpm, target_mood = shifted_target(q, energy=2, bpm_offset=0)
    assert target_bpm == 124.0 + 7.0
    neutral_bpm, neutral_mood = shifted_target(q, energy=0)
    assert target_mood[0] > neutral_mood[0]  # aggressiver
    assert target_mood[1] < neutral_mood[1]  # weniger relaxed
    assert target_mood[3] > neutral_mood[3]  # mehr arousal


def test_shifted_target_bpm_offset_unabhaengig(synthetic_library):
    q = get_track(synthetic_library, "house_a.mp3")
    target_bpm, _ = shifted_target(q, energy=0, bpm_offset=8)
    assert target_bpm == 132.0


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


# --- Transition ---------------------------------------------------------------------

def test_auto_num_tracks_grenzen(synthetic_library):
    a = get_track(synthetic_library, "house_a.mp3")
    b = get_track(synthetic_library, "house_b.mp3")
    assert auto_num_tracks(a, b) == 1  # winziger Sprung -> Minimum 1
    ambient = get_track(synthetic_library, "ambient.mp3")
    k = auto_num_tracks(a, ambient)   # riesiger Sprung -> gedeckelt
    assert 1 <= k <= 8
    assert k > 1


def test_plan_transition_kette_konsistent(synthetic_library):
    a = get_track(synthetic_library, "house_a.mp3")
    b = get_track(synthetic_library, "techno.mp3")
    rows = plan_transition(a, b, synthetic_library, num_tracks=2)
    assert rows[0]["track"]["filepath"] == "house_a.mp3"
    assert rows[-1]["track"]["filepath"] == "techno.mp3"
    assert len(rows) == 4  # A + 2 + B
    filepaths = [r["track"]["filepath"] for r in rows]
    assert len(set(filepaths)) == len(filepaths)  # keine Doppelten
    assert rows[0]["deltas"] is None
    for row in rows[1:]:
        assert row["deltas"]["emb_dist"] >= 0
        assert "d_bpm" in row["deltas"]
        assert "key_rel" in row["deltas"]


def test_plan_transition_auto(synthetic_library):
    a = get_track(synthetic_library, "house_a.mp3")
    b = get_track(synthetic_library, "ambient.mp3")
    rows = plan_transition(a, b, synthetic_library, num_tracks=None)
    assert rows[0]["track"]["filepath"] == "house_a.mp3"
    assert rows[-1]["track"]["filepath"] == "ambient.mp3"
    assert len(rows) >= 3


def test_plan_transition_library_kleiner_als_kette(synthetic_library):
    a = get_track(synthetic_library, "house_a.mp3")
    b = get_track(synthetic_library, "house_b.mp3")
    rows = plan_transition(a, b, synthetic_library, num_tracks=8)
    # nur 4 andere Tracks verfuegbar -> Kette bricht sauber ab statt zu crashen
    assert rows[0]["track"]["filepath"] == "house_a.mp3"
    assert rows[-1]["track"]["filepath"] == "house_b.mp3"
    assert len(rows) <= 2 + 4
