import numpy as np
import pytest

from selecta.library import Library, compact_csv, encode_embedding


def make_row(filepath, artist, title, bpm, key, emb, **mood):
    row = {
        "artist": artist, "title": title, "bpm": str(bpm), "key": key,
        "error": "", "embedding": encode_embedding(np.array(emb, dtype=np.float32)),
    }
    defaults = {
        "aggressive": 0.3, "happy": 0.1, "sad": 0.2, "relaxed": 0.8, "party": 0.2,
        "danceable": 0.99, "approachability": 0.3, "engagement": 0.8,
        "arousal": 6.0, "valence": 6.0,
    }
    defaults.update(mood)
    row.update({k: str(v) for k, v in defaults.items()})
    return filepath, row


@pytest.fixture
def synthetic_library(tmp_path):
    """Kleine Library mit klar getrennten Embedding-Clustern:
    housey (Achse 0), techig (Achse 1), ambient (Achse 2)."""
    rows = dict([
        make_row("house_a.mp3", "HouseArtist", "Groove A", 124, "7m", [1.0, 0.1, 0.0], arousal=6.0, aggressive=0.2),
        make_row("house_b.mp3", "HouseArtist", "Groove B", 126, "8m", [0.95, 0.15, 0.0], arousal=6.3, aggressive=0.25),
        make_row("house_fast.mp3", "PushArtist", "Peak Time", 132, "7m", [0.9, 0.2, 0.0], arousal=7.2, aggressive=0.5),
        make_row("house_slow.mp3", "ChillArtist", "Warmup", 118, "7m", [0.9, 0.05, 0.1], arousal=4.8, aggressive=0.05, relaxed=0.95),
        make_row("techno.mp3", "TechArtist", "Hammer", 140, "2m", [0.1, 1.0, 0.0], arousal=7.5, aggressive=0.8),
        make_row("ambient.mp3", "AmbientArtist", "Drift", 80, "3d", [0.0, 0.1, 1.0], arousal=2.0, aggressive=0.01, relaxed=0.99, danceable=0.2),
    ])
    music_dir = tmp_path / "musik"
    music_dir.mkdir()
    csv_path = music_dir / "library_analysis.csv"
    compact_csv(csv_path, rows)
    return Library(music_dir)


def get_track(library, filepath):
    return next(t for t in library.tracks if t["filepath"] == filepath)
