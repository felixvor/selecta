"""Demo-Library fuer README-Screenshots und das VHS-Demo-GIF.

Erzeugt einen Ordner mit ~15 frei erfundenen Tracks: leere .mp3-Dummies
(damit Datei-Zaehler und Status-Badges echte Werte zeigen) plus eine
library_analysis.csv mit handgebauten Embedding-Clustern -- die TUI laedt
rein aus der CSV, es wird nie Audio dekodiert. Dadurch sind alle Demo-
Assets jederzeit reproduzierbar, ohne echte (urheberrechtlich geschuetzte)
Musik ins Repo zu legen.

Cluster-Idee wie in tests/conftest.py: House auf Achse 0, Techno auf
Achse 1, Ambient auf Achse 2; Bridge-Tracks mischen Achsen, damit der
Transition-Modus im Demo sichtbar sinnvolle Kandidaten zeigt.

Aufruf:  python scripts/demo_library.py [zielordner]   (Default: ./demo_library)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selecta.library import compact_csv, encode_embedding

import numpy as np

# Embedding-Geometrie: Achse 0 = House, Achse 1 = Techno, Achse 2 = Ambient.
# Alle Vektoren tragen eine Grundkomponente auf den fremden Achsen, damit die
# Cosine-Landschaft realistisch aussieht (innerhalb eines Clusters ~0.95+,
# House<->Techno ~0.6, Bridge-Tracks ~0.85 zu beiden Seiten) -- vollstaendig
# orthogonale Cluster ergaeben negative Demo-Scores, die es bei echten
# discogs_track-Embeddings nicht gibt.
# (artist, title, bpm, key, embedding, genres, vibes, year, arousal, aggressive, relaxed, danceable, valence)
DEMO_TRACKS = [
    ("Marlow Vance", "Basement Language", 122, "5m", [1.00, 0.32, 0.12],
     "Deep House", "deep|groovy", "2019", 5.2, 0.15, 0.80, 0.95, 5.8),
    ("Ferra Noire", "Velvet Circuit", 125, "6m", [1.00, 0.38, 0.08],
     "Tech House|Deep House", "dark|hypnotic", "2021", 5.9, 0.25, 0.65, 0.97, 5.1),
    ("Nightshift Cartel", "After Hours Arithmetic", 126, "5m", [0.98, 0.42, 0.06],
     "Tech House", "groovy", "2022", 6.1, 0.30, 0.55, 0.98, 5.5),
    ("Kilim", "Copper Sun", 120, "4m", [1.00, 0.25, 0.22],
     "Deep House|Organic House", "melodic|warm", "2018", 4.8, 0.10, 0.88, 0.90, 6.4),
    ("Doppler Youth", "Glasshouse", 124, "12m", [0.95, 0.36, 0.10],
     "Acid House", "acid|energetic", "1994", 6.3, 0.35, 0.50, 0.96, 5.9),
    ("Stray Voltage", "Amber Alerts", 128, "6m", [0.92, 0.52, 0.04],
     "Tech House|Minimal", "dark|driving", "2023", 6.6, 0.42, 0.40, 0.97, 4.9),
    ("Ports of Call", "Tidal Memory", 118, "9m", [0.96, 0.22, 0.30],
     "Progressive House", "melodic|dreamy", "2017", 4.5, 0.08, 0.90, 0.85, 6.6),
    ("Vera Lux", "Midnight Cafeteria", 123, "5m", [1.00, 0.30, 0.14],
     "Deep House", "groovy|warm", "2020", 5.4, 0.18, 0.75, 0.94, 6.0),
    # Bridges: House->Techno (fuer den Transition-Screenshot)
    ("Cassette Era", "Concrete Flowers", 129, "6m", [0.85, 0.72, 0.04],
     "Tech House|Techno", "dark|driving", "2022", 6.9, 0.52, 0.30, 0.96, 4.6),
    ("Motor Poetry", "Third Shift", 131, "7m", [0.72, 0.86, 0.02],
     "Techno|Tech House", "hypnotic|dark", "2021", 7.1, 0.60, 0.25, 0.95, 4.3),
    # Techno-Cluster
    ("Verratt", "Iron Meridian", 136, "8m", [0.40, 1.00, 0.02],
     "Techno|Peak Time Techno", "dark|driving", "2023", 7.6, 0.78, 0.12, 0.94, 3.8),
    ("Cold Assembly", "Pressure Test", 138, "9m", [0.34, 1.00, 0.04],
     "Hard Techno", "dark|industrial", "2024", 7.9, 0.88, 0.08, 0.92, 3.4),
    ("Signal Warden", "Tunnel Vision", 134, "8m", [0.44, 0.98, 0.08],
     "Techno", "hypnotic", "2020", 7.3, 0.70, 0.18, 0.95, 4.0),
    # Warmup/Ambient
    ("Lumen Field", "Sleepwater", 96, "2m", [0.28, 0.12, 1.00],
     "Ambient|Downtempo", "dreamy|deep", "2016", 2.4, 0.03, 0.98, 0.40, 6.2),
    ("Inner Coast", "Fern Light", 108, "3m", [0.42, 0.10, 0.96],
     "Downtempo|Organic House", "warm|melodic", "2019", 3.6, 0.06, 0.95, 0.70, 6.8),
]


def create_demo_library(target_dir) -> Path:
    """Legt den Demo-Ordner an (idempotent) und liefert seinen Pfad."""
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for (artist, title, bpm, key, emb, genres, vibes, year,
         arousal, aggressive, relaxed, danceable, valence) in DEMO_TRACKS:
        filepath = target_dir / f"{artist} - {title}.mp3"
        filepath.touch()
        encoded = encode_embedding(np.array(emb, dtype=np.float32))
        rows[str(filepath)] = {
            "artist": artist, "title": title, "bpm": str(bpm), "key": key,
            "error": "", "embedding": encoded, "effnet_embedding": encoded,
            "genres": genres, "vibes": vibes, "year": year,
            "arousal": str(arousal), "aggressive": str(aggressive),
            "relaxed": str(relaxed), "danceable": str(danceable),
            "valence": str(valence),
            "happy": "0.2", "sad": "0.1", "party": "0.5",
            "approachability": "0.4", "engagement": "0.7",
        }
    compact_csv(target_dir / "library_analysis.csv", rows)
    return target_dir


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "demo_library"
    print(create_demo_library(dest))
