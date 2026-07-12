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


# Zweite, kleinere Crate fuer Multi-Library-Screenshots: Der Launcher soll
# im README nicht wie ein Ein-Ordner-Tool aussehen. Klanglich ein Warmup-/
# Listening-Regal (Ambient/Breaks), das sich mit dem Ambient-Cluster der
# Haupt-Crate ueberlappt -- so zeigt die Suche ueber beide Libraries
# hinweg sinnvolle Nachbarn.
WARMUP_TRACKS = [
    ("Riverbed Static", "Moth Season", 92, "1m", [0.20, 0.08, 1.00],
     "Ambient", "dreamy|calm", "2015", 2.1, 0.02, 0.99, 0.30, 6.5),
    ("Halden Loop", "Winter Balcony", 100, "2m", [0.30, 0.10, 0.98],
     "Downtempo|Ambient", "deep|meditative", "2018", 2.9, 0.04, 0.97, 0.55, 6.1),
    ("Cloud Ledger", "Sunday Geometry", 112, "3m", [0.46, 0.14, 0.92],
     "Downtempo|Trip Hop", "warm|groovy", "2021", 3.9, 0.08, 0.90, 0.75, 6.7),
    ("Fjara", "Salt Radio", 115, "10m", [0.55, 0.12, 0.85],
     "Organic House|Downtempo", "melodic|warm", "2022", 4.2, 0.09, 0.88, 0.80, 6.9),
    ("Paper Tigers Sleep", "Low Orbit Lullaby", 84, "12m", [0.16, 0.06, 1.00],
     "Ambient|Drone", "space|meditative", "2014", 1.8, 0.01, 0.99, 0.20, 5.9),
    ("Miro Delta", "Breakfast in Reverse", 118, "4m", [0.60, 0.20, 0.80],
     "Breaks|Downtempo", "fun|retro", "2020", 4.6, 0.12, 0.82, 0.85, 7.0),
]


# --- Prozedurale Auffuellung -------------------------------------------------
# Die kuratierten Tracks oben bleiben als Anker (make_screens/demo.tape
# suchen "velvet"/"pressure"), aber eine 15-Track-Library sieht im README
# nach Spielzeug aus. Der Generator fuellt beide Crates auf realistische
# Groesse auf: plausible Artist/Titel-Kombinationen, Cluster-Embeddings mit
# Rauschen, BPM/Key/Moods passend zum Cluster. Seed fest -> reproduzierbar.

_ARTIST_A = [
    "Aiden", "Marlow", "Selia", "Ronan", "Petra", "Idris", "Nova", "Casper",
    "Lena", "Viktor", "Sana", "Ilias", "Mara", "Theo", "Nyra", "Bruno",
    "Alva", "Dario", "Femi", "Greta", "Janto", "Kaia", "Loris", "Mio",
]
_ARTIST_B = [
    "Vance", "Kessler", "Duarte", "Lindqvist", "Okafor", "Marino", "Reyes",
    "Falk", "Sørensen", "Baptiste", "Klein", "Navarro", "Petrov", "Sato",
    "Whitfield", "Andersson", "Costa", "Meier", "Oduya", "Silva",
]
_ARTIST_SOLO = [
    "Kilowatt Social", "Paper Antenna", "Bassment Trust", "Modular Grief",
    "Hotel Neon", "Klangfabrik", "Southbound Freight", "Tape Loop City",
    "The Attic Committee", "Grey Harbour", "Analog Weather", "Fern & Firmament",
    "Night Bus Cartel", "Ceramic Youth", "Ostkreuz Kollektiv", "Velour System",
    "Dune Office", "Rotor y Rumba", "Cold Latitude", "Marble Arcade",
]
_TITLE_ADJ = [
    "Broken", "Velvet", "Neon", "Silent", "Rolling", "Hidden", "Electric",
    "Slow", "Golden", "Concrete", "Distant", "Burning", "Liquid", "Northern",
    "Peculiar", "Static", "Tidal", "Undone", "Vertical", "Weightless",
]
_TITLE_NOUN = [
    "Corridor", "Signal", "Harbour", "Motion", "Pattern", "Reunion",
    "Shelter", "Tension", "Voltage", "Window", "Afterglow", "Circuit",
    "Dispatch", "Elevation", "Frequency", "Garden", "Horizon", "Interlude",
    "Junction", "Kingdom", "Mirage", "Nocturne", "Orbit", "Postcard",
]
_TITLE_SUFFIX = ["", "", "", "", " - Original Mix", " - Extended Mix",
                 " - Radio Edit", " - Dub", " - Remix"]

# (embedding-basis, bpm-spanne, arousal-spanne, aggressive-spanne,
#  relaxed-spanne, genres-pool, vibes-pool, jahr-spanne)
_CLUSTERS = {
    "deep": ([1.00, 0.30, 0.12], (120, 125), (4.6, 6.2), (0.08, 0.30), (0.60, 0.90),
             ["Deep House", "Deep House|Tech House", "Deep House|Organic House"],
             ["deep|groovy", "warm|melodic", "groovy", "deep|dark"], (2013, 2024)),
    "tech": ([0.94, 0.46, 0.05], (124, 129), (5.8, 6.9), (0.25, 0.50), (0.35, 0.65),
             ["Tech House", "Tech House|Minimal", "Minimal|Deep Tech", "Tech House|Deep House"],
             ["dark|driving", "groovy|hypnotic", "dark", "energetic"], (2016, 2025)),
    "techno": ([0.38, 1.00, 0.03], (132, 140), (6.9, 8.0), (0.55, 0.90), (0.05, 0.30),
               ["Techno", "Peak Time Techno", "Hard Techno", "Techno|Industrial"],
               ["dark|driving", "hypnotic", "dark|industrial", "powerful"], (2018, 2025)),
    "warmup": ([0.30, 0.10, 0.97], (84, 118), (1.8, 4.6), (0.01, 0.12), (0.80, 0.99),
               ["Ambient", "Downtempo", "Downtempo|Trip Hop", "Organic House|Downtempo",
                "Ambient|Drone", "Breaks|Downtempo"],
               ["dreamy|calm", "deep|meditative", "warm|melodic", "space", "fun|retro"],
               (2010, 2023)),
    "dnb": ([0.22, 0.90, 0.16], (170, 176), (6.5, 7.8), (0.40, 0.75), (0.15, 0.45),
            ["Drum n Bass", "Drum n Bass|Jungle", "Liquid Funk", "Neurofunk"],
            ["energetic|fast", "dark|heavy", "melodic|uplifting"], (2004, 2025)),
    "disco": ([0.90, 0.14, 0.30], (105, 122), (4.8, 6.4), (0.05, 0.25), (0.55, 0.85),
              ["Disco", "Disco|Edits", "Nu-Disco", "Funk|Disco"],
              ["fun|groovy", "retro|happy", "sexy|groovy", "upbeat"], (1977, 2024)),
}


def _generate_tracks(rng, cluster_counts):
    """Zufaellige, aber plausibel benannte Tracks je Cluster."""
    tracks = []
    used_names = set()
    for cluster, count in cluster_counts.items():
        base, bpm_r, ar_r, ag_r, rel_r, genres, vibes, years = _CLUSTERS[cluster]
        for _ in range(count):
            while True:
                if rng.random() < 0.45:
                    artist = rng.choice(_ARTIST_SOLO)
                else:
                    artist = f"{rng.choice(_ARTIST_A)} {rng.choice(_ARTIST_B)}"
                title = f"{rng.choice(_TITLE_ADJ)} {rng.choice(_TITLE_NOUN)}{rng.choice(_TITLE_SUFFIX)}"
                if (artist, title) not in used_names:
                    used_names.add((artist, title))
                    break
            emb = [max(0.0, v + rng.gauss(0, 0.06)) for v in base]
            bpm = rng.randint(*bpm_r)
            key = f"{rng.randint(1, 12)}{'m' if rng.random() < 0.72 else 'd'}"
            arousal = round(rng.uniform(*ar_r), 1)
            aggressive = round(rng.uniform(*ag_r), 2)
            relaxed = round(rng.uniform(*rel_r), 2)
            danceable = round(rng.uniform(0.88, 0.99) if cluster != "warmup"
                              else rng.uniform(0.2, 0.8), 2)
            valence = round(rng.uniform(4.5, 7.0), 1)
            tracks.append((artist, title, bpm, key, emb,
                           rng.choice(genres), rng.choice(vibes),
                           str(rng.randint(*years)), arousal, aggressive,
                           relaxed, danceable, valence))
    return tracks


def _pad512(emb) -> np.ndarray:
    """3-dim-Cluster-Vektor -> 512-dim (Rest 0). Noetig, seit die Demo-Crate
    auch ECHT analysierte Dateien enthalten kann (--seed-audio): deren
    Embeddings sind 512-dim, und gemischte Dimensionen fliegen aus der
    Library (Mehrheits-Dimension gewinnt). Cosine innerhalb der Demo-Tracks
    aendert sich durch Zero-Padding nicht."""
    vec = np.zeros(512, dtype=np.float32)
    vec[: len(emb)] = emb
    return vec


def create_demo_library(target_dir, tracks=DEMO_TRACKS, csv_skip=0) -> Path:
    """Legt den Demo-Ordner an (idempotent) und liefert seinen Pfad.
    csv_skip: so viele Tracks bekommen KEINE CSV-Zeile (Dateien ohne
    Analyse -- fuer offene/unanalysierte Ordner im Launcher-Bild)."""
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for i, (artist, title, bpm, key, emb, genres, vibes, year,
            arousal, aggressive, relaxed, danceable, valence) in enumerate(tracks):
        filepath = target_dir / f"{artist} - {title}.mp3"
        filepath.touch()
        if i < csv_skip:
            continue
        encoded = encode_embedding(_pad512(emb))
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


def create_demo_crates(base_dir) -> tuple[Path, Path]:
    """Beide Demo-Crates unter base_dir: (House, Warmup) -- kuratierte
    Anker-Tracks plus prozedurale Auffuellung auf realistische Groesse."""
    import random
    base = Path(base_dir).resolve()
    rng = random.Random(2026)
    house = DEMO_TRACKS + _generate_tracks(rng, {"deep": 68, "tech": 52, "techno": 41})
    warmup = WARMUP_TRACKS + _generate_tracks(rng, {"warmup": 57})
    return (
        create_demo_library(base / "House", tracks=house),
        create_demo_library(base / "Warmup", tracks=warmup),
    )


# 7 Libraries in Echteinsatz-Groesse fuers Demo-GIF: (name, aktiv,
# cluster-mix, csv_skip). csv_skip > 0 laesst Dateien unanalysiert
# (gelber/roter Status im Launcher -- eine gewachsene Sammlung ist nie
# komplett durchanalysiert). House/Techno sind die aktiven Sets.
DEMO_UNIVERSE = [
    ("House", True, {"deep": 300, "tech": 248, "techno": 120}, 0),
    ("Techno & Peaktime", True, {"techno": 418}, 0),
    ("Warmup & Downtempo", False, {"warmup": 195}, 18),
    ("Drum & Bass", False, {"dnb": 241}, 0),
    ("Ambient & Listening", False, {"warmup": 154}, 0),
    ("Disco & Edits", False, {"disco": 96}, 0),
    ("Crates/Festival 2026", False, {"tech": 42}, 42),
]


def create_demo_universe(base_dir) -> list[tuple[Path, bool]]:
    """Alle 7 Demo-Libraries unter base_dir; Rueckgabe [(pfad, aktiv)].
    Die kuratierten Anker-Tracks liegen in House (velvet/pressure-Suchen
    aus make_screens/demo.tape) bzw. Warmup & Downtempo."""
    import random
    base = Path(base_dir).resolve()
    rng = random.Random(2026)
    out = []
    for name, active, mix, skip in DEMO_UNIVERSE:
        tracks = _generate_tracks(rng, mix)
        if name == "House":
            tracks = DEMO_TRACKS + tracks
        elif name == "Warmup & Downtempo":
            tracks = WARMUP_TRACKS + tracks
        out.append((create_demo_library(base / name, tracks=tracks, csv_skip=skip), active))
    return out


def seed_real_audio(crate_dir: Path, source_dir: Path, count: int = 4) -> None:
    """Kopiert ein paar ECHTE Audiodateien in die Crate -- umbenannt auf
    fiktive Artist/Titel-Namen und mit gestrippten ID3-Tags, OHNE CSV-Zeile.

    Zweck: Im Demo-GIF soll der Analyse-Lauf echte ✓-Ergebniszeilen mit
    Genre-Chips zeigen (die leeren Dummy-Dateien koennen nur '≡ complete').
    Ohne Tags faellt der Titel auf den fiktiven Dateinamen zurueck und
    BPM/Key werden sichtbar selbst geschaetzt (~-Praefix). Die Quelle
    bleibt privat -- ins Repo kommt nur das gerenderte GIF."""
    import random
    import shutil

    rng = random.Random(7)
    candidates = sorted(p for p in Path(source_dir).rglob("*.mp3"))
    names = [
        "Velour System - Copper Skyline", "Nyra Falk - Half Past Blue",
        "Bassment Trust - Late Checkout", "Mio Navarro - Glasshouse Dub",
        "Grey Harbour - Nocturne Nineteen", "Tape Loop City - Border Lights",
    ]
    for i, src in enumerate(rng.sample(candidates, min(count, len(candidates)))):
        dest = crate_dir / f"{names[i % len(names)]}.mp3"
        shutil.copyfile(src, dest)
        try:
            from mutagen.id3 import ID3
            tags = ID3(str(dest))
            tags.delete(str(dest))
        except Exception:
            pass


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else "demo_library"
    universe = create_demo_universe(base)
    house = universe[0][0]
    # Optional: --seed-audio QUELLE [N] mischt echte Dateien in die
    # House-Crate (fuer den Analyse-Teil des Demo-GIFs, siehe demo.tape).
    if "--seed-audio" in sys.argv:
        idx = sys.argv.index("--seed-audio")
        source = sys.argv[idx + 1]
        count = int(sys.argv[idx + 2]) if len(sys.argv) > idx + 2 else 4
        seed_real_audio(house, Path(source), count)
    # --state DIR: libraries.json fuer ein isoliertes HOME schreiben (das
    # Demo-GIF startet im Launcher, ohne die echte Nutzer-Config anzufassen).
    if "--state" in sys.argv:
        import json
        state_home = Path(sys.argv[sys.argv.index("--state") + 1])
        cfg = state_home / ".local" / "share" / "selecta"
        cfg.mkdir(parents=True, exist_ok=True)
        (cfg / "libraries.json").write_text(json.dumps({"libraries": [
            {"path": str(path), "active": active} for path, active in universe
        ]}), encoding="utf-8")
    for path, active in universe:
        print(("[x] " if active else "[ ] ") + str(path))
