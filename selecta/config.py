"""Zentrale Konstanten: Modelle, CSV-Schema, Ranking-Gewichte, Energie-Stufen."""

MODELS_BASE = "https://essentia.upf.edu/models"

MODEL_FILES = {
    "discogs-effnet-bs64-1.pb": f"{MODELS_BASE}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    # Kontrastiv auf Track-Aehnlichkeit trainierte Variante (statt Genre-Klassifikation).
    # PartitionedCall:0 ist der 512-dim Projektionsraum, in dem der Kontrastiv-Loss
    # aehnliche Tracks zusammenzieht -- das ist der eigentliche "Similarity-Vektor".
    "discogs_track_embeddings-effnet-bs64-1.pb": f"{MODELS_BASE}/feature-extractors/discogs-effnet/discogs_track_embeddings-effnet-bs64-1.pb",
    "msd-musicnn-1.pb": f"{MODELS_BASE}/feature-extractors/musicnn/msd-musicnn-1.pb",
    "mood_aggressive-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb",
    "mood_happy-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb",
    "mood_sad-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb",
    "mood_relaxed-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb",
    "mood_party-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/mood_party/mood_party-discogs-effnet-1.pb",
    "danceability-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/danceability/danceability-discogs-effnet-1.pb",
    "approachability_regression-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/approachability/approachability_regression-discogs-effnet-1.pb",
    "engagement_regression-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/engagement/engagement_regression-discogs-effnet-1.pb",
    "deam-msd-musicnn-2.pb": f"{MODELS_BASE}/classification-heads/deam/deam-msd-musicnn-2.pb",
}

MIN_MODEL_BYTES = 1024
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 30

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aiff", ".aif", ".ogg"}

# Fester Dateiname pro Musik-Ordner -- 'analyze' und die Suche verwenden
# denselben, dadurch braucht es nie eine separate CSV-Pfad-Eingabe.
CSV_FILENAME = "library_analysis.csv"

CSV_FIELDNAMES = [
    "filepath",
    "artist",
    "title",
    "bpm",
    "key",
    "aggressive",
    "happy",
    "sad",
    "relaxed",
    "party",
    "danceable",
    "approachability",
    "engagement",
    "arousal",
    "valence",
    "embedding",
    "status",
]

FLOAT_FIELDS = [
    "aggressive", "happy", "sad", "relaxed", "party", "danceable",
    "approachability", "engagement", "arousal", "valence",
]

# --- Ranking ---------------------------------------------------------------

# Gewichte fuer die Penalty-Terme im Score (Cosine-Similarity minus Abzuege).
W_BPM = 0.5
W_KEY = 0.3
W_MOOD = 0.4

# Wie viele Ergebniszeilen gerendert werden (Liste ist scrollbar).
TOP_N = 100

# --- Energie-Achse (Target-Shifting) ----------------------------------------

# Pro Energie-Stufe e (-3..+3) wird das Suchziel verschoben, nicht der Score
# belohnt -- sonst gewinnt immer der extremste Track der Library.
ENERGY_MIN = -3
ENERGY_MAX = 3
BPM_PER_ENERGY_STEP = 3.5
AROUSAL_PER_ENERGY_STEP = 0.4
AGGRESSIVE_PER_ENERGY_STEP = 0.08
RELAXED_PER_ENERGY_STEP = 0.08

# BPM-Feintuning (',' / '.') unabhaengig von der Energie-Stufe.
BPM_FINETUNE_STEP = 4

# Wertebereich des DEAM-Modells (arousal/valence).
AROUSAL_VALENCE_RANGE = (1.0, 9.0)

# --- Transition-Planer -------------------------------------------------------

TRANSITION_MAX_TRACKS = 8
# auto-k: ein Zwischentrack pro "spuerbarem Sprung" in BPM / Embedding / Arousal.
TRANSITION_BPM_PER_STEP = 4.0
TRANSITION_EMB_DIST_PER_STEP = 0.12
TRANSITION_AROUSAL_PER_STEP = 0.7
