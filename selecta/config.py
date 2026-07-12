"""Zentrale Konstanten: Modelle, CSV-Schema, Ranking-Gewichte, Energie-Stufen."""

import os

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
    # Genre/Vibe-Tagging: 400 Discogs-Styles bzw. 56 Jamendo-Mood/Theme-Tags.
    # Die .json-Dateien liefern die Klassennamen zu den Modell-Outputs.
    "genre_discogs400-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_discogs400-discogs-effnet-1.json": f"{MODELS_BASE}/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
    "mtg_jamendo_moodtheme-discogs-effnet-1.pb": f"{MODELS_BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "mtg_jamendo_moodtheme-discogs-effnet-1.json": f"{MODELS_BASE}/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.json",
}

MIN_MODEL_BYTES = 1024
MIN_METADATA_BYTES = 128  # die .json-Label-Dateien sind deutlich kleiner als die Modelle
MAX_DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 30

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".aiff", ".aif", ".ogg"}

# Parallele Worker fuer den billigen Tag-/BPM-/Key-Pfad (Prozesse, kein TF):
# RhythmExtractor/KeyExtractor sind single-threaded, der Audio-Decode von
# /mnt/* ist I/O-lastig -- Parallelitaet skaliert hier fast linear.
# Halbe Kernzahl: Hyperthreading bringt bei DSP wenig, und der Rechner
# bleibt waehrend eines Backfills benutzbar. Die teure Voll-Analyse bleibt
# bewusst sequenziell (TensorFlow parallelisiert einen Forward-Pass intern
# schon ueber die Kerne, mehr Prozesse = nur mehr RAM und Modell-Ladezeit).
ANALYSIS_WORKERS = max(1, (os.cpu_count() or 2) // 2)

# Fester Dateiname pro Musik-Ordner -- 'analyze' und die Suche verwenden
# denselben, dadurch braucht es nie eine separate CSV-Pfad-Eingabe.
CSV_FILENAME = "library_analysis.csv"

CSV_FIELDNAMES = [
    "filepath",
    "artist",
    "title",
    "bpm",
    "key",
    # "1", wenn der Key von uns geschaetzt wurde (compute_key) statt aus
    # einem DJ-Software-Tag zu stammen; leer bei Tag oder unbekannt.
    # Anzeige dann gedimmt mit ~-Praefix; ein spaeter auftauchender Tag
    # ueberschreibt Wert UND loescht das Flag. Alte CSVs ohne die Spalte
    # lesen sich als leer -- kein Migrationscode noetig.
    "key_estimated",
    "year",
    "genres",
    "vibes",
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
    # Gemitteltes discogs-effnet-Embedding (1280-dim), auf dem die
    # Classification-Heads laufen. Persistiert, damit kuenftige neue Heads
    # als reiner CSV-Backfill laufen koennen (Head auf dem Mittel ist eine
    # Naeherung ggue. Mittel der Head-Outputs pro Patch, fuer Tags ok) --
    # ohne erneuten Audio-Decode. Dient zugleich als Vollstaendigkeits-
    # Marker fuer das Genre/Vibe-Schema (siehe missing_parts).
    "effnet_embedding",
    "error",
]

FLOAT_FIELDS = [
    "aggressive", "happy", "sad", "relaxed", "party", "danceable",
    "approachability", "engagement", "arousal", "valence",
]

# --- Key-Schaetzung ----------------------------------------------------------

# Profil fuer Essentias KeyExtractor, wenn kein DJ-Software-Tag vorliegt.
# Gemessen mit scripts/key_eval.py (Testset 41 + HumanMusic-Stichprobe 80,
# Referenz = Rekordbox/Traktor-Tags): edmm 75-80% exakt, 0% Halbton-Fehler.
# Bewusst OHNE Tuning-Korrektur (TuningFrequencyExtractor verschlechterte
# die Trefferquote) und fest 440 Hz -- der historische "85% ein Halbton
# daneben"-Befund war ein Notationsvergleichs-Artefakt, kein Modellfehler.
KEY_PROFILE = "edmm"
# Notation der geschaetzten Keys: "openkey" ('9m') oder "camelot" ('4A').
# Sollte zur Notation der eigenen DJ-Software-Tags passen -- fuers Ranking
# ist es egal (harmonic_distance vergleicht kanonisch ueber Pitch-Classes),
# aber die Liste liest sich einheitlicher.
KEY_NOTATION = "openkey"

# --- Ranking ---------------------------------------------------------------

# Gewichte fuer die Penalty-Terme im Score (Cosine-Similarity minus Abzuege).
W_BPM = 0.5
W_KEY = 0.3
W_MOOD = 0.4

# Wie viele Ergebniszeilen gerendert werden (Liste ist scrollbar).
TOP_N = 100

# Farbschwellen fuer die Transition-Score-Spalten: >= Schwelle -> Stil,
# darunter rot. Absolute Werte -- je nach Library-Dichte ggf. nachkalibrieren.
SCORE_COLOR_STEPS = [(0.9, "green"), (0.8, "yellow"), (0.7, "orange1")]

# --- Energie-Achse (Target-Shifting) ----------------------------------------

# Pro Energie-Stufe e (-3..+3) wird das Suchziel verschoben, nicht der Score
# belohnt -- sonst gewinnt immer der extremste Track der Library.
# +-6 deckt ab, wo die Ziel-Verschiebung physisch saettigt (aggressive/
# relaxed klemmen bei 0/1, arousal bei 9) -- mehr Stufen waeren tote Tasten.
ENERGY_MIN = -6
ENERGY_MAX = 6
BPM_PER_ENERGY_STEP = 3.5
AROUSAL_PER_ENERGY_STEP = 0.4
AGGRESSIVE_PER_ENERGY_STEP = 0.08
RELAXED_PER_ENERGY_STEP = 0.08

# Wertebereich des DEAM-Modells (arousal/valence).
AROUSAL_VALENCE_RANGE = (1.0, 9.0)

# --- Genre-/Vibe-Tagging -----------------------------------------------------

GENRE_MODEL_JSON = "genre_discogs400-discogs-effnet-1.json"
VIBE_MODEL_JSON = "mtg_jamendo_moodtheme-discogs-effnet-1.json"

# Discogs-Styles heissen "Electronic---Acid House"; angezeigt wird nur der
# Teil nach dem Separator.
GENRE_LABEL_SEPARATOR = "---"
# Top-1 wird immer uebernommen (sonst waere 'genres' bei unsicheren Tracks
# leer und die Zeile saehe unfertig aus); weitere Styles nur ab Schwelle.
GENRE_MAX = 2
GENRE_MIN_PROB = 0.10

# Jamendo-Mood/Theme ist multi-label (Sigmoid, typisch kleine Aktivierungen).
VIBE_MAX = 3
VIBE_MIN_PROB = 0.10
# DJ-relevante Teilmenge der 56 Jamendo-Tags -- der Rest (children, christmas,
# corporate, trailer, ...) ist Produktionsmusik-Vokabular und waere nur Rauschen.
VIBE_WHITELIST = {
    "calm", "cool", "dark", "deep", "dream", "emotional", "energetic",
    "epic", "fast", "fun", "groovy", "happy", "heavy", "meditative",
    "melancholic", "melodic", "party", "powerful", "relaxing", "retro",
    "romantic", "sad", "sexy", "slow", "soft", "space", "summer",
    "upbeat", "uplifting",
}

# Trennzeichen fuer Mehrfachwerte in den CSV-Feldern 'genres'/'vibes'.
TAG_SEPARATOR = "|"

# Textfarben fuer Genre-Chips (Rich-Farbnamen, gerendert auf dunklem Pill);
# die Zuordnung ist ein stabiler Hash auf den Style-Namen, damit derselbe
# Style immer dieselbe Farbe traegt. Bewusst mittelhelle Toene: sichtbar,
# aber leiser als das Track-Label darueber.
GENRE_CHIP_COLORS = [
    "orchid", "dark_orange", "cornflower_blue", "dark_sea_green4", "medium_purple",
    "hot_pink3", "steel_blue1", "gold3", "dark_cyan", "grey66",
]
