"""Ranking und Transition-Bruecken.

Kern: Cosine-Similarity der Track-Embeddings, minus Penalties fuer BPM-,
Key- und Mood-Abweichung. Die Energie-Achse verschiebt das Suchziel
(Target-Shifting) statt Extreme zu belohnen.
"""

import numpy as np

from .config import (
    AGGRESSIVE_PER_ENERGY_STEP,
    AROUSAL_PER_ENERGY_STEP,
    AROUSAL_VALENCE_RANGE,
    BPM_PER_ENERGY_STEP,
    RELAXED_PER_ENERGY_STEP,
    TOP_N,
    W_BPM,
    W_KEY,
    W_MOOD,
)

MOOD_DIMS = ["aggressive", "relaxed", "danceable"]


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Notennamen -> Pitch-Class (0=C ... 11=B), inkl. enharmonischer Schreibweisen.
# Essentias Key-Algorithmen liefern Namen wie "A", "Eb", "F#".
NOTE_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


def note_to_camelot(tonic: str, scale: str) -> str:
    """Notenname + Modus (Essentia-Ausgabe, z.B. 'A'/'minor') -> Camelot-Code
    ('8A'). Anker: C-Dur = 8B, A-Moll = 8A; +1 auf dem Rad = Quinte hoch.
    Leerer String, wenn der Notenname unbekannt ist."""
    pc = NOTE_PC.get(tonic.strip())
    if pc is None or scale not in ("major", "minor"):
        return ""
    if scale == "minor":
        pc = (pc + 3) % 12  # Parallel-Dur bestimmt die Rad-Position
    num = ((pc * 7) % 12 + 7) % 12 + 1
    return f"{num}{'A' if scale == 'minor' else 'B'}"


def note_to_openkey(tonic: str, scale: str) -> str:
    """Wie note_to_camelot, aber Open-Key-Notation ('1m'/'1d'; Anker:
    C-Dur = 1d, A-Moll = 1m -- das Rad ist gegenueber Camelot um 7
    Positionen verschoben)."""
    camelot = note_to_camelot(tonic, scale)
    if not camelot:
        return ""
    num = (int(camelot[:-1]) - 8) % 12 + 1
    return f"{num}{'m' if scale == 'minor' else 'd'}"


def key_to_pitch_class(raw: str):
    """Getaggten Key (Open Key oder Camelot, via parse_key) -> (pitch_class,
    'major'|'minor') der Tonika. Noetig, um Keys aus verschiedenen Notationen
    auf einer kanonischen Ebene zu vergleichen -- parse_key allein reicht
    nicht, weil Open Key '8m' (Bb-Moll) und Camelot '8A' (A-Moll) dieselbe
    Nummer tragen. None, wenn nicht parsbar."""
    parsed = parse_key(raw)
    if parsed is None:
        return None
    num, mode = parsed
    letter = raw.strip()[-1].lower()
    # Rad-Position -> Pitch-Class der Dur-Tonika; Camelot-Anker 8B=C,
    # Open-Key-Anker 1d=C. Moll: relative Tonika 3 Halbtoene tiefer.
    offset = 8 if letter in ("a", "b") else 1
    pc_major = ((num - offset) * 7) % 12
    if mode == "minor":
        return ((pc_major + 9) % 12, "minor")
    return (pc_major, "major")


def parse_key(raw: str):
    """Parst Open-Key ('7m'/'12d', wie z.B. Rekordbox es schreibt) oder
    Camelot-Notation ('8A'/'8B'). Rueckgabe: (nummer 1-12, 'major'|'minor')
    oder None, wenn nicht parsbar."""
    if not raw:
        return None
    raw = raw.strip()
    if len(raw) < 2:
        return None
    letter = raw[-1].lower()
    try:
        num = int(raw[:-1])
    except ValueError:
        return None
    if not (1 <= num <= 12):
        return None
    if letter in ("d", "b"):    # Open Key 'dur' bzw. Camelot 'B' = Dur/Major
        mode = "major"
    elif letter in ("m", "a"):  # Open Key 'moll' bzw. Camelot 'A' = Moll/Minor
        mode = "minor"
    else:
        return None
    return (num, mode)


def _wheel_position(pc: int, mode: str) -> int:
    """Position auf dem Quintenrad (0-11); Moll liegt auf der Position
    seines Relativ-Durs (Camelot/Open Key: gleiche Nummer = Relativpaar)."""
    if mode == "minor":
        pc = (pc + 3) % 12
    return (pc * 7) % 12


def harmonic_distance(key1_raw, key2_raw):
    """Grobe Distanz auf dem Quintenzirkel: 0=identisch, 1=eng verwandt
    (Relativtonart oder Nachbarquinte), 2=noch harmonisch mixbar, hoeher =
    deutlicherer Bruch. None, wenn eine Tonart nicht geparst werden konnte
    (=> neutral, kein Bonus/Malus).

    Vergleicht kanonisch ueber Pitch-Classes statt ueber die rohen
    Rad-Nummern -- Open Key und Camelot sind um 7 Positionen gegeneinander
    verschoben, gemischte Notationen (z.B. Traktor-Tags neben geschaetzten
    Keys in anderer Notation) waeren sonst systematisch falsch."""
    k1, k2 = key_to_pitch_class(key1_raw), key_to_pitch_class(key2_raw)
    if k1 is None or k2 is None:
        return None
    n1, m1 = _wheel_position(*k1), k1[1]
    n2, m2 = _wheel_position(*k2), k2[1]
    if n1 == n2 and m1 == m2:
        return 0
    if n1 == n2 and m1 != m2:
        return 1
    diff = (n1 - n2) % 12
    diff = min(diff, 12 - diff)
    if m1 == m2 and diff == 1:
        return 1
    if diff == 1:
        return 2
    return 3 + diff


def relative_bpm_distance(bpm_target, bpm_candidate):
    """Relative BPM-Differenz zum Ziel. Beruecksichtigt Halb-/Doppeltempo."""
    bpm_target, bpm_candidate = _to_float(bpm_target), _to_float(bpm_candidate)
    if not bpm_target or not bpm_candidate:
        return None
    candidates = [
        abs(bpm_target - bpm_candidate),
        abs(bpm_target - bpm_candidate * 2),
        abs(bpm_target - bpm_candidate / 2),
    ]
    return min(candidates) / bpm_target


def _normalize_av(value):
    lo, hi = AROUSAL_VALENCE_RANGE
    v = _to_float(value)
    v = lo if v is None else v
    return (v - lo) / (hi - lo)


def mood_vector_from_values(aggressive, relaxed, danceable, arousal, valence) -> np.ndarray:
    return np.array(
        [
            aggressive or 0.0,
            relaxed or 0.0,
            danceable or 0.0,
            _normalize_av(arousal),
            _normalize_av(valence),
        ],
        dtype=np.float32,
    )


def mood_vector(row: dict) -> np.ndarray:
    return mood_vector_from_values(
        _to_float(row.get("aggressive")),
        _to_float(row.get("relaxed")),
        _to_float(row.get("danceable")),
        row.get("arousal"),
        row.get("valence"),
    )


# Welche Mood-Dimensionen die Energie-Achse tatsaechlich verschiebt
# (aggressive, relaxed, arousal) -- danceable/valence bleiben beim
# Target-Shifting fix und werden bei aktiver Energie ausgeblendet
# (Reihenfolge wie mood_vector: aggressive, relaxed, danceable,
# arousal, valence).
ENERGY_DIM_MASK = np.array([1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)


def mood_scales(tracks) -> np.ndarray:
    """Standardabweichung pro Mood-Dimension ueber die Library (auf
    mood_vector-Skala). Grundlage der z-normierten Energie-Distanz:
    ohne Normierung dominiert die Dimension mit der zufaellig groessten
    Roh-Spanne, obwohl z.B. danceable in einer DJ-Library praktisch
    konstant ist (gemessen: std 0.011) und nichts aussagt."""
    vecs = np.stack([mood_vector(t) for t in tracks])
    return np.maximum(vecs.std(axis=0), 1e-3)


def energy_mood_distance(track_vec: np.ndarray, target_mood: np.ndarray, scales: np.ndarray) -> float:
    """Mood-Distanz bei aktiver Energie-Achse: z-normiert und nur ueber die
    tatsaechlich verschobenen Dimensionen (ENERGY_DIM_MASK), skaliert auf
    die Groessenordnung der 5-dim-Distanz.

    Datengetrieben gewaehlt (scripts/energy_eval.py, 1426 echte Tracks):
    ggue. der rohen 5-dim-Distanz steigt die Monotonie der Energie-Antwort
    (Spearman 0.79 -> 0.96) und die Zahl neu aufgedeckter Tracks pro
    Energie-Stufe verdoppelt sich (discovery@10 3.3 -> 8.1), waehrend die
    mittlere Embedding-Aehnlichkeit der Top-10 nur minimal faellt
    (0.944 -> 0.918)."""
    zdiff = (track_vec - target_mood) / scales
    return float(np.linalg.norm(zdiff * ENERGY_DIM_MASK)) * float(np.sqrt(5.0 / 3.0))


def shifted_target(query: dict, energy: int = 0):
    """Suchziel aus der Query, verschoben um die Energie-Stufe.

    Rueckgabe: (target_bpm | None, target_mood_vector).
    """
    av_lo, av_hi = AROUSAL_VALENCE_RANGE
    bpm_q = _to_float(query.get("bpm"))
    target_bpm = None
    if bpm_q:
        target_bpm = bpm_q + BPM_PER_ENERGY_STEP * energy

    aggressive = _to_float(query.get("aggressive")) or 0.0
    relaxed = _to_float(query.get("relaxed")) or 0.0
    arousal = _to_float(query.get("arousal"))
    arousal = av_lo if arousal is None else arousal

    target_mood = mood_vector_from_values(
        min(1.0, max(0.0, aggressive + AGGRESSIVE_PER_ENERGY_STEP * energy)),
        min(1.0, max(0.0, relaxed - RELAXED_PER_ENERGY_STEP * energy)),
        _to_float(query.get("danceable")),
        min(av_hi, max(av_lo, arousal + AROUSAL_PER_ENERGY_STEP * energy)),
        query.get("valence"),
    )
    return target_bpm, target_mood


def _bpm_cutoffs(q_bpm, bpm_offset):
    """Harte Filtergrenzen (floor, ceiling) aus Query-BPM + Offset. Der Anker
    wird gerundet (BPM liegt nicht immer als ganze Zahl vor -- eigene Analyse
    liefert z.B. "128.1"), sonst waere der tatsaechliche Cutoff minimal
    daneben ggue. dem, was App-Header und BPM-Stepping anzeigen."""
    if not q_bpm or not bpm_offset:
        return None, None
    cutoff = round(q_bpm) + bpm_offset
    return (cutoff, None) if bpm_offset > 0 else (None, cutoff)


def _combined_score(cos_sim, bpm_pen, key_pen, mood_dist, w_bpm=W_BPM, w_key=W_KEY, w_mood=W_MOOD):
    score = cos_sim
    if bpm_pen is not None:
        score -= w_bpm * min(bpm_pen, 1.0)
    if key_pen is not None:
        score -= w_key * (key_pen / 8.0)
    score -= w_mood * (mood_dist / 2.5)
    return score


def rank_similar(query: dict, library, energy: int = 0, bpm_offset: float = 0.0, top: int = TOP_N):
    """Rankt die ganze Library gegen die (ggf. verschobene) Query.

    library: Library-Objekt (tracks + zeilennormalisierte matrix).
    bpm_offset: harter Tempo-Filter relativ zur Query-BPM -- unabhaengig von
    der Energie-Achse, die weiterhin weich auf Score/Mood wirkt.
    >0 blendet alles UNTER query_bpm+offset aus (nur gleich/schneller),
    <0 blendet alles UEBER query_bpm+offset aus (nur gleich/langsamer),
    0 = kein Filter. Tracks ohne BPM-Tag fallen bei aktivem Filter raus,
    da ihre Tauglichkeit nicht verifizierbar ist.

    Rueckgabe: Liste von dicts mit track, score, cos_sim und den
    Anzeige-Deltas relativ zur Query (nicht zum verschobenen Ziel).
    """
    if library.matrix is None:
        return []

    q_emb = query["_embedding"]
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    cos_all = library.matrix @ q_norm

    target_bpm, target_mood = shifted_target(query, energy)

    q_bpm = _to_float(query.get("bpm"))
    q_arousal = _to_float(query.get("arousal"))
    q_aggressive = _to_float(query.get("aggressive"))
    q_valence = _to_float(query.get("valence"))

    bpm_floor, bpm_ceiling = _bpm_cutoffs(q_bpm, bpm_offset)

    # Bei energy == 0 bleibt das Ranking exakt wie bisher (rohe 5-dim-
    # Distanz); erst die Energie-Taste schaltet auf die z-normierte
    # Subspace-Distanz um -- siehe energy_mood_distance.
    scales = mood_scales(library.tracks) if energy != 0 else None

    results = []
    for i, track in enumerate(library.tracks):
        if track["filepath"] == query["filepath"]:
            continue

        t_bpm = _to_float(track.get("bpm"))
        if bpm_floor is not None and (t_bpm is None or t_bpm < bpm_floor):
            continue
        if bpm_ceiling is not None and (t_bpm is None or t_bpm > bpm_ceiling):
            continue

        bpm_pen = relative_bpm_distance(target_bpm, track.get("bpm"))
        key_pen = harmonic_distance(query.get("key"), track.get("key"))
        if scales is not None:
            mood_dist = energy_mood_distance(mood_vector(track), target_mood, scales)
        else:
            mood_dist = float(np.linalg.norm(mood_vector(track) - target_mood))
        score = _combined_score(float(cos_all[i]), bpm_pen, key_pen, mood_dist)

        t_arousal = _to_float(track.get("arousal"))
        t_aggressive = _to_float(track.get("aggressive"))
        t_valence = _to_float(track.get("valence"))

        results.append({
            "track": track,
            "score": score,
            "cos_sim": float(cos_all[i]),
            # Rohe Penalty-Terme fuer die Score-Zerlegung in der Detail-
            # Zeile ("warum steht der hier?") -- None = neutral/unbekannt.
            "bpm_pen": bpm_pen,
            "mood_dist": mood_dist,
            "key_pen": key_pen,
            "d_bpm": (t_bpm - q_bpm) if (t_bpm and q_bpm) else None,
            "d_arousal": (t_arousal - q_arousal) if (t_arousal is not None and q_arousal is not None) else None,
            "d_aggressive": (t_aggressive - q_aggressive) if (t_aggressive is not None and q_aggressive is not None) else None,
            "d_valence": (t_valence - q_valence) if (t_valence is not None and q_valence is not None) else None,
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top]


# ---------------------------------------------------------------------------
# Transition (Bruecke von A nach B)
# ---------------------------------------------------------------------------

def pair_score_parts(track_a: dict, track_b: dict) -> dict:
    """Wie pair_score, aber mit den rohen Einzelterme (cos, bpm_pen, key_pen,
    mood_dist) statt nur der Summe -- Grundlage der Bridge-Warum-Zeile und
    der Slash-Zellen (BPM/Key je zu A und zu B)."""
    a, b = track_a["_embedding"], track_b["_embedding"]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    bpm_pen = relative_bpm_distance(track_a.get("bpm"), track_b.get("bpm"))
    key_pen = harmonic_distance(track_a.get("key"), track_b.get("key"))
    mood_dist = float(np.linalg.norm(mood_vector(track_a) - mood_vector(track_b)))
    score = _combined_score(cos, bpm_pen, key_pen, mood_dist)
    return {"score": score, "cos_sim": cos, "bpm_pen": bpm_pen,
            "key_pen": key_pen, "mood_dist": mood_dist}


def pair_score(track_a: dict, track_b: dict) -> float:
    """Direkter Uebergangs-Score zwischen zwei konkreten Tracks: Embedding-
    Cosine minus BPM/Key/Mood-Penalties, ohne Energie-Verschiebung."""
    return pair_score_parts(track_a, track_b)["score"]


def rank_bridge(query: dict, target: dict, library, bpm_offset: float = 0.0, top: int = TOP_N):
    """Brueckenkandidaten zwischen aktuellem Track (A) und Transition-Ziel (B).

    Pro Kandidat der direkte Uebergangs-Score zu beiden Seiten, sortiert nach
    der SCHWAECHEREN Seite (min) -- der Engpass entscheidet, ob eine Bruecke
    funktioniert. Eine Summen-Sortierung wuerde 0.95/0.55 ueber 0.74/0.72
    ranken, obwohl die B-Seite unbrauchbar ist. B laeuft selbst als Kandidat
    mit (score_b == 1.0) und steht damit genau dann oben, wenn der
    Direktsprung die beste Option ist. bpm_offset filtert hart relativ zu A,
    wie in rank_similar."""
    if library.matrix is None:
        return []

    q_bpm = _to_float(query.get("bpm"))
    bpm_floor, bpm_ceiling = _bpm_cutoffs(q_bpm, bpm_offset)

    results = []
    for track in library.tracks:
        if track["filepath"] == query["filepath"]:
            continue
        t_bpm = _to_float(track.get("bpm"))
        if bpm_floor is not None and (t_bpm is None or t_bpm < bpm_floor):
            continue
        if bpm_ceiling is not None and (t_bpm is None or t_bpm > bpm_ceiling):
            continue
        parts_a = pair_score_parts(query, track)
        parts_b = pair_score_parts(track, target)
        results.append({
            "track": track,
            "score_a": parts_a["score"],
            "score_b": parts_b["score"],
            # Einzelterme zu beiden Seiten -- Grundlage der Bridge-Warum-
            # Zeile ("→A ... →B ...") und der Slash-Zellen.
            "parts_a": parts_a,
            "parts_b": parts_b,
            "d_bpm": (t_bpm - q_bpm) if (t_bpm and q_bpm) else None,
        })
    results.sort(key=lambda r: min(r["score_a"], r["score_b"]), reverse=True)
    return results[:top]
