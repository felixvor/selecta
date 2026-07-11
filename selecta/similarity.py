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


def harmonic_distance(key1_raw, key2_raw):
    """Grobe Distanz auf dem Quintenzirkel: 0=identisch, 1=eng verwandt
    (Paralleltonart oder Nachbarquinte), 2=noch harmonisch mixbar, hoeher =
    deutlicherer Bruch. None, wenn eine Tonart nicht geparst werden konnte
    (=> neutral, kein Bonus/Malus)."""
    k1, k2 = parse_key(key1_raw), parse_key(key2_raw)
    if k1 is None or k2 is None:
        return None
    n1, m1 = k1
    n2, m2 = k2
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
        mood_dist = float(np.linalg.norm(mood_vector(track) - target_mood))
        score = _combined_score(float(cos_all[i]), bpm_pen, key_pen, mood_dist)

        t_arousal = _to_float(track.get("arousal"))
        t_aggressive = _to_float(track.get("aggressive"))
        t_valence = _to_float(track.get("valence"))

        results.append({
            "track": track,
            "score": score,
            "cos_sim": float(cos_all[i]),
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

def pair_score(track_a: dict, track_b: dict) -> float:
    """Direkter Uebergangs-Score zwischen zwei konkreten Tracks: Embedding-
    Cosine minus BPM/Key/Mood-Penalties, ohne Energie-Verschiebung."""
    a, b = track_a["_embedding"], track_b["_embedding"]
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    bpm_pen = relative_bpm_distance(track_a.get("bpm"), track_b.get("bpm"))
    key_pen = harmonic_distance(track_a.get("key"), track_b.get("key"))
    mood_dist = float(np.linalg.norm(mood_vector(track_a) - mood_vector(track_b)))
    return _combined_score(cos, bpm_pen, key_pen, mood_dist)


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
        results.append({
            "track": track,
            "score_a": pair_score(query, track),
            "score_b": pair_score(track, target),
            "d_bpm": (t_bpm - q_bpm) if (t_bpm and q_bpm) else None,
        })
    results.sort(key=lambda r: min(r["score_a"], r["score_b"]), reverse=True)
    return results[:top]
