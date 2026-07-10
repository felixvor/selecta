"""Ranking und Transition-Planung.

Kern: Cosine-Similarity der Track-Embeddings, minus Penalties fuer BPM-,
Key- und Mood-Abweichung. Die Energie-Achse verschiebt das Suchziel
(Target-Shifting) statt Extreme zu belohnen.
"""

import math

import numpy as np

from .config import (
    AGGRESSIVE_PER_ENERGY_STEP,
    AROUSAL_PER_ENERGY_STEP,
    AROUSAL_VALENCE_RANGE,
    BPM_PER_ENERGY_STEP,
    RELAXED_PER_ENERGY_STEP,
    TOP_N,
    TRANSITION_AROUSAL_PER_STEP,
    TRANSITION_BPM_PER_STEP,
    TRANSITION_EMB_DIST_PER_STEP,
    TRANSITION_MAX_TRACKS,
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


def shifted_target(query: dict, energy: int = 0, bpm_offset: float = 0.0):
    """Suchziel aus der Query, verschoben um Energie-Stufe und BPM-Offset.

    Rueckgabe: (target_bpm | None, target_mood_vector).
    """
    av_lo, av_hi = AROUSAL_VALENCE_RANGE
    bpm_q = _to_float(query.get("bpm"))
    target_bpm = None
    if bpm_q:
        target_bpm = bpm_q + BPM_PER_ENERGY_STEP * energy + bpm_offset

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
    Rueckgabe: Liste von dicts mit track, score, cos_sim und den
    Anzeige-Deltas relativ zur Query (nicht zum verschobenen Ziel).
    """
    if library.matrix is None:
        return []

    q_emb = query["_embedding"]
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    cos_all = library.matrix @ q_norm

    target_bpm, target_mood = shifted_target(query, energy, bpm_offset)

    q_bpm = _to_float(query.get("bpm"))
    q_arousal = _to_float(query.get("arousal"))
    q_aggressive = _to_float(query.get("aggressive"))
    q_valence = _to_float(query.get("valence"))

    results = []
    for i, track in enumerate(library.tracks):
        if track["filepath"] == query["filepath"]:
            continue

        bpm_pen = relative_bpm_distance(target_bpm, track.get("bpm"))
        key_pen = harmonic_distance(query.get("key"), track.get("key"))
        mood_dist = float(np.linalg.norm(mood_vector(track) - target_mood))
        score = _combined_score(float(cos_all[i]), bpm_pen, key_pen, mood_dist)

        t_bpm = _to_float(track.get("bpm"))
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
# Transition-Planer
# ---------------------------------------------------------------------------

def auto_num_tracks(track_a: dict, track_b: dict) -> int:
    """Zwischentrack-Anzahl aus der Groesse des Sprungs A->B: ein Track pro
    spuerbarem Schritt in BPM, Embedding-Distanz oder Arousal."""
    a_emb = track_a["_embedding"]
    b_emb = track_b["_embedding"]
    a_n = a_emb / (np.linalg.norm(a_emb) + 1e-9)
    b_n = b_emb / (np.linalg.norm(b_emb) + 1e-9)
    emb_dist = 1.0 - float(np.dot(a_n, b_n))

    bpm_a, bpm_b = _to_float(track_a.get("bpm")), _to_float(track_b.get("bpm"))
    d_bpm = abs(bpm_a - bpm_b) if (bpm_a and bpm_b) else 0.0
    ar_a, ar_b = _to_float(track_a.get("arousal")), _to_float(track_b.get("arousal"))
    d_arousal = abs(ar_a - ar_b) if (ar_a is not None and ar_b is not None) else 0.0

    k = max(
        math.ceil(d_bpm / TRANSITION_BPM_PER_STEP),
        math.ceil(emb_dist / TRANSITION_EMB_DIST_PER_STEP),
        math.ceil(d_arousal / TRANSITION_AROUSAL_PER_STEP),
    ) - 1
    return max(1, min(TRANSITION_MAX_TRACKS, k))


def _chain_deltas(prev: dict, cur: dict):
    """Anzeige-Differenzen zwischen zwei aufeinanderfolgenden Kettengliedern."""
    p = prev["_embedding"] / (np.linalg.norm(prev["_embedding"]) + 1e-9)
    c = cur["_embedding"] / (np.linalg.norm(cur["_embedding"]) + 1e-9)
    emb_dist = 1.0 - float(np.dot(p, c))
    bpm_p, bpm_c = _to_float(prev.get("bpm")), _to_float(cur.get("bpm"))
    d_bpm = (bpm_c - bpm_p) if (bpm_p and bpm_c) else None
    key_rel = harmonic_distance(prev.get("key"), cur.get("key"))
    return {"emb_dist": emb_dist, "d_bpm": d_bpm, "key_rel": key_rel}


def plan_transition(track_a: dict, track_b: dict, library, num_tracks=None):
    """Baut eine Kette A -> t1..tk -> B ueber lineare Wegpunkte im
    Embedding-/BPM-/Mood-Raum; pro Wegpunkt greedy der beste unbenutzte Track.

    num_tracks: int oder None (= auto).
    Rueckgabe: Liste von Zeilen {track, deltas|None}, erste Zeile ist A.
    """
    k = num_tracks if num_tracks else auto_num_tracks(track_a, track_b)
    k = max(1, min(TRANSITION_MAX_TRACKS, k))

    a_emb = track_a["_embedding"] / (np.linalg.norm(track_a["_embedding"]) + 1e-9)
    b_emb = track_b["_embedding"] / (np.linalg.norm(track_b["_embedding"]) + 1e-9)
    bpm_a, bpm_b = _to_float(track_a.get("bpm")), _to_float(track_b.get("bpm"))
    mood_a, mood_b = mood_vector(track_a), mood_vector(track_b)

    used = {track_a["filepath"], track_b["filepath"]}
    chain = [track_a]
    prev = track_a

    for i in range(1, k + 1):
        t = i / (k + 1)
        way_emb = (1 - t) * a_emb + t * b_emb
        way_emb = way_emb / (np.linalg.norm(way_emb) + 1e-9)
        way_bpm = (1 - t) * bpm_a + t * bpm_b if (bpm_a and bpm_b) else None
        way_mood = (1 - t) * mood_a + t * mood_b

        cos_all = library.matrix @ way_emb
        best_idx, best_score = None, None
        for j, track in enumerate(library.tracks):
            if track["filepath"] in used:
                continue
            bpm_pen = relative_bpm_distance(way_bpm, track.get("bpm"))
            key_pen = harmonic_distance(prev.get("key"), track.get("key"))
            mood_dist = float(np.linalg.norm(mood_vector(track) - way_mood))
            score = _combined_score(float(cos_all[j]), bpm_pen, key_pen, mood_dist)
            if best_score is None or score > best_score:
                best_idx, best_score = j, score

        if best_idx is None:
            break  # Library kleiner als die Kette
        step = library.tracks[best_idx]
        used.add(step["filepath"])
        chain.append(step)
        prev = step

    chain.append(track_b)

    rows = [{"track": chain[0], "deltas": None}]
    for prev_t, cur_t in zip(chain, chain[1:]):
        rows.append({"track": cur_t, "deltas": _chain_deltas(prev_t, cur_t)})
    return rows
