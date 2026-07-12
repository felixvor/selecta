"""Diagnose der Energie-Achse: Bewirkt die +/-Energie-Taste wirklich eine
hoerbare Bewegung nach oben/unten -- oder dreht sie nur an Werten?

Misst auf echten Library-CSVs drei Varianten des Rankings:

  A "current"   -- shifted_target + volle 5-dim Mood-Distanz (Ist-Zustand)
  B "zscore"    -- wie A, aber Mood-Dimensionen per Library-Std z-normiert
  C "subspace"  -- bei energy != 0 zaehlen in der Mood-Distanz nur die
                   tatsaechlich verschobenen Dimensionen (aggressive,
                   relaxed, arousal); danceable/valence werden ausgeblendet

Metriken pro Variante (ueber zufaellige Query-Tracks gemittelt):

  churn@10      -- wie viele der Top-10 sich ggue. energy=0 aendern
                   (0 = Taste wirkungslos, 10 = komplett andere Liste)
  discovery@10  -- wie viele Top-10-Tracks bei energy=+/-e NICHT in den
                   Top-50 von energy=0 lagen ("deckt neue Songs auf")
  direction     -- Spearman-artige Monotonie: steigt der Energie-Proxy
                   der Ergebnisliste (z(bpm)+z(arousal)+z(aggressive)
                   -z(relaxed), Mittel ueber Top-10) monoton mit e?
                   1.0 = perfekt monoton, 0 = Zufall.

Aufruf (in WSL, venv aktiv):
    python scripts/energy_eval.py /mnt/g/Media/Musik/House/HumanMusic [...]
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from selecta.config import (
    AGGRESSIVE_PER_ENERGY_STEP,
    AROUSAL_PER_ENERGY_STEP,
    AROUSAL_VALENCE_RANGE,
    BPM_PER_ENERGY_STEP,
    RELAXED_PER_ENERGY_STEP,
    W_BPM,
    W_KEY,
    W_MOOD,
)
from selecta.library import Library
from selecta.similarity import (
    _to_float,
    harmonic_distance,
    mood_vector,
    relative_bpm_distance,
    shifted_target,
)

ENERGIES = [-6, -3, 0, 3, 6]
TOP = 10
BASELINE_POOL = 50  # "neu aufgedeckt" = nicht in den Top-50 von e=0
N_QUERIES = 30
SEED = 7


def zstats(tracks):
    """(mean, std) pro Mood-Dimension + BPM ueber die Library."""
    dims = ["aggressive", "relaxed", "danceable", "arousal", "valence", "bpm"]
    stats = {}
    for d in dims:
        vals = [v for t in tracks if (v := _to_float(t.get(d))) is not None]
        arr = np.array(vals, dtype=np.float64)
        stats[d] = (float(arr.mean()), float(arr.std() or 1.0))
    return stats


def energy_proxy(track, stats):
    """Skalarer 'wie energetisch ist dieser Track'-Wert: z(bpm) + z(arousal)
    + z(aggressive) - z(relaxed). Nur fuer die Messung, kein Ranking-Input."""
    total, n = 0.0, 0
    for dim, sign in (("bpm", 1), ("arousal", 1), ("aggressive", 1), ("relaxed", -1)):
        v = _to_float(track.get(dim))
        if v is None:
            continue
        mean, std = stats[dim]
        total += sign * (v - mean) / std
        n += 1
    return total / n if n else 0.0


def mood_distance(track, target_mood, variant, stats, energy):
    """Mood-Distanz je Variante. target_mood ist der 5-dim Vektor aus
    shifted_target (Reihenfolge: aggressive, relaxed, danceable,
    arousal_norm, valence_norm)."""
    tv = mood_vector(track)
    diff = tv - target_mood
    if variant == "current":
        return float(np.linalg.norm(diff))
    if variant == "zscore":
        av_lo, av_hi = AROUSAL_VALENCE_RANGE
        scales = []
        for i, dim in enumerate(["aggressive", "relaxed", "danceable", "arousal", "valence"]):
            _, std = stats[dim]
            if dim in ("arousal", "valence"):
                std = std / (av_hi - av_lo)  # Vektor ist auf 0..1 normiert
            scales.append(std)
        return float(np.linalg.norm(diff / np.array(scales, dtype=np.float32)))
    if variant == "subspace":
        if energy == 0:
            return float(np.linalg.norm(diff))
        # Nur die von der Energie-Achse verschobenen Dimensionen zaehlen;
        # skaliert, damit die Groessenordnung zur 5-dim-Distanz passt.
        mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return float(np.linalg.norm(diff * mask)) * np.sqrt(5.0 / 3.0)
    if variant == "z_subspace":
        # z-normiert UND (bei energy != 0) nur die verschobenen Dimensionen.
        av_lo, av_hi = AROUSAL_VALENCE_RANGE
        scales = []
        for dim in ["aggressive", "relaxed", "danceable", "arousal", "valence"]:
            _, std = stats[dim]
            if dim in ("arousal", "valence"):
                std = std / (av_hi - av_lo)
            scales.append(std)
        zdiff = diff / np.array(scales, dtype=np.float32)
        if energy == 0:
            return float(np.linalg.norm(zdiff))
        mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        return float(np.linalg.norm(zdiff * mask)) * np.sqrt(5.0 / 3.0)
    raise ValueError(variant)


def rank(query, lib, energy, variant, stats, top):
    q_emb = query["_embedding"]
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    cos_all = lib.matrix @ q_norm
    target_bpm, target_mood = shifted_target(query, energy)
    results = []
    for i, track in enumerate(lib.tracks):
        if track["filepath"] == query["filepath"]:
            continue
        bpm_pen = relative_bpm_distance(target_bpm, track.get("bpm"))
        key_pen = harmonic_distance(query.get("key"), track.get("key"))
        mood = mood_distance(track, target_mood, variant, stats, energy)
        score = float(cos_all[i])
        if bpm_pen is not None:
            score -= W_BPM * min(bpm_pen, 1.0)
        if key_pen is not None:
            score -= W_KEY * (key_pen / 8.0)
        score -= W_MOOD * (mood / 2.5)
        results.append((score, float(cos_all[i]), track))
    results.sort(key=lambda r: -r[0])
    return [(t, c) for _, c, t in results[:top]]


def spearman(xs, ys):
    """Rangkorrelation ohne scipy."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank_i, i in enumerate(order):
            r[i] = float(rank_i)
        return r
    rx, ry = np.array(ranks(xs)), np.array(ranks(ys))
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate(lib, stats):
    rng = random.Random(SEED)
    complete = [
        t for t in lib.tracks
        if all(_to_float(t.get(d)) is not None
               for d in ("aggressive", "relaxed", "arousal", "bpm"))
    ]
    queries = rng.sample(complete, min(N_QUERIES, len(complete)))
    variants = ["current", "zscore", "subspace", "z_subspace"]
    agg = {v: {"churn": [], "discovery": [], "direction": [], "cos": []} for v in variants}

    for q in queries:
        for variant in variants:
            base10 = rank(q, lib, 0, variant, stats, TOP)
            base50 = {t["filepath"] for t, _ in rank(q, lib, 0, variant, stats, BASELINE_POOL)}
            base_set = {t["filepath"] for t, _ in base10}
            proxies = []
            churn, disc, cosims = [], [], []
            for e in ENERGIES:
                top = rank(q, lib, e, variant, stats, TOP)
                paths = {t["filepath"] for t, _ in top}
                proxies.append(np.mean([energy_proxy(t, stats) for t, _ in top]))
                cosims.append(np.mean([c for _, c in top]))
                if e != 0:
                    churn.append(len(paths - base_set))
                    disc.append(len(paths - base50))
            agg[variant]["churn"].append(np.mean(churn))
            agg[variant]["discovery"].append(np.mean(disc))
            agg[variant]["direction"].append(spearman(ENERGIES, proxies))
            agg[variant]["cos"].append(np.mean(cosims))
    return agg, len(queries)


def show_example(lib, stats, variant):
    """Eine Beispiel-Query mit Top-5 bei e=-6/0/+6 -- zum Draufschauen."""
    rng = random.Random(SEED)
    complete = [
        t for t in lib.tracks
        if all(_to_float(t.get(d)) is not None
               for d in ("aggressive", "relaxed", "arousal", "bpm"))
    ]
    q = rng.choice(complete)
    print(f"\n  Beispiel-Query: {q.get('artist')} - {q.get('title')}"
          f"  [{q.get('bpm')} BPM, arousal {q.get('arousal')}]  ({variant})")
    for e in (-6, 0, 6):
        top = [t for t, _ in rank(q, lib, e, variant, stats, 5)]
        print(f"    e={e:+d}:")
        for t in top:
            print(f"      {energy_proxy(t, stats):+5.2f}  {t.get('bpm'):>6} BPM"
                  f"  ar {t.get('arousal')}  {t.get('artist')} - {t.get('title')}")


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        sys.exit(1)
    lib = Library(dirs)
    print(f"{len(lib.tracks)} Tracks aus {len(dirs)} Ordner(n)")
    stats = zstats(lib.tracks)
    print("Library-Statistik (mean/std):")
    for d, (m, s) in stats.items():
        print(f"  {d:>12}: {m:7.3f} +/- {s:.3f}")

    agg, n = evaluate(lib, stats)
    print(f"\nMetriken ueber {n} Queries, Energien {ENERGIES}, Top-{TOP}:")
    print(f"  {'Variante':<11} {'churn@10':>9} {'discovery@10':>13} {'direction':>10} {'cos@10':>8}")
    for v, m in agg.items():
        print(f"  {v:<11} {np.mean(m['churn']):>9.2f} {np.mean(m['discovery']):>13.2f}"
              f" {np.mean(m['direction']):>10.2f} {np.mean(m['cos']):>8.3f}")

    for v in ("zscore", "z_subspace"):
        show_example(lib, stats, v)


if __name__ == "__main__":
    main()
