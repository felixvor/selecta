"""Essentia-Analyse: Modell-Download, Feature-Extraktion, Analyse-Loop.

Der Analyse-Loop ist callback-basiert (log/progress/cancel), damit er sowohl
headless (CLI, print) als auch im Textual-Worker-Thread (UI-Updates) laeuft.
"""

import json
import os
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

# TensorFlow-Logs auf Fehler reduzieren -- muss VOR jedem
# TensorFlow/Essentia-Import gesetzt sein (die passieren hier lazy).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")

from .config import (
    CSV_FIELDNAMES,
    DOWNLOAD_TIMEOUT_SECONDS,
    GENRE_LABEL_SEPARATOR,
    GENRE_MAX,
    GENRE_MIN_PROB,
    GENRE_MODEL_JSON,
    MAX_DOWNLOAD_ATTEMPTS,
    MIN_METADATA_BYTES,
    MIN_MODEL_BYTES,
    MODEL_FILES,
    TAG_SEPARATOR,
    VIBE_MAX,
    VIBE_MIN_PROB,
    VIBE_MODEL_JSON,
    VIBE_WHITELIST,
)
from .library import (
    compact_csv,
    csv_path_for,
    encode_embedding,
    find_audio_files,
    load_csv_data,
    read_existing_tags,
)

import csv as csv_module


class FilteredStderr:
    """Schreibt alle stderr-Ausgaben ausser Zeilen mit bestimmten Mustern weiter
    (Essentia/TensorFlow loggen auf C++-Ebene an Python vorbei)."""

    def __init__(self, stream, filter_strings):
        self.stream = stream
        self.filter_strings = filter_strings
        self.buffer = ""

    def write(self, msg):
        self.buffer += msg
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if not any(s in line for s in self.filter_strings):
                self.stream.write(line + "\n")

    def flush(self):
        if self.buffer:
            if not any(s in self.buffer for s in self.filter_strings):
                self.stream.write(self.buffer)
            self.buffer = ""
        self.stream.flush()


@contextmanager
def filtered_stderr(filter_strings):
    original = sys.stderr
    sys.stderr = FilteredStderr(original, filter_strings)
    try:
        yield
    finally:
        sys.stderr.flush()
        sys.stderr = original


def _min_bytes(filename: str) -> int:
    """Plausibilitaets-Minimum pro Dateityp: .pb-Modelle sind >1KB, die
    .json-Label-Dateien koennen deutlich kleiner sein."""
    return MIN_MODEL_BYTES if filename.endswith(".pb") else MIN_METADATA_BYTES


def _download(url: str, dest: Path) -> None:
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(tmp_dest, "wb") as out_file:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

    actual_size = tmp_dest.stat().st_size
    if actual_size < _min_bytes(dest.name) or (total > 0 and actual_size != total):
        tmp_dest.unlink(missing_ok=True)
        raise IOError(f"Download unvollstaendig: erwartet {total} Bytes, erhalten {actual_size} Bytes")

    tmp_dest.replace(dest)


def download_models(models_dir: Path, log=print):
    """Laedt fehlende Modell-Dateien. Wirft RuntimeError statt sys.exit, damit
    die TUI den Fehler anzeigen kann (essentia.upf.edu ist gelegentlich down)."""
    models_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        name for name in MODEL_FILES
        if not (models_dir / name).exists() or (models_dir / name).stat().st_size < _min_bytes(name)
    ]
    if not missing:
        return

    log(f"{len(missing)} von {len(MODEL_FILES)} Modell-Dateien fehlen, lade herunter ...")
    for filename in missing:
        dest = models_dir / filename
        url = MODEL_FILES[filename]
        if dest.exists():
            dest.unlink()

        last_error = None
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            log(f"[{attempt}/{MAX_DOWNLOAD_ATTEMPTS}] Lade {filename} ...")
            try:
                _download(url, dest)
                log(f"{filename}: fertig ({dest.stat().st_size / 1e6:.1f} MB)")
                last_error = None
                break
            except Exception as e:
                last_error = e
                log(f"Versuch {attempt} fehlgeschlagen: {e}")
                time.sleep(2)

        if last_error is not None:
            raise RuntimeError(
                f"Konnte {filename} nicht laden ({last_error}). "
                f"essentia.upf.edu spaeter erneut versuchen."
            )


def compute_bpm(filepath: Path) -> str:
    """BPM selbst berechnen (RhythmExtractor2013 -- schneller Standard-
    algorithmus, kein TensorFlow-Modell noetig). Eigener 44.1kHz-Load, weil
    die Embedding-Modelle mit 16kHz arbeiten, RhythmExtractor2013 aber die
    volle Samplerate braucht.

    Kein Key-Pendant hier bewusst: Essentias KeyExtractor liegt gegen 41
    Rekordbox-getaggte Referenztracks bei 85% exakt einen Halbton daneben --
    quer durch alle Tonart-Profile (bgate/edma/edmm/temperley/krumhansl/
    shaath). BPM stimmt dagegen im Schnitt auf 0.09 BPM genau. Ein falsch
    geschaetzter Key sieht in der Liste wie ein echter aus und kann beim
    harmonischen Mixen crashen -- lieber "?" als eine Zahl, der man nicht
    ansieht, dass sie ratet."""
    from essentia.standard import MonoLoader, RhythmExtractor2013

    audio = MonoLoader(filename=str(filepath))()
    bpm, _ticks, _confidence, _estimates, _intervals = RhythmExtractor2013(method="multifeature")(audio)
    return f"{bpm:.1f}"


def load_model_labels(models_dir: Path, json_filename: str) -> list[str]:
    """Klassennamen aus der Metadata-JSON eines Essentia-Modells."""
    with open(models_dir / json_filename, encoding="utf-8") as f:
        return json.load(f)["classes"]


def pick_genres(probs, labels: list[str]) -> str:
    """Anzeigefertiger Genre-String aus den Discogs400-Wahrscheinlichkeiten.

    Top-1 immer (damit 'genres' nach einer Analyse nie leer ist), weitere
    Styles nur ueber der Schwelle, maximal GENRE_MAX. Der Discogs-Parent
    ('Electronic---Acid House') wird abgeschnitten -- in einer DJ-Library
    traegt er keine Information."""
    order = sorted(range(len(labels)), key=lambda i: -float(probs[i]))
    picked = []
    for rank, i in enumerate(order[:GENRE_MAX]):
        if rank > 0 and float(probs[i]) < GENRE_MIN_PROB:
            break
        name = labels[i].split(GENRE_LABEL_SEPARATOR)[-1]
        if name not in picked:
            picked.append(name)
    return TAG_SEPARATOR.join(picked)


def pick_vibes(probs, labels: list[str]) -> str:
    """Vibe-Tags aus den Jamendo-Mood/Theme-Aktivierungen: nur Whitelist,
    nur ueber der Schwelle, maximal VIBE_MAX, staerkste zuerst. Darf leer
    sein -- nicht jeder Track hat einen klaren Vibe."""
    scored = [
        (float(probs[i]), label)
        for i, label in enumerate(labels)
        if label in VIBE_WHITELIST and float(probs[i]) >= VIBE_MIN_PROB
    ]
    scored.sort(key=lambda pair: -pair[0])
    return TAG_SEPARATOR.join(label for _, label in scored[:VIBE_MAX])


class EssentiaAnalyzer:
    def __init__(self, models_dir: Path):
        from essentia.standard import (
            MonoLoader,
            TensorflowPredictEffnetDiscogs,
            TensorflowPredictMusiCNN,
            TensorflowPredict2D,
        )
        import essentia

        essentia.log.warningActive = False
        essentia.log.infoActive = False

        self.MonoLoader = MonoLoader
        m = lambda name: str(models_dir / name)  # noqa: E731

        self.effnet_embedding = TensorflowPredictEffnetDiscogs(graphFilename=m("discogs-effnet-bs64-1.pb"), output="PartitionedCall:1")
        self.track_embedding = TensorflowPredictEffnetDiscogs(graphFilename=m("discogs_track_embeddings-effnet-bs64-1.pb"), output="PartitionedCall:0")
        self.musicnn_embedding = TensorflowPredictMusiCNN(graphFilename=m("msd-musicnn-1.pb"), output="model/dense/BiasAdd")

        self.mood_aggressive = TensorflowPredict2D(graphFilename=m("mood_aggressive-discogs-effnet-1.pb"), output="model/Softmax")
        self.mood_happy = TensorflowPredict2D(graphFilename=m("mood_happy-discogs-effnet-1.pb"), output="model/Softmax")
        self.mood_sad = TensorflowPredict2D(graphFilename=m("mood_sad-discogs-effnet-1.pb"), output="model/Softmax")
        self.mood_relaxed = TensorflowPredict2D(graphFilename=m("mood_relaxed-discogs-effnet-1.pb"), output="model/Softmax")
        self.mood_party = TensorflowPredict2D(graphFilename=m("mood_party-discogs-effnet-1.pb"), output="model/Softmax")
        self.danceability = TensorflowPredict2D(graphFilename=m("danceability-discogs-effnet-1.pb"), output="model/Softmax")
        self.approachability = TensorflowPredict2D(graphFilename=m("approachability_regression-discogs-effnet-1.pb"), output="model/Identity")
        self.engagement = TensorflowPredict2D(graphFilename=m("engagement_regression-discogs-effnet-1.pb"), output="model/Identity")
        self.arousal_valence = TensorflowPredict2D(graphFilename=m("deam-msd-musicnn-2.pb"), output="model/Identity")

        # Genre/Vibe-Heads. genre_discogs400 ist anders exportiert als die
        # uebrigen Heads und braucht explizite Input-/Output-Knoten.
        self.genre = TensorflowPredict2D(
            graphFilename=m("genre_discogs400-discogs-effnet-1.pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        self.moodtheme = TensorflowPredict2D(graphFilename=m("mtg_jamendo_moodtheme-discogs-effnet-1.pb"), output="model/Sigmoid")
        self.genre_labels = load_model_labels(models_dir, GENRE_MODEL_JSON)
        self.vibe_labels = load_model_labels(models_dir, VIBE_MODEL_JSON)

    # Etappen der Voll-Analyse -- Reihenfolge und Anzahl muessen zu den
    # stage()-Aufrufen in analyze() passen. Ein echter Sub-Fortschritt ist
    # nicht moeglich (TensorFlow meldet innerhalb eines Forward-Pass nichts),
    # aber die Etappengrenzen sind ehrlich und ticken sichtbar durch.
    STAGES = [
        "Audio dekodieren (16 kHz)",
        "EffNet-Embedding (Basis für Mood/Genre)",
        "Similarity-Embedding (discogs_track)",
        "MusiCNN-Embedding (Basis für Arousal)",
        "Mood-/Arousal-Heads auswerten",
        "Genre- & Vibe-Tags ableiten",
    ]

    def analyze(self, filepath: Path, stage=None) -> dict:
        """stage(step, steps, label) wird an jeder Etappengrenze gerufen --
        fuer die Live-Statuszeile im AnalyzeModal."""
        def at(step):
            if stage:
                stage(step, len(self.STAGES), self.STAGES[step - 1])

        at(1)
        audio = self.MonoLoader(filename=str(filepath), sampleRate=16000, resampleQuality=4)()

        at(2)
        effnet_emb = self.effnet_embedding(audio)
        at(3)
        track_emb = self.track_embedding(audio)
        at(4)
        musicnn_emb = self.musicnn_embedding(audio)

        at(5)
        agg = self.mood_aggressive(effnet_emb).mean(axis=0)[0]
        happy = self.mood_happy(effnet_emb).mean(axis=0)[0]
        sad = self.mood_sad(effnet_emb).mean(axis=0)[0]
        relaxed = self.mood_relaxed(effnet_emb).mean(axis=0)[0]
        party = self.mood_party(effnet_emb).mean(axis=0)[0]
        dance = self.danceability(effnet_emb).mean(axis=0)[0]
        approach = self.approachability(effnet_emb).mean(axis=0)[0]
        engage = self.engagement(effnet_emb).mean(axis=0)[0]
        av = self.arousal_valence(musicnn_emb).mean(axis=0)
        at(6)
        genre_probs = self.genre(effnet_emb).mean(axis=0)
        vibe_probs = self.moodtheme(effnet_emb).mean(axis=0)

        valence, arousal = float(av[0]), float(av[1])

        return {
            "genres": pick_genres(genre_probs, self.genre_labels),
            "vibes": pick_vibes(vibe_probs, self.vibe_labels),
            "aggressive": round(float(agg), 4),
            "happy": round(float(happy), 4),
            "sad": round(float(sad), 4),
            "relaxed": round(float(relaxed), 4),
            "party": round(float(party), 4),
            "danceable": round(float(dance), 4),
            "approachability": round(float(approach), 4),
            "engagement": round(float(engage), 4),
            "arousal": round(arousal, 3),
            "valence": round(valence, 3),
            "embedding": encode_embedding(track_emb.mean(axis=0)),
            "effnet_embedding": encode_embedding(effnet_emb.mean(axis=0)),
        }


def _missing_parts(row: dict | None) -> set:
    """Was fehlt einer CSV-Zeile noch? 'embedding' = volle Analyse noetig
    (teuer, TensorFlow-Modelle), 'tags' = BPM evtl. nachtragbar (billig,
    kein Modell). Massgeblich ist allein, ob ein Embedding vorhanden ist --
    'error' ist reine Debug-Info und fliesst nicht in diese Entscheidung ein.

    Nur BPM triggert 'tags', nicht Key/Jahr: die werden nie selbst berechnet
    (siehe compute_bpm-Docstring), eine Zeile ohne solche Tags waere sonst
    fuer immer 'offen' und wuerde bei jedem Ctrl+A erneut angefasst, ohne je
    fertig zu werden. Sobald BPM einmal gesetzt ist, ist die Zeile endgueltig
    fertig -- Key/Jahr werden dabei nur "kostenlos" mitgenommen, falls der
    Tag inzwischen da ist, aber nicht eigens dafuer nachverfolgt.

    Das effnet_embedding ist der Vollstaendigkeits-Marker des Genre/Vibe-
    Schemas: Zeilen aus aelteren CSVs (vor genres/vibes) haben zwar ein
    Track-Embedding, aber kein effnet_embedding -- sie laufen einmal durch
    die volle Analyse. 'genres' selbst taugt nicht als Marker, weil 'vibes'
    legitim leer sein darf und ein leeres Pflichtfeld die Zeile fuer immer
    offen halten wuerde."""
    if row is None or not row.get("embedding") or not row.get("effnet_embedding"):
        return {"embedding"}
    if not row.get("bpm"):
        return {"tags"}
    return set()


def _done_event(kind: str, filepath: Path, row: dict, secs: float) -> dict:
    """Strukturiertes Ergebnis eines Datei-Durchlaufs -- Grundlage fuer die
    Ergebniszeile im Log (TUI rendert Chips, headless eine Textzeile).

    kind: "full" (Voll-Analyse), "tags" (nur BPM/Tags nachgetragen),
    "complete" (war schon fertig), "error"."""
    return {
        "event": "done",
        "kind": kind,
        "name": filepath.name,
        "genres": row.get("genres", ""),
        "vibes": row.get("vibes", ""),
        "year": row.get("year", ""),
        "bpm": row.get("bpm", ""),
        "key": row.get("key", ""),
        "arousal": row.get("arousal", ""),
        "aggressive": row.get("aggressive", ""),
        "danceable": row.get("danceable", ""),
        "secs": round(secs, 1),
        "error": row.get("error", ""),
    }


def _human_line(info: dict) -> str:
    """Ergebniszeile fuer den headless-Lauf (eine Zeile pro Datei)."""
    name = info["name"]
    if info["kind"] == "error":
        return f"✗ {name}: {info['error']}"
    if info["kind"] == "complete":
        return f"≡ {name}  (vollstaendig)"
    if info["kind"] == "tags":
        return f"~ {name}  BPM {info['bpm'] or '?'} nachgetragen  ({info['secs']}s)"
    tags = " · ".join(part for part in (
        (info["genres"] or "").replace(TAG_SEPARATOR, " | "),
        " ".join(v for v in (info["vibes"] or "").split(TAG_SEPARATOR) if v),
        info["year"],
    ) if part)
    return (f"✓ {name}  {tags}  ·  {info['bpm'] or '?'} BPM {info['key'] or '?'}"
            f"  ·  arous {info['arousal']} aggr {info['aggressive']}"
            f" dance {info['danceable']}  ({info['secs']}s)")


def run_analysis(music_dir, models_dir, log=print, progress=None, cancelled=None, status=None):
    """Analysiert alle Audiodateien in music_dir (Resume ueber die CSV).

    Jede Datei im Ordner wird durchlaufen und geloggt -- auch wenn sie
    bereits vollstaendig ist, damit der Log/Fortschrittsbalken nie
    kommentarlos bei "0 offen" stehen bleibt, sondern sichtbar durch die
    ganze Library rattert. Nachgeholt wird pro Datei nur, was in der CSV
    tatsaechlich fehlt: fehlt das Similarity-Embedding, laeuft die volle
    (teure) Analyse; ansonsten reicht ein billiger Tag-Re-Check (BPM/Key aus
    der Datei, z.B. von Rekordbox/Traktor gesetzt) plus Rhythm-Extractor-
    Fallback ohne TensorFlow-Modelle. Der Tag-Re-Check laeuft bei JEDER
    Datei, auch bereits vollstaendigen -- ein zwischenzeitlich in der Datei
    gesetzter oder geaenderter Tag gilt als hochwertiger als unsere eigene
    Schaetzung und ueberschreibt den CSV-Wert.

    log(msg): Textzeile fuer die Ausgabe.
    progress(done, total): Fortschritt ueber alle Dateien im Ordner.
    cancelled(): True -> zwischen zwei Dateien sauber abbrechen.
    status(dict): strukturierte Events ("track" = Datei beginnt,
        "stage" = Etappe innerhalb der Voll-Analyse, "done" = Ergebnis,
        siehe _done_event). Ohne status-Callback werden die Ergebnisse
        stattdessen als lesbare Zeilen ueber log() ausgegeben -- die TUI
        nutzt status (via --porcelain), der headless-Lauf log.

    Rueckgabe: (neu_analysiert, fehler) -- zaehlt nur Dateien, an denen
    tatsaechlich etwas berechnet wurde, nicht die bereits vollstaendigen.
    """
    music_dir = Path(music_dir)
    models_dir = Path(models_dir)
    csv_path = csv_path_for(music_dir)

    if not music_dir.exists():
        raise RuntimeError(f"Musikordner nicht gefunden: {music_dir}")

    all_files = find_audio_files(music_dir)
    log(f"{len(all_files)} Audiodateien in {music_dir}")

    csv_data = load_csv_data(csv_path)
    compact_csv(csv_path, csv_data, prune_missing=True)

    todo = [(f, _missing_parts(csv_data.get(str(f)))) for f in all_files]
    open_count = sum(1 for _, missing in todo if missing)
    log(f"{len(all_files) - open_count} bereits vollstaendig, {open_count} offen.")
    if progress:
        progress(0, len(all_files))

    analyzer = None
    if any("embedding" in missing for _, missing in todo):
        download_models(models_dir, log=log)
        log("Lade Essentia-Modelle (einmalig pro Lauf) ...")
        analyzer = EssentiaAnalyzer(models_dir)

    def emit_done(kind, filepath, row, secs):
        info = _done_event(kind, filepath, row, secs)
        if status:
            status(info)
        else:
            log(_human_line(info))

    def stage_cb(step, steps, label):
        if status:
            status({"event": "stage", "step": step, "steps": steps, "label": label})

    scanned = 0
    analyzed = 0
    errors = 0
    csv_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not csv_exists:
            writer.writeheader()

        for filepath, missing in todo:
            if cancelled and cancelled():
                log("Abgebrochen -- bereits analysierte Tracks sind gespeichert.")
                break

            t0 = time.time()
            existing = csv_data.get(str(filepath)) or {}
            row = {"filepath": str(filepath), **{k: existing.get(k, "") for k in CSV_FIELDNAMES[1:]}}

            if "embedding" in missing:
                if status:
                    status({"event": "track", "index": scanned + 1,
                            "total": len(all_files), "name": filepath.name})
                try:
                    row.update(read_existing_tags(filepath))
                    row.update(analyzer.analyze(filepath, stage=stage_cb))
                    row["error"] = ""
                except Exception as e:
                    row = {k: "" for k in CSV_FIELDNAMES}
                    row["filepath"] = str(filepath)
                    row["error"] = str(e)
                    errors += 1
                    emit_done("error", filepath, row, time.time() - t0)
                    writer.writerow(row)
                    f.flush()
                    scanned += 1
                    analyzed += 1
                    if progress:
                        progress(scanned, len(all_files))
                    continue

                emit_done("full", filepath, row, time.time() - t0)
                writer.writerow(row)
                f.flush()
                scanned += 1
                analyzed += 1
                if progress:
                    progress(scanned, len(all_files))
                continue

            # Kein Embedding noetig -- trotzdem bei JEDER Datei pruefen, ob
            # DJ-Software (Rekordbox/Traktor) inzwischen einen BPM/Key-Tag
            # gesetzt oder geaendert hat. Deren Analyse gilt als hochwertiger
            # als unsere eigene Schaetzung und gewinnt daher immer, wenn ein
            # Tag vorhanden ist -- nicht nur als Luecken-Fuellung.
            fresh_tags = read_existing_tags(filepath)
            changed = False
            if fresh_tags.get("bpm") and fresh_tags["bpm"] != row.get("bpm"):
                row["bpm"] = fresh_tags["bpm"]
                changed = True
            if fresh_tags.get("key") and fresh_tags["key"] != row.get("key"):
                row["key"] = fresh_tags["key"]
                changed = True
            if fresh_tags.get("year") and fresh_tags["year"] != row.get("year"):
                row["year"] = fresh_tags["year"]
                changed = True

            if not row.get("bpm"):
                try:
                    row["bpm"] = compute_bpm(filepath)
                    changed = True
                except Exception as e:
                    log(f"BPM nicht ermittelbar bei {filepath.name}: {e}")

            scanned += 1
            if changed:
                emit_done("tags", filepath, row, time.time() - t0)
                writer.writerow(row)
                f.flush()
                analyzed += 1
            else:
                emit_done("complete", filepath, row, time.time() - t0)
            if progress:
                progress(scanned, len(all_files))

    return analyzed, errors
