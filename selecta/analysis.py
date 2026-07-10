"""Essentia-Analyse: Modell-Download, Feature-Extraktion, Analyse-Loop.

Der Analyse-Loop ist callback-basiert (log/progress/cancel), damit er sowohl
headless (CLI, print) als auch im Textual-Worker-Thread (UI-Updates) laeuft.
"""

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
    MAX_DOWNLOAD_ATTEMPTS,
    MIN_MODEL_BYTES,
    MODEL_FILES,
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
    if actual_size < MIN_MODEL_BYTES or (total > 0 and actual_size != total):
        tmp_dest.unlink(missing_ok=True)
        raise IOError(f"Download unvollstaendig: erwartet {total} Bytes, erhalten {actual_size} Bytes")

    tmp_dest.replace(dest)


def download_models(models_dir: Path, log=print):
    """Laedt fehlende Modell-Dateien. Wirft RuntimeError statt sys.exit, damit
    die TUI den Fehler anzeigen kann (essentia.upf.edu ist gelegentlich down)."""
    models_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        name for name in MODEL_FILES
        if not (models_dir / name).exists() or (models_dir / name).stat().st_size < MIN_MODEL_BYTES
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

    def analyze(self, filepath: Path) -> dict:
        audio = self.MonoLoader(filename=str(filepath), sampleRate=16000, resampleQuality=4)()

        effnet_emb = self.effnet_embedding(audio)
        track_emb = self.track_embedding(audio)
        musicnn_emb = self.musicnn_embedding(audio)

        agg = self.mood_aggressive(effnet_emb).mean(axis=0)[0]
        happy = self.mood_happy(effnet_emb).mean(axis=0)[0]
        sad = self.mood_sad(effnet_emb).mean(axis=0)[0]
        relaxed = self.mood_relaxed(effnet_emb).mean(axis=0)[0]
        party = self.mood_party(effnet_emb).mean(axis=0)[0]
        dance = self.danceability(effnet_emb).mean(axis=0)[0]
        approach = self.approachability(effnet_emb).mean(axis=0)[0]
        engage = self.engagement(effnet_emb).mean(axis=0)[0]
        av = self.arousal_valence(musicnn_emb).mean(axis=0)

        valence, arousal = float(av[0]), float(av[1])

        return {
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
        }


def run_analysis(music_dir, models_dir, log=print, progress=None, cancelled=None):
    """Analysiert alle Audiodateien in music_dir (Resume ueber die CSV).

    log(msg): Textzeile fuer die Ausgabe.
    progress(done, total): Fortschritt ueber alle Dateien.
    cancelled(): True -> zwischen zwei Dateien sauber abbrechen.

    Rueckgabe: (neu_analysiert, fehler).
    """
    music_dir = Path(music_dir)
    models_dir = Path(models_dir)
    csv_path = csv_path_for(music_dir)

    if not music_dir.exists():
        raise RuntimeError(f"Musikordner nicht gefunden: {music_dir}")

    download_models(models_dir, log=log)

    all_files = find_audio_files(music_dir)
    log(f"{len(all_files)} Audiodateien in {music_dir}")

    csv_data = load_csv_data(csv_path)
    compact_csv(csv_path, csv_data, prune_missing=True)

    def needs_analysis(filepath_str):
        values = csv_data.get(filepath_str)
        if values is None:
            return True
        # Backfill: alte Zeilen ohne Similarity-Embedding neu analysieren.
        return values.get("status") == "ok" and not values.get("embedding")

    todo = [f for f in all_files if needs_analysis(str(f))]
    log(f"{len(all_files) - len(todo)} bereits analysiert, {len(todo)} offen.")
    if progress:
        progress(0, len(todo))
    if not todo:
        return 0, 0

    log("Lade Essentia-Modelle (einmalig pro Lauf) ...")
    analyzer = EssentiaAnalyzer(models_dir)

    done = 0
    errors = 0
    csv_exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if not csv_exists:
            writer.writeheader()

        for filepath in todo:
            if cancelled and cancelled():
                log("Abgebrochen -- bereits analysierte Tracks sind gespeichert.")
                break

            t0 = time.time()
            row = {"filepath": str(filepath)}
            row.update(read_existing_tags(filepath))
            try:
                values = analyzer.analyze(filepath)
                row.update(values)
                row["status"] = "ok"
                dt = time.time() - t0
                log(f"{filepath.name}  ({dt:.1f}s)")
            except Exception as e:
                row = {k: "" for k in CSV_FIELDNAMES}
                row["filepath"] = str(filepath)
                row["status"] = f"error: {e}"
                errors += 1
                log(f"FEHLER bei {filepath.name}: {e}")

            writer.writerow(row)
            f.flush()
            done += 1
            if progress:
                progress(done, len(todo))

    return done, errors
