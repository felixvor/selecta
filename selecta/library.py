"""CSV-Persistenz und Library-Zugriff: Laden, Kompaktieren, Embedding-Codierung."""

import base64
import csv
import os
from pathlib import Path

import numpy as np

from .config import AUDIO_EXTENSIONS, CSV_FIELDNAMES, CSV_FILENAME, FLOAT_FIELDS


def encode_embedding(vector) -> str:
    arr = np.asarray(vector, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_embedding(text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype=np.float32)


def csv_path_for(music_dir) -> Path:
    return Path(music_dir) / CSV_FILENAME


def load_csv_data(csv_path: Path) -> dict:
    """Liest bereits verarbeitete Metadaten aus einer vorhandenen CSV."""
    processed = {}
    if not csv_path.exists():
        return processed
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row.pop("filepath")
            for key in FLOAT_FIELDS:
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            processed[filepath] = row
    return processed


def compact_csv(csv_path: Path, csv_data: dict, prune_missing: bool = False):
    """Schreibt die CSV dedupliziert neu (letzter Stand pro Datei).

    Wird vor jedem Analyse-Lauf ausgefuehrt, damit Backfill-Durchlaeufe nicht
    zu doppelten Zeilen pro Datei fuehren. Mit prune_missing=True fliegen
    Zeilen raus, deren Datei nicht mehr auf der Platte existiert.
    """
    if not csv_data:
        return
    if prune_missing:
        for filepath in [fp for fp in csv_data if not Path(fp).exists()]:
            del csv_data[filepath]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for filepath, values in csv_data.items():
            row = {"filepath": filepath}
            for key in CSV_FIELDNAMES[1:]:
                row[key] = values.get(key, "")
            writer.writerow(row)


def find_audio_files(music_dir: Path):
    files = []
    for root, _dirs, filenames in os.walk(music_dir):
        for name in filenames:
            if Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                files.append(Path(root) / name)
    return sorted(files)


def missing_parts(row: dict | None) -> set:
    """Was fehlt einer CSV-Zeile noch? 'embedding' = volle Analyse noetig
    (teuer, TensorFlow-Modelle), 'tags' = BPM/Key evtl. nachtragbar (billig,
    kein TF-Modell). Massgeblich ist allein, ob ein Embedding vorhanden ist --
    'error' ist reine Debug-Info und fliesst nicht in diese Entscheidung ein.

    BPM und Key triggern 'tags': beide sind selbst berechenbar (compute_bpm/
    compute_key), die Zeile kann also immer fertig werden. Das Jahr dagegen
    nicht -- es kommt nur aus dem ID3-Tag, eine Zeile ohne Jahr waere sonst
    fuer immer 'offen'; es wird beim Tag-Re-Check nur "kostenlos"
    mitgenommen, falls der Tag inzwischen da ist.

    Das effnet_embedding ist der Vollstaendigkeits-Marker des Genre/Vibe-
    Schemas: Zeilen aus aelteren CSVs (vor genres/vibes) haben zwar ein
    Track-Embedding, aber kein effnet_embedding -- sie laufen einmal durch
    die volle Analyse. 'genres' selbst taugt nicht als Marker, weil 'vibes'
    legitim leer sein darf und ein leeres Pflichtfeld die Zeile fuer immer
    offen halten wuerde.

    Lebt hier (nicht in analysis.py), damit dir_status()/Library.status()
    exakt dasselbe Vollstaendigkeits-Kriterium verwenden wie der Analyse-
    Lauf -- vorher zaehlten die Anzeigen nur 'embedding' und meldeten
    "0 offen", obwohl die Analyse noch Zeilen anfassen wuerde."""
    if row is None or not row.get("embedding") or not row.get("effnet_embedding"):
        return {"embedding"}
    if not row.get("bpm") or not row.get("key"):
        return {"tags"}
    return set()


def dir_status(music_dir) -> tuple[int, int]:
    """(vollstaendige, gesamt) Audiodateien eines Ordners, nur aus Dateisystem
    und CSV -- ohne Embeddings zu decodieren (billig genug fuer die
    Statusspalte im Library-Screen, die alle Ordner scannt). 'Vollstaendig'
    heisst: missing_parts() haette nichts mehr zu tun -- dasselbe Kriterium
    wie der Analyse-Lauf."""
    csv_data = load_csv_data(csv_path_for(music_dir))
    files = find_audio_files(Path(music_dir))
    analyzed = sum(1 for f in files if not missing_parts(csv_data.get(str(f))))
    return analyzed, len(files)


def read_existing_tags(filepath: Path) -> dict:
    """Liest vorhandene Artist/Titel/BPM/Key-Tags (z.B. von Rekordbox gesetzt)
    in die CSV, damit die Suche spaeter keine Audiodateien oeffnen muss."""
    info = {"artist": "", "title": filepath.stem, "bpm": "", "key": "", "year": ""}
    if filepath.suffix.lower() != ".mp3":
        return info
    try:
        from mutagen.id3 import ID3
        tags = ID3(str(filepath))
        if "TPE1" in tags:
            info["artist"] = str(tags["TPE1"].text[0])
        if "TIT2" in tags:
            info["title"] = str(tags["TIT2"].text[0])
        if "TBPM" in tags:
            info["bpm"] = str(tags["TBPM"].text[0])
        if "TKEY" in tags:
            info["key"] = str(tags["TKEY"].text[0])
        # TDRC (ID3v2.4) bzw. von mutagen dorthin gemappte v2.3-TYER;
        # nur das Jahr, volle Timestamps ("1994-06-01") abschneiden.
        if "TDRC" in tags and tags["TDRC"].text:
            year = str(tags["TDRC"].text[0])[:4]
            if year.isdigit():
                info["year"] = year
    except Exception:
        pass
    return info


def track_label(track: dict) -> str:
    label = f"{track.get('artist', '')} - {track.get('title', '')}".strip(" -")
    return label or Path(track["filepath"]).stem


def prefix_aware_score(needle: str, label: str) -> float:
    """Fuzzy-Score fuer Suche-beim-Tippen: das letzte Tipp-Wort zaehlt als
    Praefix ('kolter st' matcht 'Step han' voll), alle anderen fuzzy.
    Generische Scorer (WRatio etc.) koennen das nicht und liefern bei
    gleichem Artist Gleichstaende."""
    from rapidfuzz import fuzz

    n_tokens = needle.lower().split()
    if not n_tokens:
        return 0.0
    l_tokens = label.lower().replace("-", " ").replace("(", " ").replace(")", " ").split()
    if not l_tokens:
        return 0.0
    total = 0.0
    for i, nt in enumerate(n_tokens):
        is_last = i == len(n_tokens) - 1
        best = 0.0
        for lt in l_tokens:
            if is_last and lt.startswith(nt):
                best = 100.0
                break
            best = max(best, fuzz.ratio(nt, lt))
        total += best
    return total / len(n_tokens)


def fuzzy_search(needle: str, labels: list[str], limit: int = 50, cutoff: float = 50.0) -> list[int]:
    """Indizes der besten Label-Treffer, absteigend nach Score."""
    scored = [(prefix_aware_score(needle, label), i) for i, label in enumerate(labels)]
    scored = [(s, i) for s, i in scored if s >= cutoff]
    scored.sort(key=lambda pair: (-pair[0], labels[pair[1]].lower()))
    return [i for _, i in scored[:limit]]


class Library:
    """Analysierte Tracks eines oder mehrerer Musik-Ordner, suchfertig im
    Speicher. Jeder Ordner behaelt seine eigene library_analysis.csv; hier
    werden sie nur zum Suchen zusammengefuehrt.

    tracks: Zeilen mit vorhandenem Embedding, jeweils um '_embedding'
    (float32-Vektor) ergaenzt. matrix: zeilennormalisierte (N x dim)-Matrix
    fuer vektorisierte Cosine-Similarity.
    """

    def __init__(self, music_dirs):
        if isinstance(music_dirs, (str, Path)):
            music_dirs = [music_dirs]
        self.music_dirs = [Path(d) for d in music_dirs]
        self.tracks: list[dict] = []
        self.labels: list[str] = []
        self.matrix: np.ndarray | None = None
        self.reload()

    def reload(self):
        self.tracks = []
        seen: set[str] = set()
        for music_dir in self.music_dirs:
            csv_data = load_csv_data(csv_path_for(music_dir))
            for filepath, row in csv_data.items():
                # Dedupe ueber den absoluten Pfad: verschachtelte Libraries
                # (z.B. /Musik und /Musik/House) wuerden Tracks sonst doppelt
                # in die Suche bringen.
                if filepath in seen or not row.get("embedding"):
                    continue
                seen.add(filepath)
                track = dict(row)
                track["filepath"] = filepath
                track["_embedding"] = decode_embedding(row["embedding"])
                self.tracks.append(track)
        self.labels = [track_label(t) for t in self.tracks]
        if self.tracks:
            m = np.stack([t["_embedding"] for t in self.tracks]).astype(np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            self.matrix = m / np.maximum(norms, 1e-9)
        else:
            self.matrix = None

    def status(self) -> tuple[int, int]:
        """(vollstaendige, gesamt) Audiodateien ueber alle Ordner.
        'Vollstaendig' heisst: Datei liegt in einem Ordner UND ihre Zeile
        laesst nach missing_parts() nichts mehr offen -- dasselbe Kriterium
        wie dir_status() und der Analyse-Lauf."""
        files: set[str] = set()
        for music_dir in self.music_dirs:
            files.update(str(f) for f in find_audio_files(music_dir))
        complete_paths = {
            t["filepath"] for t in self.tracks if not missing_parts(t)
        }
        analyzed = sum(1 for f in files if f in complete_paths)
        return analyzed, len(files)
