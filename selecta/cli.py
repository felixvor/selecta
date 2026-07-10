"""Einstieg: `selecta [MUSIC_DIR]` startet die TUI,
`selecta analyze --music-dir ...` laeuft headless (z.B. Uebernacht-Analyse)."""

import argparse
import sys
from pathlib import Path

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _headless_analyze(argv):
    parser = argparse.ArgumentParser(prog="selecta analyze", description="Musikordner headless analysieren")
    parser.add_argument("--music-dir", required=True, help="Ordner mit Musikdateien (rekursiv)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Ordner fuer die .pb-Modelldateien")
    # Maschinenlesbare Fortschrittszeilen ("::progress i n") -- wird vom
    # Analyse-Modal der TUI genutzt, das diesen Befehl als Subprozess treibt.
    parser.add_argument("--porcelain", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    progress = None
    if args.porcelain:
        def progress(done, total):
            print(f"::progress {done} {total}", flush=True)

    def log(msg):
        print(msg, flush=True)

    from .analysis import filtered_stderr, run_analysis

    with filtered_stderr(["No network created"]):
        try:
            done, errors = run_analysis(args.music_dir, args.models_dir, log=log, progress=progress)
        except RuntimeError as e:
            print(f"FEHLER: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"Fertig: {done} analysiert, {errors} Fehler.", flush=True)


def _run_tui(argv):
    parser = argparse.ArgumentParser(prog="selecta", description="SELECTA -- Terminal-Helper zum Auflegen")
    parser.add_argument("music_dir", nargs="?", default=None, help="Musik-Ordner (sonst Abfrage beim Start)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="Ordner fuer die .pb-Modelldateien")
    args = parser.parse_args(argv)

    if args.music_dir and not Path(args.music_dir).is_dir():
        print(f"Musikordner nicht gefunden: {args.music_dir}", file=sys.stderr)
        sys.exit(1)

    from .app import SelectaApp

    SelectaApp(music_dir=args.music_dir, models_dir=Path(args.models_dir)).run()


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "analyze":
        _headless_analyze(argv[1:])
    else:
        _run_tui(argv)


if __name__ == "__main__":
    main()
