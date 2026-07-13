"""Einstieg: `selecta [MUSIC_DIR]` startet die TUI,
`selecta analyze --music-dir ...` laeuft headless (z.B. Uebernacht-Analyse)."""

import argparse
import sys
from pathlib import Path

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _headless_analyze(argv):
    parser = argparse.ArgumentParser(prog="selecta analyze", description="Analyze a music folder headless")
    parser.add_argument("--music-dir", required=True, help="folder with music files (recursive)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="folder for the .pb model files")
    # Maschinenlesbare Fortschrittszeilen ("::progress i n") -- wird vom
    # Analyse-Modal der TUI genutzt, das diesen Befehl als Subprozess treibt.
    parser.add_argument("--porcelain", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    progress = None
    status = None
    if args.porcelain:
        import json

        def progress(done, total):
            print(f"::progress {done} {total}", flush=True)

        # Strukturierte Events (track/stage/done) fuer die Live-Statuszeile
        # und die Chip-Ergebniszeilen im AnalyzeModal.
        def status(event):
            print("::status " + json.dumps(event, ensure_ascii=False), flush=True)

    def log(msg):
        print(msg, flush=True)

    from .analysis import filtered_stderr, run_analysis

    with filtered_stderr(["No network created"]):
        try:
            done, errors = run_analysis(
                args.music_dir, args.models_dir, log=log, progress=progress, status=status
            )
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    print(f"Done: {done} analyzed, {errors} errors.", flush=True)


def _map_command(argv):
    parser = argparse.ArgumentParser(prog="selecta map", description="Render a 2D map of the library's track embeddings")
    parser.add_argument("--music-dir", action="append", required=True,
                        help="folder with an analyzed library_analysis.csv (repeatable for multiple libraries)")
    parser.add_argument("--out", default=None, help="output HTML path (default: first --music-dir/selecta_map.html)")
    parser.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = parser.parse_args(argv)

    from .map import open_in_browser, write_map

    def log(msg):
        print(msg, flush=True)

    try:
        path = write_map(args.music_dir, out_path=args.out, log=log)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(str(path), flush=True)
    if not args.no_open:
        open_in_browser(path)


def _run_tui(argv):
    parser = argparse.ArgumentParser(prog="selecta", description="SELECTA -- terminal companion for DJing")
    parser.add_argument("music_dir", nargs="?", default=None, help="music folder (otherwise the launcher asks)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR), help="folder for the .pb model files")
    args = parser.parse_args(argv)

    if args.music_dir and not Path(args.music_dir).is_dir():
        print(f"Music folder not found: {args.music_dir}", file=sys.stderr)
        sys.exit(1)

    from .app import SelectaApp

    SelectaApp(music_dir=args.music_dir, models_dir=Path(args.models_dir)).run()


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "analyze":
        _headless_analyze(argv[1:])
    elif argv and argv[0] == "map":
        _map_command(argv[1:])
    else:
        _run_tui(argv)


if __name__ == "__main__":
    main()
