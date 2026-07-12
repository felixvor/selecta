"""README-Screenshots als SVG erzeugen -- reproduzierbar statt Pixel-Blob.

Faehrt die echte TUI headless ueber Textuals Pilot (wie die Tests in
tests/test_app.py) gegen die Demo-Library aus demo_library.py und
exportiert drei SVGs nach docs/:

    launcher.svg    Library-Launcher mit Logo
    ranking.svg     Aehnlichkeits-Ranking zu einer Query
    transition.svg  Transition-Modus mit Bar und Brueckenkandidaten

Nach UI-Aenderungen einfach neu ausfuehren (in WSL, selecta-venv aktiv):
    python scripts/make_screens.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_library import create_demo_crates  # noqa: E402  (scripts/-Nachbar)

import selecta.app as app_module  # noqa: E402
from selecta.app import SelectaApp  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Fuers Launcher-Bild: plausible DJ-Libraries statt der 15-Track-Technik-Demo.
# Die Pfade existieren nicht -- die Status-Zellen werden nach dem (ins Leere
# laufenden) Scan direkt im Screen-Cache ueberschrieben. Es geht um den
# ersten Eindruck im README: so saehe das Tool ueber einer echten,
# gewachsenen Sammlung aus (Hunderte Tracks, einer davon mitten in der
# Analyse, einer inaktiv).
FAKE_LIBRARIES = [
    ("/mnt/d/Music/House", True, (687, 687)),
    ("/mnt/d/Music/Techno & Peaktime", True, (418, 418)),
    ("/mnt/d/Music/Warmup & Downtempo", False, (183, 201)),
    ("/mnt/d/Music/Disco & Edits", False, (96, 96)),
    ("/mnt/d/Music/Drum & Bass", False, (241, 241)),
    ("/mnt/d/Music/Ambient & Listening", False, (154, 154)),
    ("/mnt/d/Music/Crates/Festival 2026", False, (0, 42)),
]


def save(app: SelectaApp, name: str) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    svg = app.export_screenshot(title="SELECTA")
    (DOCS_DIR / name).write_text(svg, encoding="utf-8")
    print(f"✓ docs/{name}")


async def main() -> None:
    # Zwei Crates, damit der Launcher-Screenshot nicht wie ein
    # Ein-Ordner-Tool aussieht.
    demo_dir, warmup_dir = create_demo_crates(Path(tempfile.gettempdir()) / "selecta-demo")

    # Launcher liest die gemerkte Library-Liste -- auf eine Wegwerf-Datei
    # umbiegen, damit die echte ~/.local/share/selecta unangetastet bleibt.
    state_dir = Path(tempfile.mkdtemp(prefix="selecta-screens-"))
    app_module.LIBRARIES_FILE = state_dir / "libraries.json"
    app_module.LAST_DIR_FILE = state_dir / "last_dir"
    app_module.LIBRARIES_FILE.write_text(
        json.dumps({"libraries": [
            {"path": path, "active": active} for path, active, _ in FAKE_LIBRARIES
        ]}),
        encoding="utf-8",
    )

    # 1) Launcher (eigene App-Instanz: Basis-Screen ist der LibraryScreen)
    app = SelectaApp()
    async with app.run_test(size=(100, 26)) as pilot:
        await asyncio.sleep(1.0)  # Scan-Thread laeuft (ins Leere) durch
        # Fake-Pfade existieren nicht -> Status-Zellen direkt setzen.
        screen = app.screen
        screen._statuses.update({path: counts for path, _, counts in FAKE_LIBRARIES})
        screen._render_entries()
        await pilot.pause()
        save(app, "launcher.svg")

    # Ranking/Transition laufen weiter gegen die echte Demo-Crate.
    app_module.LIBRARIES_FILE.write_text(
        json.dumps({"libraries": [
            {"path": str(demo_dir), "active": True},
            {"path": str(warmup_dir), "active": True},
        ]}),
        encoding="utf-8",
    )

    # 2) + 3) Suche: Query waehlen, dann Transition pinnen
    app = SelectaApp(music_dir=demo_dir)
    async with app.run_test(size=(110, 34)) as pilot:
        await asyncio.sleep(1.0)  # Status-Badge (n/n analyzed) fertig scannen
        await pilot.press(*"velvet", "enter")   # Query: Ferra Noire - Velvet Circuit
        await pilot.pause()
        save(app, "ranking.svg")

        await pilot.press("ctrl+t")
        await pilot.press(*"pressure", "enter")  # Ziel B: Cold Assembly - Pressure Test
        await pilot.pause()
        save(app, "transition.svg")


if __name__ == "__main__":
    asyncio.run(main())
