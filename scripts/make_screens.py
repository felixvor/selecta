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

from demo_library import create_demo_library  # noqa: E402  (scripts/-Nachbar)

import selecta.app as app_module  # noqa: E402
from selecta.app import SelectaApp  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def save(app: SelectaApp, name: str) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    svg = app.export_screenshot(title="SELECTA")
    (DOCS_DIR / name).write_text(svg, encoding="utf-8")
    print(f"✓ docs/{name}")


async def main() -> None:
    demo_dir = create_demo_library(Path(tempfile.gettempdir()) / "selecta-demo" / "demo-crate")

    # Launcher liest die gemerkte Library-Liste -- auf eine Wegwerf-Datei
    # umbiegen, damit die echte ~/.local/share/selecta unangetastet bleibt.
    state_dir = Path(tempfile.mkdtemp(prefix="selecta-screens-"))
    app_module.LIBRARIES_FILE = state_dir / "libraries.json"
    app_module.LAST_DIR_FILE = state_dir / "last_dir"
    app_module.LIBRARIES_FILE.write_text(
        json.dumps({"libraries": [{"path": str(demo_dir), "active": True}]}),
        encoding="utf-8",
    )

    # 1) Launcher (eigene App-Instanz: Basis-Screen ist der LibraryScreen)
    app = SelectaApp()
    async with app.run_test(size=(100, 22)) as pilot:
        await asyncio.sleep(1.0)  # dir_status-Thread fuellt die TRACKS-Spalte
        await pilot.pause()
        save(app, "launcher.svg")

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
