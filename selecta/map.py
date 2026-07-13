"""Library-Map: 2D-Landschaft der Track-Embeddings als selbstaendige HTML-
Datei (kein CDN, kein Framework -- Canvas-JS inline).

Projektion: nur das rohe (bereits L2-normierte) Track-Embedding, KEINE
Metadaten in die Distanz gemischt -- BPM/Genre/Key sind Anzeige-Kanaele
(Farbe/Groesse/Tooltip), nicht Teil der Geometrie. Sonst laesst sich der
Karte nicht mehr trauen ("liegt das zusammen, weil es aehnlich klingt,
oder weil beide 128 BPM haben?").

Reihenfolge der Projektionsverfahren: pacmap > umap > PCA-Fallback (reines
numpy, keine Zusatz-Dependency). pacmap/umap sind absichtlich NICHT
Kern-Dependencies (siehe pyproject.toml Extra 'map') -- der Import passiert
lazy hier in der Funktion, damit Kern-Installation und Tests ohne sie
sauber bleiben.
"""

import html as html_escape_module
import json
import shutil
import subprocess
import zlib
from pathlib import Path

import numpy as np

from .config import GENRE_CHIP_COLORS, TAG_SEPARATOR
from .library import Library, track_label

# Feste Hex-Werte fuer die Rich-Farbnamen aus GENRE_CHIP_COLORS -- bewusst
# ein festes Mapping statt Rich zur Laufzeit zu befragen (unnoetige
# Kopplung an eine TUI-Bibliothek in einem Browser-Artefakt).
RICH_TO_HEX = {
    "orchid": "#DA70D6",
    "dark_orange": "#FF8C00",
    "cornflower_blue": "#6495ED",
    "dark_sea_green4": "#698B69",
    "medium_purple": "#9370DB",
    "hot_pink3": "#CD6090",
    "steel_blue1": "#63B8FF",
    "gold3": "#CDAD00",
    "dark_cyan": "#008B8B",
    "grey66": "#A8A8A8",
}
OTHER_COLOR = "#555555"


def _genre_color_hex(name: str) -> str:
    """Stabiler Hash Genre-Name -> Hex-Farbe -- dieselbe Hash-Logik wie
    app.genre_chip_color (crc32, damit derselbe Style-Name in TUI und Map
    dieselbe Farbe traegt), aber hier dupliziert statt importiert: map.py
    darf app.py NICHT importieren (Kreisimport, weil app.py fuer Ctrl+G
    umgekehrt aus map.py importiert)."""
    rich_name = GENRE_CHIP_COLORS[zlib.crc32(name.encode("utf-8")) % len(GENRE_CHIP_COLORS)]
    return RICH_TO_HEX.get(rich_name, OTHER_COLOR)


def project_2d(matrix: np.ndarray, log=print) -> tuple[np.ndarray, str]:
    """(N x dim) -> ((N x 2)-Koordinaten min-max-normiert auf [0,1]^2, Name
    des verwendeten Verfahrens).

    pacmap (beste globale Struktur -- "Inseln" bleiben Inseln) > umap >
    PCA-Fallback ueber SVD (keine Zusatz-Dependency, laeuft immer)."""
    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64), "PCA"
    if n == 1:
        return np.zeros((1, 2), dtype=np.float64), "PCA"

    coords = None
    label = "PCA"
    try:
        import pacmap

        reducer = pacmap.PaCMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(np.asarray(matrix, dtype=np.float32), init="pca")
        label = "PaCMAP"
    except ImportError:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(matrix)
            label = "UMAP"
        except ImportError:
            log("pacmap not installed — falling back to PCA (pip install selecta[map])")
        except Exception as e:
            log(f"umap failed ({e}) — falling back to PCA")
    except Exception as e:
        log(f"pacmap failed ({e}) — falling back to PCA")

    if coords is None:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        u, s, _vt = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :2] * s[:2]

    coords = np.asarray(coords, dtype=np.float64)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    normalized = (coords - lo) / span
    return normalized, label


def _display_key(track: dict) -> str:
    key = (track.get("key") or "").strip()
    if not key:
        return ""
    return f"~{key}" if track.get("key_estimated") else key


def build_map_html(tracks: list[dict], coords: np.ndarray, projection: str = "PCA",
                   analyzed: int | None = None, total: int | None = None) -> str:
    """Fuellt HTML_TEMPLATE mit einem JSON-Array (ein Objekt pro Track) und
    ein paar Kopfzeilen-Angaben. Reines str.replace auf Marker-Strings
    (kein f-string) -- die JS-Template-Literals im Canvas-Code bleiben so
    unangetastet."""
    points = []
    for track, (x, y) in zip(tracks, coords):
        genres = [g for g in (track.get("genres") or "").split(TAG_SEPARATOR) if g]
        vibes = [v for v in (track.get("vibes") or "").split(TAG_SEPARATOR) if v]
        top_genre = genres[0] if genres else ""
        points.append({
            "x": float(x), "y": float(y),
            "label": track_label(track),
            "bpm": track.get("bpm") or "",
            "key": _display_key(track),
            "genres": genres,
            "vibes": vibes,
            "year": track.get("year") or "",
            "arousal": track.get("arousal") or "",
            "aggressive": track.get("aggressive") or "",
            "danceable": track.get("danceable") or "",
            "color": _genre_color_hex(top_genre) if top_genre else OTHER_COLOR,
            "genre": top_genre or "other",
        })

    # Legende: die haeufigsten Top-Genres (max 12), Rest faellt unter
    # "other" -- verhindert eine 60-Zeilen-Legende bei grossen Libraries.
    counts: dict[str, int] = {}
    for p in points:
        counts[p["genre"]] = counts.get(p["genre"], 0) + 1
    top_genres = sorted(counts, key=lambda g: -counts[g])[:12]
    legend = [{"genre": g, "color": _genre_color_hex(g) if g != "other" else OTHER_COLOR}
              for g in top_genres]

    title = f"{len(points)} tracks"
    if analyzed is not None and total is not None and analyzed < total:
        title = f"{analyzed} of {total} analyzed tracks mapped"

    page = HTML_TEMPLATE
    # Header-Marker sitzen in reinem HTML-Text (nicht in einem JSON-Block)
    # -- hier escapen wir fuer HTML, nicht fuer JSON, sonst stehen die
    # Anfuehrungszeichen von json.dumps() sichtbar in der Kopfzeile.
    page = page.replace("__SELECTA_TITLE__", html_escape_module.escape(title))
    page = page.replace("__SELECTA_PROJECTION__", html_escape_module.escape(projection))
    # </script> im JSON entschaerfen (Titel koennen Sonderzeichen tragen) --
    # der String bleibt sonst identisch, nur diese eine Sequenz wird
    # getrennt, damit der Browser sie nicht als Tag-Ende liest.
    tracks_json = json.dumps(points, ensure_ascii=False).replace("</script>", "<\\/script>")
    page = page.replace("__SELECTA_TRACKS_JSON__", tracks_json)
    legend_json = json.dumps(legend, ensure_ascii=False).replace("</script>", "<\\/script>")
    page = page.replace("__SELECTA_LEGEND_JSON__", legend_json)
    return page


def write_map(music_dirs, out_path=None, log=print) -> Path:
    """Library laden, projizieren, HTML schreiben. Default-Zielpfad: erster
    music_dir / 'selecta_map.html' -- liegt damit (WSL-Faelle) auf
    /mnt/..., sonst kommt der Windows-Browser nicht an die Datei."""
    if isinstance(music_dirs, (str, Path)):
        music_dirs = [music_dirs]
    music_dirs = [Path(d) for d in music_dirs]
    library = Library(music_dirs)
    if not library.tracks:
        raise RuntimeError("No analyzed tracks found — run 'selecta analyze' first.")

    out_path = Path(out_path) if out_path is not None else music_dirs[0] / "selecta_map.html"
    coords, projection = project_2d(library.matrix, log=log)
    analyzed, total = library.status()
    html = build_map_html(library.tracks, coords, projection=projection, analyzed=analyzed, total=total)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def open_in_browser(path: Path) -> None:
    """Oeffnen, robust fuer WSL: erst wslview (falls installiert), dann
    explorer.exe mit wslpath -w (nur fuer /mnt-Pfade moeglich), dann
    xdg-open, zuletzt Python-Standard. Fehler werden nur geloggt, nie
    weitergeworfen -- die HTML-Datei existiert so oder so."""
    path = Path(path).resolve()
    if shutil.which("wslview"):
        try:
            subprocess.run(["wslview", str(path)], check=True)
            return
        except (subprocess.CalledProcessError, OSError):
            pass
    if shutil.which("explorer.exe"):
        try:
            converted = subprocess.run(
                ["wslpath", "-w", str(path)], capture_output=True, text=True, check=True
            ).stdout.strip()
            if converted:
                subprocess.run(["explorer.exe", converted])
                return
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
    if shutil.which("xdg-open"):
        try:
            subprocess.run(["xdg-open", str(path)], check=True)
            return
        except (subprocess.CalledProcessError, OSError):
            pass
    try:
        import webbrowser

        webbrowser.open(path.as_uri())
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HTML-Template -- ein Canvas, kein CDN, dunkler Hackerlook.
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>SELECTA MAP</title>
<style>
  html, body { margin: 0; padding: 0; background: #0a0a0a; overflow: hidden;
               font-family: "Cascadia Code", "Consolas", "Menlo", monospace; }
  #header { position: fixed; top: 0; left: 0; right: 0; padding: 10px 16px;
            color: #ff3ea5; background: linear-gradient(#0a0a0a, rgba(10,10,10,0));
            font-size: 14px; letter-spacing: 1px; z-index: 10; pointer-events: none; }
  #header .dim { color: #7a7a8c; }
  canvas { display: block; cursor: grab; }
  canvas.dragging { cursor: grabbing; }
  #tooltip { position: fixed; display: none; padding: 8px 10px; background: #12121ce6;
             border: 1px solid #ff3ea5; color: #d8d8e8; font-size: 12px; line-height: 1.5;
             z-index: 20; pointer-events: none; white-space: pre; border-radius: 3px; }
  #tooltip .title { color: #7CFC9A; font-weight: bold; }
  #tooltip .dim { color: #7a7a8c; }
  #legend { position: fixed; bottom: 12px; left: 12px; color: #d8d8e8; font-size: 11px;
            z-index: 10; background: #0a0a0aa0; padding: 6px 10px; border-radius: 3px;
            max-width: 260px; }
  #legend .row { display: flex; align-items: center; margin: 2px 0; }
  #legend .dot { width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; flex: none; }
  #hint { position: fixed; bottom: 12px; right: 12px; color: #7a7a8c; font-size: 11px; z-index: 10; }
</style>
</head>
<body>
<div id="header">◤ SELECTA MAP ◢ <span class="dim">— __SELECTA_TITLE__ — __SELECTA_PROJECTION__</span></div>
<canvas id="c"></canvas>
<div id="tooltip"></div>
<div id="legend"></div>
<div id="hint">scroll: zoom &middot; drag: pan &middot; dblclick: reset</div>
<script type="application/json" id="selecta-tracks">__SELECTA_TRACKS_JSON__</script>
<script type="application/json" id="selecta-legend">__SELECTA_LEGEND_JSON__</script>
<script>
(function () {
  "use strict";
  var tracks = JSON.parse(document.getElementById("selecta-tracks").textContent);
  var legendData = JSON.parse(document.getElementById("selecta-legend").textContent);

  var WORLD = 2000; // Weltkoordinaten-Spanne, Punkte liegen in [0, WORLD]
  var canvas = document.getElementById("c");
  var ctx = canvas.getContext("2d");
  var tooltip = document.getElementById("tooltip");
  var legendEl = document.getElementById("legend");

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    draw();
  }

  // BPM-Spanne fuer die Punktgroesse (min->3px, max->7px, fehlend->3px).
  var bpmVals = tracks.map(function (t) { return parseFloat(t.bpm); }).filter(function (v) { return !isNaN(v); });
  var bpmMin = bpmVals.length ? Math.min.apply(null, bpmVals) : 0;
  var bpmMax = bpmVals.length ? Math.max.apply(null, bpmVals) : 1;
  var bpmSpan = Math.max(bpmMax - bpmMin, 1e-6);

  function pointRadius(t) {
    var v = parseFloat(t.bpm);
    if (isNaN(v)) return 3;
    return 3 + 4 * (v - bpmMin) / bpmSpan;
  }

  // Weltkoordinaten vorab berechnen (einmalig).
  var pts = tracks.map(function (t) {
    return { wx: t.x * WORLD, wy: t.y * WORLD, r: pointRadius(t), t: t };
  });

  // Offscreen-Canvas: die Punktwolke wird einmal gezeichnet, Pan/Zoom/Hover
  // blitten nur noch -- bei 1000+ Punkten mit Glow ist ein Redraw pro
  // Mousemove sonst spuerbar langsam.
  var off = document.createElement("canvas");
  off.width = WORLD;
  off.height = WORLD;
  var offCtx = off.getContext("2d");

  function renderOffscreen() {
    offCtx.clearRect(0, 0, WORLD, WORLD);
    offCtx.globalCompositeOperation = "lighter";
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      offCtx.beginPath();
      offCtx.fillStyle = p.t.color;
      offCtx.shadowColor = p.t.color;
      offCtx.shadowBlur = 6;
      offCtx.arc(p.wx, p.wy, p.r, 0, Math.PI * 2);
      offCtx.fill();
    }
    offCtx.globalCompositeOperation = "source-over";
  }
  renderOffscreen();

  // View-Transform: Weltkoordinaten -> Bildschirm.
  var view = { scale: 1, ox: 0, oy: 0 };

  function resetView() {
    var pad = 60;
    var scale = Math.min(
      (canvas.width - 2 * pad) / WORLD,
      (canvas.height - 2 * pad) / WORLD
    );
    view.scale = scale;
    view.ox = (canvas.width - WORLD * scale) / 2;
    view.oy = (canvas.height - WORLD * scale) / 2;
  }

  function worldToScreen(wx, wy) {
    return [wx * view.scale + view.ox, wy * view.scale + view.oy];
  }

  function screenToWorld(sx, sy) {
    return [(sx - view.ox) / view.scale, (sy - view.oy) / view.scale];
  }

  function draw() {
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, view.ox, view.oy, WORLD * view.scale, WORLD * view.scale);
    if (hovered) {
      var s = worldToScreen(hovered.wx, hovered.wy);
      ctx.beginPath();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.arc(s[0], s[1], hovered.r * view.scale + 3, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  // --- Hover ---------------------------------------------------------------
  var hovered = null;

  function findNearest(sx, sy) {
    var best = null, bestDist = 8 * 8; // 8px Radius im Screen-Space
    for (var i = 0; i < pts.length; i++) {
      var s = worldToScreen(pts[i].wx, pts[i].wy);
      var dx = s[0] - sx, dy = s[1] - sy;
      var d2 = dx * dx + dy * dy;
      if (d2 < bestDist) { bestDist = d2; best = pts[i]; }
    }
    return best;
  }

  function fmtTooltip(t) {
    var lines = [];
    lines.push('<span class="title">' + escapeHtml(t.label) + "</span>");
    var bpmKey = (t.bpm || "?") + " BPM";
    if (t.key) bpmKey += "  \\u00b7  " + escapeHtml(t.key);
    if (t.year) bpmKey += "  \\u00b7  " + escapeHtml(t.year);
    lines.push('<span class="dim">' + bpmKey + "</span>");
    if (t.genres && t.genres.length) lines.push(escapeHtml(t.genres.join(" | ")));
    if (t.vibes && t.vibes.length) lines.push('<span class="dim">' + escapeHtml(t.vibes.join(" \\u00b7 ")) + "</span>");
    var mood = [];
    if (t.arousal) mood.push("arous " + t.arousal);
    if (t.aggressive) mood.push("aggr " + t.aggressive);
    if (t.danceable) mood.push("dance " + t.danceable);
    if (mood.length) lines.push('<span class="dim">' + mood.join("  \\u00b7  ") + "</span>");
    return lines.join("\\n");
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  canvas.addEventListener("mousemove", function (e) {
    if (dragging) return;
    var near = findNearest(e.clientX, e.clientY);
    if (near !== hovered) {
      hovered = near;
      draw();
    }
    if (hovered) {
      tooltip.style.display = "block";
      tooltip.style.left = (e.clientX + 14) + "px";
      tooltip.style.top = (e.clientY + 14) + "px";
      tooltip.innerHTML = fmtTooltip(hovered.t);
    } else {
      tooltip.style.display = "none";
    }
  });

  // --- Zoom (Mausrad, zentriert auf den Cursor) -----------------------------
  canvas.addEventListener("wheel", function (e) {
    e.preventDefault();
    var factor = Math.pow(1.0015, -e.deltaY);
    var before = screenToWorld(e.clientX, e.clientY);
    view.scale = Math.max(0.05, Math.min(50, view.scale * factor));
    var after = worldToScreen(before[0], before[1]);
    view.ox += e.clientX - after[0];
    view.oy += e.clientY - after[1];
    draw();
  }, { passive: false });

  // --- Pan (Drag) ------------------------------------------------------------
  var dragging = false, dragStart = null, viewStart = null;

  canvas.addEventListener("mousedown", function (e) {
    dragging = true;
    canvas.classList.add("dragging");
    dragStart = [e.clientX, e.clientY];
    viewStart = { ox: view.ox, oy: view.oy };
    tooltip.style.display = "none";
  });
  window.addEventListener("mousemove", function (e) {
    if (!dragging) return;
    view.ox = viewStart.ox + (e.clientX - dragStart[0]);
    view.oy = viewStart.oy + (e.clientY - dragStart[1]);
    draw();
  });
  window.addEventListener("mouseup", function () {
    dragging = false;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("dblclick", function () {
    resetView();
    draw();
  });

  // --- Legende ---------------------------------------------------------------
  legendData.forEach(function (entry) {
    var row = document.createElement("div");
    row.className = "row";
    var dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = entry.color;
    row.appendChild(dot);
    row.appendChild(document.createTextNode(entry.genre));
    legendEl.appendChild(row);
  });

  window.addEventListener("resize", resize);
  // Reihenfolge wichtig: resetView() braucht canvas.width/height, die erst
  // ein resize()-Aufruf setzt (Canvas-Default waere sonst 300x150).
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  resetView();
  draw();
})();
</script>
</body>
</html>
"""
