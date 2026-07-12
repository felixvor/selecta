# ◤ SELECTA ◢

A terminal tool for set preparation and live DJing: finds the right next track while you play, or helps you build a selection ahead of time. One track as the query, your own library as the answer — ranked by audio embeddings computed straight from the signal, re-ranked by BPM, key and mood. Deliberately stateless, a tool that runs alongside Rekordbox & co. and answers the two questions they leave open — *what fits this track?* and *how do I get smoothly from track A to track B?*

- **Your music, wherever it came from.** SELECTA listens to the audio
  signal, not to catalog metadata. Bandcamp purchases, rips, your own
  unreleased tracks — all treated equally, no Beatport entry required.
- **Local & offline.** Analysis and search run on your own machine, no
  cloud, no account.
- **SOTA models, lightweight.** Embeddings contrastively trained on track
  similarity
  ([`discogs_track_embeddings`](https://essentia.upf.edu/models.html),
  Essentia / MTG-UPF).
- **Transparent, not a black box.** Every score is traceable (embedding
  cosine, BPM distance, harmonic distance, mood deltas), and every knob
  lives in `selecta/config.py` for you to fine-tune.
- **Genre & vibe tags, straight from the signal.** 400 Discogs sub-genre
  styles (Acid House, Deep Techno, …) and DJ-relevant vibe tags (dark,
  deep, uplifting, …) are computed per track and shown as a chip line
  under each result — plus the release year, if your files carry a tag.
- **Multiple libraries, one search.** Register your music folders once in
  the launcher, toggle any combination active, and search across all of
  them. Each folder keeps its own analysis CSV and stays portable.

## Requirements

Python 3.14 (exactly this version — `essentia-tensorflow` is only pinned
as a Python 3.14 wheel). `essentia-tensorflow` is only available as a
Linux wheel — so SELECTA does not run on macOS, and runs on Windows via
WSL2 (see below).

## Setup (Linux / WSL)

```bash
cd ~/selecta
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Setup (Windows via WSL2)

Install WSL2 with Ubuntu once (`wsl --install -d Ubuntu`, current LTS —
do **not** pin an older version number; those ship Python 3.10/3.12
instead of the required 3.14).
Then copy this folder anywhere (e.g. via USB stick) and double-click
`Selecta.bat` — on first launch it automatically sets up the venv and
dependencies inside WSL (`setup.sh`) and then starts the TUI. You can
also drag a folder onto `Selecta.bat` to pass it as the music folder.

## Running

```bash
selecta                             # TUI, opens the library launcher
selecta /path/to/your/music         # TUI, single folder ad-hoc (saved list untouched)
selecta analyze --music-dir DIR     # headless analysis (e.g. overnight)
```

### Libraries

Started without arguments, SELECTA opens the **library launcher**: your
music folders as a list, each with an active toggle. Add a folder once —
drag & drop it from your file manager into the terminal window, the path
is pasted for you — and SELECTA remembers it in
`~/.local/share/selecta/libraries.json`. **Enter** starts the search
across *all active* libraries at once; each folder keeps its own
`library_analysis.csv`, so folders stay portable and analyses resumable.
Nested folders are deduplicated in the search.

| Launcher key | Function |
|---|---|
| `Space` / click | Toggle a library active/inactive |
| `A` | Add a library (drag & drop the folder, or paste a path) |
| `D` | Remove the entry (the CSV inside the folder is kept) |
| `Ctrl+A` | Analyze the highlighted library |
| `Enter` | Start the search across all active libraries |

`Ctrl+L` brings the launcher back mid-session to switch libraries — the
current query and energy setting survive if the query track is still in
the new selection.

Analysis results are stored as `library_analysis.csv` inside the music
folder (resumable; existing CSVs are automatically backfilled with
embeddings). Models are downloaded once into `./models` (~48 MB).

CSVs written by older SELECTA versions (before genre/vibe tagging) are
picked up automatically: the next analysis run (`Ctrl+A`) re-analyzes
those rows once and fills in the new columns.

## Usage

Typing filters the library, **Enter** takes the highlighted track as the
query — the list then shows the most similar tracks. **Enter/click on a
result** makes it the new query (that's how you walk through a set).

| Key | Function |
|---|---|
| `↑` `↓` | Navigate the list (detail line at the bottom shows the highlighted track) |
| `←` `→` | Energy axis −6…+6: shifts the search target (BPM/arousal/hardness) |
| `,` `.` | BPM filter: hard cutoff (only faster or slower tracks), each press snaps to the next BPM value present in the library |
| `Ctrl+A` | Analyze all active libraries in sequence (confirmation → progress + live per-track log: analysis stages while running, then genre/vibe tags and scores per finished track) |
| `Ctrl+T` | Pin a transition target (fuzzy search + Enter): the list shows bridge tracks between the current query and the target |
| `Ctrl+L` | Back to the library launcher (switch libraries mid-session) |
| `Esc` | Clear the search / go back |
| `Ctrl+C` / `Ctrl+Q` | Quit |

Letters always type into the search. `←→,.` only act when the search
field is **empty** — with text in the field they move the cursor or type
as usual.

Result columns: BPM with Δ (green = directly mixable), KEY (green =
harmonic), SCORE (embedding cosine minus penalties), ΔENERG/ΔHARD/ΔMOOD
as Δ×10 integers (+4 = +0.4) relative to the query.

Below each track a chip line shows its sub-genre tags (colored pills),
vibe tags and year (dimmed) — rows without tags stay single-line. Tags
are display-only; the ranking is not affected (genre similarity is
already captured by the embedding cosine).

**Transition mode** (`Ctrl+T`): pin a target track B, and the list shows
bridge candidates with SCORE→A and SCORE→B (≥0.9 green, ≥0.8 yellow,
≥0.7 orange, red below), sorted by the weaker side — the bottleneck
decides whether a bridge works. B itself is ranked along with the rest
and rises to the top once the direct jump is the best option. **Enter**
on a candidate makes it the new query (B stays pinned — that's how you
work your way from a slow opener to a hard closer), **Enter on B**,
**Esc** or pressing `Ctrl+T` again ends the mode. The BPM filter
(`,` `.`) stays active; the energy axis is disabled in transition mode.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tuning constants (energy step sizes, weights, top-N): `selecta/config.py`.

## Dependencies & license

Uses [Essentia](https://essentia.upf.edu) (`essentia-tensorflow`),
distributed by the Music Technology Group (UPF) under AGPLv3
(non-commercial) or a commercial license — see the
[licensing information](https://essentia.upf.edu/licensing_information.html)
if you want to use this project commercially.
