# ◤ SELECTA ◢

A terminal tool for set preparation and live DJing: finds the right next
track while you play, or helps you build a selection ahead of time. One
track as the query, your own library as the answer — ranked by audio
embeddings computed straight from the signal, re-ranked by BPM, key and
mood. Deliberately stateless, with no playlists or sets of its own: a tool
that runs alongside Rekordbox & co. and answers the two questions they
leave open — *what fits this track?* and *how do I get smoothly from
track A to track B?*

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
selecta /path/to/your/music         # TUI, folder as argument
selecta                             # TUI, prompts for the folder
selecta analyze --music-dir DIR     # headless analysis (e.g. overnight)
```

Analysis results are stored as `library_analysis.csv` inside the music
folder (resumable; existing CSVs are automatically backfilled with
embeddings). Models are downloaded once into `./models` (~43 MB).

## Usage

Typing filters the library, **Enter** takes the highlighted track as the
query — the list then shows the most similar tracks. **Enter/click on a
result** makes it the new query (that's how you walk through a set).

| Key | Function |
|---|---|
| `↑` `↓` | Navigate the list (detail line at the bottom shows the highlighted track) |
| `←` `→` | Energy axis −6…+6: shifts the search target (BPM/arousal/hardness) |
| `,` `.` | BPM filter: hard cutoff (only faster or slower tracks), each press snaps to the next BPM value present in the library |
| `Ctrl+A` | Analyze the folder (confirmation → progress + log; new tracks, backfill of older CSVs) |
| `Ctrl+T` | Pin a transition target (fuzzy search + Enter): the list shows bridge tracks between the current query and the target |
| `Esc` | Clear the search / go back |
| `Ctrl+C` / `Ctrl+Q` | Quit |

Letters always type into the search. `←→,.` only act when the search
field is **empty** — with text in the field they move the cursor or type
as usual.

Result columns: BPM with Δ (green = directly mixable), KEY (green =
harmonic), SCORE (embedding cosine minus penalties), ΔENERG/ΔHARD/ΔMOOD
as Δ×10 integers (+4 = +0.4) relative to the query.

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
