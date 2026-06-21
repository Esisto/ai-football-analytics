# Football AI — Workflow

How to run the system. Everything is automatic and dynamic per match — the
only human input is **two optional tools**: *multi-select* and *correction*.
The pitch homography is produced automatically **from our trained model**, so
there is no manual pitch clicking.

---

## 1. Fully automatic run (one command)

```bash
python main.py --video <match>.mp4 --all --device cuda:0
```

This runs the whole pipeline end to end:

```
detect → track → dynamic team/role refinement →
homography from our model (+ camera-motion smoothing) →
minimap + skeleton-position video →
speed/distance → player performance → heatmaps →
tactics → possession → final match report
```

The team colours, the two teams, and the pitch geometry are all discovered
from the footage — nothing is configured per video. You already get a complete
result and all reports from this single command.

---

## 2. The two manual tools (optional, human-in-the-loop)

Both tools only **refine** the automatic result; they never replace the
pipeline and the automatic output is always kept underneath. They are the
screens that get wired into the front-end (Flutter) app so an operator can open
them and adjust.

### a) Multi-select

Label whole teams / referees / goalkeepers fast by clicking the boxes on a few
representative frames.

```bash
python -m manual_correction initialize --video <match>.mp4
```

- Click the players you want, then assign them in one action to
  **Team 0 / Team 1 / Referee / Goalkeeper**.
- Do a few frames for good coverage, then save.

### b) Correction

Fix any single track's role/team and type a **player name** (the name then
appears on the video and in every analytics output).

```bash
python -m manual_correction review --video <match>.mp4
```

- Pick a track, set its role/team, and type the **player name**.
- Save.

> The name and the corrections are stored in the corrections file and overlaid
> on the automatic roles — so they survive, are auditable, and are read by the
> reports and the final video.

---

## 3. Re-run so your manual edits apply

After multi-select / correction, re-run **reusing the same tracks** so the
track ids (which your labels are keyed to) stay identical:

```bash
python main.py --video <match>.mp4 --all --reuse-tracks --device cuda:0
```

`--reuse-tracks` skips re-detection/re-tracking and reuses the existing
identities, so your manual teams, corrections and player names are applied,
then the minimap, skeleton video and all reports are regenerated.

> Rule of thumb: run `--all` once → use the manual tools → re-run with
> `--reuse-tracks`. Plain `--all` again would re-number the tracks and your
> manual edits would no longer match.

---

## 4. What you get

The run produces (per video name):

- **Video** — `<match>_final_with_minimap.mp4` (skeleton positions, team
  colours that follow the shirt, player names, tactical minimap, possession
  bar, per-player speed/distance bar).
- **Reports** — JSON **and** TXT for: speed & distance, player performance,
  team tactics, possession, and the final match report.
- **Heatmaps** — team and best-player PNGs.

Every report exists as both machine-readable **JSON** (for the app) and
human-readable **TXT** (to read directly).

---

## 5. Front-end (Flutter) integration

The app reads/writes plain JSON, so the two manual tools map cleanly onto app
screens:

- **Read** `<match>_roles_final.json` and the analytics JSONs to show players,
  teams, names, ratings and tactics.
- **Write** the corrections file (role / team / **player name**) from the app's
  multi-select and correction screens.
- Re-run with `--reuse-tracks` to apply and regenerate everything.
