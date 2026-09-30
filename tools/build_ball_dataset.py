"""Build one merged BALL-only dataset from several public futsal datasets.

Downloads each Roboflow Universe project (latest version, YOLO format), keeps
only the ball class (renamed to class 0 "ball"), drops every other label, and
merges everything under datasets/futsal_ball_merged/ with per-source prefixes.
Sources whose licence is not in ALLOWED_LICENCES are skipped automatically.

    export ROBOFLOW_API_KEY=...        # your key, never commit it
    python tools/build_ball_dataset.py

The ScaptureSports dataset already in datasets/ is reused without downloading.
A SOURCES.md with attribution for every included dataset is written next to data.yaml.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
OUT = DATASETS / "futsal_ball_merged"
# Commercial-friendly only: non-commercial (NC) licences are rejected.
ALLOWED_LICENCES = {"CC BY 4.0", "CC0 1.0", "Public Domain", "MIT"}
BALL_WORDS = ("ball", "bola", "pallone", "balon", "balón", "football", "futsal_ball")

# (workspace, project). Moedjaheed is deliberately excluded: TV broadcast footage.
SOURCES = [
    ("scapturesports-ball-detectiontest", "futsal-ball-detection-p0e0s"),
    ("david-carvalho", "futsal-ball-detection"),
    ("futsalplayersannotations", "futsal-annotations"),
    ("itmade-esr43", "futsal-pvbvz"),
    ("begin-sdol0", "futsal-6cpe2"),
    ("remmas-workspace", "futsal-player-tracking-02"),
    ("fiwwww", "project-futsal"),
]


def ball_ids(names) -> list[int]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return [int(i) for i, n in items
            if any(w in str(n).lower() for w in BALL_WORDS) and "player" not in str(n).lower()]


def fetch(rf, workspace: str, project: str) -> Path | None:
    local = sorted(DATASETS.glob(f"{project}-v*"))
    if local:
        return local[-1]
    try:
        proj = rf.workspace(workspace).project(project)
        versions = proj.versions()
        if not versions:
            print(f"  skip {project}: no versions")
            return None
        v = max(int(str(x.version).split("/")[-1]) for x in versions)
        target = DATASETS / f"{project}-v{v}"
        proj.version(v).download("yolov11", location=str(target))
        return target
    except Exception as exc:  # private/removed projects, export errors...
        print(f"  skip {project}: {exc}")
        return None


def main() -> int:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY first.")
    from roboflow import Roboflow

    rf = Roboflow(api_key=key)
    if OUT.exists():
        shutil.rmtree(OUT)
    counts, credits = {}, []
    for workspace, project in SOURCES:
        print(f"== {workspace}/{project}")
        src = fetch(rf, workspace, project)
        if src is None or not (src / "data.yaml").is_file():
            continue
        meta = yaml.safe_load((src / "data.yaml").read_text())
        licence = (meta.get("roboflow") or {}).get("license", "unknown")
        if licence not in ALLOWED_LICENCES:
            print(f"  skip: licence '{licence}' not allowed")
            continue
        ids = ball_ids(meta.get("names", []))
        if not ids:
            print(f"  skip: no ball class in {meta.get('names')}")
            continue
        n_img = n_box = 0
        for split in ("train", "valid", "test"):
            img_dir, lbl_dir = src / split / "images", src / split / "labels"
            if not img_dir.is_dir():
                continue
            out_split = "valid" if split == "valid" else ("test" if split == "test" else "train")
            (OUT / out_split / "images").mkdir(parents=True, exist_ok=True)
            (OUT / out_split / "labels").mkdir(parents=True, exist_ok=True)
            for img in img_dir.iterdir():
                if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                lbl = lbl_dir / f"{img.stem}.txt"
                lines = []
                if lbl.is_file():
                    for line in lbl.read_text().splitlines():
                        parts = line.split()
                        if len(parts) == 5 and int(parts[0]) in ids:  # boxes only, no polygons
                            lines.append("0 " + " ".join(parts[1:]))
                name = f"{project}__{img.name}"
                shutil.copy2(img, OUT / out_split / "images" / name)
                (OUT / out_split / "labels" / f"{Path(name).stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""))
                n_img += 1
                n_box += len(lines)
        counts[project] = (n_img, n_box)
        url = (meta.get("roboflow") or {}).get("url", f"https://universe.roboflow.com/{workspace}/{project}")
        credits.append(f"- {workspace}/{project} — {licence} — {url} — {n_img} images, {n_box} balls")
        print(f"  kept {n_img} images, {n_box} ball boxes ({licence})")

    (OUT / "data.yaml").write_text(yaml.safe_dump({
        "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": 1, "names": ["ball"]}, sort_keys=False))
    (OUT / "SOURCES.md").write_text("# Sources (attribution required)\n\n" + "\n".join(credits) + "\n")
    total = sum(c[0] for c in counts.values())
    print(f"\nMerged dataset: {OUT}  ({total} images from {len(counts)} sources)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
