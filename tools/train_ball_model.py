"""Fine-tune a dedicated futsal BALL detector on the Mac (Apple MPS).

Default dataset: "Futsal Ball Detection" by ScaptureSports on Roboflow Universe
(~4.8k images, fixed high indoor camera, class "ball", licence CC BY 4.0 —
attribution required if you redistribute the model or data):
https://universe.roboflow.com/scapturesports-ball-detectiontest/futsal-ball-detection-p0e0s

Downloading needs a free Roboflow account. Put YOUR key in the environment
(never commit it):  export ROBOFLOW_API_KEY=...

    python tools/train_ball_model.py                  # download + train
    python tools/train_ball_model.py --data datasets/x/data.yaml   # skip download

Result: weights/futsal_ball.pt, used by the web app / mac_smoke_test --ball-model.
Persons still come from the generic COCO model; this model only finds the ball.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
WORKSPACE, PROJECT = "scapturesports-ball-detectiontest", "futsal-ball-detection-p0e0s"


def download(version: int) -> Path:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY (free account at roboflow.com → Settings → API Keys).")
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("Missing package: python -m pip install roboflow")
    target = DATASETS / f"{PROJECT}-v{version}"
    ds = (Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT)
          .version(version).download("yolov11", location=str(target)))
    return Path(ds.location) / "data.yaml"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune a futsal ball detector")
    parser.add_argument("--data", default=None, help="Existing YOLO data.yaml (skips download)")
    parser.add_argument("--version", type=int, default=3, help="Roboflow dataset version")
    parser.add_argument("--base", default="yolo11s.pt", help="Starting COCO checkpoint")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960, help="Balls are tiny: keep this high")
    parser.add_argument("--batch", type=int, default=8, help="Lower if the Mac runs out of memory")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Use a fraction of the images for a quick trial run (e.g. 0.1)")
    args = parser.parse_args(argv)

    data = Path(args.data) if args.data else download(args.version)
    if not data.is_file():
        sys.exit(f"data.yaml not found: {data}")

    from ultralytics import YOLO

    run = YOLO(args.base).train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, fraction=args.fraction, project=str(ROOT / "runs" / "ball"),
        name="futsal_ball", exist_ok=True, patience=10, workers=2,
        # Keep left/right flips; no vertical flips (gravity/perspective matter).
        fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=5, plots=True)
    best = Path(run.save_dir) / "weights" / "best.pt"
    out = ROOT / "weights" / "futsal_ball.pt"
    shutil.copy2(best, out)
    print(f"Ball model saved: {out}")
    print("Dataset: ScaptureSports 'Futsal Ball Detection', Roboflow Universe, CC BY 4.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
