"""Mac smoke test: generic COCO people/ball tracking, no custom checkpoints.

This is NOT the full futsal pipeline, and COCO sports-ball recall may be poor.
It checks Apple MPS, video decode, detector/tracker and MP4 output locally.

python tools/mac_smoke_test.py --source recordings/allenamento.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline Mac person/ball tracking test")
    parser.add_argument("--source", required=True, help="Local video path (MP4/MOV)")
    parser.add_argument("--output", default=None, help="Annotated MP4 output path")
    parser.add_argument("--model", default="yolo11n.pt", help="Generic COCO detector; downloads on first run")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--max-frames", type=int, default=300, help="Default: first 300 video frames; 0 = whole clip")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)
    src = Path(args.source)
    if not src.is_file():
        parser.error(f"Video not found: {src}")
    if args.max_frames < 0 or args.imgsz <= 0:
        parser.error("--max-frames must be >= 0 and --imgsz positive")
    out = Path(args.output) if args.output else Path("outputs") / f"{src.stem}_smoke.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Cannot decode video: {src}. Try converting HEVC MOV to H.264 MP4.", file=sys.stderr)
        return 2
    writer = None
    processed = 0
    people_total = 0
    balls_total = 0
    unique_ids: set[int] = set()
    try:
        from ultralytics import YOLO

        device = pick_device(args.device)
        model = YOLO(args.model)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (0.1 <= fps <= 240):
            fps = 30.0
        started = time.perf_counter()
        while args.max_frames == 0 or processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # COCO: 0=person, 32=sports ball (not a futsal-specialised checkpoint).
            results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  classes=[0, 32], device=device, imgsz=args.imgsz,
                                  conf=0.25, verbose=False)
            result = results[0]
            cls = result.boxes.cls.int().cpu().tolist() if result.boxes is not None else []
            people_total += cls.count(0)
            balls_total += cls.count(32)
            if result.boxes is not None and result.boxes.id is not None:
                unique_ids.update(int(i) for i in result.boxes.id.cpu().tolist())
            annotated = result.plot()
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write MP4: {out}")
            writer.write(annotated)
            processed += 1
        elapsed = time.perf_counter() - started
        if processed == 0:
            print("No frames decoded.", file=sys.stderr)
            return 3
        print(f"Device: {device}")
        print(f"Frames: {processed}; elapsed: {elapsed:.1f}s; throughput: {processed/elapsed:.2f} FPS")
        print(f"Person detections: {people_total}; sports-ball detections: {balls_total}")
        print(f"Distinct tracked IDs (not distinct people): {len(unique_ids)}")
        print(f"Annotated video: {out.resolve()}")
        return 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    raise SystemExit(main())
