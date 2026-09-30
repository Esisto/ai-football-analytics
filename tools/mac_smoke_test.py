"""Mac smoke test: generic COCO people/ball tracking, no custom checkpoints.

This is NOT the full futsal pipeline, and COCO sports-ball recall may be poor.
It checks Apple MPS, video decode, detector/tracker and MP4 output locally.

python tools/mac_smoke_test.py --source recordings/allenamento.mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

TRACKER = Path(__file__).with_name("futsal_bytetrack.yaml")
PERSON, BALL = 0, 32  # COCO class ids


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def to_browser_h264(src: Path, dst: Path) -> bool:
    """Re-encode OpenCV's mp4v output to H.264 so browsers can play it."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(dst)]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return done.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def tiled_ball_boxes(model, frame, grid: int, device: str, imgsz: int,
                     conf: float, overlap: float = 0.15, classes=(BALL,)) -> list[tuple[int, int, int, int, float]]:
    """Detect balls on an overlapping grid x grid split at full resolution.

    Far balls in 4K are a few pixels once the whole frame is resized to imgsz;
    each tile keeps far more of them. Slower: grid*grid extra inferences.
    grid=1 runs once on the whole frame; classes=None keeps every class
    (for a dedicated ball-only model).
    """
    h, w = frame.shape[:2]
    tw, th = w / grid, h / grid
    boxes, scores = [], []
    for row in range(grid):
        for col in range(grid):
            x0 = max(0, int(col * tw - overlap * tw))
            y0 = max(0, int(row * th - overlap * th))
            x1 = min(w, int((col + 1) * tw + overlap * tw))
            y1 = min(h, int((row + 1) * th + overlap * th))
            res = model.predict(frame[y0:y1, x0:x1], classes=list(classes) if classes else None, device=device,
                                imgsz=imgsz, conf=conf, verbose=False)[0]
            for (bx0, by0, bx1, by1), score in zip(res.boxes.xyxy.cpu().tolist(),
                                                   res.boxes.conf.cpu().tolist()):
                boxes.append([int(bx0) + x0, int(by0) + y0, int(bx1 - bx0), int(by1 - by0)])
                scores.append(float(score))
    if not boxes:
        return []
    keep = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.4)
    return [(boxes[i][0], boxes[i][1], boxes[i][0] + boxes[i][2],
             boxes[i][1] + boxes[i][3], scores[i]) for i in (int(k) for k in keep)]


def overlaps(box, others, thr: float = 0.3) -> bool:
    x0, y0, x1, y1 = box[:4]
    for o0, p0, o1, p1 in others:
        iw = max(0, min(x1, o1) - max(x0, o0))
        ih = max(0, min(y1, p1) - max(y0, p0))
        inter = iw * ih
        union = (x1 - x0) * (y1 - y0) + (o1 - o0) * (p1 - p0) - inter
        if union > 0 and inter / union > thr:
            return True
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline Mac person/ball tracking test")
    parser.add_argument("--source", required=True, help="Local video path (MP4/MOV)")
    parser.add_argument("--output", default=None, help="Annotated MP4 output path")
    parser.add_argument("--model", default="yolo11n.pt", help="Generic COCO detector; downloads on first run")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--max-frames", type=int, default=300, help="Default: first 300 video frames; 0 = whole clip")
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Inference size; 4K balls vanish at 640")
    parser.add_argument("--person-conf", type=float, default=0.25)
    parser.add_argument("--ball-conf", type=float, default=0.10,
                        help="Lower than persons: small, blurred balls score low")
    parser.add_argument("--ball-tiles", type=int, default=0, choices=[0, 2, 3],
                        help="Extra full-resolution ball search on an NxN grid (0 = off; slow)")
    parser.add_argument("--ball-model", default=None,
                        help="Dedicated ball detector (e.g. weights/futsal_ball.pt from "
                             "tools/train_ball_model.py); replaces COCO ball detection")
    args = parser.parse_args(argv)
    src = Path(args.source)
    if not src.is_file():
        parser.error(f"Video not found: {src}")
    if args.max_frames < 0 or args.imgsz <= 0:
        parser.error("--max-frames must be >= 0 and --imgsz positive")
    if not (0 < args.person_conf < 1 and 0 < args.ball_conf < 1):
        parser.error("--person-conf and --ball-conf must be between 0 and 1")
    out = Path(args.output) if args.output else Path("outputs") / f"{src.stem}_smoke.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_mp4v.mp4")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"Cannot decode video: {src}. Try converting HEVC MOV to H.264 MP4.", file=sys.stderr)
        return 2
    writer = None
    processed = 0
    people_total = 0
    balls_total = 0
    tiled_balls = 0
    unique_ids: set[int] = set()
    try:
        from ultralytics import YOLO

        device = pick_device(args.device)
        model = YOLO(args.model)
        ball_model = None
        if args.ball_model:
            if not Path(args.ball_model).is_file():
                raise FileNotFoundError(f"Ball model not found: {args.ball_model}")
            ball_model = YOLO(args.ball_model)
        track_classes = [PERSON] if ball_model else [PERSON, BALL]
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (0.1 <= fps <= 240):
            fps = 30.0
        started = time.perf_counter()
        while args.max_frames == 0 or processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # COCO: 0=person, 32=sports ball (not a futsal-specialised checkpoint).
            # Track at the lowest threshold, then apply per-class confidence.
            results = model.track(frame, persist=True, tracker=str(TRACKER),
                                  classes=track_classes, device=device, imgsz=args.imgsz,
                                  conf=min(args.person_conf, args.ball_conf), verbose=False)
            result = results[0]
            if result.boxes is not None and len(result.boxes):
                cls_t, conf_t = result.boxes.cls, result.boxes.conf
                keep = (((cls_t == PERSON) & (conf_t >= args.person_conf))
                        | ((cls_t == BALL) & (conf_t >= args.ball_conf)))
                result = result[keep.cpu().numpy()]
            cls = result.boxes.cls.int().cpu().tolist() if result.boxes is not None else []
            people_total += cls.count(PERSON)
            balls_total += cls.count(BALL)
            if result.boxes is not None and result.boxes.id is not None:
                unique_ids.update(int(i) for i in result.boxes.id.cpu().tolist())
            annotated = result.plot()
            if ball_model is not None:
                for x0, y0, x1, y1, score in tiled_ball_boxes(
                        ball_model, frame, max(1, args.ball_tiles), device,
                        args.imgsz, args.ball_conf, classes=None):
                    balls_total += 1
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(annotated, f"ball {score:.2f}", (x0, max(0, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif args.ball_tiles:
                found = [tuple(b) for b, c in zip(result.boxes.xyxy.cpu().tolist(), cls)
                         if c == BALL] if result.boxes is not None else []
                for box in tiled_ball_boxes(model, frame, args.ball_tiles, device,
                                            args.imgsz, args.ball_conf):
                    if overlaps(box, found):
                        continue
                    tiled_balls += 1
                    x0, y0, x1, y1, score = box
                    cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 165, 255), 2)
                    cv2.putText(annotated, f"ball tile {score:.2f}", (x0, max(0, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot write MP4: {raw}")
            writer.write(annotated)
            processed += 1
        elapsed = time.perf_counter() - started
        if processed == 0:
            print("No frames decoded.", file=sys.stderr)
            return 3
        print(f"Device: {device}")
        print(f"Frames: {processed}; elapsed: {elapsed:.1f}s; throughput: {processed/elapsed:.2f} FPS")
        print(f"Person detections: {people_total}; sports-ball detections: {balls_total}")
        if args.ball_tiles and ball_model is None:
            print(f"Extra untracked ball detections from {args.ball_tiles}x{args.ball_tiles} tiles: {tiled_balls}")
        print(f"Distinct tracked IDs (not distinct people): {len(unique_ids)}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if to_browser_h264(raw, out):
        raw.unlink()
    else:
        raw.replace(out)
        print("Warning: ffmpeg H.264 export unavailable; mp4v output may not play "
              "in the browser (open it with QuickTime/VLC).", file=sys.stderr)
    print(f"Annotated video: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
