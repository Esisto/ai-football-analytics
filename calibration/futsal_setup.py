"""Manual futsal setup for fixed single-camera recordings.

Example:
python -m calibration.futsal_setup --video training.mp4 --length 38 --width 19 --camera-position corner

For multipurpose floors select ONLY points belonging to futsal; different
colour lines may represent different sports. Save a separate profile per video.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from calibration.calibration_ui import CalibrationApp, CalibrationSession, grab_frame
from calibration.futsal_pitch import CAMERA_POSITIONS, FutsalPitchDimensions, FutsalPitchModel


def profile_path_for_video(video_path: str | Path, output_dir: str | Path = "outputs") -> Path:
    return Path(output_dir) / f"{Path(video_path).stem}_pitch_profile.json"


def save_profile(video_path: str | Path, dims: FutsalPitchDimensions,
                 camera_position: str, output_dir: str | Path = "outputs") -> Path:
    path = profile_path_for_video(video_path, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dims.to_dict(camera_position)
    data["video"] = Path(video_path).name
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def load_profile(video_path: str | Path, output_dir: str | Path = "outputs"):
    """Return a video-specific profile, or None for legacy football recordings."""
    path = profile_path_for_video(video_path, output_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("sport") != "futsal" or data.get("video") != Path(video_path).name:
        raise ValueError(f"Invalid or mismatched futsal pitch profile: {path}")
    dims = FutsalPitchDimensions(length=float(data["length_m"]),
                                 width=float(data["width_m"]))
    if data.get("camera_position") not in CAMERA_POSITIONS:
        raise ValueError(f"Unknown camera position in {path}")
    return dims


def main(argv=None):
    parser = argparse.ArgumentParser(description="Set up a variable-size futsal pitch")
    parser.add_argument("--video", required=True)
    parser.add_argument("--length", type=float, required=True, help="Playing surface length in metres")
    parser.add_argument("--width", type=float, required=True, help="Playing surface width in metres")
    parser.add_argument("--camera-position", choices=CAMERA_POSITIONS, default="unknown")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--frame", type=int, default=1)
    args = parser.parse_args(argv)
    dims = FutsalPitchDimensions(args.length, args.width)
    video = Path(args.video)
    if not video.is_file():
        parser.error(f"Video not found: {video}")
    frame = grab_frame(video, args.frame)
    if frame is None:
        parser.error(f"Cannot read frame {args.frame}: {video}")
    cal_path = Path(args.output_dir) / f"{video.stem}_pitch_calibration.json"
    session = CalibrationSession(video, args.frame, cal_path, FutsalPitchModel(dims))
    # An existing calibration may belong to a different geometry. Never silently
    # reuse its field-metre correspondences against a new court size.
    profile = profile_path_for_video(video, args.output_dir)
    if cal_path.exists() and profile.exists():
        previous = load_profile(video, args.output_dir)
        if previous != dims:
            parser.error("A different court profile already exists. Use a fresh output directory.")
    elif cal_path.exists() and not profile.exists():
        parser.error("An existing calibration has no court profile. Use a fresh output directory.")
    elif profile.exists() and not cal_path.exists():
        # Recover from closing the UI without saving any reference points.
        previous = load_profile(video, args.output_dir)
        if previous != dims:
            parser.error("Previous court profile uses different dimensions. Use a fresh output directory.")
    saved_profile = save_profile(video, dims, args.camera_position, args.output_dir)
    if not dims.is_regulation_size():
        print("Note: training court outside FIFA regulation dimensions; physical measurements are used.")
    print(f"Profile saved: {saved_profile}")
    print("Select 4+ visibly distinct, non-collinear FUTSAL reference points; 8+ preferred.")
    CalibrationApp(session, frame).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
