# Futsal single-camera prototype

This is an additive, experimental extension of the existing football pipeline.
The original 105 x 68 m football path remains the default if no video-specific
futsal profile exists. Existing detection/tracking code remains unchanged.

## First recording

Use an iPhone fixed in landscape mode at the sideline, a corner, or behind a
goal. Keep the lens, zoom, and framing fixed. Aim for a full unobstructed pitch;
record a short empty-pitch clip before training and a 10-15 minute continuous
drill. Save the original recording locally, with participants' appropriate
permission and special safeguards if minors are filmed.

## Court setup

The actual futsal playing area must be measured or reliably known. A multipurpose
floor may have basketball, volleyball and futsal markings simultaneously: NEVER
assume all visible lines are futsal. Use the *futsal* corner/intersection labels
only; give the user a manual override rather than guessing from line colours.

FIFA 2025/26: domestic 25-42 x 16-25 m, international 38-42 x 20-25 m.
A non-regulation training area is still accepted if its physical measurements
are supplied. Default profile is 40 x 20 m.

```bash
python -m calibration.futsal_setup \
  --video recordings/training.mp4 \
  --length 38 --width 19 --camera-position behind_goal
```

This launches the existing native calibration UI with futsal-specific named
landmarks, and writes two files under outputs/:

- `training_pitch_profile.json`: sport, physical dimensions, camera position;
- `training_pitch_calibration.json`: operator-labelled image/field landmarks
  (saved from the UI after 4+ points).

For a different location use `--camera-position corner`, `sideline`, or `other`.
Camera position is informative metadata, NOT a hard-coded projection.

Four DISTINCT and non-collinear ground-plane points are the mathematical
minimum, ideally use eight or more spread across the visible court.
Two goalposts alone cannot determine an entire field homography. If the
view does not reveal enough futsal landmarks, retain detection/tracking but
do not emit metrically trustworthy speeds and distances. The user may
calibrate another frame or reposition the camera. Do not click apparent
intersections belonging to basketball or volleyball.

```bash
python main.py --video recordings/training.mp4 --all --device mps
```

The existing pipeline discovers the saved calibration and uses the futsal
minimap dimensions. The supplied football weights are NOT necessarily
futsal-trained, and the 11-a-side automatic keypoint calibration is NOT
validated for futsal. Start with manual calibration and evaluate detection,
identity switches, ball recall and processing throughput on real video.

The video-specific profile is deliberately separate from calibration JSON
to preserve legacy file compatibility. Recalibrating after changing the
court dimensions requires a fresh output directory or an intentional reset,
not silent reuse of metre coordinates from a different court.

## Limitations before commercial use

- This PR adds futsal court geometry and an offline manual setup path. It
  does not implement iPhone live ingest or a production GUI.
- Multiple coloured court lines require operator selection; there is no
  validated line-type classifier yet.
- A strongly oblique/behind-goal view reduces far-field measurement accuracy.
  Low reprojection error on calibration points alone is not validation against
  unseen points, especially when points cluster in one area.
- Detection/ID tracking can work when geometric calibration fails; metric
  analytics must be flagged uncalibrated in that case.
- Check Ultralytics and model-weights licensing separately from the
  repository's MIT code before distributing a proprietary product.

## Tests

```bash
python -m pytest tests/test_futsal_pitch.py -q
python -m pytest tests/ -q
```
