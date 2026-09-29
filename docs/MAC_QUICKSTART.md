# Mac M5 quick start (experimental futsal branch)

This branch can run without merging draft PR #1. Python 3.12 (Apple Silicon build)
is the target development environment. macOS Terminal commands:

```bash
xcode-select --install
# Install Python 3.12 for macOS from https://www.python.org/downloads/
git clone -b feature/futsal-flexible-calibration https://github.com/Esisto/ai-football-analytics.git
cd ai-football-analytics
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
python -m pytest tests/test_futsal_pitch.py -q
python -m pytest tests/ -q
```

The MPS test should say `True` on a supported Apple Silicon/PyTorch setup.
If false, inspect Python architecture, PyTorch build and OS version. A PyTorch
CPU fallback is possible but slower. Native Tkinter UI check:
`python -m tkinter`. Use a Python build containing Tcl/Tk if unavailable.

The full football repository currently does NOT contain `weights/best.pt` in
Git and this fork currently has no GitHub Release assets. Do NOT assume that
`gh release download v1.0.0` works. The full four-class pipeline needs a
compatible checkpoint. Generic COCO weights have different labels and must NOT
be renamed to `weights/best.pt`.

For a first local smoke test, the following script downloads a generic
`yolo11n.pt` detector and tracks COCO persons (class 0) and sports balls
(class 32). This is a standalone preview, not the calibrated full app:

```bash
mkdir -p recordings outputs
# Copy an H.264 MP4 to recordings/allenamento.mp4
python tools/mac_smoke_test.py --source recordings/allenamento.mp4 --max-frames 300
open outputs/allenamento_smoke.mp4
```

The output reports effective video-processing FPS and a marked-up video.
`--max-frames 0` processes the entire video; `--device cpu` provides a
CPU fallback. Weak sports-ball detection is expected with generic COCO weights.

With valid compatible `weights/best.pt` from its rights holder, first test
the real pipeline on a short segment:

```bash
python main.py --video recordings/allenamento.mp4 --tracking --max-frames 300 --device mps
```

For futsal use per-video calibration with the correct physical court size:

```bash
python -m calibration.futsal_setup --video recordings/allenamento.mp4 \
  --length 38 --width 19 --camera-position corner
```

The UI needs four or more distinct non-collinear FUTSAL landmarks, not
basketball/volleyball markings. If insufficient landmarks are visible,
track without interpreting pixel motion as metres. Full analysis after
compatible weights and valid calibration:

```bash
python main.py --video recordings/allenamento.mp4 --all --device mps
```

The original video should remain on the user's Mac and should not be
committed to Git. Film only with the required permissions, especially
where minors participate.
