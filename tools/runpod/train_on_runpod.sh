#!/usr/bin/env bash
# Run INSIDE a RunPod GPU pod (PyTorch template). Trains the futsal ball model
# and leaves weights at /workspace/futsal_ball.pt for download to the Mac.
#   export ROBOFLOW_API_KEY=...   (set in the pod, never commit it)
#   bash train_on_runpod.sh [extra train_ball_model.py args]
set -euo pipefail
cd /workspace
if [ ! -d ai-football-analytics ]; then
  git clone -b feature/futsal-flexible-calibration https://github.com/Esisto/ai-football-analytics.git
fi
cd ai-football-analytics && git pull --ff-only
# RunPod images mark system Python as externally managed (PEP 668), and a plain
# `pip install ultralytics` upgrades torch to a build newer than the host driver
# (seen 2026-09-30: torch 2.14/cu130 on a CUDA 12.8 driver -> no GPU, cuDNN errors).
# Pin the image's own torch/torchvision so pip never replaces them.
python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" > /tmp/pin.txt
python -m pip install -q --break-system-packages -c /tmp/pin.txt "ultralytics>=8.3.0" roboflow
python -c "import torch; assert torch.cuda.is_available(), 'CUDA broken after pip install'"
# Ultralytics' OpenCV needs these on slim images.
command -v apt-get >/dev/null && (apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null) || true
nvidia-smi --query-gpu=name,memory.total --format=csv
python tools/train_ball_model.py --device 0 --workers 8 --batch 32 "$@"
cp weights/futsal_ball.pt /workspace/futsal_ball.pt
cp -r runs/ball/futsal_ball /workspace/futsal_ball_run
echo "DONE: /workspace/futsal_ball.pt  (remember to STOP/TERMINATE the pod)"
