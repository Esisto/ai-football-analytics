# Addestrare il modello pallone su RunPod (GPU a ore)

Solo l'addestramento va nel cloud; l'analisi dei video resta sul Mac.

## 1. Una tantum: chiave SSH
Copia la chiave **pubblica** del Mac e incollala in RunPod → Settings → SSH Public Keys:

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

## 2. Crea il pod
- Pods → Deploy → GPU **RTX 4090** (o A5000/L4 se esaurita), **datacenter EU** se usi video tuoi.
- Template: **RunPod PyTorch** (2.x). Disco container 20 GB, volume 20 GB.
- Deploy. Quando è "Running": Connect → copia il comando **"SSH over exposed TCP"**
  (`ssh root@<IP> -p <PORTA> -i ~/.ssh/id_ed25519`).

## 3. Addestra
```bash
ssh root@<IP> -p <PORTA> -i ~/.ssh/id_ed25519
export ROBOFLOW_API_KEY=<la tua chiave>
curl -sL https://raw.githubusercontent.com/Esisto/ai-football-analytics/feature/futsal-flexible-calibration/tools/runpod/train_on_runpod.sh | bash
```
Varianti: `... | bash -s -- --imgsz 1280 --epochs 60` oppure `--base yolo11m.pt`.

## 4. Riporta il modello sul Mac
```bash
scp -P <PORTA> -i ~/.ssh/id_ed25519 root@<IP>:/workspace/futsal_ball.pt weights/futsal_ball.pt
```

## 5. Spegni il pod
RunPod → **Terminate** (Stop continua a fatturare il volume). Il pod acceso ma inattivo si paga.

Dataset: ScaptureSports "Futsal Ball Detection", Roboflow Universe, CC BY 4.0 (citare se si ridistribuisce).
