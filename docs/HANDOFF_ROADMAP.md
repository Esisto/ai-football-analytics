# CyberFutsal AI — Handoff e roadmap

Documento vivo: stato reale del branch `feature/futsal-flexible-calibration`,
metodo di affinamento dei modelli e prossimi passi. Aggiornato al 30/09/2026.

## 1. Stato attuale (verificato)

| Area | Stato |
|---|---|
| Dashboard Streamlit (`web_app.py`) | Funzionante sul Mac. Video da `recordings/` (nessun limite di dimensione), upload o percorso. Output H.264 visibile nel browser. |
| Persone | Modello generico COCO (YOLO11n/s/m). Buona copertura su video GoPro in angolo; ID instabili negli incroci (42 ID per ~10 persone in 10 s). |
| Pallone | Modello generico COCO: **0 rilevamenti** su 300 frame di GoPro in angolo. Serve un modello dedicato → sezione 2. |
| Modello pallone dedicato | `weights/futsal_ball.pt`, usato automaticamente dalla dashboard quando presente. Prova Mac (10% dati, 5 epoche): P 0.73, R 0.50, mAP50 0.55. |
| Calibrazione | Solo CLI Tkinter (`calibration/futsal_setup.py`); non ancora nel web. |
| Test | 433/433 passati. |

## 2. Metodo di affinamento (fine-tuning) — procedura standard

**Architettura scelta:** due modelli separati.
- Persone → YOLO COCO generico (già buono, non va riaddestrato per ora).
- Pallone → YOLO fine-tuned solo sulla classe `ball`.

Motivo: i dataset pubblici di futsal annotano spesso *solo* il pallone; addestrare
un modello unico su di essi insegnerebbe a ignorare le persone.

**Dataset base:** ScaptureSports "Futsal Ball Detection" v3, Roboflow Universe,
~4.8k immagini da camera fissa indoor, licenza **CC BY 4.0** (citare la fonte se
si ridistribuisce il modello). Scaricato in `datasets/` (ignorato da git).

**Dove addestrare:**
- Mac M5 (MPS): solo prove rapide (`--fraction 0.1 --epochs 5`, ~10 min).
- RunPod GPU a ore per l'addestramento vero: vedi `docs/RUNPOD_TRAINING.md` e
  `tools/runpod/train_on_runpod.sh`. Conviene caricare il dataset già scaricato
  (tar ~1.6 GB) invece di usare la chiave Roboflow sul pod.

**Configurazioni:**
| Nome | Base | imgsz | Epoche | Uso |
|---|---|---|---|---|
| A (veloce) | yolo11s | 960 | 40 | Baseline, inferenza rapida sul Mac |
| B (potente) | yolo11m | 1280 | fino a 80, patience 20 | Miglior recall su pallone lontano/4K; più lento in inferenza |

Augment: `fliplr=0.5`, `flipud=0` (niente capovolgimenti verticali), mosaic con
`close_mosaic` nelle ultime epoche.

**Lezioni RunPod (30/09/2026):**
- Aprire la porta SSH: `--ports "22/tcp"`, altrimenti il pod non è raggiungibile.
- Le GPU in UE possono essere esaurite; il dataset pubblico può andare anche su pod USA,
  i **video propri con atleti/minori solo su datacenter UE** e cancellati a fine sessione.
- `pip install ultralytics` sui template RunPod richiede `--break-system-packages` e
  **va vincolato alla versione di torch dell'immagine**, altrimenti torch viene aggiornato
  a una build CUDA più nuova del driver (GPU non vista, errori cuDNN). Lo script lo fa già.
- Impostare sempre un limite di spesa: il Mac elimina il pod dopo un tempo massimo
  (`sleep N; runpodctl pod delete <id>` con `caffeinate`), oltre alla chiusura manuale.
- Una L40S (48 GB) regge due addestramenti in parallelo (A + B).

**Come valutare un nuovo modello:** stessa clip di regressione
(`recordings/online_gopro_futsal_angolo.mp4`, primi 300 frame), stessa
configurazione persone; confrontare palloni rilevati **e** verificare a occhio i
falsi positivi (il conteggio da solo non basta).

## 3. Roadmap

**Adesso (P0)**
1. Completare A/B su RunPod, scegliere il modello pallone per le riprese reali.
2. Registrare le sessioni iPhone + GoPro (piano nel brief di progetto, sez. 7).
3. Annotare 300–500 frame **dei propri video** (pallone + persone, anche portiere/arbitro
   se utile) e fare un secondo fine-tuning partendo dal modello pallone migliore:
   è il passo che adatta il modello a palestra, luci e camere reali.
4. Congelare clip di regressione per scenario (passaggi, rondo, partitella).

**Poi (P1)**
5. Stabilità ID: tarare ByteTrack/BoT-SORT, filtro ROI campo (panchine, pubblico).
6. Tracking del pallone con interpolazione breve dove il rilevamento manca.
7. Calibrazione nel web con landmark manuali e validazione hold-out.
8. Squadre da colore casacca.

**Dopo (P2)**
9. Minimappa/heatmap solo con calibrazione validata; metriche con incertezza.
10. Licenze: Ultralytics (AGPL-3.0 / Enterprise) prima di qualsiasi distribuzione commerciale.
