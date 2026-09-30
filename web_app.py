"""Local browser interface for the first CyberFutsal AI Mac video test.

Run from repository root: python -m streamlit run web_app.py
This intentionally calls the standalone generic COCO smoke test: no proprietary
weights or full futsal metrics are implied. All files stay on the local Mac.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "outputs" / "web_sessions"
RECORDINGS = ROOT / "recordings"
SUPPORTED = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
MODELS = {
    "YOLO11 Nano, veloce (primo test)": "yolo11n.pt",
    "YOLO11 Small, più pesante": "yolo11s.pt",
    "YOLO11 Medium, migliore su oggetti piccoli (lento)": "yolo11m.pt",
}
CAMERA_TIPS = {
    "Laterale": "Posiziona l'iPhone rialzato, se possibile, e mantieni l'intero campo nell'inquadratura.",
    "Dietro la porta": "Evita che la rete copra gran parte dell'inquadratura: può compromettere rilevamenti e geometria.",
    "Angolo": "Mantieni ben visibili più riferimenti del campo, soprattutto quelli lontani.",
    "Altro": "Mantieni iPhone, inquadratura e zoom fermi durante la clip.",
}


def gpu_status() -> tuple[bool, str]:
    try:
        import torch
        return bool(torch.backends.mps.is_available()), str(torch.__version__)
    except Exception as exc:
        return False, f"PyTorch non disponibile: {exc}"


def read_metrics(output: str) -> dict:
    """Parse the stable stdout contract of tools/mac_smoke_test.py."""
    found = {}
    patterns = {
        "device": r"Device:\s*(\w+)",
        "frames": r"Frames:\s*(\d+)",
        "throughput": r"throughput:\s*([\d.]+)\s*FPS",
        "people": r"Person detections:\s*(\d+)",
        "ball": r"sports-ball detections:\s*(\d+)",
        "tiled_balls": r"from \d+x\d+ tiles:\s*(\d+)",
        "ids": r"Distinct tracked IDs \(not distinct people\):\s*(\d+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            found[name] = match.group(1)
    return found


def save_upload(uploaded, directory: Path) -> Path:
    """Avoid user-controlled paths and copy large uploads in small chunks."""
    ext = Path(uploaded.name).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError("Formato video non supportato.")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"video{ext}"
    uploaded.seek(0)
    with path.open("wb") as stream:
        while chunk := uploaded.read(4 * 1024 * 1024):
            stream.write(chunk)
    return path


def local_source(value: str) -> Path:
    """Resolve a user-selected local path without running shell commands."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if path.suffix.lower() not in SUPPORTED or not path.is_file():
        raise ValueError("Video non trovato o formato non supportato. Controlla il percorso.")
    return path.resolve()


def run_analysis(source: Path, output: Path, model: str, device: str,
                 max_frames: int, image_size: int, person_conf: float,
                 ball_conf: float, ball_tiles: int) -> subprocess.CompletedProcess:
    command = [
        sys.executable, str(ROOT / "tools" / "mac_smoke_test.py"),
        "--source", str(source), "--output", str(output),
        "--model", model, "--device", device,
        "--max-frames", str(max_frames), "--imgsz", str(image_size),
        "--person-conf", str(person_conf), "--ball-conf", str(ball_conf),
        "--ball-tiles", str(ball_tiles),
    ]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          timeout=7200, check=False)


st.set_page_config(page_title="CyberFutsal AI | Analisi locale", page_icon="⚽",
                   layout="wide")
st.title("⚽ CyberFutsal AI")
st.caption("Prototipo locale · Analisi video offline · Nessun upload su cloud")
st.info(
    "Prima prova: il modello YOLO generico rileva persone e palloni sportivi. "
    "Non è ancora la pipeline completa di match analysis: metriche atletiche, "
    "calibrazione web e acquisizione live arriveranno dopo la validazione dei video reali."
)

available, torch_version = gpu_status()
left, right = st.columns([1, 1])
with left:
    if available:
        st.success(f"GPU Apple MPS disponibile · PyTorch {torch_version}")
    else:
        st.warning(f"GPU MPS non disponibile, puoi usare CPU. {torch_version}")
with right:
    st.caption("I filmati, i risultati e i modelli restano nel tuo Mac. "
               "Gli eventuali pesi YOLO vengono scaricati solo al primo utilizzo.")

st.subheader("1. Scegli il filmato")
RECORDINGS.mkdir(exist_ok=True)
source_mode = st.radio("Sorgente video",
                       ["Cartella recordings", "Carica dal browser", "Altro percorso sul Mac"],
                       horizontal=True)
uploaded = None
path_input = ""
if source_mode == "Cartella recordings":
    videos = sorted((p for p in RECORDINGS.rglob("*")
                     if p.is_file() and p.suffix.lower() in SUPPORTED),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    st.caption(f"Copia i video (anche file grandi, niente limite di upload) in: `{RECORDINGS}`")
    if st.button("🔄 Aggiorna elenco"):
        st.rerun()
    if videos:
        chosen = st.selectbox(
            "Video trovati (più recenti in alto)", videos,
            format_func=lambda p: f"{p.relative_to(RECORDINGS)} · {p.stat().st_size / 1024**2:.0f} MB")
        path_input = str(chosen)
    else:
        st.warning("Nessun video nella cartella recordings. Copiaci un file e premi «Aggiorna elenco».")
elif source_mode == "Carica dal browser":
    uploaded = st.file_uploader("Seleziona MP4, MOV, M4V, AVI o MKV",
                                type=["mp4", "mov", "m4v", "avi", "mkv"])
    if uploaded is not None:
        st.caption(f"File: {uploaded.name} · {uploaded.size / (1024**2):.1f} MB")
else:
    path_input = st.text_input("Percorso del video", value="recordings/allenamento.mp4",
                               help="Puoi incollare un percorso assoluto o usarne uno relativo al progetto.")

st.subheader("2. Impostazioni")
a, b, c = st.columns(3)
with a:
    selected_model = st.selectbox("Modello AI", list(MODELS))
    device = st.selectbox("Elaborazione", ["mps", "cpu"] if available else ["cpu"])
with b:
    limit_label = st.selectbox("Durata da analizzare",
                               ["Primi 300 frame (test rapido)",
                                "Primi 900 frame", "Video intero"])
    max_frames = {"Primi 300 frame (test rapido)": 300,
                  "Primi 900 frame": 900, "Video intero": 0}[limit_label]
with c:
    image_size = st.selectbox("Risoluzione analisi", [640, 960, 1280, 1920], index=2,
                              help="Con video 4K sotto 1280 il pallone diventa troppo piccolo. "
                                   "Valori più alti aumentano il carico sulla GPU.")

with st.expander("Soglie di confidenza (avanzate)"):
    person_conf = st.slider("Soglia persone", 0.05, 0.9, 0.25, 0.05)
    ball_conf = st.slider("Soglia pallone", 0.02, 0.9, 0.10, 0.01,
                          help="Più bassa = più palloni trovati ma più falsi positivi.")
    ball_tiles = st.selectbox("Ricerca pallone a tasselli", [0, 2, 3],
                              format_func=lambda n: "Disattivata" if n == 0 else f"Griglia {n}×{n} (più lento)",
                              help="Cerca il pallone anche su porzioni del frame a piena risoluzione. "
                                   "Utile per il pallone lontano; riquadri arancioni, senza ID.")

with st.expander("Indicazioni per riprese e campi polivalenti"):
    camera = st.selectbox("Posizione della telecamera", list(CAMERA_TIPS))
    st.write(CAMERA_TIPS[camera])
    st.write(
        "Le dimensioni reali del campo e le linee corrette andranno specificate "
        "nella fase di calibrazione: non assumiamo che le linee visibili appartengano tutte al futsal. "
        "La posizione selezionata qui è soltanto un promemoria per la ripresa."
    )

st.subheader("3. Avvia l'analisi")
if st.button("▶ Analizza il video", type="primary", use_container_width=True):
    st.session_state.pop("last_result", None)
    session = WORKSPACE / uuid4().hex
    try:
        if source_mode == "Carica dal browser":
            if uploaded is None:
                raise ValueError("Carica prima un filmato.")
            source = save_upload(uploaded, session)
        else:
            source = local_source(path_input)
            session.mkdir(parents=True, exist_ok=True)
        output = session / "video_annotato.mp4"
        with st.spinner("Analisi in corso sul Mac. Il primo avvio può scaricare i pesi del modello..."):
            result = run_analysis(source, output, MODELS[selected_model],
                                  device, max_frames, image_size,
                                  person_conf, ball_conf, ball_tiles)
        logs = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
            st.error("Analisi non completata. Consulta il log per identificare il problema.")
            with st.expander("Dettagli dell'errore", expanded=True):
                st.code(logs or f"Codice di uscita: {result.returncode}")
        else:
            st.session_state["last_result"] = {
                "output": str(output), "source": str(source),
                "metrics": read_metrics(result.stdout), "logs": logs,
            }
    except ValueError as exc:
        st.warning(str(exc))
    except subprocess.TimeoutExpired:
        st.error("L'analisi ha superato il limite di esecuzione. Prova con un segmento più breve.")
    except Exception as exc:
        st.error(f"Impossibile elaborare il video: {exc}")

report = st.session_state.get("last_result")
if report:
    output_path = Path(report["output"])
    if output_path.is_file():
        st.divider()
        st.subheader("4. Risultati")
        metrics = report["metrics"]
        cols = st.columns(4)
        cols[0].metric("Frame analizzati", metrics.get("frames", "n.d."))
        cols[1].metric("Velocità (FPS)", metrics.get("throughput", "n.d."))
        cols[2].metric("Rilevamenti persone", metrics.get("people", "n.d."))
        cols[3].metric("Rilevamenti pallone", metrics.get("ball", "n.d."))
        st.caption(
            f"Device: {metrics.get('device', 'n.d.')} · "
            f"ID di tracking generati: {metrics.get('ids', 'n.d.')} "
            "(non equivalgono al numero di giocatori unici)."
            + (f" · Palloni extra da tasselli: {metrics['tiled_balls']}"
               if "tiled_balls" in metrics else "")
        )
        st.video(str(output_path))
        with output_path.open("rb") as output_stream:
            st.download_button("⬇ Scarica il video elaborato", data=output_stream,
                               file_name="cyberfutsal_analisi.mp4", mime="video/mp4")
        with st.expander("Log tecnico"):
            st.code(report["logs"])
        st.caption(f"File salvato localmente: {output_path}")

st.divider()
st.caption("Versione di prova, non usare per valutazioni mediche o atletiche individuali. "
           "Non condividere filmati di atleti o minori senza autorizzazioni appropriate.")
