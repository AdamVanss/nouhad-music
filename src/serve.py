"""Local API for the web app: HTDemucs stem separation and the Boy/Girl face CNN.

Run: python -m uvicorn src.serve:app --port 8000
"""

import base64
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from src.models.gender_cnn import CLASSES, IMAGE_SIZE, GenderCNN, normalize
from src.separate_htdemucs import load_htdemucs, separate_file

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
FINETUNE = CKPT_DIR / "htdemucs_finetune.pt"
GENDER = CKPT_DIR / "gender_cnn.pt"
JOBS = ROOT / "outputs" / "app"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".aif", ".aiff"}
MAX_AUDIO_MB = 200
MAX_IMAGE_MB = 15

JOBS.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Stems and faces")
app.mount("/files", StaticFiles(directory=JOBS), name="files")

_separator = None
_separator_name = "htdemucs fine-tune" if FINETUNE.is_file() else "htdemucs pretrained"
_separator_lock = threading.Lock()
_gender = None
_cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))


def _get_separator():
    global _separator
    if _separator is None:
        # a fine-tune is only written when it beat pretrained HTDemucs on validation
        _separator = load_htdemucs(FINETUNE if _separator_name.endswith("fine-tune") else None, device)
    return _separator


def _get_gender():
    global _gender
    if _gender is None:
        if not GENDER.is_file():
            raise HTTPException(503, "Face model is not trained yet.")
        ckpt = torch.load(GENDER, map_location="cpu", weights_only=False)
        model = GenderCNN()
        model.load_state_dict(ckpt["model"])
        _gender = (model.to(device).eval(), ckpt)
    return _gender


@app.get("/health")
def health() -> dict:
    gender = torch.load(GENDER, map_location="cpu", weights_only=False) if GENDER.is_file() else None
    return {
        "device": str(device),
        "separator": _separator_name,
        "face_model": None if gender is None else {"val_accuracy": gender["val_accuracy"]},
    }


@app.post("/separate")
def separate(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_EXT:
        raise HTTPException(400, f"Unsupported audio type '{suffix or 'none'}'.")
    data = file.file.read()
    if len(data) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(413, f"Audio is larger than {MAX_AUDIO_MB} MB.")
    job = uuid.uuid4().hex[:12]
    job_dir = JOBS / job
    job_dir.mkdir(parents=True)
    source = job_dir / f"input{suffix}"
    source.write_bytes(data)
    try:
        with _separator_lock:
            paths = separate_file(_get_separator(), source, job_dir, progress=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Could not separate this file: {exc}") from exc
    finally:
        source.unlink(missing_ok=True)
    stems = {path.stem.split("_")[-1]: f"/files/{job}/{path.name}" for path in paths}
    return {
        "job": job,
        "model": _separator_name,
        "stems": {name: stems[name] for name in ("vocals", "drums", "bass", "other")},
    }


def _crop_face(image: np.ndarray) -> tuple[np.ndarray, list[int]] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # UTKFace chips include forehead, hair edge and chin; the Haar box is tighter
    side = int(max(w, h) * 1.45)
    cx, cy = x + w / 2, y + h / 2 - h * 0.05
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    height, width = image.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - width, y0 + side - height)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    crop = image[y0 + pad : y0 + pad + side, x0 + pad : x0 + pad + side]
    return crop, [int(x), int(y), int(w), int(h)]


@app.post("/classify")
def classify(file: UploadFile = File(...)) -> dict:
    data = file.file.read()
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise HTTPException(413, f"Image is larger than {MAX_IMAGE_MB} MB.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "That file is not an image this server can read.")
    found = _crop_face(image)
    if found is None:
        raise HTTPException(422, "No face found. Use a clear, front-facing photo.")
    crop, box = found
    model, ckpt = _get_gender()
    rgb = cv2.cvtColor(cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        x = normalize(x)
        logits = model(x) + model(x.flip(-1))
        probs = torch.softmax(logits / 2, dim=1)[0].cpu().tolist()
    ok, jpg = cv2.imencode(".jpg", cv2.resize(crop, (256, 256)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "label": CLASSES[int(np.argmax(probs))],
        "probabilities": {name: round(p, 4) for name, p in zip(CLASSES, probs)},
        "box": box,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "crop": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode() if ok else None,
        "val_accuracy": ckpt.get("val_accuracy"),
    }
