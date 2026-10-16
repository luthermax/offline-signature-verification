"""API FastAPI de vérification de signature.

Lancement (depuis la racine du projet) :
    uvicorn api.main:app --reload
Puis ouvrir http://127.0.0.1:8000/docs : chaque signature se charge dans son propre champ.

Exemple avec curl :
    curl -X POST http://127.0.0.1:8000/verify \
      -F "candidate=@a_verifier.png" \
      -F "reference_1=@ref1.png" \
      -F "reference_2=@ref2.png"
"""
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from predict import ROTATION_PRESETS, SignatureVerifier  # noqa: E402

MODEL_PATH = os.getenv("SIGNATURE_MODEL", str(ROOT / "models" / "best_model_v3.pt"))
THRESHOLD = float(os.getenv("SIGNATURE_THRESHOLD", "0.489"))
MAX_BYTES = 5 * 1024 * 1024  # 5 Mo par image

app = FastAPI(
    title="Vérification de signature",
    version="0.2",
    description=(
        "Compare une signature à vérifier avec 1 à 3 signatures de référence authentiques. "
        "Prototype entraîné sur CEDAR : ne pas utiliser pour sécuriser des opérations réelles."
    ),
)
verifier = SignatureVerifier(MODEL_PATH, threshold=THRESHOLD)


async def read_image(upload: UploadFile):
    """Lit une image envoyée et la renvoie en tableau numpy (gris, RGB ou RGBA)."""
    data = await upload.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{upload.filename} dépasse 5 Mo.")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, f"{upload.filename} n'est pas une image lisible.")

    # OpenCV décode en BGR(A) : on repasse en RGB(A)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def is_provided(upload) -> bool:
    """Vrai si un fichier a réellement été envoyé (un champ vide n'en est pas un)."""
    return upload is not None and not isinstance(upload, str) and bool(upload.filename)


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD}


@app.post("/verify")
async def verify(
    candidate: UploadFile = File(..., description="Signature à vérifier"),
    reference_1: UploadFile = File(..., description="Signature de référence authentique n°1"),
    reference_2: Optional[UploadFile] = File(None, description="Référence n°2 (facultative)"),
    reference_3: Optional[UploadFile] = File(None, description="Référence n°3 (facultative)"),
    rotation: str = Form("off", description="Tolérance à la rotation : off, quarter (0/90/180/270°) ou full (pas de 15°)"),
    threshold: Optional[float] = Form(None, description="Seuil de décision. Laisser vide ou 0 pour le seuil par défaut du serveur."),
):
    if rotation not in ROTATION_PRESETS:
        raise HTTPException(400, f"rotation doit valoir : {', '.join(ROTATION_PRESETS)}.")

    # La page /docs préremplit les champs numériques avec 0 : on le lit comme « seuil par défaut »
    if threshold is not None:
        if threshold == 0:
            threshold = None
        elif not 0 < threshold <= 2:
            raise HTTPException(400, "threshold doit être vide, 0 (seuil par défaut) ou compris entre 0 et 2.")

    cand = await read_image(candidate)
    refs = [await read_image(f) for f in (reference_1, reference_2, reference_3) if is_provided(f)]
    try:
        return verifier.verify(cand, refs, threshold=threshold, rotation=rotation)
    except ValueError as exc:
        raise HTTPException(400, str(exc))