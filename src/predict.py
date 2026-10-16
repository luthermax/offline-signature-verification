"""Prédiction : compare une signature à vérifier avec une ou plusieurs références.

Utilisation en ligne de commande (depuis la racine du projet) :
    python src/predict.py signature_a_verifier.png --reference ref1.png ref2.png
    python src/predict.py signature_a_verifier.png --reference ref1.png --rotation quarter

Utilisation dans du code (API, démo) :
    verifier = SignatureVerifier("models/best_model_v3.pt")
    resultat = verifier.verify(image_a_verifier, [ref1, ref2], rotation="off")

Tolérance à la rotation (`rotation`) :
    "off"     : la signature est supposée dans le même sens que les références (défaut)
    "quarter" : essaie 0°, 90°, 180°, 270° et garde la meilleure correspondance
    "full"    : essaie 24 angles (pas de 15°) et garde la meilleure correspondance

Attention : plus on essaie d'angles, plus une falsification a de chances de
ressembler à une référence par hasard. Ces modes n'ont pas été évalués sur CEDAR.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import SiameseNetwork
from preprocessing import preprocess_array, preprocess_rotations, preprocess_signature

DEFAULT_MODEL = "models/best_model_v3.pt"
# Seuil au point EER (falsifications expertes) mesuré sur le jeu de test : 0.489
DEFAULT_THRESHOLD = 0.489

ROTATION_PRESETS = {
    "off": [0],
    "quarter": [0, 90, 180, 270],
    "full": list(range(-180, 180, 15)),
}


def to_tensor(image, device):
    """Chemin de fichier OU tableau numpy -> tenseur (1, 1, H, W)."""
    if isinstance(image, (str, Path)):
        arr = preprocess_signature(image)
    else:
        arr = preprocess_array(image)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


class SignatureVerifier:
    def __init__(self, model_path=DEFAULT_MODEL, backbone="resnet18",
                 threshold=DEFAULT_THRESHOLD, device=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self.model = SiameseNetwork(backbone=backbone, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def _embed(self, image):
        return self.model.encoder(to_tensor(image, self.device))

    @torch.no_grad()
    def _embed_rotations(self, image, angles):
        """Embeddings de la signature pour chaque angle : tenseur (nb_angles, dim)."""
        arrays = preprocess_rotations(image, angles)
        batch = torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(self.device)
        return self.model.encoder(batch)

    @torch.no_grad()
    def verify(self, candidate, references, threshold=None, rotation="off"):
        """Compare `candidate` à chaque référence ; décision sur la distance moyenne.

        Plus il y a de références, plus la décision est fiable.
        """
        thr = self.threshold if threshold is None else threshold
        if thr <= 0:
            raise ValueError("Le seuil doit être strictement positif.")
        if not references:
            raise ValueError("Il faut au moins une signature de référence.")
        if rotation not in ROTATION_PRESETS:
            raise ValueError(f"rotation inconnue : {rotation} (choisir parmi {list(ROTATION_PRESETS)})")

        angles = ROTATION_PRESETS[rotation]
        cands = self._embed_rotations(candidate, angles)  # (A, D)

        distances, best_angles = [], []
        for ref in references:
            ref_emb = self._embed(ref).expand(len(angles), -1)
            d = F.pairwise_distance(cands, ref_emb)  # une distance par angle
            best = int(torch.argmin(d))
            distances.append(d[best].item())
            best_angles.append(angles[best])

        distance = float(np.mean(distances))

        # Score indicatif : 50 % exactement au seuil, 100 % = identiques, 0 % = très différentes
        similarity = float(np.clip(1.0 - distance / (2.0 * thr), 0.0, 1.0)) * 100.0

        return {
            "accepted": bool(distance < thr),
            "distance": round(distance, 4),
            "threshold": round(float(thr), 4),
            "similarity": round(similarity, 1),
            "n_references": len(references),
            "distances": [round(d, 4) for d in distances],
            "rotation": rotation,
            "angles": best_angles,
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("candidate", help="signature à vérifier")
    p.add_argument("--reference", nargs="+", required=True, help="une ou plusieurs signatures authentiques")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--backbone", default="resnet18", choices=["cnn", "resnet18"])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--rotation", default="off", choices=list(ROTATION_PRESETS))
    args = p.parse_args()

    verifier = SignatureVerifier(args.model, args.backbone, args.threshold)
    res = verifier.verify(args.candidate, args.reference, rotation=args.rotation)

    verdict = "AUTHENTIQUE" if res["accepted"] else "REJETÉE (possible falsification)"
    print(f"Résultat : {verdict}")
    print(f"Distance : {res['distance']} (seuil : {res['threshold']}) | score indicatif : {res['similarity']} %")
    print(f"Références utilisées : {res['n_references']} | distances : {res['distances']}")
    if args.rotation != "off":
        print(f"Rotation testée : {args.rotation} | meilleurs angles : {res['angles']}")


if __name__ == "__main__":
    main()