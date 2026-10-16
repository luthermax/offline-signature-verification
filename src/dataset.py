"""Création des paires de signatures pour un réseau siamois (format CEDAR).

Structure attendue :
    data/raw/full_org/original_<personne>_<n>.png
    data/raw/full_forg/forgeries_<personne>_<n>.png

Convention des labels : 1 = paire authentique (même personne, vraies signatures)
                        0 = paire différente (falsification ou autre personne)

Nouveauté : augmentation de données (petites déformations aléatoires) pour
l'entraînement, afin que le modèle ne mémorise plus les images.
"""
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing import preprocess_signature

PATTERN = re.compile(r"(original|forgeries)_(\d+)_(\d+)", re.IGNORECASE)


def index_signatures(root):
    """Renvoie {personne: {"orig": [chemins], "forg": [chemins]}}."""
    root = Path(root)
    people = defaultdict(lambda: {"orig": [], "forg": []})
    for path in sorted(root.rglob("*.png")):
        m = PATTERN.search(path.name)
        if not m:
            continue
        kind = "orig" if m.group(1).lower() == "original" else "forg"
        people[int(m.group(2))][kind].append(path)
    return dict(people)


def split_by_person(people, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Sépare train / val / test PAR PERSONNE (jamais par image)."""
    ids = sorted(people)
    random.Random(seed).shuffle(ids)
    n_test = int(len(ids) * test_ratio)
    n_val = int(len(ids) * val_ratio)
    test_ids = ids[:n_test]
    val_ids = ids[n_test:n_test + n_val]
    train_ids = ids[n_test + n_val:]
    pick = lambda sel: {i: people[i] for i in sel}
    return pick(train_ids), pick(val_ids), pick(test_ids)


def augment_signature(img, max_rot=5.0, scale_range=(0.9, 1.1),
                      max_shift=0.05, max_shear=0.1):
    """Déformation aléatoire légère d'une signature prétraitée (H, W) dans [0, 1].

    Rotation, changement d'échelle, décalage et inclinaison : une même personne
    ne signe jamais exactement pareil, le modèle doit y être robuste.
    """
    h, w = img.shape
    angle = random.uniform(-max_rot, max_rot)
    scale = random.uniform(*scale_range)
    shear = random.uniform(-max_shear, max_shear)
    tx = random.uniform(-max_shift, max_shift) * w
    ty = random.uniform(-max_shift, max_shift) * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 1] += shear                      # inclinaison horizontale
    M[0, 2] += tx - shear * h / 2         # recentrage + décalage
    M[1, 2] += ty

    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class SignaturePairDataset(Dataset):
    """Chaque élément : (img1, img2, label) avec des tenseurs (1, H, W).

    augment=True : applique une déformation aléatoire à chaque image (train seulement).
    """

    def __init__(self, people, pairs_per_person=200, seed=0, augment=False):
        self.people = people
        self.ids = sorted(people)
        self.augment = augment
        self.pairs = self._build_pairs(pairs_per_person, seed)

    def _build_pairs(self, n, seed):
        rng = random.Random(seed)
        pairs = []
        for pid in self.ids:
            orig, forg = self.people[pid]["orig"], self.people[pid]["forg"]
            if len(orig) < 2:
                continue
            for _ in range(n // 2):
                # Paire positive : deux vraies signatures de la même personne
                a, b = rng.sample(orig, 2)
                pairs.append((a, b, 1))

                # Paire négative : falsification (souvent) ou autre personne
                if forg and rng.random() < 0.7:
                    pairs.append((rng.choice(orig), rng.choice(forg), 0))
                else:
                    other = rng.choice([i for i in self.ids if i != pid])
                    pairs.append(
                        (rng.choice(orig), rng.choice(self.people[other]["orig"]), 0)
                    )
        rng.shuffle(pairs)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        a = preprocess_signature(p1)
        b = preprocess_signature(p2)
        if self.augment:
            a = augment_signature(a)
            b = augment_signature(b)
        x1 = torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0)
        x2 = torch.from_numpy(np.ascontiguousarray(b)).unsqueeze(0)
        return x1, x2, torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    people = index_signatures("data/raw")
    train, val, test = split_by_person(people)
    print(f"Personnes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")

    ds = SignaturePairDataset(train, pairs_per_person=20, augment=True)
    x1, x2, y = ds[0]
    print(f"{len(ds)} paires | x1: {tuple(x1.shape)} | label: {int(y)}")