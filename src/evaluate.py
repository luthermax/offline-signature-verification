"""Évaluation du modèle sur des personnes jamais vues (jeu de test).

Lancement (depuis la racine du projet) :
    python src/evaluate.py
    python src/evaluate.py --model models/best_model.pt --split test

Métriques :
    FAR = % de fausses signatures acceptées
    FRR = % de vraies signatures rejetées
    EER = point où FAR = FRR (plus il est bas, mieux c'est)
"""
import argparse
import random
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de fenêtre : on enregistre l'image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import index_signatures, split_by_person
from model import SiameseNetwork
from preprocessing import preprocess_signature


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    p.add_argument("--model", default="models/best_model.pt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--backbone", default="resnet18", choices=["cnn", "resnet18"],
                   help="doit correspondre au modèle entraîné (anciens modèles : cnn)")
    p.add_argument("--max-pairs", type=int, default=150, help="paires max par personne et par type")
    p.add_argument("--out", default="models/distances.png")
    return p.parse_args()


@torch.no_grad()
def embed_all(model, paths, device):
    """Calcule l'embedding de chaque image UNE seule fois."""
    model.eval()
    embs = {}
    for path in paths:
        x = torch.from_numpy(preprocess_signature(path)).unsqueeze(0).unsqueeze(0).to(device)
        embs[path] = model.encoder(x)
    return embs


def build_distances(people, embs, max_pairs, seed=0):
    """Renvoie les distances : paires authentiques, falsifications expertes, faux aléatoires."""
    rng = random.Random(seed)
    ids = sorted(people)
    d = lambda a, b: F.pairwise_distance(embs[a], embs[b]).item()

    genuine, skilled, random_f = [], [], []
    for pid in ids:
        orig, forg = people[pid]["orig"], people[pid]["forg"]

        pos = list(combinations(orig, 2))
        rng.shuffle(pos)
        genuine += [d(a, b) for a, b in pos[:max_pairs]]

        sk = [(a, f) for a in orig for f in forg]
        rng.shuffle(sk)
        skilled += [d(a, f) for a, f in sk[:max_pairs]]

        others = [i for i in ids if i != pid]
        for _ in range(max_pairs):
            other = rng.choice(others)
            random_f.append(d(rng.choice(orig), rng.choice(people[other]["orig"])))

    return np.array(genuine), np.array(skilled), np.array(random_f)


def far_frr_curve(genuine, forged, n=1000):
    """FAR et FRR pour une série de seuils (distance < seuil => accepté)."""
    lo = min(genuine.min(), forged.min())
    hi = max(genuine.max(), forged.max())
    thresholds = np.linspace(lo, hi, n)
    far = np.array([(forged < t).mean() for t in thresholds])
    frr = np.array([(genuine >= t).mean() for t in thresholds])
    return thresholds, far, frr


def compute_eer(genuine, forged):
    thresholds, far, frr = far_frr_curve(genuine, forged)
    i = np.argmin(np.abs(far - frr))
    return (far[i] + frr[i]) / 2, thresholds[i]


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    people = index_signatures(args.data)
    _, val_p, test_p = split_by_person(people)
    eval_p = test_p if args.split == "test" else val_p
    print(f"Évaluation sur '{args.split}' : {len(eval_p)} personnes")

    model = SiameseNetwork(backbone=args.backbone, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    paths = [p for person in eval_p.values() for k in ("orig", "forg") for p in person[k]]
    embs = embed_all(model, paths, device)

    genuine, skilled, random_f = build_distances(eval_p, embs, args.max_pairs)
    print(f"Paires : {len(genuine)} authentiques | {len(skilled)} falsifications | {len(random_f)} faux aléatoires")

    eer_sk, thr_sk = compute_eer(genuine, skilled)
    eer_rd, thr_rd = compute_eer(genuine, random_f)
    print(f"\nEER (falsifications expertes) : {eer_sk * 100:.1f} %  | seuil : {thr_sk:.3f}")
    print(f"EER (faux aléatoires)         : {eer_rd * 100:.1f} %  | seuil : {thr_rd:.3f}")

    # FAR / FRR au seuil choisi pour les falsifications expertes
    far = (skilled < thr_sk).mean()
    frr = (genuine >= thr_sk).mean()
    print(f"\nAu seuil {thr_sk:.3f} : FAR = {far * 100:.1f} % | FRR = {frr * 100:.1f} %")
    print(f"Distance moyenne : authentiques {genuine.mean():.3f} | falsifications {skilled.mean():.3f} | aléatoires {random_f.mean():.3f}")

    # Graphique des distributions de distances
    plt.figure(figsize=(8, 4))
    plt.hist(genuine, bins=40, alpha=0.6, label="Authentiques")
    plt.hist(skilled, bins=40, alpha=0.6, label="Falsifications expertes")
    plt.hist(random_f, bins=40, alpha=0.4, label="Faux aléatoires")
    plt.axvline(thr_sk, color="k", linestyle="--", label=f"Seuil {thr_sk:.2f}")
    plt.xlabel("Distance entre les deux signatures")
    plt.ylabel("Nombre de paires")
    plt.legend()
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print(f"Graphique enregistré : {args.out}")


if __name__ == "__main__":
    main()