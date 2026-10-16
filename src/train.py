"""Entraînement du réseau siamois (v3 : choix du backbone, ResNet18 pré-entraîné).

Lancement (depuis la racine du projet) :
    python src/train.py --epochs 3                      # test rapide
    python src/train.py --backbone resnet18 --epochs 40 # entraînement complet (arrêt auto)
    python src/train.py --backbone cnn --epochs 40      # ancien petit CNN
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SignaturePairDataset, index_signatures, split_by_person
from model import ContrastiveLoss, SiameseNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    p.add_argument("--backbone", default="resnet18", choices=["cnn", "resnet18"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="défaut : 1e-4 pour resnet18, 3e-4 pour cnn")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--pairs-per-person", type=int, default=200)
    p.add_argument("--patience", type=int, default=8,
                   help="arrêt si la val loss ne s'améliore pas pendant N époques")
    p.add_argument("--workers", type=int, default=0,
                   help="processus de chargement des données (2 sur Colab)")
    p.add_argument("--no-augment", action="store_true", help="désactive l'augmentation")
    p.add_argument("--out", default="models/best_model.pt")
    return p.parse_args()


def run_epoch(model, loader, criterion, device, margin, optimizer=None):
    """Une passe sur les données. Entraîne si optimizer est fourni, sinon évalue."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            e1, e2 = model(x1, x2)
            loss = criterion(e1, e2, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Précision avec un seuil simple : distance < margin/2 => "authentique"
            dist = torch.nn.functional.pairwise_distance(e1, e2)
            pred = (dist < margin / 2).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.lr if args.lr is not None else (1e-4 if args.backbone == "resnet18" else 3e-4)
    print(f"Appareil : {device} | backbone : {args.backbone} | lr : {lr} | "
          f"augmentation : {not args.no_augment}")

    # Données
    people = index_signatures(args.data)
    train_p, val_p, _ = split_by_person(people)
    print(f"Personnes -> train: {len(train_p)}, val: {len(val_p)}")

    train_ds = SignaturePairDataset(
        train_p, args.pairs_per_person, seed=0, augment=not args.no_augment
    )
    val_ds = SignaturePairDataset(val_p, pairs_per_person=50, seed=1, augment=False)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=pin)

    # Modèle
    model = SiameseNetwork(backbone=args.backbone).to(device)
    criterion = ContrastiveLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(model.param_groups(lr), lr=lr,
                                  weight_decay=args.weight_decay)
    # Divise le learning rate par 2 quand la val loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # Nouvelles paires à chaque époque : le modèle voit plus de combinaisons
        train_ds.pairs = train_ds._build_pairs(args.pairs_per_person, seed=epoch)

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, args.margin, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, args.margin)
        scheduler.step(va_loss)

        saved = ""
        if va_loss < best_val:
            best_val = va_loss
            bad_epochs = 0
            torch.save(model.state_dict(), out_path)
            saved = "  <- meilleur modèle sauvegardé"
        else:
            bad_epochs += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Époque {epoch:02d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} | lr {cur_lr:.1e}{saved}"
        )

        if bad_epochs >= args.patience:
            print(f"Arrêt anticipé : pas d'amélioration depuis {args.patience} époques.")
            break

    print(f"Terminé. Meilleure val loss : {best_val:.4f} -> {out_path}")


if __name__ == "__main__":
    main()