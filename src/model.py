"""Réseau siamois : un encodeur partagé + une contrastive loss.

Convention des labels (identique à dataset.py) :
    1 = paire authentique (même personne, vraies signatures)
    0 = paire différente (falsification ou autre personne)

Deux encodeurs au choix :
    "cnn"      : petit CNN entraîné de zéro (versions v1 et v2)
    "resnet18" : ResNet18 pré-entraîné sur ImageNet, affiné sur les signatures (v3)

Les embeddings sont normalisés (norme 1) : les distances restent entre 0 et 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, kernel, pool=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SignatureEncoder(nn.Module):
    """Petit CNN entraîné de zéro : image (1, 155, 220) -> embedding normalisé."""

    def __init__(self, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32, 5),
            conv_block(32, 64, 5),
            conv_block(64, 128, 3),
            conv_block(128, 128, 3, pool=False),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x):
        emb = self.fc(self.features(x))
        return F.normalize(emb, p=2, dim=1)

    def param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr}]


class ResNetEncoder(nn.Module):
    """ResNet18 pré-entraîné : il connaît déjà traits, courbes et textures."""

    def __init__(self, emb_dim=128, pretrained=True):
        super().__init__()
        from torchvision import models  # import ici : inutile pour le "cnn"

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Identity()  # on retire le classifieur ImageNet
        self.backbone = net
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, emb_dim))

        # Statistiques ImageNet (le réseau a été entraîné avec)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = 1.0 - x                  # encre sombre sur fond clair, comme une vraie photo
        x = x.repeat(1, 3, 1, 1)     # 1 canal -> 3 canaux attendus par ResNet
        x = (x - self.mean) / self.std
        emb = self.head(self.backbone(x))
        return F.normalize(emb, p=2, dim=1)

    def param_groups(self, lr):
        # Le backbone déjà entraîné bouge doucement, la nouvelle tête apprend plus vite
        return [
            {"params": self.backbone.parameters(), "lr": lr},
            {"params": self.head.parameters(), "lr": lr * 10},
        ]


class SiameseNetwork(nn.Module):
    """Applique le MÊME encodeur aux deux images."""

    def __init__(self, emb_dim=128, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "cnn":
            self.encoder = SignatureEncoder(emb_dim)
        elif backbone == "resnet18":
            self.encoder = ResNetEncoder(emb_dim, pretrained)
        else:
            raise ValueError(f"backbone inconnu : {backbone} (choisir 'cnn' ou 'resnet18')")

    def forward(self, x1, x2):
        return self.encoder(x1), self.encoder(x2)

    def param_groups(self, lr):
        return self.encoder.param_groups(lr)

    @torch.no_grad()
    def distance(self, x1, x2):
        """Distance entre deux signatures (petite = similaires)."""
        e1, e2 = self.forward(x1, x2)
        return F.pairwise_distance(e1, e2)


class ContrastiveLoss(nn.Module):
    """Rapproche les paires authentiques, éloigne les autres d'au moins `margin`."""

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        d = F.pairwise_distance(emb1, emb2)
        loss_same = label * d.pow(2)
        loss_diff = (1 - label) * F.relu(self.margin - d).pow(2)
        return (loss_same + loss_diff).mean()


if __name__ == "__main__":
    criterion = ContrastiveLoss()
    x1 = torch.rand(4, 1, 155, 220)
    x2 = torch.rand(4, 1, 155, 220)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])

    for name in ("cnn", "resnet18"):
        model = SiameseNetwork(backbone=name, pretrained=False)  # pas de téléchargement ici
        e1, e2 = model(x1, x2)
        loss = criterion(e1, e2, y)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:9s} | embeddings : {tuple(e1.shape)} | norme : {e1.norm(dim=1)[0].item():.3f} "
              f"| loss : {loss.item():.4f} | params : {n_params:,}")