"""Prétraitement des signatures : gris -> éclairage -> binarisation -> recadrage -> redimensionnement.

Entrées possibles :
    preprocess_signature(chemin)      : depuis un fichier image
    preprocess_array(tableau)         : depuis un tableau numpy (upload API, Gradio, webcam...)
    preprocess_rotations(image, ...)  : plusieurs rotations d'une même signature (vérification)

Le paramètre `angle` (degrés) fait tourner la signature APRÈS binarisation et recadrage.
Par défaut angle=0 : le comportement de l'entraînement n'est pas modifié.
"""
from pathlib import Path

import cv2
import numpy as np

IMG_H, IMG_W = 155, 220  # taille finale (hauteur, largeur)
MAX_SIDE = 1000          # les photos de téléphone sont réduites avant traitement


def _limit_size(gray):
    """Réduit les très grandes images (photos) : plus rapide, même résultat final."""
    h, w = gray.shape
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return gray


def _remove_shadows(gray):
    """Normalise l'éclairage : le fond devient uniformément blanc.

    Indispensable pour les photos (ombres, dégradés de lumière) : sans cela,
    la binarisation prend les ombres pour de l'encre.
    """
    h, w = gray.shape
    k = max(7, (min(h, w) // 30) | 1)
    bg = cv2.dilate(gray, np.ones((k, k), np.uint8))  # efface les traits foncés
    bg = cv2.medianBlur(bg, (k * 2) | 1)               # fond estimé, lissé
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def _remove_specks(binary):
    """Supprime les petites taches isolées (grain du papier, poussière)."""
    h, w = binary.shape
    min_area = max(8, int(5e-5 * h * w))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep if keep.any() else binary


def _to_gray_uint8(img):
    """Convertit un tableau (gris, RGB ou RGBA) en niveaux de gris uint8."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        if img.size and img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 3:
        if img.shape[2] == 4:  # RGBA : le transparent devient blanc
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        elif img.shape[2] == 1:
            img = img[:, :, 0]
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _crop_to_ink(binary):
    coords = cv2.findNonZero((binary > 32).astype(np.uint8))
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    return binary[y:y + h, x:x + w]


def _binarize_crop(gray):
    """Gris uint8 -> signature binaire recadrée (encre = blanc, fond = noir)."""
    gray = _limit_size(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = _remove_shadows(gray)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = _remove_specks(binary)
    return _crop_to_ink(binary)


def _rotate_crop(binary, angle):
    """Tourne la signature binaire (fond noir) de `angle` degrés, puis recadre."""
    if angle % 90 == 0:
        out = np.ascontiguousarray(np.rot90(binary, int(angle // 90) % 4))
    else:
        h, w = binary.shape
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos) + 1
        new_h = int(h * cos + w * sin) + 1
        M[0, 2] += new_w / 2.0 - center[0]
        M[1, 2] += new_h / 2.0 - center[1]
        out = cv2.warpAffine(binary, M, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderValue=0)
    return _crop_to_ink(out)


def _fit(binary, size=(IMG_H, IMG_W)):
    """Redimensionne en gardant les proportions, centre sur un canevas, normalise."""
    target_h, target_w = size
    h, w = binary.shape
    scale = min(target_h / h, target_w / w)
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas.astype(np.float32) / 255.0  # encre ~1, fond 0


def _pipeline(gray, size=(IMG_H, IMG_W), angle=0):
    binary = _binarize_crop(gray)
    if angle % 360 != 0:
        binary = _rotate_crop(binary, angle)
    return _fit(binary, size)


def _load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image illisible : {path}")
    return img


def preprocess_signature(path, size=(IMG_H, IMG_W), angle=0):
    """Charge une image depuis un fichier et la prétraite."""
    return _pipeline(_load_gray(path), size, angle)


def preprocess_array(img, size=(IMG_H, IMG_W), angle=0):
    """Prétraite une image déjà en mémoire (gris, RGB ou RGBA)."""
    return _pipeline(_to_gray_uint8(img), size, angle)


def preprocess_rotations(image, angles, size=(IMG_H, IMG_W)):
    """Prétraite UNE seule fois, puis renvoie une version par angle (liste de (H, W))."""
    if isinstance(image, (str, Path)):
        gray = _load_gray(image)
    else:
        gray = _to_gray_uint8(image)
    binary = _binarize_crop(gray)
    return [
        _fit(_rotate_crop(binary, a) if a % 360 != 0 else binary, size)
        for a in angles
    ]


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    out = preprocess_signature(sys.argv[1])
    plt.imshow(out, cmap="gray")
    plt.title(f"Prétraité : {out.shape}")
    plt.show()