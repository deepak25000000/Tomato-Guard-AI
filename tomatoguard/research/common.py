"""Shared helpers for the TomatoGuard research pipeline (no torch import here)."""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SEED = 42


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def dhash_variants(path, size: int = 8) -> list[int]:
    """64-bit difference hash of the image plus its h-flip, v-flip and 180-degree rotation.
    Comparing against flipped variants catches duplicates created by simple augmentation."""
    with Image.open(path) as im:
        gray = ImageOps.exif_transpose(im).convert("L")
    out = []
    for variant in (gray, ImageOps.mirror(gray), ImageOps.flip(gray), gray.rotate(180)):
        small = variant.resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(small, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        value = 0
        for b in bits:
            value = (value << 1) | int(b)
        out.append(value)
    return out


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(np.bitwise_xor(a[:, None], b[None, :]))
    return _POP[x.view(np.uint8).reshape(x.shape + (8,))].sum(-1)


def near_duplicate_groups(hashes: np.ndarray, threshold: int = 5, block: int = 256) -> np.ndarray:
    """hashes: (n, 4) uint64 array (original + 3 flipped variants).
    Images whose original hash is within `threshold` bits of ANY variant of another image
    are linked; connected components become groups. Returns a group id per image."""
    n = len(hashes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    orig = hashes[:, 0]
    for start in range(0, n, block):
        rows = orig[start:start + block]
        best = None
        for v in range(hashes.shape[1]):
            d = _hamming_block(rows, hashes[:, v])
            best = d if best is None else np.minimum(best, d)
        ii, jj = np.where(best <= threshold)
        for i, j in zip(ii + start, jj):
            if i != j:
                ri, rj = find(int(i)), find(int(j))
                if ri != rj:
                    parent[rj] = ri
    roots = np.array([find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids


def assert_disjoint(frames: dict, col: str) -> None:
    """Raise if any value of `col` appears in more than one split."""
    names = list(frames)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = set(frames[names[i]][col]) & set(frames[names[j]][col])
            if overlap:
                raise AssertionError(
                    f"LEAKAGE: {len(overlap)} '{col}' value(s) shared by "
                    f"{names[i]} and {names[j]}, e.g. {sorted(overlap)[:3]}"
                )
