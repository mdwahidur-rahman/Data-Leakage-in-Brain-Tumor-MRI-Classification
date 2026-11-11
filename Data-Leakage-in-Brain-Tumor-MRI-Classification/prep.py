"""Data preparation for the brain-tumor MRI study (public 253-image dataset).

1. Read the 253 distinct images (top-level yes/no folders) and build duplicate-aware groups:
   exact byte duplicates (md5) are merged with dihedral-canonical perceptual-hash near-duplicates
   (8x8 dHash over the 8 flips/rotations, Hamming distance <= 4, single linkage).
   Groups that mix tumor and no-tumor labels cannot be labeled reliably and are excluded.
2. Build three deterministic preprocessing variants of every original image (224 x 224, uint8):
      raw   : pad-to-square + resize
      clahe : raw + CLAHE (clip 2.0, 8x8 tiles)
      full  : brain crop (Otsu + largest contour) + pad-to-square + resize + non-local-means denoising + CLAHE
3. Create A seeded augmentations per image and variant (geometric + photometric). Augmented rows are
   TRAINING-ONLY material: the evaluation code never uses an augmented row of a test group.
Outputs (OUT directory): meta.csv, images_<variant>.npy, groups.csv, prep_summary.json
"""
import os, sys, json, hashlib, collections
import numpy as np, pandas as pd, cv2
from PIL import Image

SRC = "./dataset"                  # unzipped archive (top-level yes/ and no/)
OUT = "./data"; os.makedirs(OUT, exist_ok=True)
SIZE = 224; A = int(os.environ.get("NAUG", 12)); VARIANTS = ["raw", "clahe", "full"]


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def dhash_bits(a, size=8):
    im = cv2.resize(a, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    return (im[:, 1:] > im[:, :-1]).flatten()


def canon_hash(a):
    best = None
    for k in range(4):
        for fl in (False, True):
            t = np.rot90(a, k); t = np.fliplr(t) if fl else t
            b = np.packbits(dhash_bits(np.ascontiguousarray(t))).tobytes()
            best = b if best is None or b < best else best
    return best


def ham(a, b):
    return bin(int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).count("1")


def _small(g, k=64):
    return cv2.resize(g, (k, k), interpolation=cv2.INTER_AREA).astype(np.float32)


def best_ncc(a, b):
    """Best normalized cross-correlation of two images over the 8 flips/rotations (64x64 grayscale)."""
    a = _small(a); x = a.flatten() - a.mean(); best = -1.0
    for k in range(4):
        for fl in (False, True):
            t = np.rot90(_small(b), k); t = np.fliplr(t) if fl else t; y = t.flatten() - t.mean(); d = float(np.sqrt((x @ x) * (y @ y)))
            best = max(best, float(x @ y) / d if d > 0 else 0.0)
    return best


def pad_square_resize(g, size=SIZE):
    h, w = g.shape; s = max(h, w); top = (s - h) // 2; left = (s - w) // 2
    sq = cv2.copyMakeBorder(g, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_CUBIC)


def brain_crop(g, margin=0.04):
    blur = cv2.GaussianBlur(g, (5, 5), 0); _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.erode(th, None, iterations=2); th = cv2.dilate(th, None, iterations=2)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return g
    x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    if w * h < 0.25 * g.shape[0] * g.shape[1]: return g                    # implausible mask: keep the full image
    mx, my = int(margin * g.shape[1]), int(margin * g.shape[0])
    return g[max(0, y - my): y + h + my, max(0, x - mx): x + w + mx]


_CL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def preprocess(g, variant):
    if variant == "raw": return pad_square_resize(g)
    if variant == "clahe": return _CL.apply(pad_square_resize(g))
    x = pad_square_resize(brain_crop(g))
    x = cv2.fastNlMeansDenoising(x, None, h=5, templateWindowSize=7, searchWindowSize=21)
    return _CL.apply(x)


def augment(x, rng):
    """Seeded geometric + photometric augmentation of a preprocessed 224x224 uint8 image."""
    h, w = x.shape; ang = rng.uniform(-20, 20); sc = rng.uniform(0.9, 1.1); tx, ty = rng.uniform(-0.08, 0.08, 2) * w
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc); M[:, 2] += (tx, ty)
    y = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if rng.random() < 0.5: y = y[:, ::-1]
    y = y.astype(np.float32); y = y * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15)
    y = 255.0 * np.power(np.clip(y, 0, 255) / 255.0, rng.uniform(0.8, 1.25))
    if rng.random() < 0.3: y = cv2.GaussianBlur(y, (0, 0), rng.uniform(0.5, 1.0))
    y = y + rng.normal(0, rng.uniform(0, 6), y.shape)
    return np.clip(y, 0, 255).astype(np.uint8)


def main():
    rows = []
    for cls, lab in (("yes", 1), ("no", 0)):
        for f in sorted(os.listdir(f"{SRC}/{cls}")):
            p = f"{SRC}/{cls}/{f}"; g = load_gray(p); rows.append(dict(file=f"{cls}/{f}", label=lab, md5=hashlib.md5(open(p, "rb").read()).hexdigest(), canon=canon_hash(g), gray=g))
    n = len(rows); par = list(range(n))

    def find(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    by_md5 = collections.defaultdict(list)
    for i, r in enumerate(rows): by_md5[r["md5"]].append(i)
    for v in by_md5.values():
        for j in v[1:]: par[find(j)] = find(v[0])
    cand = confirmed = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ham(rows[i]["canon"], rows[j]["canon"]) <= 8:                     # the hash proposes a candidate pair ...
                cand += 1
                if best_ncc(rows[i]["gray"], rows[j]["gray"]) >= 0.90:          # ... and image correlation must confirm it
                    confirmed += 1; par[find(i)] = find(j)
    print("near-duplicate candidates (Hamming<=8):", cand, "| confirmed by correlation >= 0.90:", confirmed, flush=True)
    root = [find(i) for i in range(n)]; members = collections.defaultdict(list)
    for i, r in enumerate(root): members[r].append(i)
    drop = {i for m in members.values() if len({rows[k]["label"] for k in m}) > 1 for i in m}
    keep = [i for i in range(n) if i not in drop]; gid = {r: k for k, r in enumerate(sorted({root[i] for i in keep}))}
    G = pd.DataFrame([dict(img=k, file=rows[i]["file"], label=rows[i]["label"], group=gid[root[i]]) for k, i in enumerate(keep)])
    G.to_csv(f"{OUT}/groups.csv", index=False)
    summ = dict(n_files=n, n_excluded_label_conflict=len(drop), excluded_files=[rows[i]["file"] for i in sorted(drop)], n_images=len(keep), n_tumor=int(G.label.sum()), n_no_tumor=int((G.label == 0).sum()),
                n_groups=int(G.group.nunique()), groups_gt1=int((G.groupby("group").size() > 1).sum()), largest_group=int(G.groupby("group").size().max()), augmentations_per_image=A, size=SIZE)
    json.dump(summ, open(f"{OUT}/prep_summary.json", "w"), indent=1); print(json.dumps(summ))
    meta = pd.DataFrame([dict(row=k * (A + 1) + a, img=k, aug=a, file=G.file[k], label=G.label[k], group=G.group[k]) for k in range(len(keep)) for a in range(A + 1)]); meta.to_csv(f"{OUT}/meta.csv", index=False)
    for v in VARIANTS:
        arr = np.zeros((len(keep) * (A + 1), SIZE, SIZE), np.uint8)
        for k, i in enumerate(keep):
            x = preprocess(rows[i]["gray"], v); arr[k * (A + 1)] = x
            for a in range(1, A + 1): arr[k * (A + 1) + a] = augment(x, np.random.default_rng(10007 * k + 31 * a + {"raw": 1, "clahe": 2, "full": 3}[v]))
        np.save(f"{OUT}/images_{v}.npy", arr); print("saved", v, arr.shape, flush=True)


if __name__ == "__main__":
    main()
