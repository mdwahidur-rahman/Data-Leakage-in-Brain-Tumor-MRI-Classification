import os, hashlib, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
B = "./dataset"; rows = []
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
# 1. are the two folder trees identical copies?
same = diff = missing = 0
for cls in ("yes", "no"):
    for f in os.listdir(f"{B}/{cls}"):
        p2 = f"{B}/brain_tumor_dataset/{cls}/{f}"
        if not os.path.exists(p2): missing += 1
        elif md5(f"{B}/{cls}/{f}") == md5(p2): same += 1
        else: diff += 1
print(f"two folder trees: identical files {same}, different {diff}, missing in copy {missing}")
def dh(a, size=8):
    im = Image.fromarray(a).resize((size + 1, size), Image.LANCZOS); x = np.asarray(im, dtype=np.int16); return (x[:, 1:] > x[:, :-1]).flatten()
def canon(img):
    best = None
    for k in range(4):
        for fl in (False, True):
            t = np.rot90(img, k); t = np.fliplr(t) if fl else t; b = np.packbits(dh(t)).tobytes(); best = b if best is None or b < best else best
    return best
for cls, lab in (("yes", 1), ("no", 0)):
    for f in sorted(os.listdir(f"{B}/{cls}")):
        p = f"{B}/{cls}/{f}"; im = Image.open(p); g = np.asarray(im.convert("L")); rows.append(dict(file=f, cls=cls, label=lab, md5=md5(p), w=im.width, h=im.height, mode=im.mode, fmt=im.format, bytes=os.path.getsize(p), mean=g.mean(), std=g.std(), canon=canon(g), dh=np.packbits(dh(g)).tobytes()))
D = pd.DataFrame(rows); print("distinct images (top-level yes/no):", len(D), "| tumor", int(D.label.sum()), "| no tumor", int((D.label == 0).sum()))
print("exact byte duplicates inside the 253:", int(D.md5.duplicated(keep=False).sum()), "files in", D.md5[D.md5.duplicated(keep=False)].nunique(), "groups; cross-class duplicates:", int(D.groupby("md5").label.nunique().gt(1).sum()))
print("formats:", D.fmt.value_counts().to_dict(), "| modes:", D["mode"].value_counts().to_dict())
print("image sizes: distinct (w,h) pairs =", D[["w", "h"]].drop_duplicates().shape[0], "| width range", D.w.min(), "-", D.w.max(), "| height range", D.h.min(), "-", D.h.max())
# 2. dihedral-canonical perceptual duplicates and near-duplicates (Hamming <= 4, single linkage)
def ham(a, b): return bin(int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).count("1")
n = len(D); cb = list(D.canon); par = list(range(n))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
for thr, name in ((0, "g0 (identical canonical hash)"), (4, "g4 (Hamming<=4)")):
    par = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if ham(cb[i], cb[j]) <= thr: par[find(i)] = find(j)
    roots = [find(i) for i in range(n)]; c = collections.Counter(roots); cross = sum(1 for r in c if D.label[[i for i in range(n) if roots[i] == r]].nunique() > 1)
    print(f"{name}: {len(c)} groups for {n} images | groups of size>1: {sum(1 for v in c.values() if v > 1)} | largest group {max(c.values())} | groups mixing tumor/no-tumor: {cross}")
# 3. metadata shortcuts: how well do file/size statistics alone separate the classes?
for col in ("w", "h", "bytes", "mean", "std"):
    D["ar"] = D.w / D.h; a = roc_auc_score(D.label, D[col]); print(f"single-feature AUC {col:6s} = {max(a, 1 - a):.3f}")
a = roc_auc_score(D.label, D.w / D.h); print(f"single-feature AUC aspect = {max(a, 1 - a):.3f}")
print("by class: median width/height/bytes:", D.groupby("cls")[["w", "h", "bytes"]].median().round(0).to_dict("index"))
print("share of images with a (w,h) pair that occurs in only one class:", round(100 * D.groupby(["w", "h"]).label.transform("nunique").eq(1).mean(), 1), "%")
D.drop(columns=["canon", "dh"]).to_csv("./results/image_audit.csv", index=False)
print("--- paper vs archive ---"); print("paper text: 1275 tumor + 1255 no tumor = 2530 images  ->  archive: 155 + 98 = 253  (ratio %.1f x; tumor %.2f x, no-tumor %.2f x)" % (2530 / 253, 1275 / 155, 1255 / 98))
print("exports: 2610 rows (1300 tumor / 1310 no tumor) = %.2f rows per archive image" % (2610 / 253), "| DenseNet export: 1690 rows (490 tumor / 1200 no tumor) = %.2f rows per archive image" % (1690 / 253))
