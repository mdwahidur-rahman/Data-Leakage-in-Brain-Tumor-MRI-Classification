"""Shared utilities: leakage-aware folds, row bookkeeping, metrics, group-cluster bootstrap.

Row layout (from prep.py): row = img * (A + 1) + aug, aug = 0 is the ORIGINAL (preprocessed) image,
aug = 1..A are seeded augmentations. The evaluation rules enforced here:
  * folds are made over IMAGES, grouped by duplicate group (StratifiedGroupKFold);
  * training rows = original + augmented rows of the training images;
  * test rows = ORIGINAL rows of the test images only (no augmented copy of a test group is ever used).
"""
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

D = "./data"; F = "./feats"; R = "./results"
import os; os.makedirs(R, exist_ok=True)
meta = pd.read_csv(f"{D}/meta.csv"); A = int(meta.aug.max())
IMG = meta[meta.aug == 0].reset_index(drop=True)[["img", "label", "group", "file"]]
N_IMG = len(IMG); Y_IMG = IMG.label.values; G_IMG = IMG.group.values
ROW_LABEL = meta.label.values; ROW_GROUP = meta.group.values


def load_feats(backbone, variant="full"):
    return np.load(f"{F}/{backbone}_{variant}.npy")


def rows_train(img_ids):
    return (np.asarray(img_ids)[:, None] * (A + 1) + np.arange(A + 1)[None, :]).ravel()


def rows_test(img_ids):
    return np.asarray(img_ids) * (A + 1)


def group_folds(n_splits=5, seeds=(0, 1)):
    """Leakage-aware outer folds over images, grouped by duplicate group. Yields (seed, fold, train_imgs, test_imgs)."""
    for s in seeds:
        for k, (tr, te) in enumerate(StratifiedGroupKFold(n_splits, shuffle=True, random_state=s).split(np.zeros(N_IMG), Y_IMG, G_IMG)):
            yield s, k, tr, te


def metric_row(y, p, thr=0.5):
    """y in {0,1} (1 = tumor), p = predicted probability of tumor. Returns confusion counts and clinical metrics."""
    y = np.asarray(y); p = np.asarray(p, float); pr = (p >= thr).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); tn = int(((pr == 0) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    n = len(y); sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan; ppv = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) and not np.isnan(sens) else 0.0; d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / d if d > 0 else 0.0; auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return dict(n=n, tp=tp, fp=fp, tn=tn, fn=fn, acc=(tp + tn) / n, err=1 - (tp + tn) / n, sens=sens, spec=spec, ppv=ppv, f1=f1, mcc=mcc, auc=auc)


def fold_table(pred):
    """pred: DataFrame with columns rep, fold, y, prob -> per-fold metric table."""
    return pd.DataFrame([dict(rep=r, fold=f, **metric_row(g.y, g.prob)) for (r, f), g in pred.groupby(["rep", "fold"])])


def group_bootstrap(pred, metrics=("acc", "sens", "spec", "f1", "mcc", "auc"), B=1000, seed=2026):
    """95% group-cluster bootstrap of pooled out-of-fold predictions (resampling duplicate groups, averaging over repeats).
    pred columns: rep, img, group, y, prob. Returns {metric: (point, lo, hi)}."""
    rng = np.random.default_rng(seed); groups = np.sort(pred.group.unique()); gi = {g: i for i, g in enumerate(groups)}; G = len(groups)
    reps = sorted(pred.rep.unique()); by = {r: pred[pred.rep == r].reset_index(drop=True) for r in reps}; idx = {r: by[r].group.map(gi).values for r in reps}
    def one(w):
        out = {m: [] for m in metrics}
        for r in reps:
            d = by[r]; ww = w[idx[r]]; y = d.y.values; p = d.prob.values; pr = (p >= 0.5).astype(int)
            tp = (ww * ((pr == 1) & (y == 1))).sum(); fp = (ww * ((pr == 1) & (y == 0))).sum(); tn = (ww * ((pr == 0) & (y == 0))).sum(); fn = (ww * ((pr == 0) & (y == 1))).sum(); n = tp + fp + tn + fn
            sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan; ppv = tp / (tp + fp) if tp + fp else 0.0
            dd = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); vals = dict(acc=(tp + tn) / n, sens=sens, spec=spec, f1=2 * ppv * sens / (ppv + sens) if (ppv + sens) and not np.isnan(sens) else 0.0, mcc=(tp * tn - fp * fn) / dd if dd > 0 else 0.0)
            if "auc" in metrics:                                                   # weighted rank AUC
                o = np.argsort(p, kind="mergesort"); ps, ys, ws = p[o], y[o], ww[o]; pos = ws * (ys == 1); neg = ws * (ys == 0); cneg = np.cumsum(neg)
                u, first = np.unique(ps, return_index=True); last = np.append(first[1:], len(ps)); negb = np.concatenate([[0], cneg])[first]; negt = np.concatenate([[0], cneg])[last] - negb
                posg = np.array([pos[a:b].sum() for a, b in zip(first, last)]); vals["auc"] = (posg * (negb + 0.5 * negt)).sum() / (pos.sum() * neg.sum()) if pos.sum() * neg.sum() > 0 else np.nan
            for m in metrics: out[m].append(vals[m])
        return {m: np.nanmean(out[m]) for m in metrics}
    point = one(np.ones(G)); draws = {m: [] for m in metrics}
    for _ in range(B):
        w = np.bincount(rng.integers(0, G, G), minlength=G).astype(float); v = one(w)
        for m in metrics: draws[m].append(v[m])
    return {m: (point[m], *np.nanpercentile(draws[m], [2.5, 97.5])) for m in metrics}
