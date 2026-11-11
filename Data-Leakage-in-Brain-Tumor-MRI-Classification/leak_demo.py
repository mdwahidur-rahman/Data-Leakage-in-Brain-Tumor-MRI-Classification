"""How much does the splitting protocol inflate accuracy on the public brain-MRI images?

One fixed classifier (standardize + logistic regression, C = 0.01) on frozen features; only the protocol changes.
  P1 augment-then-split : original + augmented rows pooled, random row-level 5-fold  (the common mistake)
  P2 image-level split  : random image-level 5-fold, augmentation applied after the split (training only)
  P3 group-aware        : duplicate-group 5-fold, augmentation after the split         (primary protocol)
  P4 group-aware, no aug: duplicate-group 5-fold, originals only                       (value of augmentation)
  P5/P6 conservative groups: connected components of feature cosine similarity >= 0.40 / 0.30 (same-patient proxy)
Test rows are original images in P2-P5; P1 tests on pooled rows, including augmented copies of training images.
usage: python leak_demo.py <backbone> <variant>
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from common import *

bb, var = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("resnet50", "full")
X = load_feats(bb, var); Xo = X[rows_test(np.arange(N_IMG))]
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))


def cluster_groups(tau):
    Z = (Xo - Xo.mean(0)) / (Xo.std(0) + 1e-6); Z /= np.linalg.norm(Z, axis=1, keepdims=True); S = Z @ Z.T; np.fill_diagonal(S, 0)
    n, lab = connected_components(csr_matrix(S >= tau), directed=False); return lab


def fit_eval(tr_rows, te_rows):
    m = clf().fit(X[tr_rows], ROW_LABEL[tr_rows]); return m.predict_proba(X[te_rows])[:, 1], ROW_LABEL[te_rows]


out = []; seeds = (0, 1, 2); preds = {}
for tau in (0.3, 0.4, 0.5):                                                        # report cluster counts so tau is transparent
    print(f"cosine tau={tau}: {len(np.unique(cluster_groups(tau)))} clusters for {N_IMG} images (duplicate groups: {len(np.unique(G_IMG))})")
GC, GC2 = cluster_groups(0.40), cluster_groups(0.30)
for s in seeds:
    # P1: pooled rows, random row-level folds
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(len(ROW_LABEL)), ROW_LABEL)):
        p, y = fit_eval(tr, te); out.append(dict(protocol="P1 augment-then-split (row-level)", rep=s, fold=k, **metric_row(y, p)))
    for name, splitter in (("P2 image-level split, augment after", lambda s: StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(N_IMG), Y_IMG)),
                           ("P3 group-aware, augment after (primary)", lambda s: StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(N_IMG), Y_IMG, G_IMG)),
                           ("P5 conservative clusters (cosine >= 0.40), augment after", lambda s: StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(N_IMG), Y_IMG, GC)),
                           ("P6 very conservative clusters (cosine >= 0.30), augment after", lambda s: StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(N_IMG), Y_IMG, GC2))):
        for k, (tr, te) in enumerate(splitter(s)):
            p, y = fit_eval(rows_train(tr), rows_test(te)); out.append(dict(protocol=name, rep=s, fold=k, **metric_row(y, p)))
            if name.startswith("P3"): preds.setdefault("P3", []).append(pd.DataFrame(dict(rep=s, fold=k, img=te, group=G_IMG[te], y=y, prob=p)))
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(np.zeros(N_IMG), Y_IMG, G_IMG)):
        p, y = fit_eval(rows_test(tr), rows_test(te)); out.append(dict(protocol="P4 group-aware, no augmentation", rep=s, fold=k, **metric_row(y, p)))
T = pd.DataFrame(out); T.to_csv(f"{R}/leak_demo_{bb}_{var}.csv", index=False)
S = T.groupby("protocol")[["acc", "sens", "spec", "mcc", "auc"]].agg(["mean", "std"]); S.columns = ["_".join(c) for c in S.columns]; S = S.reset_index(); S.to_csv(f"{R}/leak_demo_summary_{bb}_{var}.csv", index=False)
pd.set_option("display.width", 200); print((S.set_index("protocol")[["acc_mean", "acc_std", "sens_mean", "spec_mean", "mcc_mean", "auc_mean"]] * 100).round(1))
base = T[T.protocol.str.startswith("P3")].groupby("rep").acc.mean().mean(); print("inflation of P1 over primary P3 (accuracy points): %.1f" % (100 * (T[T.protocol.str.startswith("P1")].groupby("rep").acc.mean().mean() - base)))
