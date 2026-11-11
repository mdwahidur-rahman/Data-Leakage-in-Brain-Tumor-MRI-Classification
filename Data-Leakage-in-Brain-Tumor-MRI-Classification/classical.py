"""Hybrid classical track: frozen CNN/ViT features -> standardize -> PCA(256) -> one of seven classifiers.

Outer protocol: duplicate-group StratifiedGroupKFold (5 folds x 2 repeats). Training rows = originals + their augmentations;
test rows = original images of the held-out groups only. Hyperparameters are chosen by inner 3-fold GroupKFold cross-validation
on the TRAINING rows (augmented copies stay with their source group), by mean ROC AUC (ties broken toward the first, simpler setting).
usage: python classical.py <backbone> <variant> [reps]
"""
import os, sys, time, warnings, json, pickle, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from common import *

CLF = {
    "LR": (lambda: LogisticRegression(max_iter=3000), {"clf__C": [0.001, 0.01, 0.1, 1.0]}),
    "SVC": (lambda: SVC(probability=True, random_state=0), {"clf__C": [1.0, 10.0]}),
    "RF": (lambda: RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=1), {"clf__max_depth": [None, 10]}),
    "DT": (lambda: DecisionTreeClassifier(random_state=0), {"clf__max_depth": [3, 6, None]}),
    "KNN": (lambda: KNeighborsClassifier(), {"clf__n_neighbors": [3, 7, 15]}),
    "NB": (lambda: GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]}),
    "XGB": (lambda: XGBClassifier(n_estimators=120, learning_rate=0.1, subsample=0.9, colsample_bytree=0.5, tree_method="hist", n_jobs=1, random_state=0, verbosity=0), {"clf__max_depth": [3, 5]}),
}


def run(bb, var, reps=(0, 1), only=None):
    X = load_feats(bb, var); out, hp = [], []; t0 = time.time()
    os.makedirs(f'{R}/ckpt', exist_ok=True)
    for s, k, tr, te in group_folds(5, reps):
        ck = f'{R}/ckpt/classical_{bb}_{var}_{"-".join(only) if only else "all"}_s{s}_k{k}.pkl'
        if os.path.exists(ck): o_, h_ = pickle.load(open(ck, 'rb')); out += o_; hp += h_; print('resume', ck, flush=True); continue
        n0, h0 = len(out), len(hp)
        tr_rows, te_rows = rows_train(tr), rows_test(te); g = ROW_GROUP[tr_rows]; ytr = ROW_LABEL[tr_rows]
        for name, (mk, grid) in CLF.items():
            if only and name not in only: continue
            pipe = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=min(256, len(tr_rows) - 1), svd_solver="randomized", random_state=0)), ("clf", mk())])
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(3), scoring="roc_auc", n_jobs=1, refit=True).fit(X[tr_rows], ytr, groups=g)
            p = gs.predict_proba(X[te_rows])[:, 1]
            out.append(pd.DataFrame(dict(backbone=bb, variant=var, classifier=name, rep=s, fold=k, img=te, group=G_IMG[te], y=Y_IMG[te], prob=p)))
            hp.append(dict(backbone=bb, variant=var, classifier=name, rep=s, fold=k, inner_auc=gs.best_score_, **{kk.split("__")[1]: (None if v is None else float(v)) for kk, v in gs.best_params_.items()}))
        pickle.dump((out[n0:], hp[h0:]), open(ck + ".tmp", "wb")); os.replace(ck + ".tmp", ck)
        print(f"{bb}/{var} rep{s} fold{k} done  {time.time() - t0:.0f}s", flush=True)
    P = pd.concat(out, ignore_index=True); P.to_csv(f"{R}/classical_preds_{bb}_{var}.csv", index=False); pd.DataFrame(hp).to_csv(f"{R}/classical_hparams_{bb}_{var}.csv", index=False)
    rows = []
    for c, d in P.groupby("classifier"):
        ft = fold_table(d); b = group_bootstrap(d, B=500); r = dict(backbone=bb, variant=var, classifier=c)
        for m in ("acc", "sens", "spec", "f1", "mcc", "auc"): r[m] = b[m][0]; r[m + "_lo"] = b[m][1]; r[m + "_hi"] = b[m][2]; r[m + "_sd"] = ft[m].std()
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(f"{R}/classical_summary_{bb}_{var}.csv", index=False)
    pd.set_option("display.width", 200); print((S.set_index("classifier")[["acc", "sens", "spec", "mcc", "auc"]] * 100).round(1)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    bb, var = sys.argv[1], sys.argv[2]; reps = tuple(range(int(sys.argv[3]))) if len(sys.argv) > 3 else (0, 1)
    run(bb, var, reps)
