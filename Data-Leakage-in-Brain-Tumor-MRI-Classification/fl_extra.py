"""Matched references for the private runs: identical schedule (4 Poisson-sampled image-level steps per round, 20 rounds) with no noise and no clipping,
and with clipping only. Isolates the cost of clipping and of noise from the effect of the different training schedule."""
import warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from common import *; from heads import *; import fl_run as FL
from sklearn.model_selection import StratifiedGroupKFold
BB, VAR = "resnet50", "full"; X = load_feats(BB, VAR); central = pd.read_csv(f"{R}/qml_selected_{BB}_{VAR}.csv"); fsel = pd.read_csv(f"{R}/fl_selected_{BB}_{VAR}.csv"); preds = []
for s, k, tr, te in list(group_folds(5, (0,))):
    c = central[(central.rep == s) & (central.fold == k)].iloc[0]; q = int(c.q); d = fsel[(fsel.arm == "DP-LR") & (fsel.fold == k)].iloc[0]; lr, C, qs = float(d.lr), float(d["clip"]), float(d.qs)
    tr_rows, te_rows = rows_train(tr), rows_test(te); Rf = Rep(q, "lda_pca").fit(X[tr_rows], ROW_LABEL[tr_rows]); Af = np.zeros((len(ROW_LABEL), q), np.float32); Af[tr_rows] = Rf.transform(X[tr_rows]); Af[te_rows] = Rf.transform(X[te_rows]); cl = FL.partition(tr, 1000 * s + k + 7)
    for name, dp in (("Matched non-private (no clip, no noise)", dict(sigma=0.0, clip=1e9, qs=qs)), ("Clipping only (sigma = 0)", dict(sigma=0.0, clip=C, qs=qs))):
        _, _, g, _ = FL.run_fl(lambda: LinHead(q, seed=k), Af, cl, lr, 1, FL.T_LR, seed=k, dp=dp, noise_seed=k)
        preds.append(pd.DataFrame(dict(method=name, rep=s, fold=k, img=te, group=G_IMG[te], y=Y_IMG[te], prob=predict(g, Af[te_rows]))))
P = pd.concat(preds, ignore_index=True); P.to_csv(f"{R}/fl_extra_preds_{BB}_{VAR}.csv", index=False); rows = []
for m, dd in P.groupby("method"):
    b = group_bootstrap(dd, B=500); ft = fold_table(dd); r = dict(method=m)
    for x in ("acc", "sens", "spec", "f1", "mcc", "auc"): r[x] = b[x][0]; r[x + "_lo"] = b[x][1]; r[x + "_hi"] = b[x][2]; r[x + "_sd"] = ft[x].std()
    rows.append(r)
S = pd.DataFrame(rows); S.to_csv(f"{R}/fl_extra_summary_{BB}_{VAR}.csv", index=False); print((S.set_index("method")[["acc", "sens", "spec", "mcc", "auc"]] * 100).round(1))
