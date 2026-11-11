"""Tuned variational quantum classifier on frozen-backbone features, with matched classical controls.

Per outer fold (duplicate-group StratifiedGroupKFold):
  1. inner split of the TRAINING images (group-aware, 80/20): the representation is fit on inner-train rows only;
  2. grid over qubits x layers x entangler, selected by validation ROC AUC (ties -> fewer parameters);
  3. refit representation on all training rows, train the selected circuit for the selected number of epochs, predict originals of the test groups.
Controls on the identical angle inputs: logistic regression, RBF-SVM, small MLP (same inner selection).
usage: python qml_run.py <backbone> <variant> <n_repeats> [--ref12]
"""
import os, sys, time, json, pickle, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from common import *
from heads import *

torch.set_num_threads(1)
GRID = [(q, L, e) for q in (4, 6, 8) for L in (1, 2, 3) for e in ("pairs", "ring")]
EPOCHS, PATIENCE, LR = 40, 8, 0.05
if os.environ.get('QUICK'): GRID = [(4, 1, 'pairs'), (6, 2, 'pairs')]; EPOCHS = 6


def inner_split(tr_imgs, seed):
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=100 + seed); a, b = next(sgkf.split(np.zeros(len(tr_imgs)), Y_IMG[tr_imgs], G_IMG[tr_imgs])); return tr_imgs[a], tr_imgs[b]


def control_fit(name, A_tr, y_tr, A_va, y_va, A_te, seed):
    grids = {"LR": [dict(C=c) for c in (0.01, 0.1, 1, 10)], "SVM": [dict(C=c) for c in (1, 10)], "MLP": [dict(alpha=a) for a in (1e-3, 1e-1)]}
    mk = {"LR": lambda **k: LogisticRegression(max_iter=3000, class_weight="balanced", **k), "SVM": lambda **k: SVC(probability=True, class_weight="balanced", random_state=0, **k), "MLP": lambda **k: MLPClassifier((16,), max_iter=400, random_state=seed, **k)}[name]
    return grids[name], mk


def run(bb, var, reps, ref12=False):
    X = load_feats(bb, var); preds, sel = [], []; t0 = time.time(); folds = list(group_folds(5, reps))[: int(os.environ.get('MAXFOLDS', 999))]
    for s, k, tr, te in folds:
        ck = f'{R}/ckpt/qml_{bb}_{var}_s{s}_k{k}.pkl'; os.makedirs(f'{R}/ckpt', exist_ok=True)
        if os.path.exists(ck): p_, s_ = pickle.load(open(ck, 'rb')); preds += p_; sel += s_; print('resume', ck, flush=True); continue
        n0, m0 = len(preds), len(sel)
        tr_rows, te_rows = rows_train(tr), rows_test(te); itr, iva = inner_split(tr, s * 10 + k); r_itr, r_iva = rows_train(itr), rows_test(iva); ytr_i, yva_i = ROW_LABEL[r_itr], ROW_LABEL[r_iva]
        reps_cache = {}
        for q in (4, 6, 8): reps_cache[q] = Rep(q, "lda_pca").fit(X[r_itr], ytr_i); 
        best = None; trace = []
        for (q, L, e) in GRID:
            R_ = reps_cache[q]; m = VQC(q, L, e, seed=s * 100 + k); auc, ep = fit_head(m, R_.transform(X[r_itr]), ytr_i, lr=LR, epochs=EPOCHS, seed=s * 100 + k, Xv=R_.transform(X[r_iva]), yv=yva_i, patience=PATIENCE)
            trace.append(dict(q=q, L=L, ent=e, val_auc=auc, epochs=ep, params=m.n_params()))
            key = (round(auc, 3), -(q * L * 3))                                      # ties (3 decimals) -> fewer parameters
            if best is None or key > best[0]: best = (key, q, L, e, ep)
        _, q, L, e, ep = best; sel.append(dict(rep=s, fold=k, q=q, L=L, ent=e, epochs=max(ep, 5), val_auc=best[0][0], trace=json.dumps(trace)))
        Rf = Rep(q, "lda_pca").fit(X[tr_rows], ROW_LABEL[tr_rows]); A_tr, A_te = Rf.transform(X[tr_rows]), Rf.transform(X[te_rows])
        m = VQC(q, L, e, seed=s * 100 + k); fit_head(m, A_tr, ROW_LABEL[tr_rows], lr=LR, epochs=max(ep, 5), seed=s * 100 + k); p = predict(m, A_te)
        preds.append(pd.DataFrame(dict(method="VQC (tuned)", rep=s, fold=k, img=te, group=G_IMG[te], y=Y_IMG[te], prob=p)))
        # controls on the identical inputs (representation of the selected qubit count, fit on the inner split for selection, refit for the final model)
        Rc = reps_cache[q]; Ai, Av = Rc.transform(X[r_itr]), Rc.transform(X[r_iva])
        for name in ("LR", "SVM", "MLP"):
            grid, mk = control_fit(name, Ai, ytr_i, Av, yva_i, None, s)
            sc = [(roc_auc_score(yva_i, mk(**g).fit(Ai, ytr_i).predict_proba(Av)[:, 1]), -i, g) for i, g in enumerate(grid)]; g = max(sc)[2]
            preds.append(pd.DataFrame(dict(method=f"{name} on identical inputs", rep=s, fold=k, img=te, group=G_IMG[te], y=Y_IMG[te], prob=mk(**g).fit(A_tr, ROW_LABEL[tr_rows]).predict_proba(A_te)[:, 1])))
        print(f"{bb}/{var} rep{s} fold{k}: selected q={q} L={L} {e} ep={ep}  {time.time() - t0:.0f}s", flush=True)
        if ref12 and s == 0 and k < int(os.environ.get('REF12_FOLDS', 5)):                                                         # fixed 12-qubit reference (simulation cost forbids tuning at 12 qubits)
            R12 = Rep(12, "lda_pca").fit(X[tr_rows], ROW_LABEL[tr_rows]); m12 = VQC(12, 2, "pairs", seed=k); fit_head(m12, R12.transform(X[tr_rows]), ROW_LABEL[tr_rows], lr=LR, epochs=25, seed=k)
            preds.append(pd.DataFrame(dict(method="VQC 12 qubits (fixed config)", rep=s, fold=k, img=te, group=G_IMG[te], y=Y_IMG[te], prob=predict(m12, R12.transform(X[te_rows])))))
            print(f"   12-qubit reference fold{k} done {time.time() - t0:.0f}s", flush=True)
        pickle.dump((preds[n0:], sel[m0:]), open(ck + '.tmp', 'wb')); os.replace(ck + '.tmp', ck)
        P = pd.concat(preds, ignore_index=True); P.to_csv(f"{R}/qml_preds_{bb}_{var}.csv", index=False); pd.DataFrame(sel).to_csv(f"{R}/qml_selected_{bb}_{var}.csv", index=False)
    rows = []
    for mth, d in P.groupby("method"):
        ft = fold_table(d); b = group_bootstrap(d, B=500); r = dict(backbone=bb, variant=var, method=mth)
        for m in ("acc", "sens", "spec", "f1", "mcc", "auc"): r[m] = b[m][0]; r[m + "_lo"] = b[m][1]; r[m + "_hi"] = b[m][2]; r[m + "_sd"] = ft[m].std()
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(f"{R}/qml_summary_{bb}_{var}.csv", index=False); pd.set_option("display.width", 200); print((S.set_index("method")[["acc", "sens", "spec", "mcc", "auc"]] * 100).round(1)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    bb, var, nrep = sys.argv[1], sys.argv[2], int(sys.argv[3]); run(bb, var, tuple(range(nrep)), ref12="--ref12" in sys.argv)
