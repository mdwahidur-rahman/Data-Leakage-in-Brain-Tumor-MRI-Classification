"""Ablations on ResNet50 'full' features under the primary protocol (duplicate-group folds, repeat 0).
 (a) input representation for the small heads: PCA-q vs LDA(1)+PCA(q-1), q in {4, 8, 12}, logistic regression and RBF-SVM;
 (b) augmentation dose: logistic regression on 2048-D features with 0, 1, 2, 4, 8 or 12 augmented rows per training image."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from common import *
from heads import Rep
X = load_feats("resnet50", "full"); out = []; folds = list(group_folds(5, (0,)))
def pooled(rows_pred, name, **kw):
    d = pd.DataFrame(rows_pred); b = group_bootstrap(d, B=300); r = dict(name=name, **kw)
    for m in ("acc", "sens", "spec", "mcc", "auc"): r[m] = b[m][0]; r[m + "_lo"] = b[m][1]; r[m + "_hi"] = b[m][2]
    out.append(r)
for kind in ("pca", "lda_pca"):
    for q in (4, 8, 12):
        for mname, mk in (("LR", lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")), ("RBF-SVM", lambda: SVC(C=10, probability=True, class_weight="balanced", random_state=0))):
            rp = []
            for s, k, tr, te in folds:
                tr_rows, te_rows = rows_train(tr), rows_test(te); R_ = Rep(q, kind).fit(X[tr_rows], ROW_LABEL[tr_rows]); m = mk().fit(R_.transform(X[tr_rows]), ROW_LABEL[tr_rows]); p = m.predict_proba(R_.transform(X[te_rows]))[:, 1]
                rp += [dict(rep=s, img=i, group=G_IMG[i], y=Y_IMG[i], prob=pp) for i, pp in zip(te, p)]
            pooled(rp, f"rep:{kind}", q=q, model=mname); print("rep", kind, q, mname, round(out[-1]["acc"], 3), flush=True)
pd.DataFrame(out).to_csv(f"{R}/ablate_rep.csv", index=False); out = []
for a in (0, 1, 2, 4, 8, 12):
    rp = []
    for s, k, tr, te in folds:
        tr_rows = (np.asarray(tr)[:, None] * (A + 1) + np.arange(a + 1)[None, :]).ravel(); te_rows = rows_test(te)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X[tr_rows], ROW_LABEL[tr_rows]); p = m.predict_proba(X[te_rows])[:, 1]
        rp += [dict(rep=s, img=i, group=G_IMG[i], y=Y_IMG[i], prob=pp) for i, pp in zip(te, p)]
    pooled(rp, "augmentation_dose", n_aug=a); print("aug", a, round(out[-1]["acc"], 3), flush=True)
pd.DataFrame(out).to_csv(f"{R}/ablate_aug.csv", index=False); open(f"{R}/ablate_done.flag", "w").write("ok\n")
