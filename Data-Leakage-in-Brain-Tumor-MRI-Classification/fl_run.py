"""Federated learning on frozen-backbone features: K=4 cross-silo clients with non-IID (label-skewed) image partitions.

Arms (per outer fold, duplicate-group folds, repeat 0):
  FL-LR  FedAvg          logistic head, hyperparameters (lr, local epochs) selected on a pooled validation split of the training groups
  FL-LR  FedProx         same (lr, epochs), proximal weight mu = 0.1
  Local  only            each client trains alone; reported test metric = mean of the four local models' predictions
  FL-VQC FedAvg          quantum head (architecture from the central tuned run of the same fold when available), tuned lr / epochs
  DP-LR  eps in {8,3,1}  image-level DP-SGD (Poisson sampling of IMAGES, per-image clipped gradient, Gaussian noise), RDP accountant
Privacy unit = one source image (all of its augmented rows move together), so augmentation cannot multiply the privacy cost.
The representation (standardization + LDA/PCA) is fit centrally on the training rows: a simulation shortcut NOT covered by the DP guarantee.
usage: python fl_run.py <backbone> <variant>
"""
import os, sys, time, json, pickle, math, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from common import *
from heads import *
import dp_acct

torch.set_num_threads(1)
K, T_LR, T_VQC, ALPHA = 4, 20, 12, 0.5
if os.environ.get("QUICK"): T_LR, T_VQC = 4, 3


def partition(imgs, seed):
    """Dirichlet label-skew partition of images into K clients; resampled until every client has >= 8 images with both classes."""
    rng = np.random.default_rng(seed); imgs = np.asarray(imgs)
    for _ in range(500):
        cl = [[] for _ in range(K)]
        for c in (0, 1):
            ii = imgs[Y_IMG[imgs] == c]; rng.shuffle(ii); p = rng.dirichlet(ALPHA * np.ones(K)); cut = (np.cumsum(p) * len(ii)).astype(int)[:-1]
            for k, part in enumerate(np.split(ii, cut)): cl[k] += list(part)
        if all(len(x) >= 8 and len(set(Y_IMG[x])) == 2 for x in cl): return [np.array(x) for x in cl]
    raise RuntimeError("no valid partition")


def flat(m): return torch.cat([p.detach().reshape(-1) for p in m.parameters()])
def setflat(m, v):
    i = 0
    for p in m.parameters(): n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


def run_fl(mk, A, clients, lr, E, T, mu=0.0, seed=0, val=None, dp=None, batch=64, noise_seed=0):
    """FedAvg/FedProx rounds. clients: list of arrays of IMAGE ids. A: angle features of all rows [n_rows, q]. Returns (best_val_auc, best_round, model, curve).
    dp = dict(sigma, clip, qs): image-level DP-SGD (analytic per-image gradients; only valid for the logistic head)."""
    g = mk(); gv = flat(g); rows = [rows_train(c) for c in clients]; nrows = np.array([len(r) for r in rows], float); wts = torch.tensor(nrows / nrows.sum(), dtype=torch.float32)
    opts = []; locals_ = [mk() for _ in clients]
    for m in locals_: opts.append(torch.optim.Adam(m.parameters(), lr=lr))
    Aall = torch.from_numpy(A); Yall = torch.from_numpy(ROW_LABEL.astype(np.float32)); CW = class_weights(ROW_LABEL[np.concatenate(rows)]); cwmap = {}; rng = np.random.default_rng(seed); nrng = np.random.default_rng(10 ** 6 + noise_seed)
    allrows = np.concatenate(rows); cwfull = torch.zeros(len(ROW_LABEL)); cwfull[torch.from_numpy(allrows)] = CW
    best, best_t, best_v, curve = -1.0, 0, gv.clone(), []
    for t in range(1, T + 1):
        upd = []
        for ci, (m, o, r) in enumerate(zip(locals_, opts, rows)):
            setflat(m, gv); gp = [x.clone().view_as(p_) for x, p_ in zip(torch.split(gv, [p_.numel() for p_ in m.parameters()]), m.parameters())] if mu > 0 else None
            if dp is None:
                for _ in range(E):
                    perm = rng.permutation(r)
                    for i in range(0, len(perm), batch):
                        ix = torch.from_numpy(perm[i:i + batch]); o.zero_grad(); loss = (nn.functional.binary_cross_entropy_with_logits(m(Aall[ix]), Yall[ix], reduction="none") * cwfull[ix]).mean()
                        if mu > 0: loss = loss + 0.5 * mu * sum(((p_ - g_) ** 2).sum() for p_, g_ in zip(m.parameters(), gp))
                        loss.backward(); o.step()
            else:                                                                    # image-level DP-SGD, analytic per-image gradient of the weighted logistic loss
                imgs_c = clients[ci]; steps = int(round(E / dp["qs"]))
                for _ in range(steps):
                    sel = imgs_c[nrng.random(len(imgs_c)) < dp["qs"]]
                    gsum = torch.zeros(len(gv))
                    if len(sel):
                        rr = torch.from_numpy(rows_train(sel)); u = Aall[rr] - math.pi / 2; pz = torch.sigmoid(u @ m.w + m.b); gr = ((pz - Yall[rr]) * cwfull[rr]).unsqueeze(1)
                        gimg = torch.cat([gr * u, gr], 1).reshape(len(sel), A_PER, -1).mean(1); nrm = gimg.norm(dim=1, keepdim=True); gsum = (gimg * torch.clamp(dp["clip"] / (nrm + 1e-12), max=1.0)).sum(0)
                    gsum = gsum + torch.from_numpy(nrng.normal(0, dp["sigma"] * dp["clip"], len(gv)).astype(np.float32)); g_ = gsum / max(dp["qs"] * len(imgs_c), 1.0)
                    o.zero_grad(); m.w.grad = g_[:-1].clone(); m.b.grad = g_[-1].clone(); o.step()
            upd.append(flat(m))
        gv = sum(w * u_ for w, u_ in zip(wts, upd)); setflat(g, gv)
        if val is not None:
            auc = roc_auc_score(val[1], predict(g, val[0])); curve.append(auc)
            if auc > best + 1e-4: best, best_t, best_v = auc, t, gv.clone()
    if val is not None: setflat(g, best_v)
    return best, best_t, g, curve


A_PER = A + 1


def metrics_of(p, y): return metric_row(y, p)


def run(bb, var, reps=(0,)):
    X = load_feats(bb, var); preds, sel = [], []; t0 = time.time(); central = None
    if os.path.exists(f"{R}/qml_selected_{bb}_{var}.csv"): central = pd.read_csv(f"{R}/qml_selected_{bb}_{var}.csv")
    folds = list(group_folds(5, reps))[: int(os.environ.get("MAXFOLDS", 999))]
    for s, k, tr, te in folds:
        ck = f'{R}/ckpt/fl_{bb}_{var}_s{s}_k{k}.pkl'; os.makedirs(f'{R}/ckpt', exist_ok=True)
        if os.path.exists(ck): p_, s_ = pickle.load(open(ck, 'rb')); preds += p_; sel += s_; print('resume', ck, flush=True); continue
        n0, m0 = len(preds), len(sel)
        q, L, ent = 8, 2, "pairs"
        if central is not None:
            c = central[(central.rep == s) & (central.fold == k)]
            if len(c): q, L, ent = int(c.q.iloc[0]), int(c.L.iloc[0]), c.ent.iloc[0]
        itr, iva = (lambda a, b: (tr[a], tr[b]))(*next(StratifiedGroupKFold(5, shuffle=True, random_state=100 + s * 10 + k).split(np.zeros(len(tr)), Y_IMG[tr], G_IMG[tr])))
        tr_rows, te_rows = rows_train(tr), rows_test(te); r_itr, r_iva = rows_train(itr), rows_test(iva)
        Ri = Rep(q, "lda_pca").fit(X[r_itr], ROW_LABEL[r_itr]); Ai = np.zeros((len(ROW_LABEL), q), np.float32); Ai[r_itr] = Ri.transform(X[r_itr]); Ai[r_iva] = Ri.transform(X[r_iva]); val = (Ai[r_iva], ROW_LABEL[r_iva])
        cl_i = partition(itr, 1000 * s + k)
        # ---- tune FL-LR (lr x local epochs), non-private FedAvg ----
        mk_lr = lambda: LinHead(q, seed=k); res = []
        for lr in (0.03, 0.1, 0.3):
            for E in (1, 3):
                b, bt, _, _ = run_fl(mk_lr, Ai, cl_i, lr, E, T_LR, seed=k, val=val); res.append((round(b, 3), -E, lr, E, bt))
        _, _, lr_s, E_s, T_s = max(res); sel.append(dict(arm="FL-LR", rep=s, fold=k, lr=lr_s, E=E_s, rounds=T_s, q=q, trace=json.dumps([[r[2], r[3], r[0], r[4]] for r in res])))
        # ---- final models on all training images ----
        Rf = Rep(q, "lda_pca").fit(X[tr_rows], ROW_LABEL[tr_rows]); Af = np.zeros((len(ROW_LABEL), q), np.float32); Af[tr_rows] = Rf.transform(X[tr_rows]); Af[te_rows] = Rf.transform(X[te_rows]); cl = partition(tr, 1000 * s + k + 7)
        out = lambda name, p: preds.append(pd.DataFrame(dict(method=name, rep=s, fold=k, img=te, group=G_IMG[te], y=Y_IMG[te], prob=p)))
        _, _, g, _ = run_fl(mk_lr, Af, cl, lr_s, E_s, T_s, seed=k); out("FL-LR FedAvg", predict(g, Af[te_rows]))
        _, _, g, _ = run_fl(mk_lr, Af, cl, lr_s, E_s, T_s, mu=0.1, seed=k); out("FL-LR FedProx (mu=0.1)", predict(g, Af[te_rows]))
        loc = []
        for c in cl:                                                                   # local-only models
            m = LinHead(q, seed=k); fit_head(m, Af[rows_train(c)], ROW_LABEL[rows_train(c)], lr=lr_s, epochs=T_s * E_s, seed=k); loc.append(predict(m, Af[te_rows]))
        out("Local-only (mean of 4 clients)", np.mean(loc, 0))
        # ---- FL-VQC ----
        mk_q = lambda: VQC(q, L, ent, seed=k); resq = []
        for lr in (0.03, 0.1):
            for E in (1, 2):
                b, bt, _, _ = run_fl(mk_q, Ai, cl_i, lr, E, T_VQC, seed=k, val=val); resq.append((round(b, 3), -E, lr, E, bt))
        _, _, lrq, Eq, Tq = max(resq); sel.append(dict(arm="FL-VQC", rep=s, fold=k, lr=lrq, E=Eq, rounds=Tq, q=q, L=L, ent=ent, trace=json.dumps([[r[2], r[3], r[0], r[4]] for r in resq])))
        _, _, g, _ = run_fl(mk_q, Af, cl, lrq, Eq, Tq, seed=k); out("FL-VQC FedAvg", predict(g, Af[te_rows]))
        # ---- DP-LR: tune (clip, lr) at eps = 8 on validation, reuse for eps = 3, 1 ----
        qs, Edp = 0.25, 1; steps = int(round(Edp / qs)) * T_LR; sig = {e: dp_acct.sigma_for(e, [qs] * K, [steps] * K) for e in (8, 3, 1)}
        rdp = []
        for C in (0.25, 1.0):
            for lr in (0.1, 0.3):
                b, bt, _, _ = run_fl(mk_lr, Ai, cl_i, lr, Edp, T_LR, seed=k, val=val, dp=dict(sigma=sig[8], clip=C, qs=qs)); rdp.append((round(b, 3), C, lr))
        _, C_s, lr_d = max(rdp); sel.append(dict(arm="DP-LR", rep=s, fold=k, lr=lr_d, clip=C_s, qs=qs, E=Edp, rounds=T_LR, sigma8=sig[8], sigma3=sig[3], sigma1=sig[1], trace=json.dumps(rdp)))
        for e in (8, 3, 1):
            _, _, g, _ = run_fl(mk_lr, Af, cl, lr_d, Edp, T_LR, seed=k, dp=dict(sigma=sig[e], clip=C_s, qs=qs), noise_seed=k); out(f"DP-LR eps={e}", predict(g, Af[te_rows]))
        print(f"{bb}/{var} fold{k}: q={q} L={L} {ent} | LR lr={lr_s} E={E_s} T={T_s} | VQC lr={lrq} E={Eq} T={Tq} | DP C={C_s} lr={lr_d}  {time.time() - t0:.0f}s", flush=True)
        pickle.dump((preds[n0:], sel[m0:]), open(ck + '.tmp', 'wb')); os.replace(ck + '.tmp', ck)
        P = pd.concat(preds, ignore_index=True); P.to_csv(f"{R}/fl_preds_{bb}_{var}.csv", index=False); pd.DataFrame(sel).to_csv(f"{R}/fl_selected_{bb}_{var}.csv", index=False)
    rows = []
    for mth, d in P.groupby("method"):
        ft = fold_table(d); b = group_bootstrap(d, B=500); r = dict(backbone=bb, variant=var, method=mth)
        for m in ("acc", "sens", "spec", "f1", "mcc", "auc"): r[m] = b[m][0]; r[m + "_lo"] = b[m][1]; r[m + "_hi"] = b[m][2]; r[m + "_sd"] = ft[m].std()
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(f"{R}/fl_summary_{bb}_{var}.csv", index=False); pd.set_option("display.width", 200); print((S.set_index("method")[["acc", "sens", "spec", "mcc", "auc"]] * 100).round(1)); print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
