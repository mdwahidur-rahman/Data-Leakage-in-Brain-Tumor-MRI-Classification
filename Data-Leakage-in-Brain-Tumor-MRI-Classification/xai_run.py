"""Explainability for the classical, federated and quantum heads (fold 0, repeat 0), through the frozen ResNet50 backbone.

Every head is rebuilt exactly as in the cross-validation runs and its test probabilities are compared with the stored ones
(a reproduction check). The path image -> ResNet50 layer4 map A (2048x7x7) -> global average pool -> head -> logit is fully
differentiable, so Grad-CAM, Grad-CAM++ and Score-CAM use the logit of the PREDICTED class. Eigen-CAM uses A only.
Faithfulness: deletion / insertion curves (11 steps, 7x7 cells, black baseline) and a random-order reference; brain-mask mass
(fraction of the map inside the brain mask) as a sanity check. There are no tumor masks, so no localization metric is claimed.
"""
import os, sys, math, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn, cv2
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from common import *
from heads import *
import features as FE
import fl_run as FL
import dp_acct

torch.set_num_threads(1); FIG = "./figures"; os.makedirs(FIG, exist_ok=True); t0 = time.time()
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "savefig.dpi": 300})
BB, VAR = "resnet50", "full"; X = load_feats(BB, VAR); IMF = np.load(f"{D}/images_full.npy", mmap_mode="r"); IMR = np.load(f"{D}/images_raw.npy", mmap_mode="r")
bone, (MEAN, STD), sha = FE.build("resnet50"); bone.eval(); MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1); STD_T = torch.tensor(STD).view(1, 3, 1, 1)
s, k, tr, te = next(group_folds(5, (0,))); tr_rows, te_rows = rows_train(tr), rows_test(te); ytr = ROW_LABEL[tr_rows]

# ---------------------------------------------------------------- rebuild the five heads of fold 0
arms = {}
mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6; lr2k = LogisticRegression(C=0.01, max_iter=3000).fit((X[tr_rows] - mu) / sd, ytr)
mu_t, sd_t, w_t, b_t = [torch.tensor(a, dtype=torch.float32) for a in (mu, sd, lr2k.coef_[0], lr2k.intercept_[0])]
arms["Classical LR (2048-D)"] = (lambda g: ((g - mu_t) / sd_t) @ w_t + b_t, lr2k.predict_proba((X[te_rows] - mu) / sd)[:, 1])
qsel = pd.read_csv(f"{R}/qml_selected_{BB}_{VAR}.csv"); q0 = qsel[(qsel.rep == 0) & (qsel.fold == 0)].iloc[0]; q, L, ent, ep = int(q0.q), int(q0.L), q0.ent, int(q0.epochs)
Rf = Rep(q, "lda_pca").fit(X[tr_rows], ytr); P = Rf.torch_params(); Atr, Ate = Rf.transform(X[tr_rows]), Rf.transform(X[te_rows])
mq = VQC(q, L, ent, seed=0); fit_head(mq, Atr, ytr, lr=0.05, epochs=ep, seed=0)
arms[f"VQC central ({q}q, {L}L, {ent})"] = (lambda g, m=mq: m(torch_rep(g, P)), predict(mq, Ate))
fsel = pd.read_csv(f"{R}/fl_selected_{BB}_{VAR}.csv"); fsel = fsel[(fsel.rep == 0) & (fsel.fold == 0)].set_index("arm")
Af = np.zeros((len(ROW_LABEL), q), np.float32); Af[tr_rows] = Atr; Af[te_rows] = Ate; cl = FL.partition(tr, 1000 * 0 + 0 + 7); mk_lr = lambda: LinHead(q, seed=0)
r = fsel.loc["FL-LR"]; _, _, g_lr, _ = FL.run_fl(mk_lr, Af, cl, float(r.lr), int(r.E), int(r.rounds), seed=0)
arms["FL-LR (FedAvg)"] = (lambda g, m=g_lr: m(torch_rep(g, P)), predict(g_lr, Ate))
r = fsel.loc["FL-VQC"]; _, _, g_q, _ = FL.run_fl(lambda: VQC(q, L, ent, seed=0), Af, cl, float(r.lr), int(r.E), int(r.rounds), seed=0)
arms[f"FL-VQC (FedAvg)"] = (lambda g, m=g_q: m(torch_rep(g, P)), predict(g_q, Ate))
r = fsel.loc["DP-LR"]; _, _, g_dp, _ = FL.run_fl(mk_lr, Af, cl, float(r.lr), int(r.E), int(r.rounds), seed=0, dp=dict(sigma=float(r.sigma8), clip=float(r["clip"]), qs=float(r.qs)), noise_seed=0)
arms["DP-LR (eps=8)"] = (lambda g, m=g_dp: m(torch_rep(g, P)), predict(g_dp, Ate))
# reproduction check against the stored cross-validation predictions
chk = []
for name, mth, f in (("VQC central", "VQC (tuned)", f"{R}/qml_preds_{BB}_{VAR}.csv"), ("FL-LR (FedAvg)", "FL-LR FedAvg", f"{R}/fl_preds_{BB}_{VAR}.csv"), ("FL-VQC (FedAvg)", "FL-VQC FedAvg", f"{R}/fl_preds_{BB}_{VAR}.csv"), ("DP-LR (eps=8)", "DP-LR eps=8", f"{R}/fl_preds_{BB}_{VAR}.csv")):
    st = pd.read_csv(f); st = st[(st.method == mth) & (st.rep == 0) & (st.fold == 0)].set_index("img").prob; key = [a for a in arms if a.startswith(name)][0]
    chk.append(dict(arm=key, max_abs_diff=float(np.abs(arms[key][1] - st.loc[te].values).max())))
pd.DataFrame(chk).to_csv(f"{R}/xai_live_check.csv", index=False); print("reproduction check (max |p_rebuilt - p_stored|):", {c["arm"]: round(c["max_abs_diff"], 8) for c in chk}, f"{time.time() - t0:.0f}s", flush=True)

# ---------------------------------------------------------------- items
yte = Y_IMG[te]; names = list(arms); ok = np.all([(arms[a][1] >= 0.5).astype(int) == yte for a in names], 0)
rng = np.random.default_rng(11); pick = {}
for c in (0, 1):
    pool = np.where(ok & (yte == c))[0]; pick[c] = list(rng.choice(pool, min(5, len(pool)), replace=False))
items = pick[1] + pick[0]; print("items (test positions):", items, "| correct under all five heads:", int(ok.sum()), "of", len(te), flush=True)


def tensor_of(row, mask01=None, base=None):
    x = torch.from_numpy(np.asarray(IMF[row], dtype=np.float32) / 255.0)
    return x


def to_net(x01):                                                  # [B,224,224] in [0,1] -> normalized 3-channel
    return (x01.unsqueeze(1).repeat(1, 3, 1, 1) - MEAN_T) / STD_T


@torch.no_grad()
def gap_of(x01):
    out = []
    for i in range(0, len(x01), 32): out.append(bone.forward_features(to_net(x01[i:i + 32])).mean((2, 3)))
    return torch.cat(out)


def norm01(m):
    m = m - m.min(); return m / m.max() if m.max() > 1e-12 else torch.zeros_like(m)


def up(c7): return nn.functional.interpolate(c7[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]


def cam_set(A, head, sign, x01):
    Ag = A.clone().requires_grad_(True); y = head(Ag.mean((2, 3)))[0] * sign; y.backward(); gr = Ag.grad[0]; Aa = A[0]
    gc = torch.relu((gr.mean((1, 2))[:, None, None] * Aa).sum(0))
    g2, g3 = gr ** 2, gr ** 3; den = 2 * g2 + Aa.sum((1, 2), keepdim=True) * g3; al = g2 / torch.where(den != 0, den, torch.ones_like(den)); gcpp = torch.relu(((al * torch.relu(gr)).sum((1, 2))[:, None, None] * Aa).sum(0))
    return gc, gcpp


def eigen_cam(A):
    M = A[0].reshape(A.shape[1], -1).T.numpy(); M = M - M.mean(0); u, s_, vt = np.linalg.svd(M, full_matrices=False); pc = torch.from_numpy((M @ vt[0]).reshape(7, 7).astype(np.float32))
    return pc if torch.corrcoef(torch.stack([pc.flatten(), A[0].mean(0).flatten()]))[0, 1] >= 0 else -pc


def score_cam(A, x01, heads, signs, top=64):
    Aa = A[0]; idx = torch.argsort(Aa.mean((1, 2)), descending=True)[:top]; masks = torch.stack([norm01(up(torch.relu(Aa[i]))) for i in idx])       # [top,224,224]
    g = gap_of(x01[None] * masks); gb = gap_of(torch.zeros(1, 224, 224)); outs = {}
    for name, h in heads.items():
        with torch.no_grad(): cic = signs[name] * (h(g) - h(gb)); w = torch.softmax(cic, 0)
        outs[name] = torch.relu((w[:, None, None] * Aa[idx]).sum(0))
    return outs


CELL = 32


def perturb_batch(x01, order, mode, steps=11):
    """order: cell indices (0..48) sorted by decreasing importance; deletion blacks out the top cells, insertion reveals only them."""
    out = []; cells = np.zeros((7, 7), bool).ravel()
    for j in range(steps):
        n = int(round(j / (steps - 1) * 49)); m = np.zeros(49, bool); m[order[:n]] = True; m = torch.from_numpy(np.kron(m.reshape(7, 7), np.ones((CELL, CELL))).astype(np.float32))
        out.append(x01 * (1 - m) if mode == "del" else x01 * m)
    return torch.stack(out)


def curve_auc(p): p = np.asarray(p); return float(np.trapezoid(p, dx=1.0 / (len(p) - 1)))


results = {}; maps = {}; faith = []; brain = []
for ii, pos in enumerate(items):
    row = int(rows_test(te)[pos]); x01 = tensor_of(row); A = bone.forward_features(to_net(x01[None])).detach(); gp = A.mean((2, 3))
    bmask = cv2.morphologyEx((np.asarray(IMR[row]) > 10).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    signs = {}; heads = {}
    for a in names:
        pr = float(arms[a][1][pos]); signs[a] = 1.0 if pr >= 0.5 else -1.0; heads[a] = arms[a][0]
    sc = score_cam(A, x01, heads, signs); eg = eigen_cam(A); M = {}
    for a in names:
        gc, gcpp = cam_set(A, heads[a], signs[a], x01); M[(a, "Grad-CAM")] = gc; M[(a, "Grad-CAM++")] = gcpp; M[(a, "Score-CAM")] = sc[a]; M[(a, "Eigen-CAM")] = eg
    ordr = np.random.default_rng(100 + ii); rnd = [ordr.permutation(49) for _ in range(5)]
    # backbone features for each distinct ordering, evaluated under every head (Eigen-CAM and random orders are shared across heads)
    cache = {}
    def feats_for(order, mode):
        key = (tuple(order), mode)
        if key not in cache: cache[key] = gap_of(perturb_batch(x01, order, mode))
        return cache[key]
    def p_of(a, g): 
        with torch.no_grad(): return torch.sigmoid(signs[a] * heads[a](g)).numpy()
    for a in names:
        rd = np.mean([[curve_auc(p_of(a, feats_for(o, "del"))), curve_auc(p_of(a, feats_for(o, "ins")))] for o in rnd], 0)
        for m in ("Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Score-CAM"):
            c7 = M[(a, m)]; order = np.argsort(-c7.flatten().numpy(), kind="stable"); d = curve_auc(p_of(a, feats_for(order, "del"))); n_ = curve_auc(p_of(a, feats_for(order, "ins")))
            full = norm01(up(c7)).numpy(); inside = float(full[bmask].sum() / max(full.sum(), 1e-12))
            faith.append(dict(item=ii, label=int(yte[pos]), arm=a, method=m, deletion=d, insertion=n_, ins_minus_del=n_ - d, rand_deletion=rd[0], rand_insertion=rd[1], brain_mass=inside, brain_area=float(bmask.mean())))
            maps[(ii, a, m)] = full
    maps[(ii, "img")] = np.asarray(IMF[row]); print(f"item {ii + 1}/{len(items)} done {time.time() - t0:.0f}s", flush=True)
F = pd.DataFrame(faith); F.to_csv(f"{R}/xai_items.csv", index=False)
S = F.groupby(["arm", "method"]).agg(insertion=("insertion", "mean"), deletion=("deletion", "mean"), ins_minus_del=("ins_minus_del", "mean"), ins_minus_del_sd=("ins_minus_del", "std"), rand_ins=("rand_insertion", "mean"), rand_del=("rand_deletion", "mean"), brain_mass=("brain_mass", "mean"), brain_area=("brain_area", "mean"), n=("item", "count")).reset_index()
S.to_csv(f"{R}/xai_faithfulness.csv", index=False)
# agreement of Grad-CAM across heads (Spearman over 7x7 cells, averaged over items) -- recompute from 224 maps pooled to 7x7
ag = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        v = [spearmanr(cv2.resize(maps[(it, names[i], "Grad-CAM")], (7, 7), interpolation=cv2.INTER_AREA).ravel(), cv2.resize(maps[(it, names[j], "Grad-CAM")], (7, 7), interpolation=cv2.INTER_AREA).ravel())[0] for it in range(len(items))]
        ag.append(dict(arm_a=names[i], arm_b=names[j], spearman_mean=float(np.nanmean(v)), spearman_sd=float(np.nanstd(v))))
pd.DataFrame(ag).to_csv(f"{R}/xai_agreement.csv", index=False)
pd.set_option("display.width", 220); print(S.round(3).to_string()); print(pd.DataFrame(ag).round(2).to_string())


# ---------------------------------------------------------------- figure grids (4 displayed items: 2 tumor, 2 no tumor)
def overlay(ax, img, cam, title=None, prob=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255); h = ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.4)
    if title: ax.set_title(title, fontsize=6.8)
    return h


show = pick[1][:2] + pick[0][:2]; sidx = [items.index(p) for p in show]
def grid(cols, fname):                                              # cols: list of (title, arm or None, method or None)
    fig, ax = plt.subplots(len(sidx), len(cols), figsize=(1.18 * len(cols) + 0.5, 1.2 * len(sidx) + 0.35)); hh = None
    for r_, ii in enumerate(sidx):
        for c_, (ttl, a, m) in enumerate(cols):
            if a is None: ax[r_, c_].imshow(maps[(ii, "img")], cmap="gray", vmin=0, vmax=255); ax[r_, c_].set_xticks([]); ax[r_, c_].set_yticks([]); ax[r_, c_].set_title(ttl if r_ == 0 else None, fontsize=6.8); [sp.set_linewidth(0.4) for sp in ax[r_, c_].spines.values()]
            else: hh = overlay(ax[r_, c_], maps[(ii, "img")], maps[(ii, a, m)], ttl if r_ == 0 else None)
        ax[r_, 0].set_ylabel(("Tumor" if yte[items[ii]] == 1 else "No tumor"), fontsize=6.8)
    fig.tight_layout(pad=0.3, w_pad=0.15, h_pad=0.2); cb = fig.colorbar(hh, ax=ax, fraction=0.02, pad=0.01); cb.ax.tick_params(labelsize=5.5); cb.set_label("relative importance", fontsize=6); fig.savefig(f"{FIG}/{fname}.png", bbox_inches="tight", facecolor="white"); plt.close(fig)


cls_a, vq_a = names[0], names[1]; fl_a, flq_a, dp_a = names[2], names[3], names[4]
grid([("Image", None, None)] + [(m, cls_a, m) for m in ("Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Score-CAM")], "fig_xai_classical")
grid([("Image", None, None), ("FL-LR\nGrad-CAM", fl_a, "Grad-CAM"), ("FL-LR\nGrad-CAM++", fl_a, "Grad-CAM++"), ("FL-LR\nScore-CAM", fl_a, "Score-CAM"), ("FL-VQC\nGrad-CAM", flq_a, "Grad-CAM"), ("DP-LR (eps=8)\nGrad-CAM", dp_a, "Grad-CAM")], "fig_xai_fl")
grid([("Image", None, None)] + [(m, vq_a, m) for m in ("Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Score-CAM")], "fig_xai_qml")
open(f"{R}/xai_done.flag", "w").write("ok\n"); print("XAI done", f"{time.time() - t0:.0f}s")
