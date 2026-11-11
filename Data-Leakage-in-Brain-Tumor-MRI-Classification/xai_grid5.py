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
print("reproduction check (max |p_rebuilt - p_stored|):", {c["arm"]: round(c["max_abs_diff"], 8) for c in chk}, f"{time.time() - t0:.0f}s", flush=True)

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



# ---- Grad-CAM of all five heads on the four displayed images (2 tumor, 2 non-tumor): same items as xai_run.py
show = pick[1][:2] + pick[0][:2]; cols = {}
fig_items = []
for pos in show:
    row = int(rows_test(te)[pos]); x01 = tensor_of(row); A = bone.forward_features(to_net(x01[None])).detach(); entry = {"img": np.asarray(IMF[row]), "label": int(yte[pos])}
    for a in names:
        pr = float(arms[a][1][pos]); sign = 1.0 if pr >= 0.5 else -1.0; gc, _ = cam_set(A, arms[a][0], sign, x01); entry[a] = norm01(up(gc)).numpy(); entry[a + "_p"] = pr
    fig_items.append(entry)
fig, ax = plt.subplots(len(fig_items), 1 + len(names), figsize=(1.15 * (1 + len(names)) + 0.6, 1.2 * len(fig_items) + 0.5)); hh = None
titles = ["Image", "Classical LR\n(2048-D)", "Quantum\n(central)", "FedAvg\nlogistic", "FedAvg\nquantum", "Private LR\n(eps = 8)"]
order = [names[0], names[1], names[2], names[3], names[4]]
for r_, e in enumerate(fig_items):
    ax[r_, 0].imshow(e["img"], cmap="gray", vmin=0, vmax=255); ax[r_, 0].set_xticks([]); ax[r_, 0].set_yticks([]); ax[r_, 0].set_ylabel("Tumor" if e["label"] == 1 else "No tumor", fontsize=6.8)
    for c_, a in enumerate(order):
        ax[r_, c_ + 1].imshow(e["img"], cmap="gray", vmin=0, vmax=255); hh = ax[r_, c_ + 1].imshow(e[a], cmap="jet", alpha=0.45, vmin=0, vmax=1); ax[r_, c_ + 1].set_xticks([]); ax[r_, c_ + 1].set_yticks([])
        ax[r_, c_ + 1].text(3, 215, f"p = {e[a + '_p']:.2f}", fontsize=5.2, color="white", va="bottom", bbox=dict(facecolor="black", alpha=0.45, pad=0.8, lw=0))
    if r_ == 0:
        for c_, t in enumerate(titles): ax[0, c_].set_title(t, fontsize=6.4)
for a_ in ax.ravel():
    for sp in a_.spines.values(): sp.set_linewidth(0.4)
fig.tight_layout(pad=0.3, w_pad=0.15, h_pad=0.2); cb = fig.colorbar(hh, ax=ax, fraction=0.02, pad=0.01); cb.ax.tick_params(labelsize=5.5); cb.set_label("relative importance", fontsize=6)
fig.savefig(f"{FIG}/fig_xai_compare.png", bbox_inches="tight", facecolor="white"); print("saved fig_xai_compare.png", f"{time.time() - t0:.0f}s")
