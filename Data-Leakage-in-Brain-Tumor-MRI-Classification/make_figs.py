"""All results figures from the stored CSVs (no hard-coded numbers). Missing inputs are skipped with a message."""
import os, json, warnings, numpy as np, pandas as pd, matplotlib
warnings.filterwarnings("ignore"); matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "./results"; FIG = "./figures"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
NAVY, TEAL, CORAL, SAND, PURP, GREY = "#2E4A6B", "#4F9D8B", "#C0504D", "#D4A56E", "#7B6B9A", "#999999"
def rd(f): p = f"{R}/{f}"; return pd.read_csv(p) if os.path.exists(p) else None
def save(fig, n): fig.savefig(f"{FIG}/{n}.png", bbox_inches="tight", facecolor="white"); plt.close(fig); print("saved", n)
def errbar(ax, x, d, col, w=0.8, **kw):
    ax.bar(x, 100 * d["acc"], w, color=col, yerr=[100 * (d["acc"] - d["acc_lo"]), 100 * (d["acc_hi"] - d["acc"])], capsize=1.5, error_kw=dict(lw=0.6), **kw)
# ---- leakage protocols
L = rd("leak_demo_summary_resnet50_full.csv")
if L is not None:
    lab = {"P1": "Augment-then-split\n(row level)", "P2": "Image-level\nsplit", "P3": "Duplicate-group\nsplit (primary)", "P4": "Group split,\nno augmentation", "P5": "Conservative\nclusters (0.40)", "P6": "Very conservative\nclusters (0.30)"}
    L["k"] = L.protocol.str[:2]; L = L.set_index("k").loc[list(lab)]; fig, ax = plt.subplots(1, 2, figsize=(7.3, 3.1)); cols = [CORAL, SAND, NAVY, GREY, TEAL, PURP]
    for a, m, t in ((ax[0], "acc", "(a) Accuracy (%)"), (ax[1], "mcc", "(b) Matthews correlation coefficient")):
        sc = 100 if m == "acc" else 1; a.bar(range(6), sc * L[m + "_mean"], 0.72, color=cols, yerr=sc * L[m + "_std"], capsize=1.5, error_kw=dict(lw=0.6)); a.set_xticks(range(6)); a.set_xticklabels([lab[k].replace(chr(10), ' ') for k in lab], fontsize=5.8, rotation=28, ha='right'); a.set_title(t, fontsize=7.5)
        for i, v in enumerate(sc * L[m + "_mean"]): a.text(i, v + (2.2 if m == "acc" else 0.03), f"{v:.1f}" if m == "acc" else f"{v:.2f}", ha="center", fontsize=6)
    ax[0].set_ylim(50, 104); ax[1].set_ylim(0, 1.08); fig.tight_layout(w_pad=1.0); save(fig, "fig_leak_protocols")
# ---- classical track
cl = [rd(f"classical_summary_{b}_full.csv") for b in ("resnet50", "inceptionv3", "xception", "vit", "densenet121")]; names = ["ResNet50", "InceptionV3", "Xception", "ViT-B/16", "DenseNet121"]
if cl[0] is not None:
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1]))
    c0 = cl[0].set_index("classifier").loc[["LR", "SVC", "KNN", "XGB", "RF", "DT", "NB"]]; errbar(ax[0], range(7), c0, [NAVY, NAVY, TEAL, TEAL, SAND, SAND, GREY]); ax[0].set_xticks(range(7)); ax[0].set_xticklabels(c0.index, fontsize=6.5); ax[0].set_ylim(30, 100); ax[0].set_ylabel("Accuracy (%)"); ax[0].set_title("(a) ResNet50 features, seven classifiers", fontsize=7.5)
    for i, v in enumerate(100 * c0["acc"]): ax[0].text(i, 100 * c0["acc_hi"].iloc[i] + 1, f"{v:.1f}", ha="center", fontsize=5.8)
    best = [(n, d.loc[d.acc.idxmax()]) for n, d in zip(names, cl) if d is not None]; errbar(ax[1], range(len(best)), pd.DataFrame([b[1] for b in best]).reset_index(drop=True), [NAVY, TEAL, PURP, CORAL, SAND][:len(best)])
    ax[1].set_xticks(range(len(best))); ax[1].set_xticklabels([f"{n}\n({b['classifier']})" for n, b in best], fontsize=5.8); ax[1].set_ylim(60, 100); ax[1].set_title("(b) Best classifier per backbone", fontsize=7.5)
    for i, (n, b) in enumerate(best): ax[1].text(i, 100 * b["acc_hi"] + 0.8, f"{100 * b['acc']:.1f}", ha="center", fontsize=5.8)
    fig.tight_layout(w_pad=1.0); save(fig, "fig_classical")
# ---- QML track
Q = rd("qml_summary_resnet50_full.csv"); S = rd("qml_selected_resnet50_full.csv")
if Q is not None and S is not None:
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.8), gridspec_kw=dict(width_ratios=[1, 1.25]))
    tr = pd.concat([pd.DataFrame(json.loads(t)).assign(fold=f) for f, t in zip(S.fold, S.trace)]); H = tr.groupby(["L", "q"]).val_auc.mean().unstack(); im = ax[0].imshow(100 * H.values, cmap="Blues", aspect="auto", vmin=100 * H.values.min() - 1, vmax=100 * H.values.max() + 0.5)
    ax[0].set_xticks(range(len(H.columns))); ax[0].set_xticklabels(H.columns); ax[0].set_yticks(range(len(H.index))); ax[0].set_yticklabels(H.index); ax[0].set_xlabel("Qubits"); ax[0].set_ylabel("Layers"); ax[0].set_title("(a) Mean validation AUC (%)\n(over folds and entanglers)", fontsize=7.2)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]): ax[0].text(j, i, f"{100 * H.values[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white" if H.values[i, j] > H.values.mean() else "black")
    order = [m for m in ["VQC (tuned)", "VQC 12 qubits (fixed config)", "LR on identical inputs", "SVM on identical inputs", "MLP on identical inputs"] if m in set(Q.method)]; q2 = Q.set_index("method").loc[order].reset_index(); errbar(ax[1], range(len(q2)), q2, [CORAL, "#E39A97", NAVY, TEAL, SAND][:len(q2)])
    ax[1].set_xticks(range(len(q2))); ax[1].set_xticklabels(["VQC\n(tuned)", "VQC 12 q\n(fixed)", "LR", "RBF-SVM", "MLP"][:len(q2)], fontsize=6.3); ax[1].set_ylim(60, 100); ax[1].set_ylabel("Accuracy (%)"); ax[1].set_title("(b) Quantum head vs controls on identical inputs", fontsize=7.2)
    for i, v in enumerate(100 * q2["acc"]): ax[1].text(i, 100 * q2["acc_hi"].iloc[i] + 0.8, f"{v:.1f}", ha="center", fontsize=5.8)
    fig.tight_layout(w_pad=1.0); save(fig, "fig_qml")
# ---- federated track
F = rd("fl_summary_resnet50_full.csv")
if F is not None:
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    order = [m for m in ["Local-only (mean of 4 clients)", "FL-LR FedAvg", "FL-LR FedProx (mu=0.1)", "FL-VQC FedAvg", "DP-LR eps=8", "DP-LR eps=3", "DP-LR eps=1"] if m in set(F.method)]; f2 = F.set_index("method").loc[order].reset_index()
    errbar(ax[0], range(len(f2)), f2, [GREY, NAVY, TEAL, CORAL, SAND, "#C9A05F", "#A67C3D"][:len(f2)]); ax[0].set_xticks(range(len(f2))); ax[0].set_xticklabels(["Local\nonly", "FedAvg\n(LR)", "FedProx\n(LR)", "FedAvg\n(VQC)", "DP\n\u03b5=8", "DP\n\u03b5=3", "DP\n\u03b5=1"][:len(f2)], fontsize=6.2); ax[0].set_ylim(40, 100); ax[0].set_ylabel("Accuracy (%)"); ax[0].set_title("(a) Federated arms, four non-IID clients", fontsize=7.5)
    for i, v in enumerate(100 * f2["acc"]): ax[0].text(i, 100 * f2["acc_hi"].iloc[i] + 0.8, f"{v:.1f}", ha="center", fontsize=5.8)
    dp = F[F.method.str.startswith("DP-LR")].copy(); dp["eps"] = dp.method.str.extract(r"eps=(\d+)").astype(float); dp = dp.sort_values("eps"); nodp = F[F.method == "FL-LR FedAvg"]
    ax[1].errorbar(dp.eps, 100 * dp.acc, yerr=[100 * (dp.acc - dp.acc_lo), 100 * (dp.acc_hi - dp.acc)], fmt="o-", color=CORAL, ms=3.5, lw=1, capsize=2, label="DP-LR (image-level DP-SGD)")
    if len(nodp): ax[1].axhline(100 * nodp.acc.iloc[0], color=NAVY, lw=0.9, ls="--", label="FedAvg without noise")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks([1, 3, 8]); ax[1].set_xticklabels(["1", "3", "8"]); ax[1].set_xlabel("Privacy budget \u03b5 (\u03b4 = 10\u207b\u2075)"); ax[1].set_ylabel("Accuracy (%)"); ax[1].set_title("(b) Privacy\u2013utility trade-off", fontsize=7.5); ax[1].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(w_pad=1.0); save(fig, "fig_fl")
# ---- all-arm comparison (Section 5.1)
rows = []
if cl[0] is not None:
    for c in ("LR", "SVC"): rows.append(("Classical " + c + " (2048-D PCA-256)", cl[0].set_index("classifier").loc[c], NAVY))
if Q is not None:
    for m, n_, c in (("VQC (tuned)", "Quantum head, tuned", CORAL), ("VQC 12 qubits (fixed config)", "Quantum head, 12 qubits", "#E39A97"), ("LR on identical inputs", "LR, identical inputs", TEAL)):
        if m in set(Q.method): rows.append((n_, Q.set_index("method").loc[m], c))
if F is not None:
    for m, n_, c in (("FL-LR FedAvg", "Federated LR", PURP), ("FL-VQC FedAvg", "Federated quantum head", "#A99BC5"), ("DP-LR eps=8", "Private LR (\u03b5 = 8)", SAND)):
        if m in set(F.method): rows.append((n_, F.set_index("method").loc[m], c))
if rows:
    fig, ax = plt.subplots(1, 2, figsize=(7.3, 2.9), sharey=True); y = np.arange(len(rows))[::-1]
    for a, m, t in ((ax[0], "acc", "(a) Accuracy (%)"), (ax[1], "mcc", "(b) MCC")):
        sc = 100 if m == "acc" else 1
        for yi, (n, r, c) in zip(y, rows): a.barh(yi, sc * r[m], 0.62, color=c, xerr=[[sc * (r[m] - r[m + "_lo"])], [sc * (r[m + "_hi"] - r[m])]], capsize=1.5, error_kw=dict(lw=0.6)); a.text(sc * r[m + "_hi"] + (0.8 if m == "acc" else 0.012), yi, f"{sc * r[m]:.1f}" if m == "acc" else f"{r[m]:.2f}", va="center", fontsize=5.8)
        a.set_title(t, fontsize=7.5)
    ax[0].set_yticks(y); ax[0].set_yticklabels([r[0] for r in rows], fontsize=6.2); ax[0].set_xlim(40, 104); ax[1].set_xlim(0, 1.08); fig.tight_layout(w_pad=0.8); save(fig, "fig_compare")
# ---- ablations
P = rd("classical_summary_resnet50_full.csv"); Pr, Pc = rd("classical_summary_resnet50_raw.csv"), rd("classical_summary_resnet50_clahe.csv"); AR, AU = rd("ablate_rep.csv"), rd("ablate_aug.csv")
if AU is not None and AR is not None:
    fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.5))
    if P is not None and Pr is not None and Pc is not None:
        for j, (c, col) in enumerate((("LR", NAVY), ("SVC", TEAL))):
            v = [d.set_index("classifier").loc[c] for d in (Pr, Pc, P)]; ax[0].bar(np.arange(3) + (j - 0.5) * 0.38, [100 * r["acc"] for r in v], 0.36, color=col, yerr=[[100 * (r["acc"] - r["acc_lo"]) for r in v], [100 * (r["acc_hi"] - r["acc"]) for r in v]], capsize=1.5, error_kw=dict(lw=0.6), label=c)
        ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["Raw", "CLAHE", "Full"]); ax[0].set_ylim(70, 95); ax[0].set_ylabel("Accuracy (%)"); ax[0].set_title("(a) Preprocessing", fontsize=7.5); ax[0].legend(frameon=False, fontsize=6)
    ax[1].errorbar(AU.n_aug, 100 * AU.acc, yerr=[100 * (AU.acc - AU.acc_lo), 100 * (AU.acc_hi - AU.acc)], fmt="o-", color=CORAL, ms=3, lw=1, capsize=2); ax[1].set_xlabel("Augmented rows per training image"); ax[1].set_ylabel("Accuracy (%)"); ax[1].set_title("(b) Augmentation dose", fontsize=7.5)
    for (kind, mname), st in {("pca", "LR"): ("o--", NAVY), ("lda_pca", "LR"): ("o-", NAVY), ("pca", "RBF-SVM"): ("s--", TEAL), ("lda_pca", "RBF-SVM"): ("s-", TEAL)}.items():
        d = AR[(AR.name == f"rep:{kind}") & (AR.model == mname)].sort_values("q"); ax[2].plot(d.q, 100 * d.acc, st[0], color=st[1], ms=3, lw=1, label=f"{'LDA+PCA' if kind == 'lda_pca' else 'PCA'}, {mname}")
    ax[2].set_xticks([4, 8, 12]); ax[2].set_xlabel("Inputs q (= qubits)"); ax[2].set_ylabel("Accuracy (%)"); ax[2].set_title("(c) Input representation", fontsize=7.5); ax[2].legend(frameon=False, fontsize=5.4, loc="lower right")
    fig.tight_layout(w_pad=1.0); save(fig, "fig_ablation")
