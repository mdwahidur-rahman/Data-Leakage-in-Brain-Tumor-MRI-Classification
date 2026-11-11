"""Figures for Section 3.1: preprocessing grid and intensity histograms, drawn from the prepared arrays (no synthetic data)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
D = "./data"; OUT = "./figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "savefig.dpi": 300})
meta = pd.read_csv(f"{D}/meta.csv"); A = int(meta.aug.max()); G = meta[meta.aug == 0].reset_index(drop=True)
V = {v: np.load(f"{D}/images_{v}.npy", mmap_mode="r") for v in ("raw", "clahe", "full")}
rng = np.random.default_rng(7); pick = [rng.choice(G.index[G.label == 1], 2, replace=False), rng.choice(G.index[G.label == 0], 2, replace=False)]; pick = [int(i) for a in pick for i in a]
cols = ["Original", "Resized (raw)", "CLAHE", "Full pipeline", "Augmentation 1", "Augmentation 2", "Augmentation 3"]
fig, ax = plt.subplots(4, 7, figsize=(7.3, 4.3))
for r, i in enumerate(pick):
    f = G.file[i]; orig = np.asarray(Image.open(f"./dataset/{f}").convert("L"))
    ims = [orig, V["raw"][i * (A + 1)], V["clahe"][i * (A + 1)], V["full"][i * (A + 1)]] + [V["full"][i * (A + 1) + a] for a in (1, 2, 3)]
    for c, im in enumerate(ims):
        ax[r, c].imshow(im, cmap="gray", vmin=0, vmax=255); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        for s in ax[r, c].spines.values(): s.set_linewidth(0.4)
        if r == 0: ax[r, c].set_title(cols[c], fontsize=7)
    ax[r, 0].set_ylabel(("Tumor" if G.label[i] == 1 else "No tumor"), fontsize=7)
fig.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.3); fig.savefig(f"{OUT}/fig_prep_grid.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
# pooled intensity histograms (originals only), with and without the black background
idx = G.index.values * (A + 1); colr = {"raw": "#2E5C8A", "clahe": "#C0504D", "full": "#4F9D8B"}; lab = {"raw": "Resized (raw)", "clahe": "CLAHE", "full": "Full pipeline"}
fig, ax = plt.subplots(1, 3, figsize=(7.3, 2.2))
for v in V:
    arr = np.asarray(V[v][idx]); h = np.bincount(arr.ravel(), minlength=256).astype(float); hn = h.copy(); hn[:8] = 0
    ax[0].plot(np.arange(256), h / h.sum(), color=colr[v], lw=0.9, label=lab[v]); ax[1].plot(np.arange(256), hn / hn.sum(), color=colr[v], lw=0.9, label=lab[v])
ax[0].set_yscale("log"); ax[0].set_title("(a) Pooled histogram (log scale)", fontsize=7.5); ax[1].set_title("(b) Brain tissue only (levels > 7)", fontsize=7.5)
for a in ax[:2]: a.set_xlabel("Gray level"); a.set_ylabel("Fraction of pixels"); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, fontsize=6.5)
sd = {v: np.asarray(V[v][idx]).reshape(len(idx), -1).std(1) for v in V}
bp = ax[2].boxplot([sd[v] for v in V], labels=["Raw", "CLAHE", "Full"], patch_artist=True, widths=0.55, medianprops=dict(color="black", lw=0.8), flierprops=dict(markersize=2))
for p, v in zip(bp["boxes"], V): p.set_facecolor(colr[v]); p.set_alpha(0.75)
ax[2].set_title("(c) Per-image contrast (SD of gray levels)", fontsize=7.5); ax[2].set_ylabel("SD"); ax[2].spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.5); fig.savefig(f"{OUT}/fig_prep_hist.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("saved fig_prep_grid.png, fig_prep_hist.png; median contrast SD raw/clahe/full:", [round(float(np.median(sd[v])), 1) for v in V])
