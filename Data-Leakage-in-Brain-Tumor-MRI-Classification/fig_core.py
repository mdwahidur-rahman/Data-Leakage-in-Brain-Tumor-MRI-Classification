"""Figure 2: core hybrid federated + quantum pipeline (input chain, variational circuit, private federated round)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300})
NAVY, TEAL, CORAL, SAND, PURP = "#2E4A6B", "#4F9D8B", "#C0504D", "#D4A56E", "#7B6B9A"
fig, ax = plt.subplots(figsize=(7.4, 4.1)); ax.set_xlim(0, 100); ax.set_ylim(-1, 59); ax.axis("off")
def box(x, y, w, h, txt, col, fs=6.4, tc="white", fc=None, lw=1.1, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=fc or col, ec=col, lw=lw)); ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else None, linespacing=1.15)
def arrow(x0, y0, x1, y1, col="#555555"): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color=col, lw=0.9))
T = dict(fontsize=7.2, fontweight="bold", color="#222222")
ax.text(1, 57.8, "(a) Leakage-safe input chain (every step fit on training rows only)", **T)
chain = [("MRI slice, preprocessed\n(+ training-only\naugmentation)", NAVY), ("Frozen backbone\nResNet50 / ViT-B/16,\nglobal pooling", TEAL), ("Standardize\nfeatures", PURP), ("LDA(1) + PCA(q\u22121)\nprojection to q inputs", CORAL), ("Rotation angles\nx\u1d62 \u2208 [0, \u03c0]", SAND)]
for i, (t, c) in enumerate(chain):
    x = 1 + i * 19.7; box(x, 46.5, 17.2, 9.6, t, c, fs=5.9)
    if i < 4: arrow(x + 17.4, 51.3, x + 19.5, 51.3)
oy = 12.0
ax.text(1, 29.6 + oy, "(b) Variational quantum classifier", **T)
wires = [24.6 + oy, 20.6 + oy, 16.6 + oy, 10.8 + oy]; names = ["q\u2080", "q\u2081", "q\u2082", "q\u2099\u208b\u2081"]
for y, n in zip(wires, names): ax.plot([4.5, 48.5], [y, y], color="#444444", lw=0.7); ax.text(2.2, y, n, fontsize=6, va="center", ha="center")
ax.plot([3.0, 3.0, 3.0], [15.4 + oy, 13.8 + oy, 12.2 + oy], ".", color="#444444", ms=2)
def gate(x, y, t, col, w=4.6): ax.add_patch(Rectangle((x - w / 2, y - 1.5), w, 3.0, fc=col, ec="#333333", lw=0.6)); ax.text(x, y, t, ha="center", va="center", fontsize=5.1, color="white")
for y in wires: gate(8.5, y, "R\u0079(x)", NAVY)
for blk, x0 in enumerate((17.5, 32.5)):
    for y in wires: gate(x0, y, "Rot", TEAL)
    for (ya, yb) in ((24.6 + oy, 20.6 + oy), (16.6 + oy, 10.8 + oy)):
        xc = x0 + 5.4; ax.plot([xc, xc], [ya, yb], color="#333333", lw=0.8); ax.add_patch(Circle((xc, ya), 0.5, fc="#333333")); ax.add_patch(Circle((xc, yb), 1.0, fc="white", ec="#333333", lw=0.8)); ax.plot([xc - 1.0, xc + 1.0], [yb, yb], color="#333333", lw=0.7); ax.plot([xc, xc], [yb - 1.0, yb + 1.0], color="#333333", lw=0.7)
    ax.text(x0 + 2.7, 28.2 + oy, "layer", fontsize=5.3, ha="center", color="#555555")
ax.text(46.8, 28.2 + oy, "\u00d7 L", fontsize=6, ha="center", color="#555555")
ax.add_patch(Rectangle((44.4, 23.1 + oy), 3.6, 3.0, fc="white", ec="#333333", lw=0.7)); ax.text(46.2, 24.6 + oy, "\u27e8Z\u27e9", fontsize=5.2, ha="center", va="center")
ax.text(26, 6.2 + oy, "logit = a\u27e8Z\u2080\u27e9 + b ,  trainable parameters: 3nL + 2", fontsize=6.0, ha="center", color="#222222")
ax.text(26, 2.8 + oy, "noiseless simulation: RY encoding \u2192 L \u00d7 (Rot + CNOT entangler) \u2192 readout", fontsize=5.5, ha="center", color="#555555")
ax.text(55, 42.0, "(c) Private federated round (image-level privacy)", **T)
cy = [34.8, 28.6, 22.4, 16.2]
for i, y in enumerate(cy):
    box(55.5, y, 14.5, 5.0, f"Client {i + 1}\nlocal images", NAVY, fs=5.5); box(72.5, y, 12.5, 5.0, "clip + noise\nlocal Adam", CORAL, fs=5.3)
    arrow(70.2, y + 2.5, 72.3, y + 2.5); arrow(85.2, y + 2.5, 89.2, 25.6)
box(89.5, 17.8, 9.5, 15.0, "Server\nFedAvg\n\u03b8 \u2190 \u03a3 w\u2096 \u03b8\u2096", TEAL, fs=5.7, bold=True)
box(55.5, 2.5, 43.5, 9.0, "RDP accountant: Poisson sampling of images, noise \u03c3C, \u03b4 = 10\u207b\u2075 \u2192 \u03b5\nGlobal \u03b8 is broadcast each round; all clients share one head\n(logistic or VQC), so every round exchanges only the head parameters", "#888888", fs=4.9, tc="#222222", fc="#F3F3F3")
fig.savefig("figures/fig_core_pipeline.png", bbox_inches="tight", facecolor="white"); print("saved")
