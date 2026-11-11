"""Figure 1: overall system block diagram of the rebuilt pipeline."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family": "serif", "savefig.dpi": 300})
NAVY, TEAL, CORAL, SAND, PURP = "#2E4A6B", "#4F9D8B", "#C0504D", "#D4A56E", "#7B6B9A"
fig, ax = plt.subplots(figsize=(7.4, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
def box(x, y, w, h, col, title, lines, fs=5.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.9", fc="white", ec=col, lw=1.3)); ax.add_patch(FancyBboxPatch((x, y + h - 6.2), w, 6.2, boxstyle="round,pad=0.2,rounding_size=0.9", fc=col, ec=col, lw=1.3))
    ax.text(x + w / 2, y + h - 3.1, title, ha="center", va="center", fontsize=6.6, color="white", fontweight="bold")
    for i, l in enumerate(lines): ax.text(x + w / 2, y + h - 9.6 - i * 3.5, l, ha="center", va="center", fontsize=fs, color="#222222")
def arrow(x0, y0, x1, y1): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color="#555555", lw=1.0))
W, G = 17.6, 3.0
mods = [("1  Public dataset", NAVY, ["253 MRI images", "155 tumor, 98 none", "audit: duplicates,", "file-size and", "intensity cues"]),
        ("2  Processing", TEAL, ["crop, denoise,", "CLAHE, resize", "duplicate groups", "(hash + corr.)", "stratified folds"]),
        ("3  Split, augment", CORAL, ["folds over groups", "12 augmentations", "of TRAINING", "images only;", "test = originals"]),
        ("4  Frozen backbones", PURP, ["ResNet50, Xception,", "InceptionV3,", "DenseNet121,", "ViT-B/16", "(224 x 224)"]),
        ("5  Three heads", SAND, ["classical (7 models)", "federated (4 clients,", "FedAvg, FedProx, DP)", "quantum (VQC, 4-8", "qubits + controls)"])]
for i, (t, c, ls) in enumerate(mods):
    x = 0.5 + i * (W + G); box(x, 17.5, W, 28, c, t, ls)
    if i < 4: arrow(x + W + 0.2, 31.5, x + W + G - 0.2, 31.5)
ax.add_patch(FancyBboxPatch((0.5, 2.0), 99, 11.5, boxstyle="round,pad=0.2,rounding_size=0.9", fc="#F3F3F3", ec="#888888", lw=1.0, ls="--"))
ax.text(50, 11.0, "Evaluation and explanation (every head, same folds)", ha="center", fontsize=6.9, fontweight="bold", color="#222222")
ax.text(50, 7.0, "accuracy, sensitivity, specificity, F1, MCC, AUC with 95% group-bootstrap intervals; communication payload and privacy budget", ha="center", fontsize=5.8, color="#222222")
ax.text(50, 3.8, "Grad-CAM, Grad-CAM++, Eigen-CAM, Score-CAM through the backbone; deletion / insertion faithfulness; brain-mask check", ha="center", fontsize=5.8, color="#222222")
arrow(92, 17.3, 92, 13.8); arrow(8, 13.6, 8, 17.2)
fig.savefig("figures/fig_system.png", bbox_inches="tight", facecolor="white"); print("saved fig_system.png")
