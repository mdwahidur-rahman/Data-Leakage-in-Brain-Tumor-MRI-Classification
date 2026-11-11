"""Preprocessing ablation on ResNet50 features: logistic regression and SVC under the primary protocol, raw vs CLAHE vs full pipeline."""
import sys; sys.argv = sys.argv[:1]
import classical
for var in ("raw", "clahe"): classical.run("resnet50", var, reps=(0,), only=("LR", "SVC"))
