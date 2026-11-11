# Data Leakage in Brain Tumor MRI Classification: A Study of Classical, Federated, and Quantum Models

Python code for the manuscript *Data Leakage in Brain Tumor MRI Classification: A Study of Classical, Federated, and Quantum Models*.
Everything runs on a single CPU core (no GPU); a full run takes a few hours.

Repository: https://github.com/mdwahidur-rahman/Data-Leakage-in-Brain-Tumor-MRI-Classification

This repository contains the Python code, the result tables, and the figures (`figures/`, 300-dpi PNG, as used in the manuscript). It contains no LaTeX.

## Data
Download the public *Brain MRI Images for Brain Tumor Detection* dataset (253 images: 155 `yes`, 98 `no`) and unzip it so that `./dataset/yes` and `./dataset/no` exist.
The archive stores every image twice (`brain_tumor_dataset/` and top-level folders); only one copy is used.

## Environment
Python 3.11; PyTorch 2.5.1 (CPU), torchvision 0.20.1, timm 0.9.16, scikit-learn, XGBoost, OpenCV, SciPy, pandas, matplotlib (see `requirements.txt`).
Pretrained weights (ResNet50, InceptionV3, Xception, DenseNet121, ViT-B/16) are downloaded from the GitHub release files of the timm model zoo by `features.py`; the SHA-256 prefix of each file is checked against the digits in its file name.

## Pipeline (run from the repository root)
| Step | Command | Output |
|---|---|---|
| 0. Dataset audit | `python audit_brain_mri_images.py` | per-image audit table |
| 1. Preprocess, group, augment | `python prep.py` | `data/` (images, groups, metadata) |
| 2. Frozen features | `python features.py resnet50 full` (also `raw`, `clahe`; other backbones: `inceptionv3`, `xception`, `vit`, `densenet121`) | `feats/` |
| 3. Leakage demonstration | `python leak_demo.py resnet50 full` | `results/leak_demo_*` |
| 4. Classical heads | `python classical.py resnet50 full 1`, `python classical_ablation.py` | `results/classical_*` |
| 5. Quantum head | `REF12_FOLDS=3 python qml_run.py resnet50 full 1 --ref12` | `results/qml_*` |
| 6. Federated heads | `python fl_run.py resnet50 full`, `python fl_extra.py` | `results/fl_*` |
| 7. Explanations | `python xai_run.py; python xai_grid5.py` | `results/xai_*`, `figures/fig_xai_*` |
| 8. Ablations | `python ablate.py` | `results/ablate_*` |
| 9. Figures | `python fig_prep.py; python fig_core.py; python fig_system.py; python make_figs.py` | `figures/` |

`run_pipeline.sh` runs the steps in order; every long step writes checkpoints under `results/ckpt/` and resumes after an interruption.

## Evaluation rules (what makes the protocol leakage-aware)
* Images are grouped by exact and near-duplicate content (perceptual hash proposes, image correlation >= 0.90 confirms): 210 groups.
* Folds are built over groups (`StratifiedGroupKFold`); augmented copies are generated only from the training images of a fold; test sets contain original images only.
* All fitting (standardization, projections, hyperparameters) uses training rows only. Uncertainty is a 95 % group-cluster bootstrap.

## Notes and limitations
Noiseless statevector simulation (pure PyTorch, with unit tests in `heads.py`); the circuit is tuned at 4-8 qubits and evaluated at 12 qubits as a fixed reference only.
Differential privacy is applied at the level of the source image and to the logistic head only. The dimensionality reduction is fitted centrally (outside the privacy guarantee).
DeiT could not be evaluated because its weights were unreachable in our environment.
The dataset has no patient identifiers or tumor masks, so groups are duplicate-based and explanations are not evaluated for localization.

## Results and figures
`results/` holds the result tables and out-of-fold predictions that the manuscript reports; `figures/` holds the figures (system diagram, core pipeline, preprocessing grid and histograms, protocol comparison, classical / quantum / federated results, ablations, and the Grad-CAM grids). `make_figs.py`, `fig_prep.py`, `fig_core.py`, `fig_system.py`, `xai_run.py` and `xai_grid5.py` regenerate them. License and citation information: to be added by the authors.
