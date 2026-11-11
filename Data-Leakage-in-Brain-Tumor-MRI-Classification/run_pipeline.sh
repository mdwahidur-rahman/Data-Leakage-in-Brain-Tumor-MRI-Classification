#!/bin/bash
set -e
python audit_brain_mri_images.py
python prep.py
for v in full raw clahe; do python features.py resnet50 $v; done
python leak_demo.py resnet50 full
python classical.py resnet50 full 1
REF12_FOLDS=3 python qml_run.py resnet50 full 1 --ref12
python fl_run.py resnet50 full
python fl_extra.py
python xai_run.py
python xai_grid5.py
python classical_ablation.py
python ablate.py
for b in inceptionv3 xception vit; do python features.py $b full; python classical.py $b full 1; done
python fig_prep.py; python fig_core.py; python fig_system.py; python make_figs.py
