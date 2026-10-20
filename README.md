# General-Purpose versus Retina-Specific Vision Foundation Models for Dementia Assessment from Retinal Fundus Photographs

This repository contains the official implementation of the study:  
**"General-purpose versus retina-specific vision foundation models for dementia assessment from retinal fundus photographs"**

The repository provides code for data preparation, model training, inference, performance evaluation, and explainability analysis used in the manuscript.

---

## 🧩 Installation

This codebase requires Python 3.11.8.  
Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

Paths in the scripts and notebooks are placeholders (`/path/to/...`); update them to match your environment.

---

## 📂 Data Preparation

Cohort construction is implemented in:

* **`data_preparation.ipynb`** — builds the development dataset for prevalent dementia detection (1:4 control sampling, 6:2:2 split without overlap of individuals) and the future-incidence cohort, and writes `dementia_detection.csv` and `dementia_prediction.csv`

The fundus photographs and clinical data are not publicly available.

---

## 🧠 Model Training

Model training can be initiated using the provided shell script:

```bash
bash dementia.sh
```

The training process relies on the following key components:

* `trainer.py` — main training loop, optimization logic and inference
* `data.py` — dataset loading and augmentation
* `models/` — foundation model backbones and adaptation strategies (linear probing, partial fine-tuning, LoRA)
* `dementia.sh` — shell script for launching training across 7 backbones × 3 adaptation strategies (21 configurations)

---

## 🔍 Inference and Performance Evaluation

Model inference and predictive performance evaluation are implemented in:

* **`trainer.py`** — saves per-image predictions for the validation set, test set and future-incidence cohort (`valid_preds.csv`, `test_preds.csv`, `prediction_preds.csv`); `--eval_only` runs inference with an existing checkpoint
* **`saliency and evaluation.ipynb`** — summarizes model performance: AUROC, sensitivity and specificity at the validation-set Youden threshold and C-index (bootstrap 95% CIs), and odds/hazard ratios adjusted for CAIDE risk factors

---

## 💡 Explainability Analysis

For saliency map generation (gradient-weighted attention rollout), regional saliency quantification, and visualization, see:

* **`saliency and evaluation.ipynb`**

---

## 📘 Citation

If you use this code or reproduce results from the study, please cite our paper:

```text
to be updated
```
