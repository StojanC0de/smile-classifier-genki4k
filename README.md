# Smile Classifier (GENKI-4K) — v1 Baseline (PyTorch)

This repository is part of my AI/ML portfolio and documents an iterative approach to building an image classifier.

**Task:** binary classification — **smile (1) vs non-smile (0)** using the **GENKI-4K** dataset.  
**v1 goal:** build a working baseline training pipeline (GPU support) and diagnose model behavior.

>> In early experiments (v1 baseline), the model tends to **overfit**. Future versions will focus on improving generalization.

---

## What’s in v1
- Custom PyTorch `Dataset` for GENKI-4K
- CNN baseline model (4 convolution blocks up to 256 channels)
- Training + validation loop
- CUDA / GPU support (if available)

---

## Dataset (not included)
This repo does **not** include the GENKI-4K dataset.

Your dataset directory should contain:
- `GENKI-4K_Labels.txt`
- `GENKI-4K_Images.txt`
- `files/` (image files)


---
## How to run

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



### 2) Set the dataset path (environment variable)

Before running the script, set the `GENKI4K_DIR` environment variable (Kali/Linux):
```bash
export GENKI4K_DIR="/path/to/your/GENKI-4K"
```
(On Windows PowerShell:)

```powershell
setx GENKI4K_DIR "C:\path\to\GENKI-4K"
```
The dataset directory should contain:

- `GENKI-4K_Labels.txt`
- `GENKI-4K_Images.txt`
- `files/` (image files)

### 3) Train the baseline model
```bash
python src/train_v1_baseline.py
```


## Baseline results & observations (v1)

When training the baseline CNN, the following behavior is typically observed:

- training accuracy increases steadily and becomes high
- validation accuracy remains significantly lower
- a clear gap appears between training and validation performance

This pattern indicates overfitting: the model learns the training data well but does not generalize as effectively to unseen images.

## Next steps (planned)

Future iterations of this project will focus on improving generalization, including:

- stratified train/validation split
- train-only data augmentation
- replacing Sigmoid + BCELoss with BCEWithLogitsLoss
- switching to AdamW with weight decay
- learning-rate scheduling and early stopping
- additional evaluation metrics (confusion matrix, precision/recall/F1)