# Smile Classifier (GENKI-4K) — PyTorch

This repository is part of my AI/ML portfolio and documents an iterative approach to building an image classifier.

**Task:** binary classification — **smile (1) vs non-smile (0)** using the **GENKI-4K** dataset.  
The project demonstrates how a simple baseline model can be progressively improved to reduce overfitting and improve generalization.

> In early experiments (v1 baseline), the model tends to **overfit**. 
> Version 2 successfully improved generalization and solved the overfitting.

---

## What was in v1 (Baseline)


The first version established a working training pipeline.

Features:

- Custom PyTorch `Dataset` for GENKI-4K
- CNN baseline model (4 convolution blocks up to 256 channels)
- Training + validation loop
- CUDA / GPU support (if available)

Observed behavior:

- training accuracy increases steadily and becomes high
- validation accuracy remains significantly lower
- a clear gap appears between training and validation performance

This pattern indicates **overfitting**.

---

## Current Version — v2 (Regularized Model)

The current implementation focuses on improving model generalization.

Key improvements include:

- train-only **data augmentation**
- **image normalization**
- **Batch Normalization** in convolution blocks
- **weight decay** regularization
- **learning rate scheduling** (`ReduceLROnPlateau`)
- **best-model checkpointing**
- safer dataset handling and file validation


These changes significantly reduce the train/validation gap observed in the baseline model.

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
python src/train_v2_regularized.py
```


## Baseline results & observations (v1)

When training the baseline CNN, the following behavior was observed:

- training accuracy increases steadily and becomes high
- validation accuracy remains significantly lower
- a clear gap appears between training and validation performance

This indicates that the model memorizes the training data but struggles to generalize to unseen images.

## v2 Results & Observations

After introducing regularization techniques:

- validation accuracy becomes significantly more stable
- the train/validation gap is reduced
- training becomes more robust thanks to augmentation and normalization

The model generalizes better to unseen images compared to the baseline.


## Future Improvements

Possible next steps for the project:

- switch to BCEWithLogitsLoss
- experiment with AdamW optimizer
- implement early stopping
- compute confusion matrix
- add precision / recall / F1 score
- create an inference script for single image prediction
