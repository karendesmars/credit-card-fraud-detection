# Credit Card Fraud Detection

**Status: work in progress, not executed yet.** This repo is private while the notebook is being validated. It will be made public once the results have been checked and reviewed.

**Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud), Machine Learning Group at ULB (Universite Libre de Bruxelles), 284,807 transactions, 492 fraud cases (0.17%)
**Tools:** Python, Pandas, scikit-learn (LogisticRegression, RandomForestClassifier, StratifiedKFold), imbalanced-learn (SMOTE)
**Goal:** Detect fraudulent credit card transactions in a heavily imbalanced dataset.

---

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv                          # Not tracked in git, see Data section below
├── notebooks/
│   └── credit_card_fraud_detection.ipynb       # Full classification workflow
├── environment.yml                             # Conda environment for this project
└── README.md
```

---

## Data

The raw dataset (`creditcard.csv`) is not included in this repo (see `.gitignore`).

To reproduce:
1. Download the dataset from Kaggle: `kaggle datasets download -d mlg-ulb/creditcardfraud`
2. Unzip into `data/`, so the notebook finds `data/creditcard.csv`

The features `V1` to `V28` are already PCA-transformed and anonymized by the original dataset authors, for confidentiality. Only `Time`, `Amount`, and `Class` (the target) keep their original meaning.

---

## Environment

This project uses its own conda environment (`environment.yml`), separate from other projects in this portfolio, to avoid dependency conflicts and keep the project reproducible for anyone cloning the repo.

```
conda env create -f environment.yml
conda activate credit-card-fraud-detection
```

The environment includes pandas, scikit-learn, matplotlib, seaborn, and `imbalanced-learn` (used for the SMOTE step).

---

## Approach

| # | Step | Focus |
|---|---|---|
| 1 | Data exploration | Load, check nulls, class balance, `Amount`/`Time` distribution by class, correlation with target |
| 2 | Preprocessing | Scale `Amount` and `Time` with `RobustScaler`, fit on the training set only; stratified train/test split |
| 3 | Baseline model | LogisticRegression with no imbalance handling, as a reference point |
| 4 | Class weights | LogisticRegression and RandomForest with `class_weight="balanced"` |
| 5 | SMOTE | Oversample the minority class on the training set only (after the split, to avoid leakage), then retrain both models |
| 6 | Model comparison | Precision, recall, f1-score, ROC-AUC, and PR-AUC (average precision) for all five models on the same test set |
| 7 | Feature importance | RandomForest feature importances (`class_weight="balanced"` version, no synthetic rows) |
| 8 | Cross-validation | Stratified 5-fold cross-validation scored on PR-AUC for the best-performing model |

See `notebooks/credit_card_fraud_detection.ipynb` for the full step-by-step workflow.

---

## Why accuracy is not used as the main metric

With only 0.17% fraud cases, a model predicting "not fraud" for every single transaction would already score above 99.8% accuracy while catching zero fraud. Precision, recall, f1-score, and PR-AUC on the fraud class are the metrics that actually matter here, which is why the evaluation function used throughout this notebook reports those instead of accuracy.

---

## Key Findings

Not written yet. This section will be filled in once the notebook has been run on the real data and the results have been reviewed.
