# Credit Card Fraud Detection

**Status: executed, results reviewed.** This repo stays private for now, until it has been fully reviewed and Karen is confident presenting it. It will be made public once ready.

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

- **RandomForest handles this imbalance much better than LogisticRegression.** With `class_weight="balanced"` or SMOTE, LogisticRegression reaches high recall (91.8%) but its precision collapses to about 6%: it correctly flags most fraud, but floods the results with false positives. RandomForest keeps a much better balance: 78.6% recall at 91.7% precision with class weights, and 83.7% recall at 86.3% precision with SMOTE.
- **RandomForest + SMOTE is the best model overall on this test set** (PR-AUC 0.870, F1 0.850), slightly ahead of RandomForest + class_weight (PR-AUC 0.863, F1 0.846). The baseline LogisticRegression, with no imbalance handling at all, comes third (PR-AUC 0.739, F1 0.724): it still works reasonably, since 0.17% fraud is imbalanced but not extreme, but it misses more fraud than either RandomForest variant (64.3% recall vs 78-84%).
- **5-fold cross-validation on the class-weighted RandomForest gives PR-AUC 0.851 (+/- 0.028)**, close to the single test-split score (0.863), which suggests the result is reasonably stable and not just a lucky split.
- **`V14`, `V10`, `V4`, `V12`, and `V17` are the most important features** for the RandomForest model, together making up more than half of the total feature importance. `Amount` barely matters (1.2% importance): the anonymized PCA components carry almost all the signal, which is expected since PCA was applied specifically to concentrate the fraud-relevant variance into a handful of components.
- **Takeaway for model choice**: if the goal is to flag as much fraud as possible regardless of false positives, a class-weighted or SMOTE-trained LogisticRegression is not usable in practice here (94% of its fraud flags are false alarms). RandomForest with SMOTE is the more balanced, production-realistic choice on this dataset.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RandomForest (SMOTE) | 0.863 | 0.837 | 0.850 | 0.973 | 0.870 |
| RandomForest (class_weight=balanced) | 0.917 | 0.786 | 0.846 | 0.957 | 0.863 |
| Baseline LogisticRegression | 0.829 | 0.643 | 0.724 | 0.957 | 0.739 |
| LogisticRegression (SMOTE) | 0.059 | 0.918 | 0.111 | 0.971 | 0.725 |
| LogisticRegression (class_weight=balanced) | 0.061 | 0.918 | 0.114 | 0.972 | 0.718 |
