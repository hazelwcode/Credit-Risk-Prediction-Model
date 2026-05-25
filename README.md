# Credit Default Risk Prediction with Explainable Machine Learning

## Overview

This repository contains the implementation of a thesis project on **credit default risk prediction** using the Home Credit Default Risk dataset.

The project focuses on:

- relational feature engineering for credit-risk data;
- baseline model comparison on a final V15 feature set;
- prediction-level ensemble learning;
- business-oriented model evaluation;
- SHAP-based explainability.

The final thesis-facing feature set is **V15**, and the final thesis-facing model is **SE-HC / SIGMA_FINAL**.

```text
SE-HC = Stacked Ensemble with Hill-Climbing Selection
SPC   = Stacked Prediction Candidate = fp_final

Final SE-HC formula:
SE-HC = 0.5 × SPC + 0.5 × V15 Multi-Seed LightGBM
```

The variants `SE_HC_SELECTED_CANDIDATES` and `SE_HC_SELECTED_CANDIDATES_MS` are additional experiments and are not presented as the final thesis model in this repository.

---

## Environment Setup

### Requirements

Recommended environment:

- Python: 3.10+
- OS: Windows / Linux / macOS
- RAM: 16GB+ recommended for feature engineering and model training
- GPU: Optional, useful for XGBoost and CatBoost training
- CUDA: Optional, only required for GPU-accelerated training

### Installation

1. Clone the repository:

```bash
git clone <your-repository-url>
cd credit_risk_prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Review additional environment notes:

```text
requirements_notes.md
```

Full retraining can be computationally expensive. Some notebooks are designed for inspection and demonstration using saved lightweight summaries rather than full end-to-end execution.

---

## Dataset

This project uses the **Home Credit Default Risk** dataset from Kaggle.

Raw data is **not included** in this repository. To reproduce the full pipeline, download the dataset from Kaggle and place the raw CSV files locally under:

```text
data/raw/
```

Expected local raw files include:

```text
application_train.csv
application_test.csv
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
credit_card_balance.csv
```
---

## Folder Structure

```text
credit_risk_prediction/
├── README.md
├── requirements.txt
├── requirements_notes.md
├── .gitignore
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_v15.ipynb
│   ├── 03_train_baselines_v15.ipynb
│   ├── 04_build_se_hc.ipynb
│   ├── 05_evaluation_and_shap.ipynb
│   └── archive/
│       └── solution_ver02.ipynb
│
├── feature_engineering/
│   ├── feature - bureau & bureau balance.py
│   ├── feature - previous_application.py
│   ├── feature - installments_payments.py
│   ├── feature - POS_CASH_balance.py
│   ├── feature - credit_card_balance.py
│   ├── feature - clustering.py
│   ├── feature - trend.py
│   └── fe_v2/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── evaluation.py
│   ├── ensemble.py
│   ├── io_utils.py
│   └── train_v15_baselines.py
│
├── docs/
│   ├── final_model_lineage.md
│   ├── code_structure.md
│   └── demo_script.md
│
└── reports/
    ├── results/
    │   ├── metrics_SE_HC.json
    │   ├── deciles_SE_HC.csv
    │   └── final_v15_baseline_comparison.csv
    └── figures/
        ├── dashboard_SE_HC.png
        ├── shap_bar_top20.png
        ├── shap_beeswarm_top20.png
        ├── shap_category_importance.png
        └── pipeline_framework.jpg
```

---

## Notebook Pipeline

### 1. Exploratory Data Analysis

```text
notebooks/01_eda.ipynb
```

This notebook explores the application-level and relational Home Credit data.

Main topics:

- target distribution and class imbalance;
- missing value patterns;
- EXT_SOURCE variables;
- demographic and financial variables;
- employment and occupation risk;
- bureau, previous application, installment, POS cash, and credit card behavior.

### 2. Feature Engineering V15

```text
notebooks/02_feature_engineering_v15.ipynb
```

This notebook documents the feature engineering pipeline that produces the final V15 feature set.

Main stages:

- script-based feature generation from relational tables;
- applicant-level aggregation by `SK_ID_CURR`;
- application preprocessing and anomaly handling;
- behavioral aggregation features;
- cross-table interaction features;
- time-window and loan-type features;
- train-test stability filtering;
- final V15 target-score features.

Final V15 matrix:

```text
1,043 columns including SK_ID_CURR and TARGET
1,041 modelling features after excluding ID and target columns
```

### 3. Baseline Training and Candidate Prediction Generation

```text
notebooks/03_train_baselines_v15.ipynb
```

This notebook has two purposes.

First, it trains and compares V15 baseline models on the same final V15 feature representation:

- Logistic Regression;
- MLP;
- LightGBM single seed;
- LightGBM multi-seed;
- XGBoost;
- CatBoost.

Second, it summarizes ensemble candidate predictors. These are OOF prediction vectors later used by the SE-HC ensemble construction stage. Candidate predictors are not all baseline models; some are stacked or historical predictors used to increase ensemble diversity.

### 4. SE-HC Ensemble Construction

```text
notebooks/04_build_se_hc.ipynb
```

This notebook builds the final thesis-facing ensemble model.

Main stages:

- candidate OOF prediction loading;
- SPC / `fp_final` explanation;
- candidate AUC inspection;
- Caruana-style hill-climbing ensemble selection;
- final `SIGMA_FINAL` construction;
- final OOF and metric output generation.

Final ensemble:

```text
SE-HC = 0.5 × SPC + 0.5 × V15 Multi-Seed LightGBM
```

### 5. Evaluation and SHAP Explainability

```text
notebooks/05_evaluation_and_shap.ipynb
```

This notebook evaluates the final SE-HC model and provides explainability analysis.

Main contents:

- OOF evaluation metrics;
- Kaggle private leaderboard benchmark;
- threshold analysis;
- confusion matrix;
- decile and lift analysis;
- baseline comparison;
- SHAP global and local explanations;
- business interpretation of risk drivers.

Because SE-HC is a blended ensemble over prediction vectors, SHAP is used to interpret a representative tree-based component or the dominant predictive logic. It is not presented as an exact additive decomposition of the final blended SE-HC score.

---

## Source Code

### Feature Engineering Scripts

The `feature_engineering/` folder contains modular scripts for table-specific relational feature generation.

| Script | Purpose |
|---|---|
| `feature - bureau & bureau balance.py` | Aggregates external bureau credit history and bureau balance behavior |
| `feature - previous_application.py` | Aggregates previous Home Credit application behavior |
| `feature - installments_payments.py` | Extracts repayment, late payment, and payment-ratio behavior |
| `feature - POS_CASH_balance.py` | Aggregates POS cash loan monthly status |
| `feature - credit_card_balance.py` | Aggregates card utilization, balance, and payment behavior |
| `feature - clustering.py` | Creates cluster-based segmentation features |
| `feature - trend.py` | Creates temporal trend features |

### Reusable Source Modules

The `src/` folder contains reusable project code.

| File | Purpose |
|---|---|
| `config.py` | Common constants, paths, seeds, target column, and categorical columns |
| `evaluation.py` | Evaluation metrics, decile table, lift, KS, and model comparison helpers |
| `ensemble.py` | SE-HC / Caruana hill-climbing helper functions |
| `io_utils.py` | I/O helpers and OOF alignment utilities |
| `train_v15_baselines.py` | V15 baseline training workflow |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare raw data

Download the Kaggle Home Credit Default Risk dataset and place the CSV files under:

```text
data/raw/
```

### 3. Run or inspect notebooks

Recommended notebook order:

```text
notebooks/01_eda.ipynb
notebooks/02_feature_engineering_v15.ipynb
notebooks/03_train_baselines_v15.ipynb
notebooks/04_build_se_hc.ipynb
notebooks/05_evaluation_and_shap.ipynb
```

Some cells require local-only artifacts such as processed V15 matrices, OOF prediction files, or SHAP arrays. Heavy cells are intended for local reproduction and should not be run during a quick demo.

### 4. Run V15 baseline script

The V15 baseline training workflow is available in:

```text
src/train_v15_baselines.py
```

Example:

```bash
python src/train_v15_baselines.py
```

This script requires local V15 feature matrices and may take a long time depending on hardware.

---

## Evaluation Results

Lightweight final results are stored in:

```text
reports/results/
```

Key files:

| File | Description |
|---|---|
| `metrics_SE_HC.json` | Final SE-HC evaluation metrics |
| `deciles_SE_HC.csv` | Final SE-HC decile table |
| `final_v15_baseline_comparison.csv` | Corrected V15 baseline comparison table |

Selected visual outputs are stored in:

```text
reports/figures/
```

Key figures:

| Figure | Description |
|---|---|
| `dashboard_SE_HC.png` | Final SE-HC evaluation dashboard |
| `shap_bar_top20.png` | Top SHAP feature importance |
| `shap_beeswarm_top20.png` | SHAP beeswarm summary |
| `shap_category_importance.png` | Feature category importance |
| `pipeline_framework.jpg` | Project pipeline/framework diagram |

---

## Metrics

The project uses multiple evaluation metrics because credit default prediction is an imbalanced classification problem.

| Metric | Purpose |
|---|---|
| ROC-AUC | Overall ranking quality |
| Gini | Credit-risk ranking metric derived from ROC-AUC |
| KS | Separation between default and non-default score distributions |
| PR-AUC | Minority-class-sensitive ranking metric |
| Lift@10% | Business usefulness in the highest-risk segment |
| Precision / Recall / F1 | Threshold-based operating performance |
| Brier / ECE | Secondary probability-quality indicators |

Accuracy is not used as the main metric because the target distribution is highly imbalanced.

---

## Model Pipeline

The final thesis-facing pipeline is:

```text
Raw Home Credit relational data
→ V15 feature engineering
→ V15 baseline model comparison
→ SPC / fp_final
→ SE-HC / SIGMA_FINAL
→ OOF evaluation, Kaggle benchmark, and SHAP explainability
```

### Final Model

```text
Final model: SE-HC / SIGMA_FINAL
```

### Final Formula

```text
SE-HC = 0.5 × SPC + 0.5 × V15 Multi-Seed LightGBM
```

### SPC

```text
SPC = Stacked Prediction Candidate = fp_final
```

SPC is an intermediate stacked prediction candidate. It is used as one component of the final SE-HC ensemble and is not the same as the final SE-HC model.

---

## Explainability

SHAP is used to explain the main predictive patterns learned by representative tree-based components of the pipeline.

Key SHAP outputs include:

- global feature importance;
- beeswarm summary;
- local explanation cases;
- feature category importance;
- business-level interpretation of major risk drivers.

Since SE-HC is a blend of candidate prediction vectors, SHAP is interpreted as component-level explanation rather than an exact decomposition of the final blended prediction.

---

## Reproducibility Notes

This repository is designed for code review, demonstration, and partial reproduction.

Full reproduction requires local artifacts that are not tracked on GitHub:

- raw Kaggle CSV files;
- processed V15 parquet matrices;
- full OOF prediction files;
- model outputs;
- SHAP arrays;
- large intermediate feature tables.

The clean notebooks document the full thesis pipeline. The archived notebook is retained for traceability of the original experimental development.

---
