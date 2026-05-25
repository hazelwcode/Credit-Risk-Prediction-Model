# Home Credit Default Risk Thesis Repository

This repository contains the cleaned bachelor thesis machine learning work for the Kaggle Home Credit Default Risk problem. The final thesis-facing model is **SE-HC: Stacked Ensemble with Hill-Climbing Selection**, implemented as `SIGMA_FINAL`.

Large raw/generated artifacts are intentionally excluded from GitHub. The repository keeps the clean notebooks, feature-engineering scripts, final lightweight summaries, selected figures, and documentation needed for thesis defense and video demonstration.

## Dataset

The project uses the Kaggle Home Credit Default Risk dataset. Raw Kaggle data is not included because of size and redistribution constraints.

Download the dataset from Kaggle and place the raw CSV files locally under:

```text
data/raw/
```

Expected raw files include `application_train.csv`, `application_test.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `installments_payments.csv`, `credit_card_balance.csv`, and `sample_submission.csv`.

## Final Pipeline

The final thesis pipeline is:

```text
EDA
-> V15 feature engineering
-> V15 baseline comparison
-> SPC / fp_final
-> SE-HC / SIGMA_FINAL
-> evaluation and SHAP explainability
```

Final naming:

- SPC = Stacked Prediction Candidate = `fp_final`
- V15 Multi-Seed LightGBM = `lgb_v15_ms`
- SE-HC = Stacked Ensemble with Hill-Climbing Selection = `SIGMA_FINAL`

Final model formula:

```text
SE-HC = 0.5 * SPC + 0.5 * V15 Multi-Seed LightGBM
```

The final thesis-facing feature representation is **V15**. Baseline comparison models are trained/evaluated on V15 for a fair algorithm-level comparison.

## Repository Structure

```text
notebooks/              clean final-pipeline notebooks and archived source notebooks
feature_engineering/    table-specific feature engineering scripts
src/                    reusable project code and V15 baseline workflow
docs/                   lineage, code-structure, and demo documentation
reports/results/        lightweight final result summaries
reports/figures/        selected thesis/demo figures
```

See `docs/code_structure.md` for the full target structure and local-only folders.

## How To Inspect The Final Pipeline

Read the clean notebooks in order:

1. `notebooks/01_eda.ipynb`
2. `notebooks/02_feature_engineering_v15.ipynb`
3. `notebooks/03_train_baselines_v15.ipynb`
4. `notebooks/04_build_se_hc.ipynb`
5. `notebooks/05_evaluation_and_shap.ipynb`

The original notebooks are kept under `notebooks/archive/` for traceability.

The V15 baseline helper is available at:

```text
src/train_v15_baselines.py
```

## Final Results

The final SE-HC / `SIGMA_FINAL` OOF ROC-AUC is:

```text
0.801688
```

The final Kaggle private leaderboard score reported for the thesis/demo is:

```text
0.79939
```

Full reproducibility requires external Kaggle data and regenerated local artifacts that are not committed to this repository.
