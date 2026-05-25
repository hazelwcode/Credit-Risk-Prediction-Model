# Requirements Notes

This repository is a cleaned thesis/demo version of the Home Credit Default Risk project.

`requirements.txt` lists the main packages needed to inspect notebooks and rerun local pipeline components. It is intentionally lightweight and not a frozen environment lockfile. Exact reproduction also requires the external Kaggle dataset and local generated artifacts that are not committed to GitHub.

Recommended setup:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Full reproducibility requires:

- Raw Home Credit Default Risk CSV files downloaded from Kaggle and placed under `data/raw/`.
- Generated V15 matrices under `processed_train_test/`.
- Local OOF prediction artifacts under `outputs/oof_predictions/`.
- Cached SHAP/model artifacts under `outputs/shap_output/` for full explainability reruns.

These artifacts are intentionally gitignored because they are large generated files or externally licensed data.
