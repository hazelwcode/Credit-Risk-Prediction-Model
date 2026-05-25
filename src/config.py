"""Shared constants and lightweight path configuration."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_TEST_DIR = PROJECT_ROOT / "processed_train_test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OOF_DIR = OUTPUT_DIR / "oof_predictions"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
METRICS_DIR = EVALUATION_DIR / "metrics"
DECILES_DIR = EVALUATION_DIR / "deciles"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_RESULTS_DIR = REPORTS_DIR / "results"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_V15_PATH = PROCESSED_TRAIN_TEST_DIR / "train_merged_v15.parquet"
TEST_V15_PATH = PROCESSED_TRAIN_TEST_DIR / "test_merged_v15.parquet"

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
EXCLUDE_COLS = [ID_COL, TARGET_COL, "index"]

RANDOM_STATE = 42
SEED = RANDOM_STATE
N_FOLDS = 5

V15_CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "BURO_CLU_LABEL",
    "PREV_CLU_LABEL",
]

