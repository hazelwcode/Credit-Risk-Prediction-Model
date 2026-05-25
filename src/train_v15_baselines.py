from __future__ import annotations

import gc
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

import lightgbm as lgb
import xgboost as xgb
import pyarrow.parquet as pq
from catboost import CatBoostClassifier


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_train_test"
OOF_DIR = ROOT / "outputs" / "oof_predictions"
METRICS_DIR = ROOT / "evaluation" / "metrics"
AUDIT_DIR = ROOT / "audit"
os.environ.setdefault("MPLCONFIGDIR", str(AUDIT_DIR / "mplconfig"))

TRAIN_FILE = DATA_DIR / "train_merged_v15.parquet"
TEST_FILE = DATA_DIR / "test_merged_v15.parquet"

TARGET = "TARGET"
ID_COL = "SK_ID_CURR"
EXCLUDE_COLS = [ID_COL, TARGET, "index"]
N_FOLDS = 5
SEED = 42

FORCE_CATEGORICAL = [
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "BURO_CLU_LABEL",
    "PREV_CLU_LABEL",
]


def log(message: str) -> None:
    print(message, flush=True)


def elapsed(start: float) -> str:
    return f"{(time.time() - start) / 60:.1f} min"


def load_v15() -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray]:
    log("Loading existing V15 matrices...")
    train = pd.read_parquet(TRAIN_FILE)
    test = pd.read_parquet(TEST_FILE)
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]

    missing_in_test = sorted(set(feature_cols) - set(test.columns))
    if missing_in_test:
        raise ValueError(f"Feature columns missing in test: {missing_in_test[:20]}")

    log(f"Train shape: {train.shape}")
    log(f"Test shape:  {test.shape}")
    log(f"Model feature count after exclusions: {len(feature_cols)}")
    log(f"Target counts: {train[TARGET].value_counts().to_dict()}")

    for col in feature_cols:
        if train[col].dtype == "object":
            combined = pd.concat([train[col].astype(str), test[col].astype(str)], ignore_index=True)
            codes, _ = pd.factorize(combined)
            train[col] = codes[: len(train)].astype(np.int32)
            test[col] = codes[len(train) :].astype(np.int32)

    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    test[feature_cols] = test[feature_cols].replace([np.inf, -np.inf], np.nan)

    y = train[TARGET].astype(int).to_numpy()
    return train, test, feature_cols, y


def load_v15_train_only() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    train = pd.read_parquet(TRAIN_FILE)
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = train[TARGET].astype(int).to_numpy()
    return train, feature_cols, y


def prepare_linear_matrix(train: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    log("Preparing dense float32 matrix with global target-independent median fill + scaling...")
    x_values = train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    del train
    gc.collect()

    for j in range(x_values.shape[1]):
        col = x_values[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            finite = col[~bad]
            fill = np.median(finite) if finite.size else 0.0
            col[bad] = fill
        mean = float(col.mean())
        std = float(col.std())
        if std > 0:
            col -= mean
            col /= std
        else:
            col -= mean
    return x_values


def save_oof(train: pd.DataFrame, y: np.ndarray, pred: np.ndarray, pred_col: str, filename: str) -> None:
    out = pd.DataFrame({ID_COL: train[ID_COL].to_numpy(), TARGET: y, pred_col: pred})
    out.to_parquet(OOF_DIR / filename, index=False)
    log(f"Saved {OOF_DIR / filename}")


def save_oof_arrays(ids: np.ndarray, y: np.ndarray, pred: np.ndarray, pred_col: str, filename: str) -> None:
    out = pd.DataFrame({ID_COL: ids, TARGET: y, pred_col: pred})
    out.to_parquet(OOF_DIR / filename, index=False)
    log(f"Saved {OOF_DIR / filename}")


def load_linear_matrix_from_disk() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    train = pd.read_parquet(TRAIN_FILE)
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    ids = train[ID_COL].to_numpy()
    y = train[TARGET].astype(int).to_numpy()
    x_values = prepare_linear_matrix(train, feature_cols)
    return ids, y, feature_cols, x_values


def load_tree_matrix_from_disk() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    train = pd.read_parquet(TRAIN_FILE)
    feature_cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    ids = train[ID_COL].to_numpy()
    y = train[TARGET].astype(int).to_numpy()
    log("Preparing tree-model float32 matrix...")
    x_values = train[feature_cols].to_numpy(dtype=np.float32, copy=True)
    del train
    gc.collect()
    x_values[~np.isfinite(x_values)] = np.nan
    return ids, y, feature_cols, x_values


def run_lr_from_disk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = time.time()
    log("\n=== Training Logistic Regression V15 (mini-batch SGD log-loss) ===")
    ids, y, _feature_cols, x_values = load_linear_matrix_from_disk()
    oof = np.zeros(len(y), dtype=np.float32)
    scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_values, y), 1):
        fold_start = time.time()
        rng = np.random.default_rng(SEED + fold)
        model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            average=True,
            random_state=SEED,
        )
        classes = np.array([0, 1], dtype=int)
        batch_size = 8192
        for _epoch in range(12):
            shuffled = tr_idx.copy()
            rng.shuffle(shuffled)
            for start_idx in range(0, len(shuffled), batch_size):
                batch = shuffled[start_idx : start_idx + batch_size]
                model.partial_fit(x_values[batch], y[batch], classes=classes)
        pred = np.zeros(len(val_idx), dtype=np.float32)
        for start_idx in range(0, len(val_idx), batch_size):
            batch = val_idx[start_idx : start_idx + batch_size]
            pred[start_idx : start_idx + len(batch)] = model.predict_proba(x_values[batch])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y[val_idx], pred)
        scores.append(auc)
        log(f"LR fold {fold}/{N_FOLDS}: AUC={auc:.5f}, time={elapsed(fold_start)}")
        del model
        gc.collect()
    log(f"LR OOF AUC={roc_auc_score(y, oof):.5f}, mean={np.mean(scores):.5f}, std={np.std(scores):.5f}, total={elapsed(start)}")
    del x_values
    gc.collect()
    return ids, y, oof


def run_mlp_from_disk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = time.time()
    log("\n=== Training MLP V15 ===")
    ids, y, _feature_cols, x_values = load_linear_matrix_from_disk()
    oof = np.zeros(len(y), dtype=np.float32)
    scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    params = {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "batch_size": 4096,
        "learning_rate_init": 1e-3,
        "max_iter": 1,
        "warm_start": True,
        "early_stopping": False,
        "random_state": SEED,
        "verbose": False,
    }
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_values, y), 1):
        fold_start = time.time()
        rng = np.random.default_rng(SEED + fold)
        model = MLPClassifier(**params)
        classes = np.array([0, 1], dtype=int)
        batch_size = 4096
        for _epoch in range(20):
            shuffled = tr_idx.copy()
            rng.shuffle(shuffled)
            for start_idx in range(0, len(shuffled), batch_size):
                batch = shuffled[start_idx : start_idx + batch_size]
                model.partial_fit(x_values[batch], y[batch], classes=classes)
        pred = np.zeros(len(val_idx), dtype=np.float32)
        for start_idx in range(0, len(val_idx), batch_size):
            batch = val_idx[start_idx : start_idx + batch_size]
            pred[start_idx : start_idx + len(batch)] = model.predict_proba(x_values[batch])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y[val_idx], pred)
        scores.append(auc)
        log(f"MLP fold {fold}/{N_FOLDS}: AUC={auc:.5f}, time={elapsed(fold_start)}")
        del model
        gc.collect()
    log(f"MLP OOF AUC={roc_auc_score(y, oof):.5f}, mean={np.mean(scores):.5f}, std={np.std(scores):.5f}, total={elapsed(start)}")
    del x_values
    gc.collect()
    return ids, y, oof


def run_lgb_single(train: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> np.ndarray:
    start = time.time()
    log("\n=== Training LightGBM V15 single seed ===")
    x_df = train[feature_cols]
    cat_features = [c for c in feature_cols if c in FORCE_CATEGORICAL]
    oof = np.zeros(len(train), dtype=np.float32)
    scores = []
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 24,
        "max_depth": 6,
        "min_child_samples": 100,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "max_bin": 63,
        "force_col_wise": True,
        "histogram_pool_size": 256,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": 4,
    }
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_df, y), 1):
        fold_start = time.time()
        dtrain = lgb.Dataset(x_df.iloc[tr_idx], label=y[tr_idx], categorical_feature=cat_features)
        dval = lgb.Dataset(x_df.iloc[val_idx], label=y[val_idx], categorical_feature=cat_features)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
        )
        pred = model.predict(x_df.iloc[val_idx], num_iteration=model.best_iteration)
        oof[val_idx] = pred
        auc = roc_auc_score(y[val_idx], pred)
        scores.append(auc)
        log(f"LGB fold {fold}/{N_FOLDS}: AUC={auc:.5f}, best_iter={model.best_iteration}, time={elapsed(fold_start)}")
        del model, dtrain, dval
        gc.collect()
    log(f"LGB single OOF AUC={roc_auc_score(y, oof):.5f}, mean={np.mean(scores):.5f}, std={np.std(scores):.5f}, total={elapsed(start)}")
    return oof


def run_lgb_single_from_disk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = time.time()
    log("\n=== Training LightGBM V15 single seed ===")
    ids, y, feature_cols, x_values = load_tree_matrix_from_disk()
    cat_features = [feature_cols.index(c) for c in feature_cols if c in FORCE_CATEGORICAL]
    oof = np.zeros(len(y), dtype=np.float32)
    scores = []
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 24,
        "max_depth": 6,
        "min_child_samples": 100,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "subsample_freq": 1,
        "max_bin": 63,
        "force_col_wise": True,
        "histogram_pool_size": 256,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": 4,
    }
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_values, y), 1):
        fold_start = time.time()
        dtrain = lgb.Dataset(x_values[tr_idx], label=y[tr_idx], categorical_feature=cat_features, free_raw_data=True)
        dval = lgb.Dataset(x_values[val_idx], label=y[val_idx], categorical_feature=cat_features, free_raw_data=True)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            valid_sets=[dval],
            valid_names=["valid"],
            callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
        )
        pred = model.predict(x_values[val_idx], num_iteration=model.best_iteration)
        oof[val_idx] = pred
        auc = roc_auc_score(y[val_idx], pred)
        scores.append(auc)
        log(f"LGB fold {fold}/{N_FOLDS}: AUC={auc:.5f}, best_iter={model.best_iteration}, time={elapsed(fold_start)}")
        del model, dtrain, dval
        gc.collect()
    log(f"LGB single OOF AUC={roc_auc_score(y, oof):.5f}, mean={np.mean(scores):.5f}, std={np.std(scores):.5f}, total={elapsed(start)}")
    del x_values
    gc.collect()
    return ids, y, oof


def run_xgb(train: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> np.ndarray:
    start = time.time()
    log("\n=== Training XGBoost V15 ===")
    x_values = train[feature_cols].to_numpy(dtype=np.float32)
    oof = np.zeros(len(train), dtype=np.float32)
    scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_values, y), 1):
        fold_start = time.time()
        model = xgb.XGBClassifier(
            n_estimators=5000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.7,
            max_bin=64,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=SEED,
            n_jobs=4,
            early_stopping_rounds=200,
        )
        model.fit(x_values[tr_idx], y[tr_idx], eval_set=[(x_values[val_idx], y[val_idx])], verbose=False)
        pred = model.predict_proba(x_values[val_idx])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y[val_idx], pred)
        scores.append(auc)
        log(f"XGB fold {fold}/{N_FOLDS}: AUC={auc:.5f}, best_iter={model.best_iteration}, time={elapsed(fold_start)}")
        del model
        gc.collect()
    log(f"XGB OOF AUC={roc_auc_score(y, oof):.5f}, mean={np.mean(scores):.5f}, std={np.std(scores):.5f}, total={elapsed(start)}")
    return oof


def run_xgb_from_disk() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = time.time()
    log("\n=== Training XGBoost V15 ===")
    ids, y, _feature_cols, x_values = load_tree_matrix_from_disk()
    oof = np.zeros(len(y), dtype=np.float32)
    scores = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_values, y), 1):
        fold_start = time.time()
        model = xgb.XGBClassifier(
            n_estimators=5000,
            learning_rate=0.02,
            max_depth=6,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.7,
            max_bin=64,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=SEED,
            n_jobs=4,
            early_stopping_rounds=200,
        )
        model.fit(x_values[tr_idx], y[tr_idx], eval_set=[(x_values[val_idx], y[val_idx])], verbose=False)
        pred = model.predict_proba(x_values[val_idx])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y[val_idx], pred)
        scores.append(auc)
        log(f"XGB fold {fold}/{N_FOLDS}: AUC={auc:.5f}, best_iter={model.best_iteration}, time={elapsed(fold_start)}")
        del model
        gc.collect()
    log(f"XGB OOF AUC={roc_auc_score(y, oof):.5f}, mean={np.mean(scores):.5f}, std={np.std(scores):.5f}, total={elapsed(start)}")
    del x_values
    gc.collect()
    return ids, y, oof


def run_catboost(train: pd.DataFrame, feature_cols: list[str], y: np.ndarray) -> np.ndarray:
    start = time.time()
    log("\n=== Training CatBoost V15 ===")
    x_df = train[feature_cols]
    cat_features = [feature_cols.index(c) for c in FORCE_CATEGORICAL if c in feature_cols]
    cat_names = [feature_cols[i] for i in cat_features]
    x_df = x_df.copy()
    for col in cat_names:
        x_df[col] = x_df[col].fillna(-999999).astype(int).astype(str)
    oof = np.zeros(len(train), dtype=np.float32)
    scores = []
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 3000,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "min_data_in_leaf": 100,
        "random_strength": 1.0,
        "bagging_temperature": 0.2,
        "border_count": 128,
        "grow_policy": "SymmetricTree",
        "od_type": "Iter",
        "od_wait": 150,
        "random_state": SEED,
        "verbose": False,
        "allow_writing_files": False,
        "task_type": "CPU",
        "thread_count": -1,
    }
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_df, y), 1):
        fold_start = time.time()
        model = CatBoostClassifier(**params)
        model.fit(
            x_df.iloc[tr_idx],
            y[tr_idx],
            eval_set=(x_df.iloc[val_idx], y[val_idx]),
            cat_features=cat_features,
            use_best_model=True,
        )
        pred = model.predict_proba(x_df.iloc[val_idx])[:, 1]
        oof[val_idx] = pred
        auc = roc_auc_score(y[val_idx], pred)
        scores.append(auc)
        log(f"CB fold {fold}/{N_FOLDS}: AUC={auc:.5f}, best_iter={model.best_iteration_}, time={elapsed(fold_start)}")
        del model
        gc.collect()
    log(f"CB OOF AUC={roc_auc_score(y, oof):.5f}, mean={np.mean(scores):.5f}, std={np.std(scores):.5f}, total={elapsed(start)}")
    return oof


def ks_stat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(np.max(tpr - fpr))


def lift_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.1) -> float:
    n_top = int(len(y_true) * k)
    top_idx = np.argsort(y_pred)[::-1][:n_top]
    return float(y_true[top_idx].mean() / y_true.mean())


def ece_metric(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            ece += abs(y_pred[mask].mean() - y_true[mask].mean()) * mask.sum() / len(y_true)
    return float(ece)


def metric_row(name: str, y_true: np.ndarray, oof: np.ndarray) -> dict[str, float | str]:
    fpr, tpr, thresholds = roc_curve(y_true, oof)
    t_star = thresholds[np.argmax(tpr - fpr)]
    y_pred_binary = (oof >= t_star).astype(int)
    auc = roc_auc_score(y_true, oof)
    return {
        "Model": name,
        "ROC-AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": ks_stat(y_true, oof),
        "PR-AUC": average_precision_score(y_true, oof),
        "Lift@10%": lift_at_k(y_true, oof, 0.1),
        "F1 (t*)": f1_score(y_true, y_pred_binary),
        "Precision": precision_score(y_true, y_pred_binary),
        "Recall": recall_score(y_true, y_pred_binary),
        "Brier": brier_score_loss(y_true, oof),
        "ECE": ece_metric(y_true, oof),
        "t*": t_star,
    }


def load_existing_oof(filename: str, pred_col: str, ref_ids: np.ndarray) -> np.ndarray:
    df = pd.read_parquet(OOF_DIR / filename)
    return df.set_index(ID_COL).loc[ref_ids, pred_col].to_numpy()


def existing_or_none(filename: str, pred_col: str, ref_ids: np.ndarray) -> np.ndarray | None:
    path = OOF_DIR / filename
    if path.exists():
        log(f"Reusing existing {path}")
        return load_existing_oof(filename, pred_col, ref_ids)
    return None


def corrected_latex_tables(metrics: pd.DataFrame) -> str:
    order_single = [
        ("Logistic Regression", "LogReg"),
        ("Multi-Layer Perceptron", "MLP"),
        ("XGBoost", "XGBoost"),
        ("CatBoost", "CatBoost"),
        ("LightGBM, single seed", "LightGBM single"),
        ("LightGBM, three-seed average", "LightGBM multi-seed"),
    ]
    metric_by_model = metrics.set_index("Model")
    single_rows = "\n".join(
        f"{label} & {metric_by_model.loc[key, 'ROC-AUC']:.4f} \\\\"
        if key in metric_by_model.index
        else f"{label} & N/A \\\\"
        for label, key in order_single
    )

    eval_models = ["LogReg", "LightGBM multi-seed", "XGBoost", "CatBoost", "MLP", "SE-HC"]
    eval_names = ["LogReg", "LightGBM", "XGBoost", "CatBoost", "MLP", "SE-HC"]
    eval_metrics = ["ROC-AUC", "Gini", "KS", "PR-AUC", "Lift@10%", "F1 (t*)", "Precision", "Recall"]
    eval_rows = []
    for metric in eval_metrics:
        vals = []
        for model in eval_models:
            if model in metric_by_model.index:
                value = metric_by_model.loc[model, metric]
                vals.append(f"{value:.2f}" if metric == "Lift@10%" else f"{value:.4f}")
            else:
                vals.append("N/A")
        eval_rows.append(f"{metric.replace('F1 (t*)', 'F1')} & " + " & ".join(vals) + " \\\\")

    candidates = [
        ("LightGBM, Stage 3, multi-seed", "LightGBM multi-seed"),
        ("Stacked ensemble, LightGBM meta-learner", "FP_FINAL"),
        ("Stacked ensemble, final aggregation", "K3_stack_v8_FINAL"),
        ("CatBoost", "CatBoost"),
        ("XGBoost", "XGBoost"),
        ("MLP", "MLP"),
        ("Logistic Regression", "LogReg"),
    ]
    candidate_rows = []
    for label, key in candidates:
        if key in metric_by_model.index:
            candidate_rows.append(f"{label} & {metric_by_model.loc[key, 'ROC-AUC']:.4f} \\\\")

    return f"""# Corrected V15 LaTeX Tables

## tab:single-models

```latex
\\begin{{table}}[htbp]
\\centering
\\caption{{Base learner performance on the full engineered feature set.}}
\\label{{tab:single-models}}
\\small
\\begin{{tabular}}{{lr}}
\\hline
\\textbf{{Model}} & \\textbf{{OOF ROC-AUC}} \\\\
\\hline
{single_rows}
\\hline
\\end{{tabular}}
\\end{{table}}
```

## tab:eval-all

```latex
\\begin{{table}}[htbp]
\\centering
\\caption{{Comprehensive evaluation across models using out-of-fold predictions.
Threshold-based metrics are reported at each model's selected operating threshold
\\(t^*\\) from Youden's J statistic.}}
\\label{{tab:eval-all}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lrrrrrr}}
\\hline
\\textbf{{Metric}} & \\textbf{{{eval_names[0]}}} & \\textbf{{{eval_names[1]}}} & \\textbf{{{eval_names[2]}}} & \\textbf{{{eval_names[3]}}} & \\textbf{{{eval_names[4]}}} & \\textbf{{{eval_names[5]}}} \\\\
\\hline
{chr(10).join(eval_rows)}
\\hline
\\end{{tabular}}
\\end{{table}}
```

## tab:candidates

```latex
\\begin{{table}}[htbp]
\\centering
\\caption{{Candidate library for hill-climbing ensemble selection.}}
\\label{{tab:candidates}}
\\small
\\begin{{tabular}}{{lr}}
\\hline
\\textbf{{Candidate predictor}} & \\textbf{{OOF ROC-AUC}} \\\\
\\hline
{chr(10).join(candidate_rows)}
\\hline
\\end{{tabular}}
\\end{{table}}
```
"""


def main() -> None:
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    train_pf = pq.ParquetFile(TRAIN_FILE)
    test_pf = pq.ParquetFile(TEST_FILE)
    train_columns = train_pf.schema.names
    test_columns = test_pf.schema.names
    feature_cols = [c for c in train_columns if c not in EXCLUDE_COLS]
    if set(feature_cols) - set(test_columns):
        raise ValueError("V15 test matrix is missing train feature columns")
    target_counts = pd.read_parquet(TRAIN_FILE, columns=[TARGET])[TARGET].value_counts().to_dict()
    sigma_ref = pd.read_parquet(OOF_DIR / "oof_SIGMA_FINAL.parquet", columns=[ID_COL, TARGET])
    ref_ids = sigma_ref[ID_COL].to_numpy()
    y = sigma_ref[TARGET].astype(int).to_numpy()
    feature_info = {
        "train_file": str(TRAIN_FILE),
        "test_file": str(TEST_FILE),
        "train_shape": [train_pf.metadata.num_rows, len(train_columns)],
        "test_shape": [test_pf.metadata.num_rows, len(test_columns)],
        "n_model_features": len(feature_cols),
        "excluded_columns": EXCLUDE_COLS,
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
    }
    (AUDIT_DIR / "v15_feature_info.json").write_text(json.dumps(feature_info, indent=2), encoding="utf-8")
    log("V15 metadata verified without full pre-load.")
    log(f"Train shape: {feature_info['train_shape']}")
    log(f"Test shape:  {feature_info['test_shape']}")
    log(f"Model feature count after exclusions: {len(feature_cols)}")
    log(f"Target counts: {target_counts}")
    del sigma_ref
    gc.collect()

    oof_lr = existing_or_none("oof_lr_v15_baseline.parquet", "oof_lr", ref_ids)
    if oof_lr is None:
        ids_lr, y_lr, oof_lr = run_lr_from_disk()
        save_oof_arrays(ids_lr, y_lr, oof_lr, "oof_lr", "oof_lr_v15_baseline.parquet")
        del ids_lr, y_lr
        gc.collect()

    oof_mlp = existing_or_none("oof_mlp_v15.parquet", "oof_mlp", ref_ids)
    if oof_mlp is None:
        ids_mlp, y_mlp, oof_mlp = run_mlp_from_disk()
        save_oof_arrays(ids_mlp, y_mlp, oof_mlp, "oof_mlp", "oof_mlp_v15.parquet")
        del ids_mlp, y_mlp
        gc.collect()

    oof_lgb_single = existing_or_none("oof_lgb_v15_single_seed.parquet", "oof_lgb", ref_ids)
    if oof_lgb_single is None:
        try:
            ids_lgb, y_lgb, oof_lgb_single = run_lgb_single_from_disk()
            save_oof_arrays(ids_lgb, y_lgb, oof_lgb_single, "oof_lgb", "oof_lgb_v15_single_seed.parquet")
            del ids_lgb, y_lgb
            gc.collect()
        except Exception as exc:
            log(f"WARNING: LightGBM single-seed V15 training failed: {type(exc).__name__}: {exc}")
            oof_lgb_single = None

    oof_xgb = existing_or_none("oof_xgb_v15.parquet", "oof_xgb", ref_ids)
    if oof_xgb is None:
        try:
            ids_xgb, y_xgb, oof_xgb = run_xgb_from_disk()
            save_oof_arrays(ids_xgb, y_xgb, oof_xgb, "oof_xgb", "oof_xgb_v15.parquet")
            del ids_xgb, y_xgb
            gc.collect()
        except Exception as exc:
            log(f"WARNING: XGBoost V15 training failed: {type(exc).__name__}: {exc}")
            oof_xgb = None

    oof_cb = existing_or_none("oof_cb_v15.parquet", "oof_cb", ref_ids)
    if oof_cb is None:
        try:
            train, feature_cols, y = load_v15_train_only()
            oof_cb = run_catboost(train, feature_cols, y)
            save_oof(train, y, oof_cb, "oof_cb", "oof_cb_v15.parquet")
            del train
            gc.collect()
        except Exception as exc:
            log(f"WARNING: CatBoost V15 training failed: {type(exc).__name__}: {exc}")
            oof_cb = None

    oof_lgb_ms = load_existing_oof("oof_lgb_v15_multiseed.parquet", "oof_lgb", ref_ids)
    oof_sigma = load_existing_oof("oof_SIGMA_FINAL.parquet", "oof_final", ref_ids)

    rows = [
        metric_row("LogReg", y, oof_lr),
        metric_row("MLP", y, oof_mlp),
    ]
    if oof_xgb is not None:
        rows.append(metric_row("XGBoost", y, oof_xgb))
    if oof_cb is not None:
        rows.append(metric_row("CatBoost", y, oof_cb))
    if oof_lgb_single is not None:
        rows.append(metric_row("LightGBM single", y, oof_lgb_single))
    rows.extend([
        metric_row("LightGBM multi-seed", y, oof_lgb_ms),
        metric_row("SE-HC", y, oof_sigma),
    ])

    for filename, pred_col, model_name in [
        ("oof_FP_FINAL.parquet", "oof_final", "FP_FINAL"),
        ("oof_K3_stack_v8_FINAL.parquet", "oof_final", "K3_stack_v8_FINAL"),
    ]:
        path = OOF_DIR / filename
        if path.exists():
            rows.append(metric_row(model_name, y, load_existing_oof(filename, pred_col, ref_ids)))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_DIR / "final_v15_baseline_comparison.csv", index=False)
    metrics.to_csv(AUDIT_DIR / "final_v15_baseline_comparison.csv", index=False)
    (AUDIT_DIR / "corrected_v15_latex_tables.md").write_text(corrected_latex_tables(metrics), encoding="utf-8")

    log("\n=== Final V15 metric comparison ===")
    log(metrics.to_string(index=False))
    log(f"\nSaved {METRICS_DIR / 'final_v15_baseline_comparison.csv'}")
    log(f"Saved {AUDIT_DIR / 'corrected_v15_latex_tables.md'}")


if __name__ == "__main__":
    main()
