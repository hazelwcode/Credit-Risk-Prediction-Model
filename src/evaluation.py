"""Reusable metric helpers for OOF model evaluation."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def gini_from_auc(auc: float) -> float:
    """Convert ROC-AUC to the corresponding Gini coefficient."""
    return float(2 * auc - 1)


def gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Gini from labels and prediction scores."""
    return gini_from_auc(roc_auc_score(y_true, y_pred))


def ks_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov separation statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(np.max(tpr - fpr))


def lift_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.1) -> float:
    """Compute lift in the top-k fraction ranked by predicted risk."""
    n_top = int(len(y_true) * k)
    top_idx = np.argsort(y_pred)[::-1][:n_top]
    return float(y_true[top_idx].mean() / y_true.mean())


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 15
) -> float:
    """Compute expected calibration error with fixed-width probability bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            ece += abs(y_pred[mask].mean() - y_true[mask].mean()) * mask.sum() / len(y_true)
    return float(ece)


def decile_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Build a risk-decile table sorted from highest to lowest predicted risk."""
    df = pd.DataFrame({"TARGET": y_true, "prediction": y_pred})
    df["decile"] = pd.qcut(df["prediction"].rank(method="first"), 10, labels=False) + 1
    rows = []
    total_defaults = df["TARGET"].sum()
    for decile in sorted(df["decile"].unique(), reverse=True):
        part = df[df["decile"] == decile]
        defaults = part["TARGET"].sum()
        rows.append(
            {
                "decile": int(11 - decile),
                "n": int(len(part)),
                "default_count": int(defaults),
                "default_rate": float(part["TARGET"].mean()),
                "capture_rate": float(defaults / total_defaults) if total_defaults else 0.0,
                "mean_score": float(part["prediction"].mean()),
                "min_score": float(part["prediction"].min()),
                "max_score": float(part["prediction"].max()),
            }
        )
    out = pd.DataFrame(rows)
    out["cumulative_capture_rate"] = out["capture_rate"].cumsum()
    return out


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model") -> dict:
    """Compute the compact metric row used by the thesis evaluation notebooks."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    t_star = thresholds[np.argmax(tpr - fpr)]
    y_pred_binary = (y_pred >= t_star).astype(int)
    auc = roc_auc_score(y_true, y_pred)
    return {
        "Model": model_name,
        "ROC-AUC": auc,
        "Gini": gini_from_auc(auc),
        "KS": ks_statistic(y_true, y_pred),
        "PR-AUC": average_precision_score(y_true, y_pred),
        "Lift@10%": lift_at_k(y_true, y_pred, 0.1),
        "Lift@20%": lift_at_k(y_true, y_pred, 0.2),
        "F1 (t*)": f1_score(y_true, y_pred_binary),
        "Precision": precision_score(y_true, y_pred_binary),
        "Recall": recall_score(y_true, y_pred_binary),
        "Brier": brier_score_loss(y_true, y_pred),
        "ECE": expected_calibration_error(y_true, y_pred),
        "t*": t_star,
    }


def compare_models(model_oofs: Mapping[str, tuple[np.ndarray, np.ndarray] | np.ndarray]) -> pd.DataFrame:
    """Evaluate several OOF vectors and return a sorted comparison table.

    Values may be either ``pred`` arrays with a shared target supplied separately by
    the caller, or ``(y_true, pred)`` tuples. For this repository, tuple form is
    preferred because each OOF artifact carries its own target column.
    """
    rows = []
    for name, value in model_oofs.items():
        if isinstance(value, tuple):
            y_true, y_pred = value
        else:
            raise ValueError("compare_models expects values as (y_true, y_pred) tuples.")
        rows.append(evaluate_model(np.asarray(y_true), np.asarray(y_pred), name))
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)

