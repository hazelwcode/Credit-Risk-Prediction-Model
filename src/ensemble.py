"""Prediction-level ensemble helpers for the SE-HC pipeline."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def caruana_hill_climb(
    candidates_oof: Mapping[str, np.ndarray],
    y: np.ndarray,
    max_iters: int = 50,
    verbose: bool = True,
) -> tuple[list[str], np.ndarray]:
    """Greedy forward selection with replacement over OOF prediction vectors.

    SE-HC uses prediction-level hill-climbing: at each iteration the candidate
    that gives the best OOF ROC-AUC for the averaged ensemble is added. Candidate
    frequency in the selected list corresponds to ensemble weight.
    """
    ensemble = []
    ensemble_pred = np.zeros(len(y))

    for iteration in range(max_iters):
        current_auc = roc_auc_score(y, ensemble_pred) if iteration > 0 else 0
        best_auc = current_auc
        best_model = None

        for name, pred in candidates_oof.items():
            if iteration == 0:
                new_pred = pred.copy()
            else:
                new_pred = (ensemble_pred * iteration + pred) / (iteration + 1)

            auc = roc_auc_score(y, new_pred)

            if auc > best_auc:
                best_auc = auc
                best_model = name

        if best_model is None:
            if verbose:
                print(f"  Iter {iteration}: No improvement, stopping")
            break

        ensemble.append(best_model)
        if iteration == 0:
            ensemble_pred = candidates_oof[best_model].copy()
        else:
            ensemble_pred = (ensemble_pred * iteration + candidates_oof[best_model]) / (
                iteration + 1
            )

        if verbose:
            print(f"  Iter {iteration + 1:2d}: +{best_model:15s} | AUC = {best_auc:.5f}")

    return ensemble, ensemble_pred


def candidate_auc_summary(candidates_oof: Mapping[str, np.ndarray], y: np.ndarray) -> pd.DataFrame:
    """Return candidate OOF ROC-AUC values sorted from strongest to weakest."""
    rows = [
        {"candidate": name, "roc_auc": roc_auc_score(y, pred)}
        for name, pred in candidates_oof.items()
    ]
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def blend_predictions(predictions: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Compute a weighted average from named prediction vectors."""
    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be positive.")
    blended = None
    for name, weight in weights.items():
        if name not in predictions:
            raise KeyError(f"Missing prediction vector for {name!r}.")
        term = np.asarray(predictions[name], dtype=float) * float(weight)
        blended = term if blended is None else blended + term
    return blended / total_weight


def model_frequencies(selected_models: list[str]) -> pd.DataFrame:
    """Convert selected model repetitions into frequency and weight columns."""
    counts = Counter(selected_models)
    total = sum(counts.values())
    rows = [
        {"candidate": name, "frequency": count, "weight": count / total}
        for name, count in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(["weight", "candidate"], ascending=[False, True])

