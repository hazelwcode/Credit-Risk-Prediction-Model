"""Lightweight I/O and OOF alignment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ID_COL, TARGET_COL


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    """Save an object as pretty JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_read_parquet(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a parquet file with a clearer missing-file error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path, **kwargs)


def safe_read_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file with a clearer missing-file error."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


def infer_prediction_column(df: pd.DataFrame, id_col: str = ID_COL, target_col: str = TARGET_COL) -> str:
    """Infer the single prediction column in an OOF dataframe."""
    excluded = {id_col, target_col}
    candidates = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one numeric prediction column, found {candidates}.")
    return candidates[0]


def check_oof_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    id_col: str = ID_COL,
    target_col: str = TARGET_COL,
) -> dict[str, bool]:
    """Check row-level ID and target alignment between two OOF dataframes."""
    same_length = len(reference) == len(candidate)
    id_aligned = same_length and np.array_equal(reference[id_col].to_numpy(), candidate[id_col].to_numpy())
    if target_col in reference.columns and target_col in candidate.columns:
        target_aligned = same_length and np.array_equal(
            reference[target_col].to_numpy(), candidate[target_col].to_numpy()
        )
    else:
        target_aligned = False
    return {
        "same_length": bool(same_length),
        "id_aligned": bool(id_aligned),
        "target_aligned": bool(target_aligned),
    }

