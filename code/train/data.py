from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


META_COLS = {"user_id", "label_daily_stress", "label_stai1", "label_stai2"}


def load_user_level_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_feature_columns(df: pd.DataFrame, drop_panas: bool, use_stai2_feature: bool) -> List[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if drop_panas and c.startswith("panas_"):
            continue
        if (not use_stai2_feature) and "stai2" in c.lower():
            continue
        cols.append(c)
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def make_xy(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    max_missing_ratio: float = 0.40,
    add_missing_indicator: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    work = df.copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work = work.dropna(subset=[target])

    X = work[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # 1) 删除全空列
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)

    # 2) 删除高缺失列（默认缺失率 > 40%）
    missing_ratio = X.isna().mean()
    high_missing_cols = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    if high_missing_cols:
        X = X.drop(columns=high_missing_cols)

    # 3) 增加缺失指示器（在填补前计算）
    if add_missing_indicator:
        na_mask = X.isna().astype(int)
        na_mask.columns = [f"{c}_missing" for c in X.columns]

    # 4) 中位数填补
    X = X.fillna(X.median(numeric_only=True))

    # 5) 再兜底：若仍有 NaN（例如某列中位数不可得），用 0
    X = X.fillna(0.0)

    if add_missing_indicator:
        X = pd.concat([X, na_mask], axis=1)

    y = work[target].to_numpy(dtype=float)
    groups = work["user_id"].to_numpy()
    return X, y, groups
