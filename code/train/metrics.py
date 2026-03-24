from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    # 兼容旧版 sklearn（无 squared 参数）
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": rmse,
        "spearman": float(rho) if rho == rho else 0.0,
    }
