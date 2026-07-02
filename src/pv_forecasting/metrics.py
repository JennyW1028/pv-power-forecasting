from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape_nonzero(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def prediction_interval_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    predictions = np.asarray(predictions)
    q10 = predictions[:, 0]
    q50 = predictions[:, 1]
    q90 = predictions[:, 2]

    coverage = np.mean((y_true >= q10) & (y_true <= q90)) * 100.0
    reliability = np.mean(
        [
            abs(np.mean(y_true <= q10) - 0.1),
            abs(np.mean(y_true <= q50) - 0.5),
            abs(np.mean(y_true <= q90) - 0.9),
        ]
    ) * 100.0
    mean_width = np.mean(q90 - q10)

    return {
        "PICP": float(coverage),
        "ReliabilityIndex": float(reliability),
        "MPIW": float(mean_width),
    }
