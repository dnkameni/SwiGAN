"""Module containing methods to compute some evaluation metrics."""

import numpy as np


def compute_rmse_per_date(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute spatial RMSE per predictions timestep."""
    n_elements = mask.sum()

    masked_y_pred = y_pred * mask
    masked_y_true = y_true * mask

    return np.sqrt(((masked_y_true - masked_y_pred) ** 2).sum(axis=(-2, -1)) / n_elements)


def compute_smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute mean symmetric absolute percentage error."""
    absolute_diff = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mape = absolute_diff / (denominator + 1e-8)
    masked_mape = mape * mask
    return 100 * masked_mape


def compute_smape_per_date(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute mean symmetric absolute percentage error."""
    n_elements = mask.sum()

    masked_mape = compute_smape(y_true, y_pred, mask)
    return masked_mape.sum(axis=(-2, -1)) / n_elements
