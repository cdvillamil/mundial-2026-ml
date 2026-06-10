"""Metricas de evaluacion probabilistica para predicciones 1X2.

Convencion de clases: 0 = home win, 1 = draw, 2 = away win.
probs: array (n, 3); actual: array (n,) con la clase real.
"""
import numpy as np

_EPS = 1e-15


def rps(probs: np.ndarray, actual: np.ndarray) -> float:
    """Ranked Probability Score promedio (menor es mejor)."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    cum_pred = np.cumsum(probs, axis=1)
    cum_true = np.cumsum(onehot, axis=1)
    # RPS = (1/(r-1)) * sum_{i=1}^{r-1} (CDF_pred_i - CDF_true_i)^2
    return float(np.mean(np.sum((cum_pred[:, :-1] - cum_true[:, :-1]) ** 2, axis=1)))


def log_loss_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    chosen = probs[np.arange(len(actual)), actual]
    return float(-np.mean(np.log(chosen)))


def brier_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == actual))
