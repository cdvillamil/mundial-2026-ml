"""Calibracion isotonica de probabilidades 1X2 + ECE."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(probs: np.ndarray, actual: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE sobre la clase predicha (confianza vs acierto)."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == actual).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(actual)
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


class IsotonicCalibrator1x2:
    """Una regresion isotonica por clase + renormalizacion."""

    def __init__(self):
        self.iso_ = [IsotonicRegression(out_of_bounds="clip") for _ in range(3)]

    def fit(self, probs: np.ndarray, actual: np.ndarray) -> "IsotonicCalibrator1x2":
        for c in range(3):
            target = (actual == c).astype(float)
            self.iso_[c].fit(probs[:, c], target)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        cal = np.column_stack([self.iso_[c].transform(probs[:, c]) for c in range(3)])
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cal / row_sums
