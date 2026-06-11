import numpy as np

from src.models.calibration import IsotonicCalibrator1x2, expected_calibration_error


def test_ece_perfect_is_zero():
    # 100 predicciones de 1.0 al evento real -> ECE ~ 0
    probs = np.tile([1.0, 0.0, 0.0], (100, 1))
    actual = np.zeros(100, dtype=int)
    assert expected_calibration_error(probs, actual) < 1e-6


def test_ece_detects_miscalibration():
    # decir 0.9 pero acertar solo 50%
    probs = np.tile([0.9, 0.05, 0.05], (100, 1))
    actual = np.array([0] * 50 + [2] * 50)
    assert expected_calibration_error(probs, actual) > 0.2


def test_isotonic_calibrator_normalizes():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 2, 2], size=300)
    actual = rng.integers(0, 3, size=300)
    cal = IsotonicCalibrator1x2().fit(probs, actual)
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out.shape == probs.shape
