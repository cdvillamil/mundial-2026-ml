import numpy as np

from src.evaluation.metrics import brier_1x2, log_loss_1x2, rps


def test_rps_perfect_prediction_is_zero():
    # prob 1.0 al resultado real
    probs = np.array([[1.0, 0.0, 0.0]])
    actual = np.array([0])  # home win
    assert rps(probs, actual) < 1e-9


def test_rps_penalizes_ordered_errors_less():
    # predecir empate cuando gano local: error "cercano"
    near = rps(np.array([[0.0, 1.0, 0.0]]), np.array([0]))
    # predecir away win cuando gano local: error "lejano"
    far = rps(np.array([[0.0, 0.0, 1.0]]), np.array([0]))
    assert far > near


def test_log_loss_perfect_is_near_zero():
    probs = np.array([[0.99, 0.005, 0.005]])
    assert log_loss_1x2(probs, np.array([0])) < 0.02


def test_brier_range():
    probs = np.array([[0.5, 0.3, 0.2]])
    b = brier_1x2(probs, np.array([0]))
    assert 0 <= b <= 2
