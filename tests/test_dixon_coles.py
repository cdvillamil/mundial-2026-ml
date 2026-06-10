import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel, dc_tau


def test_tau_correction_factors():
    # Para marcadores altos, tau = 1 (sin correccion)
    assert dc_tau(3, 2, lambda_h=1.5, lambda_a=1.2, rho=-0.1) == 1.0
    # Para 0-0 con rho<0, tau != 1
    assert dc_tau(0, 0, 1.5, 1.2, rho=-0.1) != 1.0


def _training_data():
    rng = np.random.default_rng(7)
    rows = []
    dates = pd.date_range("2020-01-01", periods=80, freq="14D")
    for d in dates:
        rows.append({"date": d, "home_team": "A", "away_team": "B",
                     "home_goals": rng.poisson(1.8), "away_goals": rng.poisson(0.8),
                     "neutral": True})
    return pd.DataFrame(rows)


def test_dixon_coles_fits_and_matrix_normalized():
    model = DixonColesModel().fit(_training_data())
    m = model.predict_score_matrix("A", "B", neutral=True)
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-6


def test_dixon_coles_predicts_proba():
    model = DixonColesModel().fit(_training_data())
    probs = model.predict_proba_1x2("A", "B", neutral=True)
    assert abs(probs.sum() - 1.0) < 1e-6
