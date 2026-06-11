import numpy as np
import pandas as pd

from src.models.gbm_poisson import GBMPoissonModel


def _training_data():
    """Equipos con Elo alto anotan mas; entrenamiento sintetico coherente."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        he = rng.uniform(1400, 2000)
        ae = rng.uniform(1400, 2000)
        # lambda crece con ventaja de Elo
        lh = np.exp(0.0 + (he - ae) / 600 + 0.2)  # +0.2 ventaja local
        la = np.exp(0.0 + (ae - he) / 600)
        rows.append({
            "home_team": "H", "away_team": "A",
            "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
            "tournament_importance": 1, "neutral": False,
            "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
        })
    return pd.DataFrame(rows)


def test_gbm_fits_and_predicts_lambdas():
    model = GBMPoissonModel().fit(_training_data())
    lh, la = model.predict_lambdas(home_elo=1900, away_elo=1500,
                                   tournament_importance=1, neutral=False)
    assert lh > la  # equipo local mucho mas fuerte anota mas
    assert lh > 0 and la > 0


def test_gbm_predicts_score_matrix():
    model = GBMPoissonModel().fit(_training_data())
    m = model.predict_score_matrix(home_elo=1700, away_elo=1700,
                                    tournament_importance=1, neutral=False)
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-6


def test_gbm_proba_1x2_sums_to_one():
    model = GBMPoissonModel().fit(_training_data())
    probs = model.predict_proba_1x2(home_elo=1800, away_elo=1500,
                                     tournament_importance=3, neutral=True)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert probs[0] > probs[2]  # local favorito gana mas
