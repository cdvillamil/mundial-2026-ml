import numpy as np
import pandas as pd

from src.models.gbm_poisson import GBMPoissonModel
from src.simulation.rates import RateProvider


def _trained_model():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        he, ae = rng.uniform(1400, 2000), rng.uniform(1400, 2000)
        lh = np.exp((he - ae) / 600 + 0.2)
        la = np.exp((ae - he) / 600)
        rows.append({"home_elo": he, "away_elo": ae, "elo_diff": he - ae,
                     "tournament_importance": 3, "neutral": True,
                     "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la)})
    return GBMPoissonModel().fit(pd.DataFrame(rows))


def test_rate_provider_caches_and_returns_matrix():
    model = _trained_model()
    elos = {"A": 1900, "B": 1500}
    rp = RateProvider(model, elos)
    m = rp.matrix("A", "B")
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-6
    # cache: misma instancia devuelta
    assert rp.matrix("A", "B") is m


def test_rate_provider_elo_diff():
    model = _trained_model()
    rp = RateProvider(model, {"A": 1900, "B": 1500})
    assert rp.elo_diff("A", "B") == 400
