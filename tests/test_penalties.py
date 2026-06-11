import numpy as np
import pandas as pd

from src.models.penalties import PenaltyModel


def test_penalty_model_fits_and_predicts():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 200, 400)
    p = 1 / (1 + 10 ** (-diffs / 800))
    home_won = (rng.random(400) < p).astype(int)
    model = PenaltyModel().fit(pd.DataFrame({"elo_diff": diffs, "home_won": home_won}))
    assert model.predict(400) > model.predict(-400)
    assert 0 <= model.predict(0) <= 1


def test_penalty_model_default_near_half_at_zero():
    rng = np.random.default_rng(1)
    diffs = rng.normal(0, 150, 300)
    home_won = (rng.random(300) < 0.5).astype(int)
    model = PenaltyModel().fit(pd.DataFrame({"elo_diff": diffs, "home_won": home_won}))
    assert abs(model.predict(0) - 0.5) < 0.15
