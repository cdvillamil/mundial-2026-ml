import numpy as np
import pandas as pd

from src.models.poisson import PoissonModel


def _training_data():
    """A es fuerte (mete muchos, recibe pocos), B debil. 40 partidos."""
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(40):
        rows.append({"home_team": "A", "away_team": "B",
                     "home_goals": rng.poisson(2.5), "away_goals": rng.poisson(0.5),
                     "neutral": True})
        rows.append({"home_team": "B", "away_team": "A",
                     "home_goals": rng.poisson(0.5), "away_goals": rng.poisson(2.5),
                     "neutral": True})
    return pd.DataFrame(rows)


def test_poisson_fits_and_predicts_matrix():
    model = PoissonModel().fit(_training_data())
    matrix = model.predict_score_matrix("A", "B", neutral=True)
    assert matrix.shape == (11, 11)
    assert abs(matrix.sum() - 1.0) < 1e-6


def test_stronger_team_wins_more_often():
    from src.models.score_matrix import outcome_probs
    model = PoissonModel().fit(_training_data())
    matrix = model.predict_score_matrix("A", "B", neutral=True)
    p_home, _, p_away = outcome_probs(matrix)
    assert p_home > p_away  # A (local) gana mas que B


def test_predict_proba_1x2():
    model = PoissonModel().fit(_training_data())
    probs = model.predict_proba_1x2("A", "B", neutral=True)
    assert abs(probs.sum() - 1.0) < 1e-6
