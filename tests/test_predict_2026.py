import numpy as np
import pandas as pd

from src.inference.predict_2026 import predict_group_matches
from src.models.gbm_poisson import GBMPoissonModel
from src.simulation.rates import RateProvider


def _model():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        he, ae = rng.uniform(1400, 2000), rng.uniform(1400, 2000)
        rows.append({"home_elo": he, "away_elo": ae, "elo_diff": he - ae,
                     "tournament_importance": 3, "neutral": True,
                     "home_goals": rng.poisson(np.exp((he - ae) / 600 + 0.2)),
                     "away_goals": rng.poisson(np.exp((ae - he) / 600))})
    return GBMPoissonModel().fit(pd.DataFrame(rows))


def test_predict_group_matches_columns_and_probs():
    groups = {"A": ["X", "Y", "Z"]}
    elos = {"X": 1900, "Y": 1700, "Z": 1500}
    rp = RateProvider(_model(), elos)
    df = predict_group_matches(groups, rp)
    # 3 partidos en un grupo de 3 (round robin)
    assert len(df) == 3
    assert {"group", "home", "away", "p_home", "p_draw", "p_away",
            "most_likely", "score_prob"} <= set(df.columns)
    # probabilidades suman ~1
    assert np.allclose(df["p_home"] + df["p_draw"] + df["p_away"], 1.0, atol=1e-3)
    # formato de marcador "i-j"
    assert df["most_likely"].str.match(r"^\d+-\d+$").all()


def test_stronger_home_has_higher_p_home():
    groups = {"A": ["X", "Z"]}
    rp = RateProvider(_model(), {"X": 2000, "Z": 1400})
    df = predict_group_matches(groups, rp)
    row = df.iloc[0]
    assert row["p_home"] > row["p_away"]
