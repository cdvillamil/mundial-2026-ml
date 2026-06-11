import numpy as np
import pandas as pd

from src.models.gbm_poisson import GBMPoissonModel
from src.simulation.monte_carlo import simulate_tournament
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


def _mini_field():
    # 8 equipos, 2 grupos de 4 (mini-torneo para test rapido)
    elos = {"A": 2000, "B": 1950, "C": 1500, "D": 1450,
            "E": 1980, "F": 1920, "G": 1480, "H": 1440}
    groups = {"A": ["A", "B", "C", "D"], "B": ["E", "F", "G", "H"]}
    return elos, groups


def test_simulate_tournament_returns_probabilities():
    model = _model()
    elos, groups = _mini_field()
    rp = RateProvider(model, elos)
    res = simulate_tournament(groups, rp, n_sims=200, n_qualify_per_group=2,
                              n_best_thirds=0, seed=1)
    # devuelve dataframe con una fila por equipo
    assert set(res["team"]) == set(elos)
    # probabilidades en [0,1]
    assert (res["p_champion"] >= 0).all() and (res["p_champion"] <= 1).all()
    # suma de P(campeon) ~ 1
    assert abs(res["p_champion"].sum() - 1.0) < 1e-6


def test_simulate_official_2026_runs():
    import yaml

    from src.config import CONFIGS_DIR
    from src.simulation.monte_carlo import simulate_official_2026

    field = yaml.safe_load((CONFIGS_DIR / "groups_2026.yaml").read_text(encoding="utf-8"))
    rp = RateProvider(_model(), {t: float(e) for t, e in field["elos"].items()})
    res = simulate_official_2026(field["groups"], rp, n_sims=200, seed=1)
    assert len(res) == 48
    assert abs(res["p_champion"].sum() - 1.0) < 1e-6
    assert (res["p_qualify"] <= 1.0).all()
    assert (res["p_qualify"] >= res["p_r16"]).all()  # acumulado monotono


def test_stronger_teams_have_higher_champion_prob():
    model = _model()
    elos, groups = _mini_field()
    rp = RateProvider(model, elos)
    res = simulate_tournament(groups, rp, n_sims=500, n_qualify_per_group=2,
                              n_best_thirds=0, seed=2).set_index("team")
    # A (Elo 2000) debe tener mayor P(campeon) que C (Elo 1500)
    assert res.loc["A", "p_champion"] > res.loc["C", "p_champion"]
