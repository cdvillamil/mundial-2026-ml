"""GBM-Poisson: HistGradientBoostingRegressor(loss=poisson) sobre features de Elo.

Sustituto justificado de LightGBM/XGBoost (Python 3.14 sin wheels estables).
Filas simetricas: cada partido -> 2 filas (perspectiva del equipo que anota).
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.dixon_coles import dc_tau
from src.models.score_matrix import outcome_probs

_FEATURES = ["self_elo", "opp_elo", "elo_diff_signed", "is_home", "tournament_importance"]


def _build_symmetric_rows(matches: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Convierte cada partido en 2 filas (local anota / visitante anota)."""
    is_home_flag = (~matches["neutral"].astype(bool)).astype(int)
    home_rows = pd.DataFrame({
        "self_elo": matches["home_elo"].to_numpy(),
        "opp_elo": matches["away_elo"].to_numpy(),
        "elo_diff_signed": (matches["home_elo"] - matches["away_elo"]).to_numpy(),
        "is_home": is_home_flag.to_numpy(),
        "tournament_importance": matches["tournament_importance"].to_numpy(),
    })
    away_rows = pd.DataFrame({
        "self_elo": matches["away_elo"].to_numpy(),
        "opp_elo": matches["home_elo"].to_numpy(),
        "elo_diff_signed": (matches["away_elo"] - matches["home_elo"]).to_numpy(),
        "is_home": np.zeros(len(matches), dtype=int),  # visitante nunca tiene ventaja local
        "tournament_importance": matches["tournament_importance"].to_numpy(),
    })
    X = pd.concat([home_rows, away_rows], ignore_index=True)
    y = np.concatenate([matches["home_goals"].to_numpy(), matches["away_goals"].to_numpy()])
    return X, y


class GBMPoissonModel:
    def __init__(self, rho: float = -0.05, max_iter: int = 200,
                 weight_recent: bool = True, xi: float = 0.0008, **kwargs):
        self.rho = rho
        self.weight_recent = weight_recent
        self.xi = xi  # decaimiento temporal por dia (~half-life 2.4 anos)
        self.model_ = HistGradientBoostingRegressor(
            loss="poisson", max_iter=max_iter, learning_rate=0.05,
            max_depth=4, min_samples_leaf=50, **kwargs
        )

    def _weights(self, matches: pd.DataFrame):
        """Peso por importancia del torneo y recencia; duplicado (filas simetricas)."""
        imp_factor = {0: 0.7, 1: 1.0, 2: 1.3, 3: 1.5}
        w = matches["tournament_importance"].map(imp_factor).fillna(1.0).to_numpy()
        if "date" in matches.columns:
            tmax = matches["date"].max()
            age = (tmax - matches["date"]).dt.days.to_numpy()
            w = w * np.exp(-self.xi * age)
        return np.concatenate([w, w])

    def fit(self, matches: pd.DataFrame) -> "GBMPoissonModel":
        X, y = _build_symmetric_rows(matches)
        sw = None
        if self.weight_recent and "tournament_importance" in matches.columns:
            sw = self._weights(matches)
        self.model_.fit(X[_FEATURES], y, sample_weight=sw)
        return self

    def predict_lambdas(self, home_elo, away_elo, tournament_importance, neutral):
        is_home = 0 if neutral else 1
        row_h = [[home_elo, away_elo, home_elo - away_elo, is_home, tournament_importance]]
        row_a = [[away_elo, home_elo, away_elo - home_elo, 0, tournament_importance]]
        lh = float(self.model_.predict(pd.DataFrame(row_h, columns=_FEATURES))[0])
        la = float(self.model_.predict(pd.DataFrame(row_a, columns=_FEATURES))[0])
        return lh, la

    def predict_score_matrix(self, home_elo, away_elo, tournament_importance,
                             neutral, max_goals=10) -> np.ndarray:
        lh, la = self.predict_lambdas(home_elo, away_elo, tournament_importance, neutral)
        ph = poisson.pmf(np.arange(max_goals + 1), lh)
        pa = poisson.pmf(np.arange(max_goals + 1), la)
        m = np.outer(ph, pa)
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= dc_tau(i, j, lh, la, self.rho)
        return m / m.sum()

    def predict_proba_1x2(self, home_elo, away_elo, tournament_importance,
                          neutral) -> np.ndarray:
        return np.array(outcome_probs(self.predict_score_matrix(
            home_elo, away_elo, tournament_importance, neutral)))
