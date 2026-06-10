"""Modelo Poisson de fuerzas ataque/defensa por equipo."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.score_matrix import outcome_probs, poisson_score_matrix


class PoissonModel:
    """Estima ataque/defensa por equipo + ventaja local global via MLE Poisson."""

    def __init__(self):
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.intercept_: float = 0.0

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams_ = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h_idx = matches["home_team"].map(idx).to_numpy()
        a_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy()
        ag = matches["away_goals"].to_numpy()
        is_home = (~matches["neutral"].astype(bool)).to_numpy().astype(float)

        # params: [intercept, home_adv, attack(n-1), defense(n-1)]
        # fijamos attack[0]=defense[0]=0 (identificabilidad) -> usamos n-1 libres c/u
        def unpack(p):
            intercept = p[0]
            home_adv = p[1]
            attack = np.concatenate([[0.0], p[2:2 + (n - 1)]])
            defense = np.concatenate([[0.0], p[2 + (n - 1):]])
            return intercept, home_adv, attack, defense

        def neg_ll(p):
            intercept, home_adv, attack, defense = unpack(p)
            log_lh = intercept + home_adv * is_home + attack[h_idx] - defense[a_idx]
            log_la = intercept + attack[a_idx] - defense[h_idx]
            lh, la = np.exp(log_lh), np.exp(log_la)
            ll = np.sum(hg * log_lh - lh) + np.sum(ag * log_la - la)
            return -ll

        p0 = np.zeros(2 + 2 * (n - 1))
        p0[0] = np.log(max(matches[["home_goals", "away_goals"]].mean().mean(), 0.1))
        res = minimize(neg_ll, p0, method="L-BFGS-B")

        intercept, home_adv, attack, defense = unpack(res.x)
        self.intercept_ = float(intercept)
        self.home_adv_ = float(home_adv)
        self.attack_ = {t: float(attack[idx[t]]) for t in teams}
        self.defense_ = {t: float(defense[idx[t]]) for t in teams}
        return self

    def _lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        adv = 0.0 if neutral else self.home_adv_
        lh = np.exp(self.intercept_ + adv + self.attack_[home] - self.defense_[away])
        la = np.exp(self.intercept_ + self.attack_[away] - self.defense_[home])
        return float(lh), float(la)

    def predict_score_matrix(self, home: str, away: str, neutral: bool = True,
                             max_goals: int = 10) -> np.ndarray:
        lh, la = self._lambdas(home, away, neutral)
        return poisson_score_matrix(lh, la, max_goals)

    def predict_proba_1x2(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        return np.array(outcome_probs(self.predict_score_matrix(home, away, neutral)))
