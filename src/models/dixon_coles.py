"""Modelo Dixon-Coles: Poisson + correccion de marcadores bajos + decaimiento temporal."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from src.models.score_matrix import outcome_probs


def dc_tau(i: int, j: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Factor de correccion de Dixon-Coles para marcadores bajos."""
    if i == 0 and j == 0:
        return 1.0 - lambda_h * lambda_a * rho
    if i == 0 and j == 1:
        return 1.0 + lambda_h * rho
    if i == 1 and j == 0:
        return 1.0 + lambda_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    """Ataque/defensa + rho (correccion) + xi (decaimiento temporal)."""

    def __init__(self, xi: float = 0.0):
        self.xi = xi
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_ = 0.0
        self.intercept_ = 0.0
        self.rho_ = 0.0

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams_ = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h_idx = matches["home_team"].map(idx).to_numpy()
        a_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy().astype(int)
        ag = matches["away_goals"].to_numpy().astype(int)
        is_home = (~matches["neutral"].astype(bool)).to_numpy().astype(float)

        # pesos por decaimiento temporal (si hay fecha)
        if "date" in matches.columns and self.xi > 0:
            tmax = matches["date"].max()
            age_days = (tmax - matches["date"]).dt.days.to_numpy()
            weights = np.exp(-self.xi * age_days)
        else:
            weights = np.ones(len(matches))

        low = (hg <= 1) & (ag <= 1)
        low_k = np.where(low)[0]

        def unpack(p):
            intercept, home_adv, rho = p[0], p[1], p[2]
            attack = np.concatenate([[0.0], p[3:3 + (n - 1)]])
            defense = np.concatenate([[0.0], p[3 + (n - 1):]])
            return intercept, home_adv, rho, attack, defense

        def neg_ll(p):
            intercept, home_adv, rho, attack, defense = unpack(p)
            log_lh = intercept + home_adv * is_home + attack[h_idx] - defense[a_idx]
            log_la = intercept + attack[a_idx] - defense[h_idx]
            lh, la = np.exp(log_lh), np.exp(log_la)
            ll_pois = hg * log_lh - lh + ag * log_la - la
            tau = np.ones(len(hg))
            for k in low_k:
                tau[k] = dc_tau(int(hg[k]), int(ag[k]), lh[k], la[k], rho)
            tau = np.clip(tau, 1e-10, None)
            ll = np.sum(weights * (ll_pois + np.log(tau)))
            return -ll

        p0 = np.zeros(3 + 2 * (n - 1))
        p0[0] = np.log(max(matches[["home_goals", "away_goals"]].mean().mean(), 0.1))
        res = minimize(neg_ll, p0, method="L-BFGS-B",
                       bounds=[(None, None), (None, None), (-0.2, 0.2)]
                              + [(None, None)] * (2 * (n - 1)))

        intercept, home_adv, rho, attack, defense = unpack(res.x)
        self.intercept_, self.home_adv_, self.rho_ = float(intercept), float(home_adv), float(rho)
        self.attack_ = {t: float(attack[idx[t]]) for t in teams}
        self.defense_ = {t: float(defense[idx[t]]) for t in teams}
        return self

    def _lambdas(self, home, away, neutral):
        adv = 0.0 if neutral else self.home_adv_
        lh = np.exp(self.intercept_ + adv + self.attack_[home] - self.defense_[away])
        la = np.exp(self.intercept_ + self.attack_[away] - self.defense_[home])
        return float(lh), float(la)

    def predict_score_matrix(self, home, away, neutral=True, max_goals=10) -> np.ndarray:
        lh, la = self._lambdas(home, away, neutral)
        ph = poisson.pmf(np.arange(max_goals + 1), lh)
        pa = poisson.pmf(np.arange(max_goals + 1), la)
        m = np.outer(ph, pa)
        # aplicar tau a las 4 celdas bajas
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= dc_tau(i, j, lh, la, self.rho_)
        return m / m.sum()

    def predict_proba_1x2(self, home, away, neutral=True) -> np.ndarray:
        return np.array(outcome_probs(self.predict_score_matrix(home, away, neutral)))
