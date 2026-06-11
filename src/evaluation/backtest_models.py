"""Backtest comparativo de modelos con split temporal."""
import sqlite3

import numpy as np
import pandas as pd

from src.config import DB_PATH, REPORTS_DIR
from src.evaluation.metrics import accuracy_1x2, brier_1x2, log_loss_1x2, rps
from src.features.elo_features import attach_pre_match_elo
from src.models.calibration import expected_calibration_error
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_poisson import GBMPoissonModel
from src.models.poisson import PoissonModel


def _load(min_date="2010-01-01"):
    con = sqlite3.connect(DB_PATH)
    matches = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral, m.tournament
           FROM matches m
           JOIN teams t1 ON t1.team_id = m.home_team_id
           JOIN teams t2 ON t2.team_id = m.away_team_id
           WHERE m.date >= ? ORDER BY m.date""",
        con, params=[min_date], parse_dates=["date"])
    elo = pd.read_sql(
        "SELECT t.name AS team, e.date, e.elo FROM elo_history e "
        "JOIN teams t ON t.team_id = e.team_id",
        con, parse_dates=["date"])
    con.close()
    return matches, elo


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def run_backtest(cutoff="2018-01-01"):
    matches, elo = _load()
    matches = attach_pre_match_elo(matches, elo)
    train = matches[matches["date"] < cutoff].copy()
    test = matches[matches["date"] >= cutoff].copy()

    poisson = PoissonModel().fit(train)
    dc = DixonColesModel().fit(train)
    gbm = GBMPoissonModel().fit(train)

    known = set(poisson.teams_)
    test_cls = test[test["home_team"].isin(known) & test["away_team"].isin(known)].copy()
    actual = np.array([_outcome(r.home_goals, r.away_goals) for r in test_cls.itertuples()])

    rows = []
    # Poisson y DC (basados en equipo)
    for name, model in [("Poisson", poisson), ("Dixon-Coles", dc)]:
        probs = np.array([model.predict_proba_1x2(r.home_team, r.away_team, bool(r.neutral))
                          for r in test_cls.itertuples()])
        rows.append((name, probs))

    # GBM (basado en Elo; funciona tambien con equipos no vistos)
    gbm_probs = np.array([
        gbm.predict_proba_1x2(r.home_elo, r.away_elo, r.tournament_importance, bool(r.neutral))
        for r in test_cls.itertuples()])
    rows.append(("GBM-Poisson", gbm_probs))

    results = []
    for name, probs in rows:
        results.append({
            "model": name, "n_test": len(test_cls),
            "RPS": round(rps(probs, actual), 4),
            "log_loss": round(log_loss_1x2(probs, actual), 4),
            "Brier": round(brier_1x2(probs, actual), 4),
            "accuracy": round(accuracy_1x2(probs, actual), 4),
            "ECE": round(expected_calibration_error(probs, actual), 4),
        })
    return pd.DataFrame(results)


def main():
    res = run_backtest()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f4_model_comparison.md"
    best = res.loc[res["RPS"].idxmin(), "model"]
    lines = [
        "# Comparacion de modelos con GBM (corte temporal 2018-01-01)",
        "", "Entrenamiento: 2010-2018. Test: 2018+. GBM usa Elo point-in-time.",
        "", res.to_markdown(index=False), "",
        f"**Mejor modelo por RPS: {best}**", "",
        "Nota: GBM-Poisson usa HistGradientBoostingRegressor(loss=poisson) de sklearn",
        "(sustituto de LightGBM/XGBoost por Python 3.14). Predice desde Elo, no identidad",
        "del equipo, por lo que generaliza a equipos no vistos en entrenamiento.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
