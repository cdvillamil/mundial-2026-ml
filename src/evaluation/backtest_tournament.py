"""Backtest del sistema completo contra mundiales pasados (2010-2022)."""
import sqlite3

import numpy as np
import pandas as pd

from src.config import DB_PATH, REPORTS_DIR
from src.evaluation.historical import (WORLD_CUPS, elos_before, extract_field,
                                       wc_matches)
from src.evaluation.metrics import accuracy_1x2, log_loss_1x2, rps
from src.features.elo_features import attach_pre_match_elo, tournament_importance
from src.models.baseline_elo import EloBaseline
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_poisson import GBMPoissonModel
from src.simulation.monte_carlo import simulate_tournament
from src.simulation.rates import RateProvider


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def _train_models(con, before_date):
    """Entrena GBM-Poisson y baseline Elo con partidos anteriores a before_date."""
    matches = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral, m.tournament
           FROM matches m JOIN teams t1 ON t1.team_id=m.home_team_id
           JOIN teams t2 ON t2.team_id=m.away_team_id
           WHERE m.date >= '2002-01-01' AND m.date < ? ORDER BY m.date""",
        con, params=[before_date], parse_dates=["date"])
    elo = pd.read_sql("SELECT t.name AS team, e.date, e.elo FROM elo_history e "
                      "JOIN teams t ON t.team_id=e.team_id WHERE e.date < ?",
                      con, params=[before_date], parse_dates=["date"])
    matches = attach_pre_match_elo(matches, elo)
    gbm = GBMPoissonModel().fit(matches)
    dc = DixonColesModel().fit(matches)

    base_df = matches.assign(outcome=[_outcome(h, a) for h, a in
                                      zip(matches["home_goals"], matches["away_goals"])])
    baseline = EloBaseline().fit(base_df[["elo_diff", "outcome"]])
    return gbm, baseline, dc


def _match_level_eval(con, year, start_date, gbm, baseline, dc):
    """Predice cada partido real del Mundial con el modelo pre-torneo."""
    m = wc_matches(con, year)
    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    elos = elos_before(con, start_date, teams)
    imp = tournament_importance("FIFA World Cup")
    dc_teams = set(dc.teams_)

    actual = np.array([_outcome(r.home_goals, r.away_goals) for r in m.itertuples()])
    gbm_probs = np.array([
        gbm.predict_proba_1x2(elos[r.home_team], elos[r.away_team], imp, bool(r.neutral))
        for r in m.itertuples()])
    base_probs = baseline.predict_proba(
        np.array([elos[r.home_team] - elos[r.away_team] for r in m.itertuples()]))
    # Ensamble GBM+Dixon-Coles (DC solo si ambos equipos tienen parametros)
    dc_probs = []
    for r in m.itertuples():
        if r.home_team in dc_teams and r.away_team in dc_teams:
            dc_probs.append(dc.predict_proba_1x2(r.home_team, r.away_team, bool(r.neutral)))
        else:
            dc_probs.append(None)
    ens_probs = np.array([
        (g + d) / 2 if d is not None else g
        for g, d in zip(gbm_probs, dc_probs)])
    return {
        "n_matches": len(m),
        "gbm_rps": round(rps(gbm_probs, actual), 4),
        "ens_rps": round(rps(ens_probs, actual), 4),
        "base_rps": round(rps(base_probs, actual), 4),
        "gbm_logloss": round(log_loss_1x2(gbm_probs, actual), 4),
        "gbm_acc": round(accuracy_1x2(gbm_probs, actual), 4),
    }


def _tournament_eval(con, year, start_date, gbm, n_sims, champion):
    """Simula el torneo con grupos reales y evalua el ranking de P(campeon)."""
    groups = extract_field(con, year)
    teams = [t for g in groups.values() for t in g]
    elos = elos_before(con, start_date, teams)
    rp = RateProvider(gbm, elos)
    res = simulate_tournament(groups, rp, n_sims=n_sims,
                              n_qualify_per_group=2, n_best_thirds=0, seed=year)
    res = res.reset_index(drop=True)
    rank = res.index[res["team"] == champion]
    champ_rank = int(rank[0]) + 1 if len(rank) else -1
    champ_prob = (float(res.loc[res["team"] == champion, "p_champion"].iloc[0])
                  if champ_rank > 0 else 0.0)
    return {
        "predicted_champion": res.iloc[0]["team"],
        "predicted_prob": round(float(res.iloc[0]["p_champion"]) * 100, 1),
        "real_champion": champion,
        "real_champ_rank": champ_rank,
        "real_champ_prob": round(champ_prob * 100, 1),
        "n_groups": len(groups),
    }


def run(n_sims: int = 20000):
    con = sqlite3.connect(DB_PATH)
    rows_match, rows_tourn = [], []
    for year, (start, champion) in WORLD_CUPS.items():
        gbm, baseline, dc = _train_models(con, start)
        me = _match_level_eval(con, year, start, gbm, baseline, dc)
        te = _tournament_eval(con, year, start, gbm, n_sims, champion)
        rows_match.append({"year": year, **me})
        rows_tourn.append({"year": year, **te})
    con.close()
    return pd.DataFrame(rows_match), pd.DataFrame(rows_tourn)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    args = parser.parse_args()

    match_df, tourn_df = run(n_sims=args.n)

    wins = int((match_df["gbm_rps"] < match_df["base_rps"]).sum())
    ens_wins = int((match_df["ens_rps"] < match_df["gbm_rps"]).sum())
    top5 = int(((tourn_df["real_champ_rank"] >= 1) & (tourn_df["real_champ_rank"] <= 5)).sum())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f6_historical_validation.md"
    lines = [
        "# Validacion historica (Mundiales 2010, 2014, 2018, 2022)",
        "", f"Simulaciones por torneo: {args.n:,}. Entrenamiento con corte temporal",
        "estricto (solo datos anteriores al inicio de cada Mundial).", "",
        "## Nivel partido (GBM-Poisson vs baseline Elo vs ensamble GBM+Dixon-Coles)",
        "", match_df.to_markdown(index=False), "",
        f"GBM gana en RPS al baseline en **{wins}/4** mundiales.",
        f"El ensamble GBM+Dixon-Coles mejora al GBM en **{ens_wins}/4** mundiales.", "",
        "## Nivel torneo (ranking de P(campeon))",
        "", tourn_df.to_markdown(index=False), "",
        f"Campeon real en **top-5** de P(campeon) en **{top5}/4** mundiales.", "",
        "## Gate F6",
        f"- (a) GBM supera baseline en RPS en >=3/4: {'PASS' if wins >= 3 else 'FAIL'} ({wins}/4)",
        f"- (b) Campeon real en top-5 en >=3/4: {'PASS' if top5 >= 3 else 'FAIL'} ({top5}/4)",
        "", "Nota: bracket por siembra y Elo congelado al inicio (ver plan). El campo y",
        "grupos son los REALES de cada torneo (reconstruidos desde la base de datos).",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
