"""Motor Monte Carlo del torneo: grupos + mejores terceros + eliminatorias."""
import itertools

import numpy as np
import pandas as pd

from src.simulation.group_stage import rank_best_thirds, rank_group
from src.simulation.knockout import propagate_bracket, seed_qualifiers
from src.simulation.match import sample_knockout, sample_score
from src.simulation.rates import RateProvider

# etapas que se contabilizan
_STAGES = ["qualify", "r16", "qf", "sf", "final", "champion"]


def _round_robin(group_teams):
    """Todos los emparejamientos del grupo (round robin)."""
    return list(itertools.combinations(group_teams, 2))


def _simulate_group(teams, rp: RateProvider, rng):
    """Simula los partidos del grupo y devuelve (orden, stats_por_equipo)."""
    results = []
    for h, a in _round_robin(teams):
        hg, ag = sample_score(rp.matrix(h, a), rng)
        results.append({"home": h, "away": a, "hg": hg, "ag": ag})
    order = rank_group(list(teams), results, rng)

    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for m in results:
        h, a, x, y = m["home"], m["away"], m["hg"], m["ag"]
        gf[h] += x; ga[h] += y; gf[a] += y; ga[a] += x
        if x > y:
            pts[h] += 3
        elif y > x:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    stats = {t: {"points": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]} for t in teams}
    return order, stats


def _quality_rank(qualifiers, rp: RateProvider):
    """Ordena los clasificados por Elo (mejor primero) para sembrar el bracket."""
    return sorted(qualifiers, key=lambda t: -rp.elos[t])


def simulate_tournament(groups: dict, rp: RateProvider, n_sims: int,
                        n_qualify_per_group: int = 2, n_best_thirds: int = 8,
                        seed: int = 0) -> pd.DataFrame:
    """Corre n_sims simulaciones y agrega P(fase) por equipo."""
    rng = np.random.default_rng(seed)
    all_teams = [t for g in groups.values() for t in g]
    counts = {t: {s: 0 for s in _STAGES} for t in all_teams}

    for _ in range(n_sims):
        qualifiers = []
        thirds = []
        for teams in groups.values():
            order, stats = _simulate_group(teams, rp, rng)
            qualifiers.extend(order[:n_qualify_per_group])
            if n_best_thirds > 0 and len(order) > n_qualify_per_group:
                third = order[n_qualify_per_group]
                thirds.append({"team": third, **stats[third]})

        if n_best_thirds > 0:
            qualifiers.extend(rank_best_thirds(thirds, n_best_thirds, rng))

        for t in qualifiers:
            counts[t]["qualify"] += 1

        ranked = _quality_rank(qualifiers, rp)
        bracket = seed_qualifiers(ranked)

        def decide(home, away):
            w = sample_knockout(rp.matrix(home, away), rng, rp.elo_diff(home, away))
            return home if w == 0 else away

        rounds = propagate_bracket(bracket, decide)
        # la ultima ronda es campeon; etiquetar desde el final hacia atras
        stage_labels = ["r16", "qf", "sf", "final", "champion"]
        for offset, label in enumerate(reversed(stage_labels)):
            idx = len(rounds) - 1 - offset
            if idx >= 0:
                for t in rounds[idx]:
                    counts[t][label] += 1

    rows = []
    for t in all_teams:
        row = {"team": t}
        for s in _STAGES:
            row[f"p_{s}"] = counts[t][s] / n_sims
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)


def _load_field_and_model():
    """Carga grupos+Elo de config y entrena el GBM con todo el historial."""
    import sqlite3

    import yaml

    from src.config import CONFIGS_DIR, DB_PATH
    from src.features.elo_features import attach_pre_match_elo
    from src.models.gbm_poisson import GBMPoissonModel

    field = yaml.safe_load((CONFIGS_DIR / "groups_2026.yaml").read_text(encoding="utf-8"))
    con = sqlite3.connect(DB_PATH)
    matches = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral, m.tournament
           FROM matches m JOIN teams t1 ON t1.team_id=m.home_team_id
           JOIN teams t2 ON t2.team_id=m.away_team_id
           WHERE m.date >= '2006-01-01' ORDER BY m.date""", con, parse_dates=["date"])
    elo = pd.read_sql("SELECT t.name AS team, e.date, e.elo FROM elo_history e "
                      "JOIN teams t ON t.team_id=e.team_id", con, parse_dates=["date"])
    con.close()
    matches = attach_pre_match_elo(matches, elo)
    model = GBMPoissonModel().fit(matches)
    return field, model


def main():
    import argparse

    from src.config import REPORTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    field, model = _load_field_and_model()
    rp = RateProvider(model, {t: float(e) for t, e in field["elos"].items()})
    res = simulate_tournament(field["groups"], rp, n_sims=args.n,
                              n_qualify_per_group=2, n_best_thirds=8, seed=args.seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f5_simulation.md"
    top = res.head(16).copy()
    for c in [c for c in res.columns if c.startswith("p_")]:
        top[c] = (top[c] * 100).round(1)
    lines = [
        f"# Simulacion Monte Carlo del Mundial 2026 ({args.n:,} corridas)",
        "", "CAMPO DE EJEMPLO sembrado por Elo (no es el sorteo oficial).",
        "Probabilidades en %. Top 16 por P(campeon).", "",
        top.to_markdown(index=False), "",
        f"**Campeon mas probable: {res.iloc[0]['team']} "
        f"({res.iloc[0]['p_champion']*100:.1f}%)**",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
