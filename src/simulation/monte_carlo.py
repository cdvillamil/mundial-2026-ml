"""Motor Monte Carlo del torneo: grupos + mejores terceros + eliminatorias."""
import itertools

import numpy as np
import pandas as pd

from src.simulation.group_stage import rank_best_thirds, rank_group
from src.simulation.knockout import propagate_bracket, seed_qualifiers
from src.simulation.match import knockout_from_cumsum, sample_from_cumsum
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
        cumsum, n = rp.cumsum(h, a)
        hg, ag = sample_from_cumsum(cumsum, n, rng)
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
            cumsum, n = rp.cumsum(home, away)
            w = knockout_from_cumsum(cumsum, n, rng, rp.elo_diff(home, away))
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


_STAGE_ORDER = {"qualify": 0, "r16": 1, "qf": 2, "sf": 3, "final": 4, "champion": 5}


def simulate_official_2026(groups: dict, rp: RateProvider, n_sims: int,
                           seed: int = 0) -> pd.DataFrame:
    """Simula el Mundial 2026 con el CUADRO OFICIAL FIFA (12 grupos, 8 terceros)."""
    from src.simulation.bracket_2026 import assign_thirds, evaluate_bracket

    rng = np.random.default_rng(seed)
    all_teams = [t for g in groups.values() for t in g]
    counts = {t: {s: 0 for s in _STAGES} for t in all_teams}

    def decide(home, away):
        cumsum, n = rp.cumsum(home, away)
        w = knockout_from_cumsum(cumsum, n, rng, rp.elo_diff(home, away),
                                 pen_p_home=rp.penalty_p_home(home, away))
        return home if w == 0 else away

    for _ in range(n_sims):
        winners, runners, thirds = {}, {}, []
        for letter, teams in groups.items():
            order, stats = _simulate_group(teams, rp, rng)
            winners[letter] = order[0]
            runners[letter] = order[1]
            third = order[2]
            thirds.append({"group": letter, "team": third, **stats[third]})

        rand = {d["group"]: rng.random() for d in thirds}
        thirds_sorted = sorted(
            thirds, key=lambda d: (-d["points"], -d["gd"], -d["gf"], rand[d["group"]]))
        best = thirds_sorted[:8]
        qual_groups = {d["group"] for d in best}
        third_team = {d["group"]: d["team"] for d in best}

        assign = assign_thirds(qual_groups)
        thirds_by_match = {mid: third_team[grp] for mid, grp in assign.items()}

        stages = evaluate_bracket(winners, runners, thirds_by_match, decide)
        for team, st in stages.items():
            reached = _STAGE_ORDER[st]
            for s in _STAGES:
                if reached >= _STAGE_ORDER[s]:
                    counts[team][s] += 1

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

    from src.models.penalties import build_from_db

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
    matches = attach_pre_match_elo(matches, elo)
    model = GBMPoissonModel().fit(matches)
    penalty_model = build_from_db(con)
    con.close()
    return field, model, penalty_model


def main():
    import argparse

    from src.config import REPORTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    field, model, penalty_model = _load_field_and_model()
    rp = RateProvider(model, {t: float(e) for t, e in field["elos"].items()},
                      penalty_model=penalty_model)
    official = bool(field.get("official"))
    if official:
        res = simulate_official_2026(field["groups"], rp, n_sims=args.n, seed=args.seed)
    else:
        res = simulate_tournament(field["groups"], rp, n_sims=args.n,
                                  n_qualify_per_group=2, n_best_thirds=8, seed=args.seed)

    # intervalo de confianza 95% de P(campeon): p +/- 1.96*sqrt(p(1-p)/N)
    p = res["p_champion"]
    se = np.sqrt(p * (1 - p) / args.n)
    res = res.assign(champ_lo=(p - 1.96 * se).clip(lower=0),
                     champ_hi=(p + 1.96 * se).clip(upper=1))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f5_simulation.md"
    top = res.head(16).copy()
    ci = top.apply(lambda r: f"{r['p_champion']*100:.1f} "
                             f"[{r['champ_lo']*100:.1f}-{r['champ_hi']*100:.1f}]", axis=1)
    for c in [c for c in res.columns if c.startswith("p_")]:
        top[c] = (top[c] * 100).round(1)
    top["campeon_% [IC95]"] = ci.values
    top = top.drop(columns=["champ_lo", "champ_hi", "p_champion"])
    fuente = ("SORTEO OFICIAL FIFA (grupos A-L) + cuadro oficial."
              if official else "CAMPO DE EJEMPLO sembrado por Elo.")
    lines = [
        f"# Simulacion Monte Carlo del Mundial 2026 ({args.n:,} corridas)",
        "", fuente, "Probabilidades en %. IC95 = intervalo de confianza por Monte Carlo.",
        "Top 16 por P(campeon).", "",
        top.to_markdown(index=False), "",
        f"**Campeon mas probable: {res.iloc[0]['team']} "
        f"({res.iloc[0]['p_champion']*100:.1f}%, IC95 "
        f"{res.iloc[0]['champ_lo']*100:.1f}-{res.iloc[0]['champ_hi']*100:.1f})**",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
