"""Utilidades para validacion historica sobre mundiales pasados."""
import sqlite3

import pandas as pd

# Mundiales objetivo: año -> (fecha de inicio, campeon real)
WORLD_CUPS = {
    2010: ("2010-06-11", "Spain"),
    2014: ("2014-06-12", "Germany"),
    2018: ("2018-06-14", "France"),
    2022: ("2022-11-20", "Argentina"),
}


def wc_matches(con: sqlite3.Connection, year: int) -> pd.DataFrame:
    """Partidos del Mundial 'year' con nombres de equipos y resultado."""
    df = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral,
                  ts.name AS shootout_winner
           FROM matches m
           JOIN teams t1 ON t1.team_id = m.home_team_id
           JOIN teams t2 ON t2.team_id = m.away_team_id
           LEFT JOIN teams ts ON ts.team_id = m.shootout_winner_id
           WHERE m.tournament = 'FIFA World Cup'
             AND substr(m.date, 1, 4) = ?
           ORDER BY m.date""",
        con, params=[str(year)], parse_dates=["date"])
    return df


def extract_groups(group_stage: pd.DataFrame, group_size: int = 4) -> dict:
    """Reconstruye grupos como componentes conexas del grafo 'jugaron entre si'.
    Espera SOLO partidos de fase de grupos (cada equipo juega group_size-1)."""
    adj: dict[str, set] = {}
    for m in group_stage.itertuples(index=False):
        adj.setdefault(m.home_team, set()).add(m.away_team)
        adj.setdefault(m.away_team, set()).add(m.home_team)

    seen = set()
    groups = {}
    gid = 0
    for team in adj:
        if team in seen:
            continue
        comp = set()
        stack = [team]
        while stack:
            t = stack.pop()
            if t in comp:
                continue
            comp.add(t)
            stack.extend(adj[t] - comp)
        seen |= comp
        groups[f"G{gid + 1}"] = sorted(comp)
        gid += 1
    return groups


def extract_field(con: sqlite3.Connection, year: int) -> dict:
    """Grupos reales (8 de 4) reconstruidos desde los primeros 48 partidos."""
    m = wc_matches(con, year)
    n_group_matches = 48  # formato 32 equipos: 8 grupos x 6 partidos
    group_stage = m.head(n_group_matches)
    return extract_groups(group_stage, group_size=4)


def champion_of(con: sqlite3.Connection, year: int) -> str:
    """Ganador de la final (ultimo partido del torneo; penales si empate)."""
    m = wc_matches(con, year)
    final = m.iloc[-1]
    if final["home_goals"] > final["away_goals"]:
        return final["home_team"]
    if final["away_goals"] > final["home_goals"]:
        return final["away_team"]
    return final["shootout_winner"]


def elos_before(con: sqlite3.Connection, date: str, teams: list[str]) -> dict:
    """Elo de cada equipo justo antes de 'date' (point-in-time)."""
    elo = pd.read_sql(
        """SELECT t.name AS team, e.elo, e.date FROM elo_history e
           JOIN teams t ON t.team_id = e.team_id
           WHERE e.date < ? AND t.name IN ({})
           ORDER BY e.date""".format(",".join(["?"] * len(teams))),
        con, params=[date] + teams)
    latest = elo.groupby("team")["elo"].last().to_dict()
    return {t: float(latest.get(t, 1500.0)) for t in teams}
